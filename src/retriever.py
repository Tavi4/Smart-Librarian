import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from database import load_book_summaries


# ---------------------------
# OpenAI and Chroma clients
# ---------------------------

def get_openai_client():
    """
    Create and return an OpenAI client using the new SDK.
    Requires OPENAI_API_KEY in the environment or .env file.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or api_key.strip() == "":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or your .env file."
        )

    client = OpenAI(api_key=api_key)
    return client


def get_chroma_client(persist_dir):
    """
    Create or open a persistent ChromaDB client in the given directory.
    """
    if persist_dir is None or str(persist_dir).strip() == "":
        raise ValueError("persist_dir must be a non-empty string path.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


# ---------------------------
# Helpers
# ---------------------------


def embed_texts(client, texts, model_name, batch_size=64):
    """
    Embed all texts in order, batching for safety. Returns a list of vectors.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings.")
    if len(texts) == 0:
        return []

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings

def cosine_score_from_distance(distance_value):
    """
    Convert Chroma cosine distance to a friendly similarity in [0, 1].
      cosine_similarity = 1 - cosine_distance  → range ~[-1,1]
      map [-1,1] → [0,1] via (sim+1)/2
    """
    d = float(distance_value)
    sim = 1.0 - d
    norm = (sim + 1.0) / 2.0
    if norm < 0.0:
        norm = 0.0
    if norm > 1.0:
        norm = 1.0
    return round(norm, 3)

# ---------------------------
# Index building
# ---------------------------

# --- Index building (cosine-only) ---

def build_index(json_path, persist_dir, collection_name, embed_model="text-embedding-3-small"):
    openai_client = get_openai_client()
    chroma_client = get_chroma_client(persist_dir)

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    books = load_book_summaries(json_path)
    if not books:
        raise RuntimeError("No books were loaded. Check the JSON file.")

    ids, documents, metadatas = [], [], []
    for idx, book in enumerate(books):
        ids.append(str(idx))
        documents.append(book.get("summary", ""))
        # Simple metadata: only title + themes as plain string
        title = book.get("title", f"Untitled #{idx}")
        themes = book.get("themes", [])
        metadatas.append({
            "title": title,
            "themes": ", ".join(themes)   # no lists
        })

    vectors = embed_texts(openai_client, documents, embed_model, batch_size=64)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    print(f"Indexed {len(ids)} books into '{collection_name}'.")


# ---------------------------
# Search
# ---------------------------


def search_books(query, persist_dir, collection_name, k=3, embed_model="text-embedding-3-small"):
    """
    Semantic search returning: {"title": str, "score": float [0..1], "themes": list[str]}
    """

    openai_client = get_openai_client()
    chroma_client = get_chroma_client(persist_dir)

    # ensure collection exists and uses cosine
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # embed query
    resp = openai_client.embeddings.create(model=embed_model, input=[query])
    qvec = resp.data[0].embedding

    # query chroma
    result = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["metadatas", "distances"]
    )

    items = []
    metadatas_list = result.get("metadatas", [[]])
    distances_list = result.get("distances", [[]])
    if not metadatas_list:
        return items

    metadatas = metadatas_list[0]
    distances = distances_list[0] if distances_list else []

    for i, meta in enumerate(metadatas):
        d = float(distances[i]) if i < len(distances) else 1.0
        score = cosine_score_from_distance(d)

        title = meta.get("title")
        themes = meta.get("themes")
        if isinstance(themes, str):
            try:
                themes = json.loads(themes)
            except Exception:
                themes = [themes]
        if themes is None:
            themes = []

        items.append({"title": title, "score": score, "themes": themes})

    return items
