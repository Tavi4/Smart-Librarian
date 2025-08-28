import json
from database import load_book_summaries
from utils import get_chroma_client
from utils import get_openai_client


def embed_texts(client, texts, model_name, batch_size=64):
    """
    Create embeddings for a list of texts (used to build/search the RAG index).
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
    Build a ChromaDB index with book summaries preparing data for RAG search
    """
    d = float(distance_value)
    sim = 1.0 - d
    norm = (sim + 1.0) / 2.0
    if norm < 0.0:
        norm = 0.0
    if norm > 1.0:
        norm = 1.0
    return round(norm, 3)

# Index building using cosine

def build_index(json_path, persist_dir, collection_name, embed_model="text-embedding-3-small"):
    """
    Build a ChromaDB index from local summaries - preparing data for RAG).
    """

    openai_client = get_openai_client()
    chroma_client = get_chroma_client(persist_dir)

    # Get or create a collection in Chroma for cosine indexing
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Loading books from JSON
    books = load_book_summaries(json_path)
    if not books:
        raise RuntimeError("No books were loaded. Check the JSON file.")

    # Prepare ids, documents (summaries), and metadata (title + themes)
    ids, documents, metadatas = [], [], []
    for idx, book in enumerate(books):
        ids.append(str(idx))
        documents.append(book.get("summary", ""))
        title = book.get("title", f"Untitled #{idx}")
        themes = book.get("themes", [])
        metadatas.append({
            "title": title,
            "themes": ", ".join(themes)
        })

    # Embed summaries into vectors
    vectors = embed_texts(openai_client, documents, embed_model, batch_size=64)

    # Insert all data into the Chroma collection
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    print(f"Indexed {len(ids)} books into '{collection_name}'.")


# Searching books by the title

def search_books(query, persist_dir, collection_name, k=3, embed_model="text-embedding-3-small"):
    """
    Semantic search in Chroma index (RAG entry point).
    """
    openai_client = get_openai_client()
    chroma_client = get_chroma_client(persist_dir)

    # Ensure the target collection exists
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Embed the user query
    resp = openai_client.embeddings.create(model=embed_model, input=[query])
    qvec = resp.data[0].embedding

    # Query Chroma with the query embedding (top-k results)
    result = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["metadatas", "distances"]
    )

    # Parse results: collect metadatas and distances
    items = []
    metadatas_list = result.get("metadatas", [[]])
    distances_list = result.get("distances", [[]])
    if not metadatas_list:
        return items

    metadatas = metadatas_list[0]
    distances = distances_list[0] if distances_list else []

    # Build structured results - title, normalized score, themes
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
