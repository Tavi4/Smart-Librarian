import os
from pathlib import Path
from openai import OpenAI
import chromadb

def load_openai_api_key(path=".env"):
    """
    Load API key from environment or .env and ensures that the key is
    written correctly in case.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    raise ValueError("OPENAI_API_KEY not found in environment or .env")

def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client with key/project/org (used by retriever + app).
    """

    key = load_openai_api_key(".env")
    project = os.getenv("OPENAI_PROJECT")  # service account project (optional)
    org = os.getenv("OPENAI_ORG_ID")       # org id (optional)

    return OpenAI(
        api_key=key,
        project=project.strip() if project else None,
        organization=org.strip() if org else None,
    )

def get_chroma_client(persist_dir):
    """
    Create or open a persistent ChromaDB client in the given directory.
    """
    if persist_dir is None or str(persist_dir).strip() == "":
        raise ValueError("persist_dir must be a non-empty string path.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


