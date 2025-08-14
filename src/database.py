from pathlib import Path
import json

def load_book_summaries(json_path : str):
    """
    Loader function for data/book_summaries.json into python.
    Returns a list of {title, summary, themes}.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    return data

if __name__ == "__main__":
    books = load_book_summaries("../data/book_summaries.json")
    print(f"Loaded {len(books)} books. Example titles:")
    print(books)
