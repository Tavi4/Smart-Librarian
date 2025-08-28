import json
import os
import difflib

JSON_FILE = "../data/book_summaries.json"  # keep it simple

def _load_books():
    """
    Load raw list of books from JSON file (base loader for tools).
    """
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def get_summary_by_title(title: str) -> str:
    """
    Return verbatim summary for a book title (tool used after GPT picks a title).
    """
    if not isinstance(title, str) or not title.strip():
        return ""
    want = title.strip().lower()
    for b in _load_books():
        if b.get("title", "").strip().lower() == want:
            return b.get("summary", "")
    return ""

def _all_titles():
    """
    Collect all book titles from the dataset (used for fuzzy matching).
    """
    return [b.get("title", "").strip() for b in _load_books() if b.get("title")]

def match_title(query: str) -> str:
    """
    Fuzzy title helper: given a possibly misspelled title, return the best title or "".
    Uses difflib (stdlib) so we avoid extra deps.
    """
    if not isinstance(query, str) or not query.strip():
        return ""
    titles = _all_titles()
    best = difflib.get_close_matches(query.strip(), titles, n=1, cutoff=0.6)
    return best[0] if best else ""
