import argparse
import json
from typing import List

from utils import get_openai_client
from retriever import search_books
from tools import get_summary_by_title, match_title


PERSIST_DIR = "./data/embeddings"
COLLECTION_NAME = "books_summaries"


def _pick_from_titles(user_query: str, titles: List[str], client) -> str:
    """
    Ask GPT to pick exactly one title from the provided list.
    Returns the chosen title and its summary if possible.
    """
    if not titles:
        return ""
    # Prepare system + user messages (restrict GPT to only the given titles)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Smart Librarian. Choose exactly ONE title from the provided list "
                "that best fits the user's request. Do NOT invent titles. "
                "Respond with the chosen title on the first line, then output it's respective summary found in book_summaries.json"
            ),
        },
        {
            "role": "user",
            "content": f"User request: {user_query}\nTitles: {json.dumps(titles, ensure_ascii=False)}",
        },
    ]

    # Ask GPT to make the choice
    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    ).choices[0].message.content or ""

    # heuristic: pick the first title that appears in reply; fallback to top-1
    chosen = next((t for t in titles if t.lower() in reply.lower()), titles[0])

    # Print the recommendation and GPT’s reasoning
    print("\n=== Recommendation (RAG) ===")
    print(chosen)
    if reply.strip():
        print("\nReasoning:")
        print(reply.strip())
    return chosen


def handle_query(user_query: str):
    """
    Handle a user query end-to-end:
    - Try to resolve as a direct title (tool-only path).
    - If not found, run semantic search (RAG) + GPT to pick the best title.
    - Always fetch and display the verbatim summary from JSON.
    """

    client = get_openai_client()

    # Try title-based path first
    fixed_title = match_title(user_query) or user_query  # try fuzzy; fall back to raw
    summary = get_summary_by_title(fixed_title)
    if summary:
        print("\n=== Title Match ===")
        print(fixed_title)
        print("\n=== Summary (verbatim) ===")
        print(summary)
        return

    # If no title was found then we use RAG semantic search
    print("No direct title match. Running semantic search (RAG)…")
    candidates = search_books(user_query, PERSIST_DIR, COLLECTION_NAME, k=3)
    titles = [c.get("title") for c in candidates if c.get("title")]
    if not titles:
        print("No results found in your local library.")
        return

    # Constrain GPT to only pick one of the retrieved titles
    chosen = _pick_from_titles(user_query, titles, client)
    if not chosen:
        print("Could not select a title from candidates.")
        return

    #  Always show the verbatim summary from your JSON
    summary = get_summary_by_title(chosen)
    if summary:
        print("\n=== Summary (verbatim) ===")
        print(summary)
    else:
        print("Summary not found for the chosen title in your JSON.")


def main():
    """
    Entry point for CLI:
    - Parse arguments or prompt user input.
    - Pass the query to handle_query().
    """

    parser = argparse.ArgumentParser(description="Smart Librarian (CLI)")
    parser.add_argument(
        "query",
        nargs="*",
        help="Your request (title or theme). If omitted, you'll be prompted.",
    )
    args = parser.parse_args()

    # Either get query or prompt
    if args.query:
        user_query = " ".join(args.query).strip()
    else:
        user_query = input("Ask for a book (title or theme): ").strip()

    if not user_query:
        print("Please provide a non-empty query.")
        return

    handle_query(user_query)


if __name__ == "__main__":
    main()
