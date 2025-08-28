# Smart Librarian

## Overview
Smart Librarian is a small project to explore how an AI chatbot can recommend books using Retrieval Augmented Generation (RAG) with ChromaDB, plus tool-calling for fetching exact summaries from a local dataset.

The goal was:
1. Build a dataset of book summaries.  
2. Index them in a vector store (ChromaDB) with embeddings.  
3. Use GPT to answer user questions and call tools to fetch verbatim summaries.  
4. Provide a simple CLI interface.

---

## Steps Taken

### 1. Dataset
- Created `book_summaries.json` with 10+ books, each with:
  - `title`
  - `summary`
  - `themes`
- Implemented `database.py` to load the data.

### 2. Tools
- Built two key functions in `tools.py`:
  - `get_summary_by_title(title)` → returns the exact stored summary.  
  - `match_title(query)` → fuzzy matching for user typos.  
- Added `tool_schemas.py` to expose these functions for GPT tool-calling.

### 3. Retrieval (RAG)
- Wrote `retriever.py`:
  - `build_index` → embeds summaries and stores them in ChromaDB.  
  - `search_books` → semantic search returning top matches with scores.  
- Works conceptually, but requires API calls for embeddings.

### 4. Client Utilities
- Wrote a simplified `get_openai_client()` in `utils.py`.  
- Added `.env` support for loading API keys.  
- This makes it easy to swap between environments.

### 5. CLI Version
- Implemented `cli.py`:
  - Checks direct title matches first.  
  - Falls back to semantic search (RAG) if no match.  
  - Always shows verbatim summaries from the JSON.  
- Inline comments explain each step.

---

## Current Status
- Working now (no API required):
  - Dataset loading.  
  - Title search (exact + fuzzy).  
  - CLI runs in interactive mode.  

- Blocked (API required):
  - Semantic search (embeddings → Chroma).  
  - GPT tool-calling.  
  - Full RAG + reasoning loop.  

---

## Reflection
- Even without API access, there is have a solid offline backbone:
  - A dataset loader.  
  - Tools that guarantee summaries are verbatim.  
  - A working CLI.  

- The code is clean, modular, and well-commented — so once API access is restored, the full RAG flow will work with minimal changes.  

---

## Next Steps (when API available)
1. Generate embeddings for the dataset with `build_index`.  
2. Run full RAG + tool-calling flows in the CLI.  
3. Add optional extras (text-to-speech, filters, image generation).  
