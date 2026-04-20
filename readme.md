# Handbook RAG Chatbot

A production-ready RAG chatbot for semantic search over GitLab handbook and direction pages using natural language queries.

Live App: https://handbook-rag-chatbot.streamlit.app

---

## App UI

![App Screenshot](assets/Screenshot.png)

---

## Features

- **Chat UI** - Clean interface with sidebar chat history
- **Semantic Search** - Natural language queries over handbook content
- **Context-Aware Answers** - Every response is backed by real source material, nothing is made up
- **Shareable Sessions** - Share chats via encoded URLs
- **Streaming** - Real-time streamed responses
- **Guardrails** - Safe, scoped query handling

---

## Architecture

The pipeline is straightforward. Your query hits the retriever, gets re-ranked by a CrossEncoder to surface the most relevant results, and that context gets injected into the prompt.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Hugging Face Inference API |
| Vector DB | ChromaDB |
| Embeddings | BGE-base |
| Retrieval | LangChain |

---

## Getting Started

### Prerequisites

- Python 3.11
- A [Hugging Face](https://huggingface.co) API token with read and write access

### Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_api_token
HF_TOKEN_WRITE=your_huggingface_write_token
REPO_ID=your_username/handbook-chat-history
FILE=chat_history.json
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/handbook-rag-chatbot.git
cd handbook-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Build the local vector store
python ingest.py

# Run the app
python run.py
```

---

## Deployment

This project is deployed on **Streamlit Cloud** with the following setup:

- Prebuilt vector database - no scraping at runtime, so cold starts are fast
- Environment secrets managed via the Streamlit Cloud dashboard
- Persistent chat history stored through Hugging Face datasets

---

## Project Structure

```
handbook-rag-chatbot/
├── run.py                  # Launches the Streamlit app
├── ingest.py               # Entry point for offline ingestion pipeline
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages
├── ingestion/              # Data ingestion pipeline (offline use only)
│   ├── checkpoint.py       # Crawl state tracking & recovery
│   ├── chunker.py          # Intelligent text chunking for embeddings
│   ├── crawler.py          # URL discovery & sitemap traversal
│   └── scraper.py          # HTML parsing & content extraction
├── rag/                    # Core RAG pipeline
│   ├── chain.py            # RAG orchestration (LLM + retrieval)
│   └── retriever.py        # Vector search + re-ranking
├── utils/                  # Shared utilities
│   ├── chat_store.py       # Chat history persistence (HF datasets)
│   └── logger.py           # Centralized logging
├── vectorstore/            # Vector database handling
│   ├── loader.py           # Loads prebuilt vectorDB
│   └── store.py            # Embedding storage & ingestion logic
├── config/
│   └── settings.py         # Central configuration (models, paths, env)
└── app/
    └── ui.py               # Streamlit UI (main entry point)
```

---

## Highlights

- **Shareable chats** - Sessions are encoded into URLs so conversations are easy to pass around
- **Persistent history** - Chat logs are stored via Hugging Face datasets, nothing is lost between sessions
- **Hybrid retrieval** - MMR combined with CrossEncoder re-ranking meaningfully improves result quality over plain similarity search
- **Modular architecture** - Clean separation of retrieval, ranking, and generation

---

