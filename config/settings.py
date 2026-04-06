"""Central configuration for the GitLab Chatbot."""

import os
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

# Paths and file locations
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH     = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db"))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "ingest_checkpoint.json")

# Log file location
LOG_FILE        = os.getenv("LOG_FILE", os.path.join(BASE_DIR, "ingest.log"))

# URLs to scrape for content
SEED_URLS: list[str] = [
    "https://handbook.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
]

ALLOWED_URL_PREFIXES: list[str] = [
    "https://handbook.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
]

SITEMAP_URLS: list[str] = [
    "https://handbook.gitlab.com/sitemap.xml",
    "https://handbook.gitlab.com/sitemap_index.xml",
    "https://about.gitlab.com/sitemap.xml",
    "https://about.gitlab.com/sitemap_index.xml",
]

MAX_PAGES        = int(os.getenv("MAX_PAGES", 5000))
MAX_RETRIES      = 3
REQUEST_TIMEOUT  = 20
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", 0.8))
CHECKPOINT_EVERY = 10

SKIP_EXTENSIONS: tuple = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz",
    ".css", ".js", ".xml", ".json", ".ico",
    ".mp4", ".mp3", ".wav", ".woff", ".woff2", ".ttf",
)

SKIP_PATH_PATTERNS: list[str] = [
    r"/tags/", r"/categories/", r"/feed",
    r"/sitemap", r"/search", r"/404", r"\?",
]

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Configure Playwright for JavaScript-heavy pages
JS_RENDER_THRESHOLD      = int(os.getenv("JS_RENDER_THRESHOLD", 500))
PLAYWRIGHT_TIMEOUT_MS    = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", 20000))
PLAYWRIGHT_SCROLL_PAUSE_MS = int(os.getenv("PLAYWRIGHT_SCROLL_PAUSE_MS", 1000))
PLAYWRIGHT_BLOCKED_RESOURCES: list[str] = ["image", "media", "font", "stylesheet"]

# How to split up documents into chunks for embedding
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 900))

# How much the chunks should overlap to maintain context
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

MIN_CHUNK_LENGTH   = int(os.getenv("MIN_CHUNK_LENGTH", 60))
MIN_PAGE_LENGTH    = int(os.getenv("MIN_PAGE_LENGTH", 200))
MIN_SECTION_LENGTH = int(os.getenv("MIN_SECTION_LENGTH", 100))
MAX_HEADING_DEPTH  = int(os.getenv("MAX_HEADING_DEPTH", 3))
PREPEND_BREADCRUMB = os.getenv("PREPEND_BREADCRUMB", "true").lower() == "true"

# Which embedding model to use for converting text to vectors
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# How many documents to store and search in the vector database
BATCH_SIZE      = 100
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 5))

# Filter out chunks that aren't relevant enough
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", -3.0))

# Which cross-encoder model to use for ranking retrieved chunks
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# LLM settings
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "huggingface")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Limit response length to avoid rambling
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.2))

# Keep context within reasonable size for the model
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))

# Protect against really long user inputs
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", 500))

# App display settings
APP_TITLE    = "GitLab Handbook Chatbot"
APP_ICON     = "🦊"
APP_SUBTITLE = "Ask anything about GitLab's Handbook and Direction pages."
