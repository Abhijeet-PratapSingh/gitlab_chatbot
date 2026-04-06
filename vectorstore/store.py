import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
    BATCH_SIZE,
    CHROMA_PATH,
    EMBED_MODEL,
)
from ingestion.checkpoint import Checkpoint
from utils.logger import get_logger

log = get_logger("vectorstore")

def load_embeddings() -> HuggingFaceEmbeddings:
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("Embedding model loaded")
    return embeddings

def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    log.info(f"Loading ChromaDB: {CHROMA_PATH}")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="gitlab_handbook",
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = _collection_count(vectorstore)
    log.info(f"ChromaDB ready - {count} chunks in collection")
    return vectorstore


def _collection_count(vectorstore: Chroma) -> int:
    try:
        result = vectorstore.get(include=[])
        return len(result.get("ids", []))
    except Exception as e:
        log.debug(f"Collection count failed: {e}")
        return -1

def store_chunks(
    vectorstore: Chroma,
    chunks: list[str],
    metadatas: list[dict],
    checkpoint: Checkpoint,
) -> int:
    new_chunks: list[str]  = []
    new_metas:  list[dict] = []

    for chunk, meta in zip(chunks, metadatas):
        if not checkpoint.is_ingested(meta["chunk_hash"]):
            new_chunks.append(chunk)
            new_metas.append(meta)

    if not new_chunks:
        log.debug("All chunks already ingested - skipping")
        return 0

    total_written = 0
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch_chunks = new_chunks[i : i + BATCH_SIZE]
        batch_metas  = new_metas[i : i + BATCH_SIZE]

        try:
            vectorstore.add_texts(texts=batch_chunks, metadatas=batch_metas)
            for meta in batch_metas:
                checkpoint.mark_ingested(meta["chunk_hash"])
            total_written += len(batch_chunks)
            log.debug(f"Wrote batch {i // BATCH_SIZE + 1} — {len(batch_chunks)} chunks")
        except Exception as e:
            log.error(f"Failed to write batch at index {i}: {e}")

    log.debug(f"Stored {total_written}/{len(new_chunks)} new chunks")
    return total_written

def health_check(vectorstore: Chroma) -> dict:
    try:
        count  = _collection_count(vectorstore)
        status = "ok" if count > 0 else "empty"
        result = {
            "status":      status,
            "chunk_count": count,
            "chroma_path": CHROMA_PATH,
            "embed_model": EMBED_MODEL,
        }
        if status == "empty":
            log.warning("ChromaDB is empty - run ingest.py first")
        else:
            log.info(f"Health check OK - {count} chunks available")
        return result
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e), "chroma_path": CHROMA_PATH}