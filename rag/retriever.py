from dataclasses import dataclass
from typing import Optional

from langchain_community.vectorstores import Chroma

from config.settings import (
    RETRIEVER_TOP_K,
    MIN_RERANK_SCORE,
    CROSS_ENCODER_MODEL,
)
from utils.logger import get_logger

log = get_logger("retriever")

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
    log.debug("CrossEncoder available - re-ranking enabled")
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False
    log.info("sentence-transformers CrossEncoder not found - re-ranking disabled")

@dataclass
class RetrievedChunk:
    content: str
    source: str
    section_title: str
    breadcrumb: str
    block_type: str
    score: float
    chunk_index: int
    total_chunks: int

    def citation(self) -> str:
        parts = []
        if self.breadcrumb:
            parts.append(self.breadcrumb)
        if self.source:
            parts.append(self.source)
        return " | ".join(parts) if parts else "GitLab Handbook"

class Retriever:
    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = RETRIEVER_TOP_K,
    ):
        self.vectorstore   = vectorstore
        self.top_k         = top_k  
        self._cross_encoder: Optional[object] = None
        self._cross_encoder_loaded = False

    def _load_cross_encoder(self) -> None:
        if self._cross_encoder_loaded:
            return
        if not _CROSS_ENCODER_AVAILABLE:
            self._cross_encoder_loaded = True
            return
        try:
            log.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
            log.info("Cross-encoder loaded")
        except Exception as e:
            log.warning(f"Could not load cross-encoder: {e}. Re-ranking disabled.")
            self._cross_encoder = None
        self._cross_encoder_loaded = True

    def _stage1_recall(self, query: str, search_type: str, candidate_k: int) -> list:
        try:
            if search_type == "mmr":
                docs = self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=candidate_k,
                    fetch_k=candidate_k * 4,
                    lambda_mult=0.6,
                )
            else:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=candidate_k,
                )
            log.debug(f"Stage 1 [{search_type}]: recalled {len(docs)} candidates")
            return docs
        except Exception as e:
            log.error(f"Stage 1 recall failed: {e}")
            return []

    def _stage2_rerank(self, query: str, docs: list) -> list[tuple[object, float]]:
        if not self._cross_encoder:
            return [(doc, 0.0) for doc in docs]
        try:
            pairs  = [(query, doc.page_content) for doc in docs]
            scores = self._cross_encoder.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            log.debug(
                f"Stage 2: re-ranked {len(docs)} - "
                f"top score: {ranked[0][1]:.3f}"
            )
            return ranked
        except Exception as e:
            log.warning(f"Re-ranking failed: {e}. Using Stage 1 order.")
            return [(doc, 0.0) for doc in docs]

    def retrieve(
        self,
        query: str,
        search_type: str = "mmr",
        top_k: Optional[int] = None,          
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        effective_top_k = top_k if top_k is not None else self.top_k
        candidate_k     = effective_top_k * 3

        self._load_cross_encoder()

        candidates = self._stage1_recall(query, search_type, candidate_k)
        if not candidates:
            log.warning("No candidates retrieved - ChromaDB may be empty")
            return []

        ranked = self._stage2_rerank(query, candidates)

        results: list[RetrievedChunk] = []
        for doc, score in ranked[:effective_top_k]:
            if self._cross_encoder and float(score) < MIN_RERANK_SCORE:
                log.debug(
                    f"Dropping low-score chunk "
                    f"(score={score:.3f} < threshold={MIN_RERANK_SCORE})"
                )
                continue

            meta = doc.metadata or {}
            results.append(RetrievedChunk(
                content       = doc.page_content,
                source        = meta.get("source", ""),
                section_title = meta.get("section_title", ""),
                breadcrumb    = meta.get("breadcrumb", ""),
                block_type    = meta.get("block_type", "text"),
                score         = float(score),
                chunk_index   = meta.get("chunk_index", 0),
                total_chunks  = meta.get("total_chunks", 1),
            ))

        log.info(
            f"Retrieved {len(results)} chunks [{search_type}] "
            f"(dropped {effective_top_k - len(results)} below threshold) — "
            f"'{query[:60]}{'...' if len(query) > 60 else ''}'"
        )
        return results

    def retrieve_as_context(
        self,
        query: str,
        search_type: str = "mmr",
        top_k: Optional[int] = None,          
    ) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retrieve(query, search_type=search_type, top_k=top_k)
        if not chunks:
            return "", []

        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}]"
            if chunk.breadcrumb:
                header += f" {chunk.breadcrumb}"
            context_parts.append(f"{header}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)
        return context, chunks