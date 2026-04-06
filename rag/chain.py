import re
from dataclasses import dataclass
from typing import Generator, Optional

from config.settings import (
    HF_MODEL,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    MAX_CONTEXT_CHARS,
    MAX_QUERY_CHARS,
    TEMPERATURE,
)
from rag.retriever import Retriever, RetrievedChunk
from utils.logger import get_logger

log = get_logger("chain")

@dataclass
class RAGResponse:
    answer: str
    chunks: list[RetrievedChunk]
    query: str
    guardrail_triggered: bool = False
    guardrail_reason: str = ""

    def citations(self) -> list[str]:
        seen, result = set(), []
        for chunk in self.chunks:
            c = chunk.citation()
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?instructions?|"
    r"you\s+are\s+now|"
    r"disregard\s+(the\s+)?(above|previous|context|system)|"
    r"forget\s+(all\s+)?instructions?|"
    r"jailbreak|DAN\b|system\s*:\s*|<\s*/?system\s*>)",
    re.IGNORECASE,
)


def _sanitise_query(query: str) -> str:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"Query too long ({len(query)} chars). Max is {MAX_QUERY_CHARS}."
        )
    if _INJECTION_RE.search(query):
        raise ValueError("Query contains a disallowed pattern.")
    return query.strip()

_OUT_OF_SCOPE_RE = re.compile(
    r"\b(recipe|cook|food|weather|sport|movie|music|celebrity|"
    r"stock|crypto|bitcoin|investment|trading|"
    r"medical|diagnosis|symptom|treatment|drug|medicine|"
    r"legal advice|lawsuit|sue|attorney|"
    r"political|election|vote|president|parliament)\b",
    re.IGNORECASE,
)


def _is_out_of_scope(query: str) -> tuple[bool, str]:
    match = _OUT_OF_SCOPE_RE.search(query)
    if match:
        return True, (
            f"This question is outside the scope of the GitLab Handbook "
            f"(matched: '{match.group()}')."
        )
    return False, ""

_SYSTEM_CONTENT = (
    "You are a helpful assistant for GitLab employees and candidates. "
    "Answer questions strictly based on the GitLab Handbook context provided. "
    "If the context does not contain the answer, say so clearly. "
    "Never make up facts. Be concise and use markdown formatting where helpful."
)


def _build_messages(query: str, context: str) -> list[dict]:
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"
        log.warning(f"Context truncated to {MAX_CONTEXT_CHARS} chars")

    user_content = (
        f"Context from the GitLab Handbook:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system",  "content": _SYSTEM_CONTENT},
        {"role": "user",    "content": user_content},
    ]

def _load_client():
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError("Run: pip install huggingface-hub")

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set in your .env file.\n"
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    log.info(f"HuggingFace InferenceClient ready - model: {HF_MODEL}")
    return client


def _generate(client, messages: list[dict]) -> str:
    response = client.chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def _stream_tokens(client, messages: list[dict]) -> Generator[str, None, None]:
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    ):
        try:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

        except Exception as e:
            log.debug(f"Skipping malformed chunk: {e}")
            continue

class RAGChain:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._client   = _load_client()

    def stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_type: str = "mmr",
    ) -> Generator[tuple[Optional[str], Optional[list]], None, None]:
        try:
            query = _sanitise_query(query)
        except ValueError as e:
            yield str(e), None
            yield None, []
            return

        out_of_scope, reason = _is_out_of_scope(query)
        if out_of_scope:
            yield (
                f"This question is outside the scope of the "
                f"GitLab Handbook.\n\n{reason}"
            ), None
            yield None, []
            return

        context, chunks = self.retriever.retrieve_as_context(
            query, search_type=search_type, top_k=top_k
        )
        if not chunks:
            yield (
                "I could not find relevant information in the GitLab Handbook "
                "for your question. Please try rephrasing."
            ), None
            yield None, []
            return

        messages = _build_messages(query, context)
        log.info(f"Streaming — '{query[:60]}'")
        streamed_any = False
        try:
            for token in _stream_tokens(self._client, messages):
                streamed_any = True
                yield token, None

        except Exception as e:
            log.warning(f"Streaming failed: {e} - falling back to non-streaming")

            if not streamed_any:
                try:
                    answer = _generate(self._client, messages)
                    yield answer, None
                except Exception as e2:
                    log.error(f"Fallback also failed: {e2}")
                    yield (
                        f"Could not generate a response.\n\n"
                        f"**Error**: {e2}\n\n"
                        f"Ensure your `HF_TOKEN` is valid and `{HF_MODEL}` "
                        f"is accessible on the free tier."
                    ), None
            else:
                yield f"\n\nStream interrupted: {e}", None

        # Sentinel
        yield None, chunks

    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_type: str = "mmr",
    ) -> RAGResponse:
        try:
            query = _sanitise_query(query)
        except ValueError as e:
            return RAGResponse(
                answer=str(e), chunks=[], query=query,
                guardrail_triggered=True, guardrail_reason=str(e),
            )

        out_of_scope, reason = _is_out_of_scope(query)
        if out_of_scope:
            return RAGResponse(
                answer=f"Outside scope: {reason}",
                chunks=[], query=query,
                guardrail_triggered=True, guardrail_reason=reason,
            )

        context, chunks = self.retriever.retrieve_as_context(
            query, search_type=search_type, top_k=top_k
        )
        if not chunks:
            return RAGResponse(
                answer="Could not find relevant information. Please try rephrasing.",
                chunks=[], query=query,
            )

        messages = _build_messages(query, context)
        answer   = _generate(self._client, messages)
        return RAGResponse(answer=answer, chunks=chunks, query=query)