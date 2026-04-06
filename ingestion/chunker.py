import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import nltk

for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.tokenize import sent_tokenize

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    MIN_SECTION_LENGTH,
    MAX_HEADING_DEPTH,
    PREPEND_BREADCRUMB,
)
from utils.logger import get_logger

log = get_logger("chunker")

BlockType = Literal["text", "table", "code", "list"]


@dataclass
class Block:
    content: str
    block_type: BlockType


@dataclass
class Section:
    title: str
    level: int              
    blocks: list[Block] = field(default_factory=list)
    breadcrumb: list[str] = field(default_factory=list)

    @property
    def full_body(self) -> str:
        return "\n\n".join(b.content for b in self.blocks)

    @property
    def breadcrumb_prefix(self) -> str:
        if PREPEND_BREADCRUMB and self.breadcrumb:
            return f"[{' > '.join(self.breadcrumb)}]\n\n"
        return ""

_CODE_RE  = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
_TABLE_RE = re.compile(r"(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n)*)", re.MULTILINE)
_LIST_RE  = re.compile(
    r"((?:^[ \t]*(?:[-*+]|\d+\.)[ \t]+.+\n?)+)",
    re.MULTILINE,
)

_PLACEHOLDER_TPL = "__BLOCK_{index}__"
_PLACEHOLDER_RE  = re.compile(r"__BLOCK_(\d+)__")


def _extract_special_blocks(text: str) -> tuple[str, list[Block]]:
    blocks: list[Block] = []
    result = text

    def replace(match: re.Match, block_type: BlockType) -> str:
        idx = len(blocks)
        blocks.append(Block(content=match.group(0).strip(), block_type=block_type))
        return f"\n{_PLACEHOLDER_TPL.format(index=idx)}\n"

    result = _CODE_RE.sub(lambda m: replace(m, "code"), result)
    result = _TABLE_RE.sub(lambda m: replace(m, "table"), result)
    result = _LIST_RE.sub(lambda m: replace(m, "list"), result)

    return result, blocks


def _restore_block(placeholder: str, blocks: list[Block]) -> Block | None:
    match = _PLACEHOLDER_RE.match(placeholder.strip())
    if match:
        idx = int(match.group(1))
        if idx < len(blocks):
            return blocks[idx]
    return None

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _parse_sections(markdown: str) -> list[Section]:
    matches = [
        m for m in _HEADING_RE.finditer(markdown)
        if len(m.group(1)) <= MAX_HEADING_DEPTH
    ]
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []

    def make_blocks(raw_body: str) -> list[Block]:
        cleaned, specials = _extract_special_blocks(raw_body.strip())
        text_blocks: list[Block] = []
        for segment in re.split(r"\n{2,}", cleaned):
            s = segment.strip()
            if not s:
                continue
            block = _restore_block(s, specials)
            text_blocks.append(
                block if block else Block(content=s, block_type="text")
            )
        return text_blocks

    if not matches:
        blocks = make_blocks(markdown)
        return [Section(title="", level=0, blocks=blocks, breadcrumb=[])]

    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        blocks = make_blocks(preamble)
        sections.append(Section(title="Preamble", level=0, blocks=blocks, breadcrumb=[]))

    for i, match in enumerate(matches):
        level      = len(match.group(1))
        title      = match.group(2).strip()
        body_start = match.end()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        raw_body   = markdown[body_start:next_start]

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        breadcrumb = [h[1] for h in heading_stack] + [title]
        heading_stack.append((level, title))

        blocks = make_blocks(raw_body)
        if blocks:
            sections.append(Section(
                title=title,
                level=level,
                blocks=blocks,
                breadcrumb=breadcrumb,
            ))

    return sections

def _merge_short_sections(sections: list[Section]) -> list[Section]:
    """
    Merge sections shorter than MIN_SECTION_LENGTH into the next sibling.
    Prevents useless stub chunks (e.g. a heading with one line of body).
    """
    if not sections:
        return sections

    merged: list[Section] = []
    buffer: Section | None = None

    for section in sections:
        if buffer is None:
            buffer = section
            continue

        if len(buffer.full_body.strip()) < MIN_SECTION_LENGTH:
            buffer = Section(
                title=section.title,
                level=section.level,
                blocks=buffer.blocks + section.blocks,
                breadcrumb=section.breadcrumb,
            )
        else:
            merged.append(buffer)
            buffer = section

    if buffer:
        merged.append(buffer)

    return merged

def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _sentence_aware_split(text: str, prefix: str) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return [prefix + text] if text.strip() else []

    chunks: list[str]        = []
    current: list[str]       = []
    current_len              = len(prefix)

    for sentence in sentences:
        sentence_len = len(sentence) + 1    # +1 for joining space

        if current_len + sentence_len > CHUNK_SIZE and current:
            chunks.append(prefix + " ".join(current))

            # Sentence-level overlap — carry trailing sentences that fit
            overlap: list[str] = []
            overlap_len        = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 <= CHUNK_OVERLAP:
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current     = overlap
            current_len = len(prefix) + overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(prefix + " ".join(current))

    return chunks

def _section_to_chunks(section: Section) -> list[tuple[str, BlockType]]:
    prefix  = section.breadcrumb_prefix
    results: list[tuple[str, BlockType]] = []

    for block in section.blocks:
        if block.block_type in ("table", "code", "list"):
            chunk_text = (prefix + block.content).strip()
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                results.append((chunk_text, block.block_type))
        else:
            full = prefix + block.content
            if len(full) <= CHUNK_SIZE:
                chunk_text = full.strip()
                if len(chunk_text) >= MIN_CHUNK_LENGTH:
                    results.append((chunk_text, "text"))
            else:
                for sub in _sentence_aware_split(block.content, prefix):
                    sub = sub.strip()
                    if len(sub) >= MIN_CHUNK_LENGTH:
                        results.append((sub, "text"))

    return results

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_metadata(
    chunk_text: str,
    source_url: str,
    section: Section,
    block_type: BlockType,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    return {
        "source":        source_url,
        "section_title": section.title,
        "breadcrumb":    " > ".join(section.breadcrumb),
        "heading_level": section.level,
        "block_type":    block_type,
        "chunk_index":   chunk_index,
        "total_chunks":  total_chunks,
        "chunk_hash":    content_hash(chunk_text),
        "ingested_at":   datetime.now().isoformat(),
    }

def chunk(text: str, source_url: str) -> tuple[list[str], list[dict]]:
    """
    Intelligently chunk a markdown document into contextually rich pieces.

    Pipeline:
      1. _parse_sections()        — heading tree + special block extraction
      2. _merge_short_sections()  — merge stub sections into siblings
      3. _section_to_chunks()     — atomic tables/code/lists; Punkt-split text
      4. _make_metadata()         — enrich every chunk with full context info

    Returns (chunks, metadatas) — parallel lists of equal length.
    """
    sections = _parse_sections(text)
    sections = _merge_short_sections(sections)

    all_chunks: list[str]  = []
    all_metas:  list[dict] = []

    for section in sections:
        raw_chunks = _section_to_chunks(section)
        total      = len(raw_chunks)

        for idx, (chunk_text, block_type) in enumerate(raw_chunks):
            meta = _make_metadata(
                chunk_text   = chunk_text,
                source_url   = source_url,
                section      = section,
                block_type   = block_type,
                chunk_index  = idx,
                total_chunks = total,
            )
            all_chunks.append(chunk_text)
            all_metas.append(meta)

    log.debug(
        f"Chunked '{source_url}' - "
        f"{len(sections)} sections - {len(all_chunks)} chunks"
    )
    return all_chunks, all_metas