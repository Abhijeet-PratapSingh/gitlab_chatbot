"""Handles crawl state persistence and recovery."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.settings import CHECKPOINT_FILE
from utils.logger import get_logger

log = get_logger("checkpoint")

@dataclass
class Checkpoint:
    # URLs processed successfully
    visited_urls: set[str]     = field(default_factory=set)
    # URLs that failed after all retries
    failed_urls: set[str]      = field(default_factory=set)
    ingested_hashes: set[str]  = field(default_factory=set)
    # URLs discovered but not yet processed (queue)
    queued_urls: list[str]     = field(default_factory=list)

    # totals
    total_chunks: int          = 0
    total_pages: int           = 0

    # Timestamps
    started_at: str            = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str]  = None

    def save(self) -> None:
        """Save checkpoint to disk"""
        self.updated_at = datetime.now().isoformat()

        data = {
            "visited_urls":    list(self.visited_urls),
            "failed_urls":     list(self.failed_urls),
            "ingested_hashes": list(self.ingested_hashes),
            "queued_urls":     self.queued_urls,
            "total_chunks":    self.total_chunks,
            "total_pages":     self.total_pages,
            "started_at":      self.started_at,
            "updated_at":      self.updated_at,
        }

        checkpoint_dir = os.path.dirname(CHECKPOINT_FILE) or "."

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=checkpoint_dir,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, CHECKPOINT_FILE)

            log.debug(
                "Checkpoint saved successfully | "
                f"Visited: {len(self.visited_urls)}, "
                f"Queued: {len(self.queued_urls)}, "
                f"Chunks: {self.total_chunks}"
            )

        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if rename failed
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def load(cls) -> "Checkpoint":
        """Load checkpoint from disk."""
        if not os.path.exists(CHECKPOINT_FILE):
            log.info("No checkpoint found - starting fresh crawl")
            return cls()

        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            cp = cls(
                visited_urls    = set(data.get("visited_urls", [])),
                failed_urls     = set(data.get("failed_urls", [])),
                ingested_hashes = set(data.get("ingested_hashes", [])),
                queued_urls     = data.get("queued_urls", []),
                total_chunks    = data.get("total_chunks", 0),
                total_pages     = data.get("total_pages", 0),
                started_at      = data.get("started_at", datetime.now().isoformat()),
                updated_at      = data.get("updated_at"),
            )

            log.info("Checkpoint loaded - Resuming crawl from last saved state.")
            cp.print_summary()
            return cp

        except json.JSONDecodeError as e:
            log.warning(f"Checkpoint file is corrupt (JSON error): {e}")
            log.warning("Starting fresh - previous progress is lost")
            return cls()

        except Exception as e:
            log.warning(f"Could not load checkpoint: {e}")
            log.warning("Starting fresh")
            return cls()

    @classmethod
    def reset(cls) -> "Checkpoint":
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            log.info(f"Checkpoint deleted: {CHECKPOINT_FILE}")
        else:
            log.info("No checkpoint file to delete")
        return cls()

    def is_visited(self, url: str) -> bool:
        return url in self.visited_urls

    def is_failed(self, url: str) -> bool:
        return url in self.failed_urls

    def is_ingested(self, content_hash: str) -> bool:
        return content_hash in self.ingested_hashes

    def mark_visited(self, url: str) -> None:
        self.visited_urls.add(url)

    def mark_failed(self, url: str) -> None:
        self.failed_urls.add(url)
        self.visited_urls.add(url)     

    def mark_ingested(self, content_hash: str) -> None:
        self.ingested_hashes.add(content_hash)

    def print_summary(self) -> None:
        log.info("-" * 50)
        log.info(f"  Started at   : {self.started_at}")
        log.info(f"  Last updated : {self.updated_at or 'N/A'}")
        log.info(f"  Pages done   : {self.total_pages}")
        log.info(f"  Chunks stored: {self.total_chunks}")
        log.info(f"  URLs visited : {len(self.visited_urls)}")
        log.info(f"  URLs queued  : {len(self.queued_urls)}")
        log.info(f"  URLs failed  : {len(self.failed_urls)}")
        log.info("-" * 50)