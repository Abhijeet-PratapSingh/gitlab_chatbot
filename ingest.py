import argparse
import signal
import sys
import time
from collections import deque

from tqdm import tqdm

from config.settings import (
    MIN_PAGE_LENGTH,
    MAX_PAGES,
    RATE_LIMIT_DELAY,
    CHECKPOINT_EVERY,
)
from ingestion.checkpoint import Checkpoint
from ingestion.crawler import build_queue_from_sitemaps, discover_links, normalise
from ingestion.scraper import scrape, close_browser
from ingestion.chunker import chunk, content_hash
from vectorstore.store import load_embeddings, load_vectorstore, store_chunks
from utils.logger import get_logger

log = get_logger("ingest")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GitLab Handbook ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py              Full crawl (resumes if interrupted)
  python ingest.py --reset      Wipe checkpoint, start completely fresh
  python ingest.py --test       50-page test run
  python ingest.py --stats      Show checkpoint stats and exit
        """,
    )
    parser.add_argument("--reset",  action="store_true", help="Fresh crawl from scratch")
    parser.add_argument("--test",   action="store_true", help="Limit to 50 pages")
    parser.add_argument("--stats",  action="store_true", help="Show stats and exit")
    return parser.parse_args()

_checkpoint_ref: Checkpoint | None = None
_queue_ref:      deque | None      = None


def _handle_sigint(sig, frame):
    log.info("\nInterrupted - saving checkpoint before exit...")
    if _checkpoint_ref is not None and _queue_ref is not None:
        _checkpoint_ref.queued_urls = list(_queue_ref)
        _checkpoint_ref.save()
        log.info("Checkpoint saved. Resume with: python ingest.py")
    close_browser()  
    sys.exit(0)

def _crawl(
    queue: deque,
    checkpoint: Checkpoint,
    vectorstore,
    page_limit: int,
) -> dict:
    global _queue_ref
    _queue_ref = queue 

    stats = {"success": 0, "skipped": 0, "failed": 0, "new_chunks": 0}

    with tqdm(desc="Crawling", unit="page", dynamic_ncols=True) as pbar:
        while queue and checkpoint.total_pages < page_limit:
            url = queue.popleft()

            if checkpoint.is_visited(url):
                continue

            pbar.set_description(
                f"Crawling [{checkpoint.total_pages + 1}/{page_limit}]"
            )
            pbar.set_postfix({
                "queued":  len(queue),
                "chunks":  checkpoint.total_chunks,
                "failed":  stats["failed"],
            })

            log.info(f"[{checkpoint.total_pages + 1}] {url}")

            text, soup = scrape(url)
            time.sleep(RATE_LIMIT_DELAY)
            checkpoint.mark_visited(url)

            if not text or len(text) < MIN_PAGE_LENGTH:
                log.warning(f"Skipping (too short or failed): {url}")
                checkpoint.mark_failed(url)
                stats["failed"] += 1
                pbar.update(1)
                continue

            page_hash = content_hash(text)
            if checkpoint.is_ingested(page_hash):
                log.debug(f"Duplicate content skipped: {url}")
                stats["skipped"] += 1
                pbar.update(1)
                continue

            chunks, metas = chunk(text, url)
            added = store_chunks(vectorstore, chunks, metas, checkpoint)

            checkpoint.total_chunks += added
            checkpoint.total_pages  += 1
            stats["success"]        += 1
            stats["new_chunks"]     += added

            if soup:
                new_links = [
                    link for link in discover_links(soup, url)
                    if not checkpoint.is_visited(link) and link not in queue
                ]
                if new_links:
                    queue.extend(new_links)
                    log.debug(f"BFS added {len(new_links)} new links")

            log.info(
                f"  +{added} chunks | "
                f"total {checkpoint.total_chunks} | "
                f"queue {len(queue)}"
            )

            if checkpoint.total_pages % CHECKPOINT_EVERY == 0:
                checkpoint.queued_urls = list(queue)
                checkpoint.save()

            pbar.update(1)

    return stats


def main() -> None:
    global _checkpoint_ref

    args = _parse_args()

    log.info("=" * 65)
    log.info("  GitLab Handbook - Ingestion Pipeline")
    log.info("=" * 65)

    if args.stats:
        cp = Checkpoint.load()
        cp.print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    if args.reset:
        log.info("--reset: wiping checkpoint and starting fresh")
        checkpoint = Checkpoint.reset()
    else:
        checkpoint = Checkpoint.load()

    _checkpoint_ref = checkpoint   

    page_limit = 50 if args.test else MAX_PAGES
    if args.test:
        log.info("--test mode: limited to 50 pages")

    log.info("Loading embedding model (downloads on first run)...")
    embeddings  = load_embeddings()
    vectorstore = load_vectorstore(embeddings)

    if checkpoint.queued_urls:
        log.info(f"Resuming ({len(checkpoint.queued_urls)} URLs pending)")
        queue: deque = deque(checkpoint.queued_urls)
    else:
        queue = build_queue_from_sitemaps()

    log.info(f"Queue: {len(queue)} URLs | Page cap: {page_limit}")
    log.info("=" * 65)

    try:
        stats = _crawl(queue, checkpoint, vectorstore, page_limit)
    except Exception as e:
        log.error(f"Crawl failed with unhandled error: {e}")
        checkpoint.queued_urls = list(queue)
        checkpoint.save()
        raise
    finally:
        close_browser()

    checkpoint.queued_urls = list(queue)
    checkpoint.save()

    log.info("=" * 65)
    log.info("  Ingestion complete")
    log.info(f"  Pages crawled  : {stats['success']}")
    log.info(f"  Pages skipped  : {stats['skipped']}  (duplicate content)")
    log.info(f"  Pages failed   : {stats['failed']}")
    log.info(f"  New chunks     : {stats['new_chunks']}")
    log.info(f"  Total chunks   : {checkpoint.total_chunks}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()