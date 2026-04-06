import re
import requests
from collections import deque
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from config.settings import (
    ALLOWED_URL_PREFIXES,
    HEADERS,
    REQUEST_TIMEOUT,
    SEED_URLS,
    SITEMAP_URLS,
    SKIP_EXTENSIONS,
    SKIP_PATH_PATTERNS,
)
from utils.logger import get_logger

log = get_logger("crawler")

def normalise(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl().rstrip("/")


def is_crawlable(url: str) -> bool:
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not any(url.startswith(prefix) for prefix in ALLOWED_URL_PREFIXES):
            return False

        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            return False

        full = parsed.path + ("?" + parsed.query if parsed.query else "")
        if any(re.search(pat, full) for pat in SKIP_PATH_PATTERNS):
            return False

        return True
    except Exception:
        return False

_NS_STRIP_RE = re.compile(r'xmlns[^"]*"[^"]*"|\s+xmlns\S+="[^"]*"')

def _strip_namespaces(xml_bytes: bytes) -> bytes:
    text = xml_bytes.decode("utf-8", errors="replace")
    text = _NS_STRIP_RE.sub("", text)
    return text.encode("utf-8")

def _fetch_sitemap_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        log.warning(f"Sitemap HTTP {code}: {url}")
    except Exception as e:
        log.warning(f"Sitemap fetch failed ({url}): {e}")
    return None


def _parse_urlset(root: ElementTree.Element) -> list[str]:
    
    urls: list[str] = []

    for url_elem in root.findall("url"):
        loc = url_elem.find("loc")
        if loc is None or not loc.text:
            continue

        raw_url   = loc.text.strip()
        page_url  = normalise(raw_url)

        if not is_crawlable(page_url):
            log.debug(f"Skipping non-crawlable URL: {page_url}")
            continue

        urls.append(page_url)

    return urls


def _parse_sitemapindex(root: ElementTree.Element, visited: set[str]) -> list[str]:
    urls: list[str] = []

    for sitemap_elem in root.findall("sitemap"):
        loc = sitemap_elem.find("loc")
        if loc is None or not loc.text:
            continue

        child_url = loc.text.strip()
        if child_url in visited:
            continue

        log.info(f"  Found child sitemap: {child_url}")
        child_urls = _fetch_and_parse_sitemap(child_url, visited)
        urls.extend(child_urls)

    return urls


def _fetch_and_parse_sitemap(url: str, visited: set[str], depth: int = 0) -> list[str]:
    MAX_DEPTH = 3

    if depth > MAX_DEPTH:
        log.warning(f"Sitemap depth limit reached: {url}")
        return []

    if url in visited:
        return []
    visited.add(url)

    raw = _fetch_sitemap_bytes(url)
    if not raw:
        return []

    try:
        # strip namespaces to simplify parsing
        clean = _strip_namespaces(raw)
        root  = ElementTree.fromstring(clean)
    except ElementTree.ParseError as e:
        log.warning(f"Sitemap XML parse error ({url}): {e}")
        return []
    except Exception as e:
        log.warning(f"Sitemap parse failed ({url}): {e}")
        return []
    tag = root.tag.lower()

    if "urlset" in tag:
        log.debug(f"Parsing urlset: {url}")
        return _parse_urlset(root)

    elif "sitemapindex" in tag:
        log.info(f"Parsing sitemap index: {url}")
        return _parse_sitemapindex(root, visited)

    else:
        log.warning(f"Unknown sitemap root tag '{root.tag}': {url}")
        return []

def build_queue_from_sitemaps() -> deque[str]: 
    all_urls: list[str]       = []
    visited_sitemaps: set[str] = set()

    log.info("=" * 55)
    log.info("Building crawl queue from sitemaps")
    log.info("=" * 55)

    for sitemap_url in SITEMAP_URLS:
        log.info(f"Fetching: {sitemap_url}")
        found = _fetch_and_parse_sitemap(sitemap_url, visited_sitemaps)

        if found:
            log.info(f"{len(found)} crawlable URLs")
            all_urls.extend(found)
        else:
            log.warning(f"No URLs found or sitemap unreachable")

    unique = list(dict.fromkeys(all_urls))

    if unique:
        log.info(f"Total: {len(unique)} unique URLs queued")
        log.info("=" * 55)
        return deque(unique)

    log.warning("All sitemaps failed- falling back to SEED_URLS")
    log.warning("BFS link discovery will handle coverage")
    log.info("=" * 55)
    return deque(normalise(u) for u in SEED_URLS)

def discover_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    seen:  set[str]  = set()
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue

        absolute = normalise(urljoin(base_url, href))
        if absolute not in seen and is_crawlable(absolute):
            seen.add(absolute)
            links.append(absolute)

    log.debug(f"BFS found {len(links)} crawlable links from {base_url}")
    return links