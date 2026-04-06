import re
import time
import atexit
from typing import Optional

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from config.settings import (
    HEADERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    MIN_PAGE_LENGTH,
    JS_RENDER_THRESHOLD,
    PLAYWRIGHT_TIMEOUT_MS,
    PLAYWRIGHT_SCROLL_PAUSE_MS,
    PLAYWRIGHT_BLOCKED_RESOURCES,
)
from utils.logger import get_logger

log = get_logger("scraper")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    PLAYWRIGHT_AVAILABLE = True
    log.debug("Playwright available - dynamic fallback enabled")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    log.info(
        "Playwright not installed - running in static-only mode. "
        "To enable: pip install playwright && playwright install chromium"
    )

_pw_instance = None
_browser     = None


def _get_browser():
    global _pw_instance, _browser
    if not PLAYWRIGHT_AVAILABLE:
        return None
    if _browser is None:
        try:
            _pw_instance = sync_playwright().start()
            _browser = _pw_instance.chromium.launch(headless=True)
            log.debug("Playwright browser launched")
        except Exception as e:
            log.error(f"Failed to launch Playwright browser: {e}")
            return None
    return _browser


def close_browser() -> None:
    global _pw_instance, _browser
    if _browser is not None:
        try:
            _browser.close()
            log.debug("Playwright browser closed")
        except Exception:
            pass
        _browser = None
    if _pw_instance is not None:
        try:
            _pw_instance.stop()
        except Exception:
            pass
        _pw_instance = None

atexit.register(close_browser)

def _fetch_static(url: str) -> Optional[requests.Response]:
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout:
            log.warning(f"Timeout [{attempt}/{MAX_RETRIES}]: {url}")

        except requests.exceptions.ConnectionError:
            log.warning(f"Connection error [{attempt}/{MAX_RETRIES}]: {url}")

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            log.warning(f"HTTP {code} [{attempt}/{MAX_RETRIES}]: {url}")
            if e.response and 400 <= e.response.status_code < 500:
                return None

        except Exception as e:
            log.error(f"Unexpected error [{attempt}/{MAX_RETRIES}] {url}: {e}")

        time.sleep(2 ** attempt)

    log.error(f"Static fetch failed after {MAX_RETRIES} attempts: {url}")
    return None

def _block_assets(route, request):
    if request.resource_type in PLAYWRIGHT_BLOCKED_RESOURCES:
        route.abort()
    else:
        route.continue_()


def _fetch_dynamic(url: str) -> Optional[str]:
    browser = _get_browser()
    if browser is None:
        return None

    log.info(f"Dynamic render (Playwright): {url}")
    page = None
    try:
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            java_script_enabled=True,
        )
        page = context.new_page()
        page.route("**/*", _block_assets)

        page.goto(
            url,
            wait_until="networkidle",
            timeout=PLAYWRIGHT_TIMEOUT_MS,
        )

        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(PLAYWRIGHT_SCROLL_PAUSE_MS)

        return page.content()

    except PWTimeout:
        log.warning(f"Playwright timeout ({PLAYWRIGHT_TIMEOUT_MS}ms): {url}")
    except Exception as e:
        log.error(f"Playwright error {url}: {e}")
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass

    return None

def _parse_html(html: str) -> tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "script", "style",
                     "header", "aside", "noscript", "iframe",
                     "form", "button", "input", "select", "textarea"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        try:
            if (
                tag.get("aria-hidden") == "true"
                or tag.get("hidden")
                or tag.get("role") in ("navigation", "banner", "contentinfo")
            ):
                tag.decompose()
        except Exception:
            pass

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"^(content|main|body|article)", re.I))
        or soup.find(class_=re.compile(
            r"(handbook-content|direction-content|page-body|"
            r"post-body|doc-content|content-body|markdown-body)",
            re.I,
        ))
        or soup.body
    )

    raw  = md(str(main), heading_style="ATX", strip=["img", "a"])
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^\s*[-*|]\s*$", "", text, flags=re.MULTILINE)

    return text.strip(), soup

def _is_thin(text: str) -> bool:
    return len(text.strip()) < JS_RENDER_THRESHOLD


def scrape(url: str) -> tuple[str, Optional[BeautifulSoup]]:
    response    = _fetch_static(url)
    static_html = response.text if response else None
    static_text = ""
    static_soup = None

    if static_html:
        static_text, static_soup = _parse_html(static_html)
        if not _is_thin(static_text):
            log.debug(f"Static OK ({len(static_text)} chars): {url}")
            return static_text, static_soup
        log.info(f"Static thin ({len(static_text)} chars) - trying dynamic: {url}")

    dynamic_html = _fetch_dynamic(url)
    if dynamic_html:
        dynamic_text, dynamic_soup = _parse_html(dynamic_html)
        if len(dynamic_text) > len(static_text) * 1.2 or len(dynamic_text) >= MIN_PAGE_LENGTH:
            log.info(f"Dynamic render succeeded ({len(dynamic_text)} chars): {url}")
            return dynamic_text, dynamic_soup
        log.warning(f"Dynamic also thin ({len(dynamic_text)} chars): {url}")

    if static_text:
        log.warning(f"Using thin static content as fallback: {url}")
        return static_text, static_soup

    log.error(f"Both tiers failed: {url}")
    return "", None