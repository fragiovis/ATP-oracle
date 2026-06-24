"""Base HTTP scraper with rate limiting, retries, caching, and checkpointing."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests
from fake_useragent import UserAgent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

_ua = UserAgent()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ScraperError(Exception):
    """Raised when all retries are exhausted or a non-retryable error occurs."""


class RateLimitError(ScraperError):
    """Raised on HTTP 429."""


# ---------------------------------------------------------------------------
# HTML cache
# ---------------------------------------------------------------------------

class HTMLCache:
    """Simple file-based HTML cache keyed on URL hash."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 72):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / f"{h}.html"

    def get(self, url: str) -> Optional[str]:
        path = self._key(url)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return path.read_text(encoding="utf-8")

    def set(self, url: str, html: str) -> None:
        self._key(url).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class Checkpoint:
    """Track which items have already been scraped so runs can be resumed."""

    def __init__(self, name: str, checkpoint_dir: Path):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._path = checkpoint_dir / f"{name}.json"
        self._done: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._done = set(data.get("done", []))

    def _save(self) -> None:
        self._path.write_text(json.dumps({"done": list(self._done)}, indent=2))

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        self._done.add(key)
        self._save()

    def reset(self) -> None:
        self._done.clear()
        self._save()


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket style with random jitter)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, requests_per_minute: int, min_delay: float, max_delay: float):
        self.min_interval = 60.0 / requests_per_minute
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_request_time: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        jitter = random.uniform(self.min_delay, self.max_delay)
        required = max(self.min_interval, jitter)
        sleep_for = max(0.0, required - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request_time = time.time()


# ---------------------------------------------------------------------------
# Base scraper
# ---------------------------------------------------------------------------

class BaseScraper:
    """
    Reusable HTTP client with:
    - Rotating user-agent headers
    - Configurable rate limiting + random jitter
    - Exponential backoff on 429/403/5xx
    - Optional HTML caching
    - Checkpoint / resume support
    """

    def __init__(self, checkpoint_name: str = "base"):
        cfg = settings.scraper
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(
            cfg.requests_per_minute,
            cfg.min_delay_seconds,
            cfg.max_delay_seconds,
        )
        self.cache: Optional[HTMLCache] = (
            HTMLCache(cfg.cache_dir, cfg.cache_ttl_hours) if cfg.enable_cache else None
        )
        self.checkpoint = Checkpoint(checkpoint_name, cfg.checkpoint_dir)
        self._max_retries = cfg.max_retries
        self._backoff = cfg.backoff_multiplier
        self._retry_codes = set(cfg.retry_status_codes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_headers(self) -> dict[str, str]:
        return {
            "User-Agent": _ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    def _fetch_raw(self, url: str) -> str:
        """Single attempt – raises requests.HTTPError on bad status codes."""
        self.rate_limiter.wait()
        headers = self._make_headers()
        if settings.scraper.log_requests:
            logger.debug("GET %s", url)
        resp = self.session.get(url, headers=headers, timeout=30)
        if resp.status_code in self._retry_codes:
            resp.raise_for_status()  # triggers tenacity retry
        resp.raise_for_status()
        return resp.text

    def _fetch_with_retry(self, url: str) -> str:
        """Fetch with exponential backoff via tenacity."""

        @retry(
            retry=retry_if_exception_type(requests.HTTPError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._backoff, min=2, max=120),
            reraise=True,
        )
        def _inner() -> str:
            return self._fetch_raw(url)

        return _inner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_html(self, url: str) -> str:
        """
        Fetch ``url`` and return the HTML string.

        Checks the disk cache first; on cache miss fetches and caches
        the response.  Raises ``ScraperError`` if all retries fail.
        """
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        try:
            html = self._fetch_with_retry(url)
        except Exception as exc:
            logger.error("Failed to fetch %s after retries: %s", url, exc)
            raise ScraperError(f"Could not fetch {url}") from exc

        if self.cache:
            self.cache.set(url, html)

        return html
