"""Playwright-based browser client for scraping Cloudflare-protected sites.

Uses a headless Chromium browser to render pages, bypassing JavaScript
challenges that block plain HTTP requests.  Includes rate limiting,
retry logic, and optional HTML caching.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from config.settings import settings
from scraper.base import HTMLCache, RateLimiter, Checkpoint

logger = logging.getLogger(__name__)


class PlaywrightBrowser:
    """Headless Chromium wrapper with rate limiting, retries, and caching."""

    def __init__(self, checkpoint_name: str = "browser"):
        cfg = settings.scraper
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

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the browser (call once before scraping)."""
        if self._browser is not None:
            return
        logger.info("Launching headless Chromium via Playwright…")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        # Stealth tweaks: remove webdriver flag
        self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)
        logger.info("Browser ready.")

    def stop(self) -> None:
        """Close the browser and release resources."""
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_html(self, url: str, wait_selector: str | None = None, timeout_ms: int = 30000) -> str:
        """
        Navigate to *url* and return the fully-rendered HTML.

        Parameters
        ----------
        url : str
            The page to fetch.
        wait_selector : str | None
            CSS selector to wait for before capturing HTML (ensures JS rendered).
        timeout_ms : int
            Max time to wait for the page / selector.

        Returns the page HTML or raises after max retries.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self.rate_limiter.wait()
            try:
                html = self._fetch_page(url, wait_selector, timeout_ms)
                # Cache the result
                if self.cache:
                    self.cache.set(url, html)
                return html
            except Exception as exc:
                last_error = exc
                wait_s = min(2 ** attempt + random.uniform(0, 2), 60)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s – retrying in %.1fs",
                    attempt, self._max_retries, url, exc, wait_s,
                )
                time.sleep(wait_s)

        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str, wait_selector: str | None, timeout_ms: int) -> str:
        """Single attempt to load a page and return its HTML."""
        if self._context is None:
            self.start()

        page: Page = self._context.new_page()
        try:
            logger.debug("Navigating to %s", url)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if resp and resp.status in (403, 503):
                # Cloudflare challenge — wait for it to resolve
                logger.debug("Got %d, waiting for Cloudflare challenge…", resp.status)
                page.wait_for_timeout(5000)
                # Check if we got through
                if "challenge" in page.title().lower() or "just a moment" in page.content().lower():
                    page.wait_for_timeout(10000)

            # Wait for dynamic content to render
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
                except Exception:
                    logger.debug("Selector %s not found, using page as-is", wait_selector)

            # Extra settle time for JS rendering
            page.wait_for_timeout(2000)

            html = page.content()

            # Quick sanity check
            if len(html) < 500 and ("access denied" in html.lower() or "blocked" in html.lower()):
                raise RuntimeError(f"Access denied for {url}")

            return html
        finally:
            page.close()
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self.rate_limiter.wait()
            try:
                html = self._fetch_page(url, wait_selector, timeout_ms)
                # Cache the result
                if self.cache:
                    self.cache.set(url, html)
                return html
            except Exception as exc:
                last_error = exc
                wait_s = min(2 ** attempt + random.uniform(0, 2), 60)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s – retrying in %.1fs",
                    attempt, self._max_retries, url, exc, wait_s,
                )
                time.sleep(wait_s)

        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str, wait_selector: str | None, timeout_ms: int) -> str:
        """Single attempt to load a page and return its HTML."""
        if self._context is None:
            self.start()

        page: Page = self._context.new_page()
        try:
            logger.debug("Navigating to %s", url)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if resp and resp.status in (403, 503):
                # Cloudflare challenge — wait for it to resolve
                logger.debug("Got %d, waiting for Cloudflare challenge…", resp.status)
                page.wait_for_timeout(5000)
                # Check if we got through
                if "challenge" in page.title().lower() or "just a moment" in page.content().lower():
                    page.wait_for_timeout(10000)

            # Wait for dynamic content to render
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
                except Exception:
                    logger.debug("Selector %s not found, using page as-is", wait_selector)

            # Extra settle time for JS rendering
            page.wait_for_timeout(2000)

            html = page.content()

            # Quick sanity check
            if len(html) < 500 and ("access denied" in html.lower() or "blocked" in html.lower()):
                raise RuntimeError(f"Access denied for {url}")

            return html
        finally:
            page.close()
