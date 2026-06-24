"""Fetch the current ATP singles rankings via simple HTTP (no Playwright).

Scrapes atptour.com rankings page and returns a list of
  {'rank': int, 'player_name': str, 'snapshot_date': str}

Player IDs are NOT available here (ATP uses 4-char slugs, not Sackmann IDs).
Resolution to Sackmann numeric IDs is done downstream in the CLI via
_build_name_to_id_map / _resolve_name.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_URL = "https://www.atptour.com/en/rankings/singles?rankRange=1-{n}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_atp_rankings(top_n: int = 200) -> list[dict[str, Any]]:
    """
    Return current ATP top-N singles rankings as a list of dicts.

    Each dict: {'rank': int, 'player_name': str, 'snapshot_date': str (ISO)}.
    Returns empty list on failure.
    """
    url = _URL.format(n=top_n)
    logger.info("Fetching ATP live rankings (top %d): %s", top_n, url)
    try:
        resp = requests.get(url, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Could not fetch ATP rankings: %s", exc)
        return []

    return _parse(resp.text, top_n)


def _parse(html: str, top_n: int) -> list[dict[str, Any]]:
    import re
    soup = BeautifulSoup(html, "html.parser")
    today = date.today().isoformat()
    seen_ranks: set[int] = set()
    results: list[dict[str, Any]] = []

    rows = soup.select("table.mega-table tbody tr") or soup.select("tbody tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Rank: strip non-digit characters (change indicators like "+1", "-2")
        rank_raw = re.sub(r"[^\d]", "", cells[0].get_text(strip=True))
        if not rank_raw:
            continue
        try:
            rank = int(rank_raw)
        except ValueError:
            continue
        if rank < 1 or rank > top_n:
            continue

        # Player name: strip leading change indicators (e.g. "1Taylor Fritz" → "Taylor Fritz")
        name_raw = cells[1].get_text(strip=True)
        name = re.sub(r"^[\d\+\-]+", "", name_raw).strip()
        if not name:
            continue

        # Each rank appears twice: first abbreviated ("J. Sinner"), then full ("Jannik Sinner").
        # Keep the FULL name (longer string) — overwrite abbreviated if already seen.
        if rank in seen_ranks:
            existing = next(r for r in results if r["rank"] == rank)
            if len(name) > len(existing["player_name"]):
                existing["player_name"] = name
        else:
            seen_ranks.add(rank)
            results.append({"rank": rank, "player_name": name, "snapshot_date": today})

    results.sort(key=lambda x: x["rank"])
    logger.info("Parsed %d players from ATP rankings page.", len(results))
    return results
