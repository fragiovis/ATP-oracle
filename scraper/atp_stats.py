"""Scrape matches + detailed statistics from ATP Tour via the Hawkeye API.

Replaces tennis-data.co.uk as the source for recent matches (2025+).
Unlike tennis-data, this source provides full per-match serve stats
(aces, double faults, serve %, break points, return stats, etc.).

Uses Playwright to load match-stats pages and intercept the internal
Hawkeye API response (``/-/Hawkeye/MatchStats/Complete/{year}/{event}/{match}``).

Workflow:
  1. Load the results-archive page for a given year to discover all tournaments.
  2. For each tournament, load the results page to find match-stats URLs.
  3. For each match, navigate to the stats page and capture the Hawkeye JSON.
  4. Parse the JSON into:
     - A match record (compatible with the pipeline's match schema)
     - Two stat records (one per player, compatible with match_stats schema)

The Hawkeye API is behind Cloudflare and requires a real browser session.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from scraper.browser import PlaywrightBrowser

logger = logging.getLogger(__name__)

ATP_BASE = "https://www.atptour.com"
RESULTS_ARCHIVE_URL = f"{ATP_BASE}/en/scores/results-archive?year={{year}}"

# Map ATP EventType to our tournament_category
EVENT_TYPE_MAP = {
    "GS": "Grand Slam",
    "1000": "ATP 1000",
    "500": "ATP 500",
    "250": "ATP 250",
    "F": "ATP Finals",
    "O": "Olympics",
}


class ATPStatsScraper:
    """Scrape matches and detailed statistics from atptour.com using Playwright.

    Returns two lists:
      - matches:     match records compatible with the pipeline (same schema as Sackmann/tennis-data)
      - match_stats: per-player stat records compatible with match_player_stats
    """

    def __init__(self):
        self._browser = PlaywrightBrowser(checkpoint_name="atp_stats")
        self._consecutive_blocks = 0
        self._max_consecutive_blocks = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_year(
        self, year: int, max_tournaments: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Scrape all matches + stats for a given year.

        Returns ``{"matches": [...], "match_stats": [...]}``.
        """
        all_matches: list[dict[str, Any]] = []
        all_stats: list[dict[str, Any]] = []

        with self._browser:
            tournaments = self._discover_tournaments(year)
            if max_tournaments:
                tournaments = tournaments[:max_tournaments]
            logger.info("Year %d: found %d tournaments to scrape.", year, len(tournaments))

            for i, (tourney_name, tourney_url) in enumerate(tournaments, 1):
                logger.info("[%d/%d] Scraping: %s", i, len(tournaments), tourney_name)

                ck_key = f"{year}_{tourney_url}"
                if self._browser.checkpoint.is_done(ck_key):
                    logger.info("  Skip (checkpoint): %s", tourney_name)
                    continue

                match_urls = self._discover_match_stats_urls(tourney_url)
                logger.info("  Found %d matches.", len(match_urls))

                self._fresh_context()

                blocked = False
                for j, stats_path in enumerate(match_urls, 1):
                    match_ck = f"{year}_{stats_path}"
                    if self._browser.checkpoint.is_done(match_ck):
                        continue

                    result = self._scrape_match(stats_path, tourney_name, year)
                    if result:
                        all_matches.append(result["match"])
                        all_stats.extend(result["stats"])
                        self._browser.checkpoint.mark_done(match_ck)
                        self._consecutive_blocks = 0
                    elif self._consecutive_blocks >= self._max_consecutive_blocks:
                        logger.warning("  Too many Cloudflare blocks – stopping tournament.")
                        blocked = True
                        break

                    if j % 10 == 0:
                        logger.info("  Progress: %d/%d matches", j, len(match_urls))

                if not blocked:
                    self._browser.checkpoint.mark_done(ck_key)
                logger.info(
                    "  Done %s: %d matches, %d stats total.",
                    tourney_name, len(all_matches), len(all_stats),
                )
                if blocked:
                    logger.warning("Cloudflare rate limit. Resume later with checkpointing.")
                    break

        logger.info("Year %d complete: %d matches, %d stats.", year, len(all_matches), len(all_stats))
        return {"matches": all_matches, "match_stats": all_stats}

    def scrape_tournament(
        self, results_url: str, year: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Scrape matches + stats for a single tournament."""
        all_matches: list[dict] = []
        all_stats: list[dict] = []

        with self._browser:
            match_urls = self._discover_match_stats_urls(results_url)
            logger.info("Found %d matches.", len(match_urls))
            self._fresh_context()

            for stats_path in match_urls:
                match_ck = f"{year}_{stats_path}"
                if self._browser.checkpoint.is_done(match_ck):
                    continue
                result = self._scrape_match(stats_path, "", year)
                if result:
                    all_matches.append(result["match"])
                    all_stats.extend(result["stats"])
                    self._browser.checkpoint.mark_done(match_ck)
                    self._consecutive_blocks = 0
                elif self._consecutive_blocks >= self._max_consecutive_blocks:
                    logger.warning("Cloudflare rate limit. Resume later.")
                    break

        return {"matches": all_matches, "match_stats": all_stats}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_tournaments(self, year: int) -> list[tuple[str, str]]:
        """Return [(name, results_url_path), ...] for all tournaments in a year."""
        url = RESULTS_ARCHIVE_URL.format(year=year)
        html = self._browser.get_html(url, wait_selector="[class*=tournament], a[href*=results]")
        logger.debug("Results archive URL: %s (fetched %d chars)", url, len(html) if html is not None else 0)
        if not html or len(html) < 200:
            logger.debug("Results archive page appears small/empty — first 500 chars:\n%s", (html or "")[:500])
        soup = BeautifulSoup(html, "lxml")

        tournaments = []
        seen = set()
        # Restrict to archive links that include the year and an event slug
        pattern = re.compile(r"/scores/archive/\d{4}/[^/]+/[^/]+")
        for link in soup.find_all("a", href=pattern):
            href = link.get("href", "")
            if href in seen:
                continue
            seen.add(href)
            # Prefer semantic tournament name from surrounding markup, otherwise derive from the href
            name = self._find_tournament_name(link)
            if not name:
                parts = href.split("/")
                # Typical path: /en/scores/archive/2025/rome/438/results -> tournament slug at index 5
                name = parts[5] if len(parts) > 5 else href
            tournaments.append((name, href))

        if not tournaments:
            logger.warning("No tournaments discovered on results archive for year %d. Check selector or page layout.", year)
            logger.debug("Results archive sample HTML:\n%s", (html or "")[:2000])

        logger.debug("Discovered %d tournaments (sample 10): %s", len(tournaments), [t[0] for t in tournaments[:10]])
        return tournaments

    def _discover_match_stats_urls(self, results_path: str) -> list[str]:
        """Return list of match-stats URL paths from a tournament results page."""
        url = ATP_BASE + results_path if results_path.startswith("/") else results_path

        self._browser.start()
        page = self._browser._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(10000)
            html = page.content()
            logger.debug("Tournament results page: %s fetched (%d chars)", url, len(html) if html else 0)
        finally:
            page.close()

        soup = BeautifulSoup(html, "lxml")

        stats_urls = []
        seen = set()
        anchors = soup.find_all("a", href=re.compile(r"(match-stats|stats-centre)/archive/"))
        logger.debug("Found %d candidate anchors for match-stats links on %s", len(anchors), url)
        for link in anchors:
            href = link.get("href", "")
            if href and href not in seen:
                seen.add(href)
                stats_urls.append(href)

        if not stats_urls:
            logger.warning("No match-stats links found on %s. Page may have changed or require JS-rendered content.", url)
            logger.debug("Tournament results page snippet:\n%s", (html or "")[:2000])
        else:
            logger.debug("Discovered %d match-stats urls (sample 10): %s", len(stats_urls), stats_urls[:10])

        return stats_urls

    @staticmethod
    def _find_tournament_name(link_el) -> str:
        for parent in link_el.parents:
            name_el = parent.select_one(
                "[class*=tournament-name], [class*=event-name], "
                "[class*=title], h3, h4"
            )
            if name_el:
                text = name_el.get_text(strip=True)
                if text and len(text) < 80:
                    return text
            if parent.name in ("body", "html", "section"):
                break
        return ""

    # ------------------------------------------------------------------
    # Match scraping
    # ------------------------------------------------------------------

    def _fresh_context(self) -> None:
        """Create a fresh browser context (required: ATP caching blocks Hawkeye XHR on reuse)."""
        # Delegate context creation to PlaywrightBrowser
        if hasattr(self._browser, "fresh_context"):
            self._browser.fresh_context()
        else:
            # Fallback: attempt to recreate the low-level context
            if self._browser._context:
                self._browser._context.close()
            self._browser._context = self._browser._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            try:
                self._browser._context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', { get: () => false });"
                )
            except Exception:
                pass

    def _scrape_match(
        self, stats_path: str, tournament_name: str, year: int
    ) -> dict[str, Any] | None:
        """
        Navigate to a match-stats page, intercept the Hawkeye API,
        and return ``{"match": {...}, "stats": [{...}, {...}]}`` or None.
        """
        import time
        import random

        url = ATP_BASE + stats_path if stats_path.startswith("/") else stats_path

        if self._browser._browser is None:
            self._browser.start()

        # Backoff or normal delay
        if self._consecutive_blocks > 0:
            backoff = min(30 * (2 ** (self._consecutive_blocks - 1)), 300) + random.uniform(0, 10)
            logger.info("Cloudflare backoff: %.0fs…", backoff)
            time.sleep(backoff)
        else:
            time.sleep(random.uniform(10.0, 20.0))

        self._fresh_context()

        hawkeye_data: dict[str, Any] = {}
        cloudflare_blocked = False
        page = self._browser._context.new_page()

        try:
            def capture_response(response):
                nonlocal cloudflare_blocked
                try:
                    resp_url = response.url.lower()
                except Exception:
                    resp_url = ""
                try:
                    status = response.status
                except Exception:
                    status = None
                logger.debug("Response captured: %s status=%s", resp_url, status)
                if "hawkeye" in resp_url and "matchstats" in resp_url:
                    try:
                        body = response.text()
                        logger.debug("Hawkeye response size: %d", len(body) if body else 0)
                        if body and body != "null":
                            hawkeye_data["json"] = json.loads(body)
                    except Exception as exc:
                        logger.exception("Hawkeye capture error parsing JSON: %s", exc)
                if "challenge-platform" in resp_url or "cloudflare" in resp_url:
                    cloudflare_blocked = True

            page.on("response", capture_response)
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            logger.debug("Navigated to %s — page title: %s", url, page.title())

            # Cloudflare challenge — wait for resolution
            if cloudflare_blocked and "json" not in hawkeye_data:
                logger.debug("Cloudflare challenge, waiting…")
                page.wait_for_timeout(15000)
                if "json" not in hawkeye_data:
                    page.wait_for_timeout(10000)

            if "json" not in hawkeye_data:
                page.wait_for_timeout(8000)

            result = None
            if "json" in hawkeye_data:
                logger.debug("Hawkeye JSON payload present (keys: %s)", list(hawkeye_data["json"].keys())[:10])
                result = _parse_hawkeye_full(hawkeye_data["json"], stats_path, tournament_name, year)
                if result:
                    self._consecutive_blocks = 0
            elif cloudflare_blocked:
                logger.warning("Cloudflare blocked: %s", stats_path)
                self._consecutive_blocks += 1
            else:
                logger.debug("No Hawkeye data for %s — will capture page snippet for debugging", stats_path)
                try:
                    # capture page HTML snippet for debugging
                    snippet = page.content()[:2000]
                    logger.debug("Page snippet for %s:\n%s", stats_path, snippet)
                except Exception:
                    logger.debug("Could not capture page snippet for %s", stats_path)

        except Exception as exc:
            logger.warning("Failed to load %s: %s", stats_path, exc)
            result = None
        finally:
            page.close()

        return result


# =========================================================================
# Hawkeye JSON → match record + stat records
# =========================================================================

def _parse_hawkeye_full(
    data: dict, stats_path: str, tournament_name: str, year: int
) -> dict[str, Any] | None:
    """
    Convert the Hawkeye JSON into:
      - A match record (compatible with pipeline match schema)
      - Two stat records (one per player, compatible with match_player_stats)

    Returns ``{"match": {...}, "stats": [{...}, {...}]}`` or None.
    """
    try:
        match_info = data.get("Match", {})
        tourney_info = data.get("Tournament", {})
        player_team = match_info.get("PlayerTeam", {})
        opponent_team = match_info.get("OpponentTeam", {})

        # Extract IDs from path: .../archive/YEAR/EVENT_ID/MATCH_ID
        parts = stats_path.rstrip("/").split("/")
        event_id = parts[-2] if len(parts) >= 2 else ""
        match_id_short = parts[-1] if parts else ""
        match_id = f"atp_{year}_{event_id}_{match_id_short}"

        # Player info
        player_info = player_team.get("Player", {}) or {}
        opponent_info = opponent_team.get("Player", {}) or {}

        player_a_name = f"{player_info.get('PlayerFirstName', '')} {player_info.get('PlayerLastName', '')}".strip()
        player_b_name = f"{opponent_info.get('PlayerFirstName', '')} {opponent_info.get('PlayerLastName', '')}".strip()
        player_a_id = player_info.get("PlayerId", "")
        player_b_id = opponent_info.get("PlayerId", "")

        if not player_a_name or not player_b_name:
            return None

        # Winner
        winner_id = match_info.get("WinningPlayerId") or match_info.get("Winner", "")

        # Tournament info
        event_display = tourney_info.get("EventDisplayName") or tourney_info.get("TournamentName") or tournament_name
        surface = (tourney_info.get("Court") or "").lower()
        event_type = tourney_info.get("EventType", "250")
        category = EVENT_TYPE_MAP.get(event_type, "ATP 250")

        # Best of: Grand Slams = 5, rest = 3
        best_of = 5 if category == "Grand Slam" else 3

        # Match date: prefer tournament start date (Hawkeye doesn't provide exact match date)
        match_date = (tourney_info.get("StartDate") or "")[:10]

        # Round
        round_info = match_info.get("Round", {})
        round_short = round_info.get("ShortName", "") if isinstance(round_info, dict) else str(round_info)

        # Scoreline from set scores
        scoreline = _build_scoreline(player_team, opponent_team)

        # Indoor detection
        indoor = "indoor" in surface or surface == "carpet"

        # ---- Match record ----
        match_record = {
            "match_id": match_id,
            "match_date": match_date,
            "year": year,
            "tournament_id": tourney_info.get("EventId", event_id),
            "tournament_name": event_display,
            "tournament_category": category,
            "round": round_short,
            "surface": surface,
            "indoor": indoor,
            "best_of": best_of,
            "player_a_id": player_a_id,
            "player_b_id": player_b_id,
            "player_a_name": player_a_name,
            "player_b_name": player_b_name,
            "winner_id": winner_id,
            "scoreline": scoreline,
            "match_time": match_info.get("MatchTimeTotal", ""),
            "source": "atp_hawkeye",
        }

        # ---- Stat records (one per player) ----
        stats = []
        pa_stats = _extract_player_stats(player_team, set_number=0)
        if pa_stats:
            stats.append({"match_id": match_id, "player_id": player_a_id, **pa_stats})

        pb_stats = _extract_player_stats(opponent_team, set_number=0)
        if pb_stats:
            stats.append({"match_id": match_id, "player_id": player_b_id, **pb_stats})

        return {"match": match_record, "stats": stats}

    except Exception as exc:
        logger.debug("Hawkeye parse error for %s: %s", stats_path, exc)
        return None


def _build_scoreline(player_team: dict, opponent_team: dict) -> str:
    """Reconstruct '6-3 7-6(5) 6-4' from PlayerTeam/OpponentTeam SetScores."""
    p_sets = {s["SetNumber"]: s for s in player_team.get("SetScores", []) if s.get("SetNumber", 0) > 0}
    o_sets = {s["SetNumber"]: s for s in opponent_team.get("SetScores", []) if s.get("SetNumber", 0) > 0}

    if not p_sets:
        return ""

    parts = []
    for set_num in sorted(p_sets.keys()):
        p_score_raw = p_sets.get(set_num, {}).get("SetScore")
        o_score_raw = o_sets.get(set_num, {}).get("SetScore")
        if p_score_raw is None or o_score_raw is None:
            continue

        p_score = int(p_score_raw)
        o_score = int(o_score_raw)

        # Tiebreak detection: Hawkeye reports both sides as 7 in a tiebreak set.
        # In standard notation, the loser has 6 games (the TB counts as 1 game for the winner).
        p_tb = p_sets.get(set_num, {}).get("TieBreakScore")
        o_tb = o_sets.get(set_num, {}).get("TieBreakScore")
        is_tiebreak = p_tb is not None or o_tb is not None

        if is_tiebreak and p_score == 7 and o_score == 7:
            # Determine who won the tiebreak: the winner has the higher TB score
            if (p_tb or 0) > (o_tb or 0):
                o_score = 6  # Loser gets 6
            else:
                p_score = 6  # Loser gets 6

        set_str = f"{p_score}-{o_score}"

        if is_tiebreak:
            loser_tb = min(p_tb or 99, o_tb or 99)
            if loser_tb < 99:
                set_str += f"({loser_tb})"

        parts.append(set_str)

    return " ".join(parts)


def _extract_player_stats(team: dict, set_number: int = 0) -> dict[str, Any]:
    """Extract stats from a team's SetScores entry (set_number=0 = match totals).

    Returns a dict compatible with the Sackmann match_stats schema.
    """
    set_scores = team.get("SetScores", [])
    stats_entry = None
    for ss in set_scores:
        if ss.get("SetNumber") == set_number and ss.get("Stats"):
            stats_entry = ss["Stats"]
            break

    if not stats_entry:
        return {}

    result: dict[str, Any] = {}

    # Service stats
    svc = stats_entry.get("ServiceStats", {})
    if svc:
        result["aces"] = _get_number(svc.get("Aces"))
        result["double_faults"] = _get_number(svc.get("DoubleFaults"))
        # Raw counts for Sackmann compatibility
        first_serve_in = _get_dividend(svc.get("FirstServe"))
        service_pts_total = _get_divisor(svc.get("FirstServe"))
        result["service_pts_total"] = service_pts_total
        result["first_serve_in"] = first_serve_in
        result["first_serve_pct"] = _get_pct(svc.get("FirstServe"))
        result["first_serve_pts_won"] = _get_dividend(svc.get("FirstServePointsWon"))
        result["first_serve_pts_won_pct"] = _get_pct(svc.get("FirstServePointsWon"))
        result["second_serve_pts_won"] = _get_dividend(svc.get("SecondServePointsWon"))
        result["second_serve_pts_won_pct"] = _get_pct(svc.get("SecondServePointsWon"))
        result["bp_saved"] = _get_dividend(svc.get("BreakPointsSaved"))
        result["bp_faced"] = _get_divisor(svc.get("BreakPointsSaved"))
        result["bp_saved_pct"] = _get_pct(svc.get("BreakPointsSaved"))
        result["service_games"] = _get_number(svc.get("ServiceGamesPlayed"))
        result["serve_rating"] = _get_number(svc.get("ServeRating"))

    # Return stats
    ret = stats_entry.get("ReturnStats", {})
    if ret:
        result["first_return_won"] = _get_dividend(ret.get("FirstServeReturnPointsWon"))
        result["first_return_won_pct"] = _get_pct(ret.get("FirstServeReturnPointsWon"))
        result["second_return_won"] = _get_dividend(ret.get("SecondServeReturnPointsWon"))
        result["second_return_won_pct"] = _get_pct(ret.get("SecondServeReturnPointsWon"))
        result["bp_converted"] = _get_dividend(ret.get("BreakPointsConverted"))
        result["bp_opportunities"] = _get_divisor(ret.get("BreakPointsConverted"))
        result["bp_converted_pct"] = _get_pct(ret.get("BreakPointsConverted"))
        result["return_games"] = _get_number(ret.get("ReturnGamesPlayed"))
        result["return_rating"] = _get_number(ret.get("ReturnRating"))

    # Point stats
    pts = stats_entry.get("PointStats", {})
    if pts:
        result["total_service_pts_won"] = _get_dividend(pts.get("TotalServicePointsWon"))
        result["total_return_pts_won"] = _get_dividend(pts.get("TotalReturnPointsWon"))
        result["total_pts_won"] = _get_dividend(pts.get("TotalPointsWon"))
        result["total_pts"] = _get_divisor(pts.get("TotalPointsWon"))

    # Derived: service_pts_won_pct (for Sackmann compat)
    svpt = result.get("service_pts_total")
    total_svc_won = result.get("total_service_pts_won")
    if svpt and svpt > 0 and total_svc_won is not None:
        result["service_pts_won_pct"] = total_svc_won / svpt

    # Skip if entirely empty
    if all(v is None for k, v in result.items()):
        return {}

    return result


# ---------------------------------------------------------------------------
# Value extractors for Hawkeye JSON stat objects
# ---------------------------------------------------------------------------

def _get_number(obj: dict | None) -> int | None:
    if obj is None:
        return None
    return obj.get("Number")


def _get_pct(obj: dict | None) -> float | None:
    if obj is None:
        return None
    pct = obj.get("Percent")
    return pct / 100.0 if pct is not None else None


def _get_dividend(obj: dict | None) -> int | None:
    if obj is None:
        return None
    return obj.get("Dividend")


def _get_divisor(obj: dict | None) -> int | None:
    if obj is None:
        return None
    return obj.get("Divisor")
