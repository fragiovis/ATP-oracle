"""
Canonicalisation of surfaces and tournament names.

Single source of truth used by the ingestion normaliser, the feature pipeline
and the serving API, so the same match never appears under two different
surfaces or two different tournament names.

Two rules are enforced here:

1. Surfaces are exactly {clay, hard, grass}. There is NO "indoor" surface:
   indoor hard and carpet courts are modelled as "hard". The feature layer
   (features/pipeline.py, features/player_profiler.py, config/settings.py)
   only ever builds clay/hard/grass columns, so an "indoor_hard" row would
   silently lose its surface one-hot.

2. Tournament names are collapsed to one canonical spelling per event, so the
   UI dropdown shows "Roland Garros" once instead of "French Open" +
   "Roland Garros", and per-tournament profiles aggregate over the full
   history of the event.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

SURFACES = ["clay", "hard", "grass"]


def canonical_surface(raw: str | None) -> str:
    """
    Map any raw surface string to one of {clay, hard, grass}.

    Indoor hard and carpet both collapse to "hard" — the indoor/outdoor
    distinction is carried separately by the boolean `indoor` column.
    Unknown or empty values default to "hard" (the most common surface).
    """
    s = (raw or "").lower().strip()
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    # hard, indoor hard, indoor, carpet, and anything unrecognised
    return "hard"


# ---------------------------------------------------------------------------
# Tournament names
# ---------------------------------------------------------------------------

# Alias (lower-cased, stripped) -> canonical display name.
# Only genuine duplicates of the SAME event are listed. Distinct events that
# merely share a city (e.g. Adelaide 1 / Adelaide 2 in the 2021 bubble season)
# are collapsed too, since they are the same tournament for modelling purposes.
_TOURNAMENT_ALIASES: dict[str, str] = {
    # ── Grand Slams ────────────────────────────────────────────────────
    "french open": "Roland Garros",
    "roland garros": "Roland Garros",
    "us open": "US Open",
    "u.s. open": "US Open",
    "australian open": "Australian Open",
    "wimbledon": "Wimbledon",

    # ── Masters 1000 ───────────────────────────────────────────────────
    "indian wells": "Indian Wells Masters",
    "indian wells masters": "Indian Wells Masters",
    "bnp paribas open": "Indian Wells Masters",
    "miami masters": "Miami Masters",
    "miami open": "Miami Masters",
    "key biscayne": "Miami Masters",
    "monte carlo": "Monte Carlo Masters",
    "monte carlo masters": "Monte Carlo Masters",
    "madrid": "Madrid Masters",
    "madrid masters": "Madrid Masters",
    "mutua madrid open": "Madrid Masters",
    "rome": "Rome Masters",
    "rome masters": "Rome Masters",
    "internazionali bnl d'italia": "Rome Masters",
    "canada masters": "Canada Masters",
    "canadian open": "Canada Masters",
    "montreal / toronto": "Canada Masters",
    "toronto indoor": "Canada Masters",
    "cincinnati": "Cincinnati Masters",
    "cincinnati masters": "Cincinnati Masters",
    "western & southern financial group masters": "Cincinnati Masters",
    "shanghai": "Shanghai Masters",
    "shanghai masters": "Shanghai Masters",
    "paris masters": "Paris Masters",
    "paris indoor": "Paris Masters",
    "bnp paribas masters": "Paris Masters",
    "hamburg": "Hamburg Open",
    "hamburg masters": "Hamburg Open",
    "hamburg open": "Hamburg Open",
    "stuttgart masters": "Stuttgart Open",
    "stuttgart indoor": "Stuttgart Open",
    "stuttgart outdoor": "Stuttgart Open",
    "stuttgart": "Stuttgart Open",
    "stuttgart open": "Stuttgart Open",
    "essen masters": "Essen Masters",
    "stockholm masters": "Stockholm",
    "stockholm": "Stockholm",

    # ── Season finals ──────────────────────────────────────────────────
    "masters": "ATP Finals",
    "masters cup": "ATP Finals",
    "masters dec": "ATP Finals",
    "tour finals": "ATP Finals",
    "atp finals": "ATP Finals",
    "next gen finals": "Next Gen Finals",
    "nextgen finals": "Next Gen Finals",

    # ── ATP 500 / 250 duplicates ───────────────────────────────────────
    "rotterdam": "Rotterdam",
    "abn amro world tennis tournament": "Rotterdam",
    "dubai": "Dubai",
    "dubai tennis championships": "Dubai",
    "doha": "Doha",
    "qatar exxon mobil open": "Doha",
    "acapulco": "Acapulco",
    "abierto mexicano": "Acapulco",
    "barcelona": "Barcelona Open",
    "barcelona open": "Barcelona Open",
    "basel": "Basel",
    "swiss indoors": "Basel",
    "vienna": "Vienna",
    "vienna open": "Vienna",
    "halle": "Halle",
    "halle open": "Halle",
    "queen's club": "Queen's Club Championships",
    "queen's club championships": "Queen's Club Championships",
    "tokyo": "Tokyo",
    "tokyo indoor": "Tokyo",
    "tokyo outdoor": "Tokyo",
    "japan open tennis championships": "Tokyo",
    "beijing": "China Open",
    "china open": "China Open",
    "washington": "Washington",
    "citi open": "Washington",
    "auckland": "ASB Classic",
    "asb classic": "ASB Classic",
    "brisbane": "Brisbane International",
    "brisbane international": "Brisbane International",
    "adelaide": "Adelaide International",
    "adelaide 1": "Adelaide International",
    "adelaide 2": "Adelaide International",
    "adelaide international": "Adelaide International",
    "sydney": "Sydney",
    "sydney indoor": "Sydney",
    "sydney outdoor": "Sydney",
    "delray beach": "Delray Beach Open",
    "delray beach open": "Delray Beach Open",
    "buenos aires": "Argentina Open",
    "argentina open": "Argentina Open",
    "rio de janeiro": "Rio Open",
    "rio open": "Rio Open",
    "atp rio de janeiro": "Rio Open",
    "santiago": "Chile Open",
    "chile open": "Chile Open",
    "umag": "Croatia Open",
    "croatia open": "Croatia Open",
    "bastad": "Nordea Open",
    "nordea open": "Nordea Open",
    "kitzbuhel": "Generali Open",
    "generali open": "Generali Open",
    "gstaad": "Suisse Open Gstaad",
    "suisse open gstaad": "Suisse Open Gstaad",
    "munich": "BMW Open",
    "bmw open": "BMW Open",
    "marrakech": "Grand Prix Hassan II",
    "casablanca": "Grand Prix Hassan II",
    "grand prix hassan ii": "Grand Prix Hassan II",
    "houston": "U.S.Men's Clay Court Championships",
    "u.s.men's clay court championships": "U.S.Men's Clay Court Championships",
    "geneva": "Geneva Open",
    "geneva open": "Geneva Open",
    "s hertogenbosch": "Rosmalen Grass Court Championships",
    "rosmalen": "Rosmalen Grass Court Championships",
    "rosmalen grass court championships": "Rosmalen Grass Court Championships",
    "eastbourne": "Eastbourne International",
    "eastbourne international": "Eastbourne International",
    "mallorca": "Mallorca Championships",
    "mallorca championships": "Mallorca Championships",
    "winston-salem": "Winston-Salem Open",
    "winston-salem open at wake forest university": "Winston-Salem Open",
    "winston-salem open": "Winston-Salem Open",
    "los cabos": "Los Cabos Open",
    "los cabos open": "Los Cabos Open",
    "chengdu": "Chengdu Open",
    "chengdu open": "Chengdu Open",
    "hangzhou": "Hangzhou Open",
    "hangzhou open": "Hangzhou Open",
    "hong kong": "Hong Kong Tennis Open",
    "hong kong tennis open": "Hong Kong Tennis Open",
    "almaty": "Almaty Open",
    "almaty open": "Almaty Open",
    "nur-sultan": "Astana",
    "astana": "Astana",
    "dallas": "Dallas Open",
    "dallas open": "Dallas Open",
    "montpellier": "Open Sud de France",
    "open sud de france": "Open Sud de France",
    "marseille": "Open 13",
    "open 13": "Open 13",
    "metz": "Open de Moselle",
    "open de moselle": "Open de Moselle",
    "antwerp": "European Open",
    "european open": "European Open",
    "bucharest": "Tiriac Open",
    "tiriac open": "Tiriac Open",
    "athens": "Hellenic Championship",
    "hellenic championship": "Hellenic Championship",
    "st petersburg": "St. Petersburg",
    "st. petersburg": "St. Petersburg",
    "belgrade": "Belgrade",
    "belgrade 2": "Belgrade",
    "cologne": "Cologne",
    "cologne 1": "Cologne",
    "cologne 2": "Cologne",
    "kuala lumpur": "Kuala Lumpur",
    "kuala lumpur-1": "Kuala Lumpur",
    "kuala lumpur-2": "Kuala Lumpur",
}


def canonical_tournament(raw: str | None) -> str:
    """
    Collapse tournament name variants to a single canonical display name.

    Unknown names are returned title-cased-as-given (whitespace stripped) so
    that at worst nothing changes; only known duplicates are merged.
    """
    if raw is None:
        return ""
    name = " ".join(str(raw).split())  # collapse internal/trailing whitespace
    if not name:
        return ""
    return _TOURNAMENT_ALIASES.get(name.lower(), name)
