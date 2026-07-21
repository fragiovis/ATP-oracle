"""FastAPI server – exposes the primary agent and the predictor over HTTP.

Endpoints:
  POST /api/ask          – natural-language query → AgentResult
  POST /api/predict      – structured MatchRequest → MatchPrediction
  GET  /api/health       – liveness check
  GET  /                 – serves app/index.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import PrimaryAgent
from serving.predictor import MatchPredictor, MatchRequest
from storage.canonical import canonical_surface, canonical_tournament

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATP Oracle",
    description="Pre-match tennis prediction system with natural-language agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
_STATIC_DIR = Path(__file__).parent.parent / "app"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Lazy singletons (loaded on first request)
# ---------------------------------------------------------------------------

_agent: PrimaryAgent | None = None
_predictor: MatchPredictor | None = None


def get_agent() -> PrimaryAgent:
    global _agent
    if _agent is None:
        logger.info("Initialising primary agent…")
        _agent = PrimaryAgent()
    return _agent


def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        logger.info("Initialising predictor…")
        _predictor = MatchPredictor()
    return _predictor


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    best_of: int = 3          # UI toggle: 3 or 5
    surface: str = "hard"     # UI surface selector (overridden by query if explicit)


class AskResponse(BaseModel):
    query: str
    answer: str
    prediction: dict[str, Any] | None
    requires_clarification: bool
    clarification_question: str | None
    intent: str | None
    entities: dict[str, Any] | None


class PredictRequest(BaseModel):
    player_a: str
    player_b: str
    surface: str = "hard"
    tournament_category: str = "ATP 250"
    round: str = "R32"
    best_of: int = 3
    match_date: str | None = None


class H2HRequest(BaseModel):
    player_a: str
    player_b: str


class PlayerStatsRequest(BaseModel):
    player_name: str


class ExplainRequest(BaseModel):
    player_a: str
    player_b: str
    surface: str = "hard"
    prediction: dict[str, Any]
    explanation: list[dict[str, Any]]  # SHAP feature importances


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

_matches_df: pd.DataFrame | None = None
_tournament_df: pd.DataFrame | None = None
_player_profiles_df: pd.DataFrame | None = None


def _get_matches() -> pd.DataFrame:
    """Load the silver matches table (cached). Retries on each call if previous load failed."""
    global _matches_df
    if _matches_df is None or _matches_df.empty:
        from storage.writer import StorageWriter
        writer = StorageWriter()
        try:
            _matches_df = writer.read_silver_table("matches")
            logger.info("Loaded %d matches for H2H lookup.", len(_matches_df))
        except Exception as e:
            logger.warning("Could not load matches table: %s", e)
            return pd.DataFrame()
    return _matches_df


def _get_tournaments() -> pd.DataFrame:
    global _tournament_df
    if _tournament_df is None or _tournament_df.empty:
        from storage.writer import StorageWriter
        writer = StorageWriter()
        try:
            _tournament_df = writer.read_tournament_profiles()
            logger.info("Loaded %d tournament profiles.", len(_tournament_df))
        except Exception as e:
            logger.warning("Could not load tournament profiles: %s", e)
            return pd.DataFrame()
    return _tournament_df


def _get_player_profiles() -> pd.DataFrame:
    global _player_profiles_df
    if _player_profiles_df is None or _player_profiles_df.empty:
        from storage.writer import StorageWriter
        writer = StorageWriter()
        try:
            _player_profiles_df = writer.read_player_profiles()
            logger.info("Loaded %d player profiles.", len(_player_profiles_df))
        except Exception as e:
            logger.warning("Could not load player profiles: %s", e)
            return pd.DataFrame()
    return _player_profiles_df


def _lookup_h2h(player_a: str, player_b: str) -> list[dict[str, Any]]:
    """Return past H2H matches between two players (tournament, score, surface)."""
    df = _get_matches()
    if df.empty:
        return []

    a_lower = player_a.lower().strip()
    b_lower = player_b.lower().strip()

    col_a = df["player_a_name"].fillna("").str.lower()
    col_b = df["player_b_name"].fillna("").str.lower()

    # Match: player A in one column and player B in the other (either order)
    mask = (
        (col_a.str.contains(a_lower, regex=False) & col_b.str.contains(b_lower, regex=False))
        | (col_a.str.contains(b_lower, regex=False) & col_b.str.contains(a_lower, regex=False))
    )
    h2h = df[mask].copy()

    if h2h.empty:
        return []

    # Sort by date descending
    if "match_date" in h2h.columns:
        h2h["match_date"] = pd.to_datetime(h2h["match_date"], errors="coerce")
        h2h = h2h.sort_values("match_date", ascending=False, na_position="last")

    results = []
    for _, row in h2h.iterrows():
        # Determine winner name
        winner_name = ""
        if row.get("winner_id") == row.get("player_a_id"):
            winner_name = row.get("player_a_name", "")
        else:
            winner_name = row.get("player_b_name", "")

        results.append({
            "tournament": row.get("tournament_name", ""),
            "surface": row.get("surface", ""),
            "date": str(row.get("match_date", ""))[:10] if pd.notnull(row.get("match_date")) else "",
            "score": row.get("scoreline", ""),
            "winner": winner_name,
        })

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "ATP Oracle API running. Frontend not found at app/index.html."})


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "atp-oracle"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Process a natural-language prediction query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = get_agent().run(
            request.query,
            ui_best_of=request.best_of,
            ui_surface=request.surface,
        )
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        query=request.query,
        answer=result.answer,
        prediction=result.prediction,
        requires_clarification=result.requires_clarification,
        clarification_question=result.clarification_question,
        intent=result.agent_response.intent if result.agent_response else None,
        entities={
            **(result.agent_response.entities.model_dump() if result.agent_response else {}),
            # Overwrite with the actual values used (UI fallback may have filled nulls)
            "best_of": result.agent_response.entities.best_of or request.best_of if result.agent_response else request.best_of,
            "surface": result.agent_response.entities.surface or request.surface if result.agent_response else request.surface,
        },
    )


@app.post("/api/h2h")
async def h2h(request: H2HRequest):
    """Return past head-to-head matches between two players."""
    try:
        matches = _lookup_h2h(request.player_a, request.player_b)
        return {"player_a": request.player_a, "player_b": request.player_b, "matches": matches}
    except Exception as exc:
        logger.exception("H2H error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tournaments")
async def get_tournaments():
    """Return list of tournaments with surface and category info."""
    df = _get_tournaments()
    if df.empty:
        return {"tournaments": []}

    matches = _get_matches()

    # Build category + surface per tournament from matches table.
    # Surface and category are decided by MAJORITY VOTE over the tournament's
    # matches, not by whichever row happens to come first — a handful of
    # mislabelled historical rows used to make e.g. the Australian Open show up
    # as "Grass". Names are canonicalised first so variants of the same event
    # (e.g. "French Open" / "Roland Garros") vote together.
    tourn_meta: dict[str, dict] = {}
    if not matches.empty and "tournament_name" in matches.columns:
        meta_src = matches[["tournament_name", "surface", "tournament_category"]].copy()
        meta_src["tournament_name"] = meta_src["tournament_name"].map(canonical_tournament)
        meta_src["surface"] = meta_src["surface"].map(canonical_surface)
        meta_src = meta_src[meta_src["tournament_name"] != ""]

        for name, grp in meta_src.groupby("tournament_name"):
            surf_votes = grp["surface"].value_counts()
            cat_votes = grp["tournament_category"].dropna().astype(str)
            cat_votes = cat_votes[cat_votes != ""].value_counts()
            tourn_meta[str(name)] = {
                "surface": str(surf_votes.idxmax()) if not surf_votes.empty else "hard",
                "category": str(cat_votes.idxmax()) if not cat_votes.empty else "ATP 250",
            }

    # Normalise: tournament_name may be a column or the index
    if "tournament_name" not in df.columns and df.index.name == "tournament_name":
        df = df.reset_index()
    elif "tournament_name" not in df.columns:
        df = df.reset_index().rename(columns={"index": "tournament_name"})

    # Collapse name variants of the same event into a single entry. Where a
    # canonical name has several profile rows, their t_* stats are combined as
    # a weighted average over t_n_matches so the UI shows one row per event.
    df = df.copy()
    df["tournament_name"] = df["tournament_name"].map(canonical_tournament)
    df = df[df["tournament_name"] != ""]

    _stat_cols = [
        "t_avg_aces_per_match", "t_avg_games_per_match", "t_avg_sets_played",
        "t_tiebreak_pct", "t_avg_service_pts_won_pct", "t_avg_return_pts_won_pct",
    ]
    if df["tournament_name"].duplicated().any():
        df["_w"] = pd.to_numeric(df.get("t_n_matches"), errors="coerce").fillna(0.0)
        agg_rows = []
        for name, grp in df.groupby("tournament_name"):
            w = grp["_w"]
            total = float(w.sum())
            row = {"tournament_name": name, "t_n_matches": total}
            for c in _stat_cols:
                if c not in grp.columns:
                    continue
                vals = pd.to_numeric(grp[c], errors="coerce")
                ok = vals.notna() & (w > 0)
                row[c] = float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else 0.0
            agg_rows.append(row)
        df = pd.DataFrame(agg_rows)

    tournaments = []
    for _, row_data in df.iterrows():
        tname = str(row_data.get("tournament_name", ""))
        if not tname:
            continue
        def _f(key: str) -> float:
            v = row_data.get(key)
            try:
                f = float(v)
                return 0.0 if (f != f) else f  # NaN check: NaN != NaN
            except (TypeError, ValueError):
                return 0.0

        meta = tourn_meta.get(tname, {})
        t = {
            "name": tname,
            "surface": meta.get("surface", "hard"),
            "category": meta.get("category", "ATP 250"),
            "t_avg_aces_per_match": _f("t_avg_aces_per_match"),
            "t_avg_games_per_match": _f("t_avg_games_per_match"),
            "t_avg_sets_played": _f("t_avg_sets_played"),
            "t_tiebreak_pct": _f("t_tiebreak_pct"),
            "t_avg_service_pts_won_pct": _f("t_avg_service_pts_won_pct"),
            "t_avg_return_pts_won_pct": _f("t_avg_return_pts_won_pct"),
            "t_n_matches": int(_f("t_n_matches")),
        }
        tournaments.append(t)

    tournaments.sort(key=lambda x: (x["category"], x["name"]))
    return {"tournaments": tournaments}


@app.post("/api/player_stats")
async def get_player_stats(request: PlayerStatsRequest):
    """Return profile stats for a player by name."""
    df = _get_player_profiles()
    if df.empty:
        return {"found": False, "stats": {}}

    name_query = request.player_name.strip().lower()

    # player profiles are indexed by player_id; look for name column
    player_row = None
    if "player_name" in df.columns:
        name_col = df["player_name"].fillna("").str.lower()
        # Try exact match first, then partial
        exact = df[name_col == name_query]
        if not exact.empty:
            player_row = exact.iloc[0]
        else:
            # surname match
            parts = name_query.split()
            surname = parts[-1] if parts else name_query
            partial = df[name_col.str.contains(surname, regex=False)]
            if not partial.empty:
                player_row = partial.iloc[0]

    if player_row is None:
        return {"found": False, "stats": {}}

    # Extract only pa_* numeric stats
    stat_cols = [c for c in df.columns if c.startswith("pa_") and pd.api.types.is_numeric_dtype(df[c])]
    stats = {}
    for c in stat_cols:
        v = player_row.get(c)
        if pd.notna(v):
            stats[c] = round(float(v), 4)

    # Include elo if present
    for elo_col in ["elo", "elo_clay", "elo_hard", "elo_grass"]:
        if elo_col in df.columns:
            v = player_row.get(elo_col)
            if pd.notna(v):
                stats[elo_col] = round(float(v), 1)

    return {
        "found": True,
        "player_name": str(player_row.get("player_name", request.player_name)),
        "stats": stats,
    }


@app.post("/api/explain")
async def explain(request: ExplainRequest):
    """Call Groq LLM to produce a natural-language explanation of the predictions."""
    from groq import Groq
    from config.settings import settings

    api_key = os.environ.get("GROQ_API_KEY", "") or settings.agent.groq_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")
    groq_client = Groq(api_key=api_key)

    pa, pb = request.player_a, request.player_b

    pred = request.prediction
    prob_a = pred.get("prob_player_a_wins") or pred.get("win_probability_player_a") or pred.get("winner_prob")
    prob_b = pred.get("win_probability_player_b")
    predicted_winner = pred.get("predicted_winner", pa if (prob_a or 0) >= 0.5 else pb)

    if prob_a is not None:
        prob_winner = float(prob_a) if predicted_winner == pa else (float(prob_b) if prob_b is not None else 1 - float(prob_a))
    else:
        prob_winner = None
    prob_str = f"{prob_winner*100:.1f}%" if prob_winner is not None else "N/A"

    # Sort features: biggest absolute SHAP impact first
    sorted_feats = sorted(request.explanation, key=lambda f: abs(f.get("value", 0)), reverse=True)

    # Build feature lines WITHOUT raw SHAP numbers — only label, value, and who it favors.
    # Hiding SHAP signs prevents the LLM from re-summing and overriding the predicted winner.
    winner_feats = []
    loser_feats = []
    for f in sorted_feats:
        fv = f.get("feature_value")
        val_str = f"{fv:.3f}" if fv is not None else "N/A"
        favors = pa if f["favors"] == "player_a" else pb
        line = f"  - {f['label']}: {val_str} (favorisce {favors})"
        if favors == predicted_winner:
            winner_feats.append(line)
        else:
            loser_feats.append(line)

    feat_text = ""
    if winner_feats:
        feat_text += f"Fattori che favoriscono {predicted_winner}:\n" + "\n".join(winner_feats[:4])
    if loser_feats:
        feat_text += f"\n\nFattori che favoriscono {pa if predicted_winner == pb else pb}:\n" + "\n".join(loser_feats[:2])

    system_msg = (
        f"Sei un analista tennistico. Il modello di machine learning ha già determinato che "
        f"{predicted_winner} vincerà con probabilità {prob_str}. "
        f"NON mettere in discussione questo risultato e NON calcolare nulla autonomamente. "
        f"Il tuo UNICO compito è scrivere 3-4 frasi in italiano che spieghino PERCHÉ {predicted_winner} "
        f"è favorito, usando i dati forniti come supporto narrativo. "
        f"Inizia la risposta con '{predicted_winner} è favorito' oppure 'Il modello prevede {predicted_winner}'."
    )

    user_prompt = (
        f"Partita: {pa} vs {pb} su {request.surface}.\n"
        f"Vincitore previsto: {predicted_winner} (probabilità {prob_str}).\n\n"
        f"{feat_text}\n\n"
        f"Scrivi la spiegazione in 3-4 frasi."
    )

    try:
        response = groq_client.chat.completions.create(
            model=settings.agent.groq_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        return {"explanation": text}
    except Exception as exc:
        logger.exception("Groq explain error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
async def predict_structured(request: PredictRequest):
    """Run prediction from a structured match request."""
    try:
        req = MatchRequest(
            player_a=request.player_a,
            player_b=request.player_b,
            surface=request.surface,
            tournament_category=request.tournament_category,
            round=request.round,
            best_of=request.best_of,
            match_date=request.match_date,
        )
        pred = get_predictor().predict(req)
        return pred.to_dict()
    except Exception as exc:
        logger.exception("Predictor error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("serving.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start()
