"""Shared data contracts for the primary agent layer.

These types define the structured representation the agent produces
*before* the serving layer runs inference.  Keeping them in a separate
module lets us swap the NLP backend without touching the serving layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

Intent = Literal[
    "predict_match_winner_probability",
    "predict_match_winner_label",
    "predict_player_aces",
    "predict_total_aces",
    "predict_total_games",
    "predict_deciding_set_probability",
    "predict_tiebreak_probability",
    "predict_first_set_winner_probability",
    "unknown",
]

# Maps agent intent → model_id used by the serving layer
INTENT_TO_MODEL: dict[str, str] = {
    "predict_match_winner_probability": "winner",
    "predict_match_winner_label": "winner",
    "predict_player_aces": "player_aces",
    "predict_total_aces": "total_aces",
    "predict_total_games": "total_games",
    "predict_deciding_set_probability": "deciding_set",
    "predict_tiebreak_probability": "tiebreak",
    "predict_first_set_winner_probability": "first_set",
}


# ---------------------------------------------------------------------------
# Extracted entities
# ---------------------------------------------------------------------------

class MatchEntities(BaseModel):
    """Structured entities extracted from the user's natural-language query."""

    player_a: str | None = Field(None, description="First player name")
    player_b: str | None = Field(None, description="Second player name")
    surface: str | None = Field(None, description="Court surface (clay/hard/grass)")
    tournament_category: str | None = Field(None, description="ATP 250/500/1000/GS")
    round: str | None = Field(None, description="Tournament round, e.g. SF, F, QF")
    best_of: int | None = Field(None, description="3 or 5")
    match_date: str | None = Field(None)


# ---------------------------------------------------------------------------
# Agent response contract
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """Normalised output produced by the primary agent for every user query."""

    intent: Intent = "unknown"
    entities: MatchEntities = Field(default_factory=MatchEntities)
    requires_clarification: bool = False
    clarification_question: str | None = None
    raw_query: str = ""
