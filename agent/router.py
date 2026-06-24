"""Router: map an AgentResponse to a MatchRequest and call the predictor.

The router applies default values for missing entities, then calls the
serving layer.  It is intentionally decoupled from the NLP parser so
either side can be replaced independently.
"""

from __future__ import annotations

import logging

from agent.schema import AgentResponse, INTENT_TO_MODEL
from config.settings import settings
from serving.predictor import MatchPredictor, MatchRequest, MatchPrediction

logger = logging.getLogger(__name__)


class PredictionRouter:
    """
    Takes a structured AgentResponse and produces a MatchPrediction.

    If the response requires clarification, returns None instead of
    calling the predictor.
    """

    def __init__(self):
        self._predictor = MatchPredictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        agent_response: AgentResponse,
        ui_best_of: int = 3,
        ui_surface: str = "hard",
    ) -> MatchPrediction | None:
        """
        Route the agent response to the correct predictor.

        Returns
        -------
        MatchPrediction  – if prediction can be made
        None             – if clarification is required
        """
        if agent_response.requires_clarification:
            logger.info(
                "Clarification required: %s", agent_response.clarification_question
            )
            return None

        req = self._build_request(agent_response, ui_best_of=ui_best_of, ui_surface=ui_surface)
        if req is None:
            return None

        logger.info(
            "Routing intent=%s players=(%s, %s) surface=%s best_of=%d",
            agent_response.intent,
            req.player_a,
            req.player_b,
            req.surface,
            req.best_of,
        )

        return self._predictor.predict(req)

    # ------------------------------------------------------------------
    # Request construction with defaults
    # ------------------------------------------------------------------

    def _build_request(
        self,
        resp: AgentResponse,
        ui_best_of: int = 3,
        ui_surface: str = "hard",
    ) -> MatchRequest | None:
        e = resp.entities
        cfg = settings.agent

        player_a = e.player_a
        player_b = e.player_b

        if not player_a or not player_b:
            logger.warning("Cannot route – missing player names.")
            return None

        # Query takes precedence over UI selectors; UI is the fallback
        surface  = (e.surface or ui_surface or cfg.default_surface or "hard").lower()
        best_of  = e.best_of or ui_best_of or cfg.default_best_of

        return MatchRequest(
            player_a=player_a,
            player_b=player_b,
            surface=surface,
            tournament_category=e.tournament_category or "ATP 250",
            round=e.round or "R32",
            best_of=best_of,
            match_date=e.match_date,
        )
