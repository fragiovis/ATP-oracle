"""Primary agent: orchestrates the full NL → prediction pipeline.

Responsibilities:
  1. Accept natural-language query.
  2. Call QueryParser to get a structured AgentResponse.
  3. If clarification is needed, return the clarification question.
  4. Otherwise, call PredictionRouter to get a MatchPrediction.
  5. Format the final answer as a human-readable string + structured dict.

This is the single entry point that user-facing code (CLI, API, etc.)
should call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.parser import QueryParser
from agent.router import PredictionRouter
from agent.schema import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Unified output of the primary agent for a single user query."""

    query: str
    agent_response: AgentResponse
    prediction: dict[str, Any] | None
    answer: str                   # human-readable text
    requires_clarification: bool
    clarification_question: str | None


class PrimaryAgent:
    """
    The primary agent that translates a natural-language tennis prediction
    query into a structured prediction result.

    Usage::

        agent = PrimaryAgent()
        result = agent.run("Who is favored between Sinner and Alcaraz on clay?")
        print(result.answer)
    """

    def __init__(self):
        self._parser = QueryParser()
        self._router = PredictionRouter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, ui_best_of: int = 3, ui_surface: str = "hard") -> AgentResult:
        """Process a user query end-to-end."""
        logger.info("Agent received query: %r", query)

        # Step 1 – Parse
        agent_resp = self._parser.parse(query)
        logger.debug("Parsed: intent=%s entities=%s", agent_resp.intent, agent_resp.entities)

        # Step 2 – Clarification check
        if agent_resp.requires_clarification:
            question = agent_resp.clarification_question or "Could you provide more details?"
            return AgentResult(
                query=query,
                agent_response=agent_resp,
                prediction=None,
                answer=question,
                requires_clarification=True,
                clarification_question=question,
            )

        # Step 3 – Route → predict
        match_prediction = self._router.route(
            agent_resp, ui_best_of=ui_best_of, ui_surface=ui_surface
        )

        if match_prediction is None:
            return AgentResult(
                query=query,
                agent_response=agent_resp,
                prediction=None,
                answer="I couldn't determine the players or prediction type. Could you rephrase?",
                requires_clarification=True,
                clarification_question="Please specify the two players and what you'd like to predict.",
            )

        pred_dict = match_prediction.to_dict()

        # Step 4 – Format answer
        answer = self._format_answer(agent_resp, pred_dict)

        return AgentResult(
            query=query,
            agent_response=agent_resp,
            prediction=pred_dict,
            answer=answer,
            requires_clarification=False,
            clarification_question=None,
        )

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------

    def _format_answer(self, resp: AgentResponse, pred: dict[str, Any]) -> str:
        intent = resp.intent
        e = resp.entities
        pa = e.player_a or "Player A"
        pb = e.player_b or "Player B"

        parts: list[str] = []

        if intent == "predict_match_winner_probability":
            p_a = pred.get("win_probability_player_a")
            p_b = pred.get("win_probability_player_b")
            if p_a is not None:
                parts.append(
                    f"{pa} has a {p_a:.1%} chance of winning; "
                    f"{pb} has a {p_b:.1%} chance."
                )

        elif intent == "predict_match_winner_label":
            winner = pred.get("predicted_winner")
            p = pred.get("win_probability_player_a") or pred.get("win_probability_player_b")
            if winner:
                parts.append(f"Predicted winner: {winner}.")

        elif intent == "predict_player_aces":
            v = pred.get("expected_aces_player_a")
            if v is not None:
                parts.append(f"{pa} is expected to hit approximately {v:.1f} aces.")

        elif intent == "predict_total_aces":
            v = pred.get("expected_total_aces")
            if v is not None:
                parts.append(f"Expected total aces in the match: {v:.1f}.")

        elif intent == "predict_total_games":
            v = pred.get("expected_total_games")
            if v is not None:
                parts.append(f"Expected total games in the match: {v:.1f}.")

        elif intent == "predict_deciding_set_probability":
            v = pred.get("prob_deciding_set")
            if v is not None:
                parts.append(f"Probability of a deciding set: {v:.1%}.")

        elif intent == "predict_tiebreak_probability":
            v = pred.get("prob_tiebreak")
            if v is not None:
                parts.append(f"Probability of at least one tie-break: {v:.1%}.")

        elif intent == "predict_first_set_winner_probability":
            v = pred.get("prob_player_a_first_set")
            if v is not None:
                parts.append(
                    f"{pa} is favored to win the first set with probability {v:.1%} "
                    f"(implied: {pb} {1-v:.1%})."
                )

        if not parts:
            parts.append("Prediction complete. See structured output for details.")

        # Append surface/context context if known
        ctx_parts = []
        if e.surface:
            ctx_parts.append(f"surface: {e.surface}")
        if e.tournament_category:
            ctx_parts.append(e.tournament_category)
        if ctx_parts:
            parts.append(f"[Context: {', '.join(ctx_parts)}]")

        return " ".join(parts)
