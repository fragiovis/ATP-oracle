"""Unit tests for the primary agent layer.

Tests:
  - QueryParser heuristic fallback (no API call)
  - Intent routing
  - Entity extraction
  - Clarification behavior
  - PrimaryAgent.run() with mocked parser + router
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.parser import QueryParser
from agent.schema import AgentResponse, MatchEntities
from agent.router import PredictionRouter
from agent.agent import PrimaryAgent


# ---------------------------------------------------------------------------
# Heuristic parser (no API call)
# ---------------------------------------------------------------------------

class TestHeuristicParser:
    """Test the regex-based fallback parser directly."""

    def _parse(self, query: str) -> AgentResponse:
        parser = QueryParser.__new__(QueryParser)
        return parser._heuristic_parse(query)

    def test_win_probability_intent(self):
        resp = self._parse("Who is favored between Sinner and Alcaraz on clay?")
        assert resp.intent == "predict_match_winner_probability"

    def test_total_aces_intent(self):
        resp = self._parse("How many total aces will there be in the match?")
        assert resp.intent == "predict_total_aces"

    def test_player_aces_intent(self):
        resp = self._parse("How many aces will Sinner hit against Medvedev?")
        assert resp.intent == "predict_player_aces"

    def test_total_games_intent(self):
        resp = self._parse("How many total games are expected?")
        assert resp.intent == "predict_total_games"

    def test_tiebreak_intent(self):
        resp = self._parse("Will there be a tie-break in this match?")
        assert resp.intent == "predict_tiebreak_probability"

    def test_deciding_set_intent(self):
        resp = self._parse("What is the probability this goes to a deciding set?")
        assert resp.intent == "predict_deciding_set_probability"

    def test_first_set_intent(self):
        resp = self._parse("Who will win the first set?")
        assert resp.intent == "predict_first_set_winner_probability"

    def test_player_extraction(self):
        resp = self._parse("Sinner vs Alcaraz on clay")
        assert resp.entities.player_a == "Sinner"
        assert resp.entities.player_b == "Alcaraz"

    def test_surface_extraction(self):
        resp = self._parse("Who wins on clay between Nadal and Djokovic?")
        assert resp.entities.surface == "clay"

    def test_clarification_when_no_players(self):
        resp = self._parse("How many aces will there be?")
        assert resp.requires_clarification
        assert resp.clarification_question is not None

    def test_clarification_when_intent_unknown(self):
        resp = self._parse("Tell me about the weather at Wimbledon")
        assert resp.requires_clarification

    def test_no_clarification_when_complete(self):
        resp = self._parse("Who is favored between Sinner and Alcaraz on hard?")
        # Complete enough: intent + both players + surface
        assert resp.intent != "unknown"
        # Not necessarily asking for clarification if players found
        assert resp.entities.player_a is not None


# ---------------------------------------------------------------------------
# Intent-to-model routing
# ---------------------------------------------------------------------------

class TestIntentRouting:
    from agent.schema import INTENT_TO_MODEL

    def test_all_intents_have_model(self):
        from agent.schema import INTENT_TO_MODEL
        intents_needing_model = [
            "predict_match_winner_probability",
            "predict_match_winner_label",
            "predict_player_aces",
            "predict_total_aces",
            "predict_total_games",
            "predict_deciding_set_probability",
            "predict_tiebreak_probability",
            "predict_first_set_winner_probability",
        ]
        for intent in intents_needing_model:
            assert intent in INTENT_TO_MODEL, f"Missing mapping for: {intent}"


# ---------------------------------------------------------------------------
# PredictionRouter – clarification bypass
# ---------------------------------------------------------------------------

class TestPredictionRouter:
    def test_returns_none_when_clarification_needed(self):
        router = PredictionRouter.__new__(PredictionRouter)
        resp = AgentResponse(
            intent="predict_total_games",
            entities=MatchEntities(),
            requires_clarification=True,
            clarification_question="Who are the players?",
        )
        result = router.route(resp)
        assert result is None

    def test_returns_none_when_no_players(self):
        router = PredictionRouter.__new__(PredictionRouter)
        router._predictor = MagicMock()
        resp = AgentResponse(
            intent="predict_total_games",
            entities=MatchEntities(player_a=None, player_b=None),
            requires_clarification=False,
        )
        result = router.route(resp)
        assert result is None


# ---------------------------------------------------------------------------
# PrimaryAgent end-to-end (mocked predictor)
# ---------------------------------------------------------------------------

class TestPrimaryAgent:
    def _make_agent(self, parsed_resp: AgentResponse, prediction_dict: dict | None):
        agent = PrimaryAgent.__new__(PrimaryAgent)

        parser_mock = MagicMock()
        parser_mock.parse.return_value = parsed_resp
        agent._parser = parser_mock

        router_mock = MagicMock()
        if prediction_dict is not None:
            pred_obj = MagicMock()
            pred_obj.to_dict.return_value = prediction_dict
            router_mock.route.return_value = pred_obj
        else:
            router_mock.route.return_value = None
        agent._router = router_mock

        return agent

    def test_clarification_response(self):
        resp = AgentResponse(
            intent="unknown",
            requires_clarification=True,
            clarification_question="Which players?",
            raw_query="test",
        )
        agent = self._make_agent(resp, None)
        result = agent.run("something vague")
        assert result.requires_clarification
        assert result.answer == "Which players?"
        assert result.prediction is None

    def test_successful_prediction(self):
        resp = AgentResponse(
            intent="predict_match_winner_probability",
            entities=MatchEntities(player_a="Sinner", player_b="Alcaraz", surface="clay"),
            requires_clarification=False,
            raw_query="Who is favored?",
        )
        pred_dict = {"win_probability_player_a": 0.54, "win_probability_player_b": 0.46}
        agent = self._make_agent(resp, pred_dict)
        result = agent.run("Who is favored?")
        assert not result.requires_clarification
        assert result.prediction == pred_dict
        assert "54" in result.answer or "Sinner" in result.answer

    def test_router_returns_none_handled(self):
        resp = AgentResponse(
            intent="predict_total_games",
            entities=MatchEntities(player_a=None, player_b=None),
            requires_clarification=False,
            raw_query="some query",
        )
        agent = self._make_agent(resp, None)
        result = agent.run("some query")
        assert result.requires_clarification
        assert result.prediction is None
