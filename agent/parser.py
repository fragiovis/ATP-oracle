"""NLP parser: extract intent and entities from a user query.

Uses the Groq API (llama-3.3-70b-versatile) for intent detection and
entity extraction.  Falls back to a regex heuristic if the API call fails
or GROQ_API_KEY is not set.

Set your key before starting the server:
    export GROQ_API_KEY="gsk_..."
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from groq import Groq

from agent.schema import AgentResponse, Intent, MatchEntities
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt – tuned for Llama 3.3 70B
#
# Llama models can prepend conversational text before the JSON, so the prompt:
#  1. Repeats "output ONLY JSON" multiple times in different ways
#  2. Shows a complete worked example (few-shot)
#  3. Lists every allowed value explicitly
#  4. Tells the model its response MUST start with { and end with }
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a JSON extraction engine for an ATP tennis prediction API.

YOUR ONLY JOB: read the user message and return a single JSON object.
DO NOT write any text before or after the JSON.
DO NOT explain your answer.
DO NOT use markdown or code fences.
Your response MUST start with the character { and end with the character }.

──────────────────────────────────────────
EXACT OUTPUT SCHEMA
──────────────────────────────────────────
{
  "intent": <string>,
  "entities": {
    "player_a": <string or null>,
    "player_b": <string or null>,
    "surface": <string or null>,
    "tournament_category": <string or null>,
    "round": <string or null>,
    "best_of": <integer or null>,
    "match_date": <string or null>
  },
  "requires_clarification": <boolean>,
  "clarification_question": <string or null>
}

──────────────────────────────────────────
ALLOWED VALUES PER FIELD
──────────────────────────────────────────
intent — pick EXACTLY one of these strings:
  "predict_match_winner_probability"   → user asks who is favored, win odds, win chances
  "predict_match_winner_label"         → user asks who wins, who will win, predict the winner
  "predict_player_aces"                → user asks how many aces ONE specific player will hit
  "predict_total_aces"                 → user asks total aces in the match (both players)
  "predict_total_games"                → user asks how many games total in the match
  "predict_deciding_set_probability"   → user asks about deciding set, 3rd set, 5th set, full distance
  "predict_tiebreak_probability"       → user asks about tie-break, tiebreak
  "predict_first_set_winner_probability" → user asks who wins the first set
  "unknown"                            → cannot determine

surface — one of: "clay", "hard", "grass", or null
tournament_category — one of: "ATP 250", "ATP 500", "ATP 1000", "Grand Slam", or null
round — one of: "R128", "R64", "R32", "R16", "QF", "SF", "F", or null
best_of — integer 3 or 5, or null
match_date — "YYYY-MM-DD" string, or null

──────────────────────────────────────────
RULES
──────────────────────────────────────────
- Set requires_clarification = true ONLY when:
    (a) intent is "unknown" AND you cannot guess it, OR
    (b) BOTH player_a AND player_b are null AND the intent requires players
- When requires_clarification is true, set clarification_question to the
  single most important question (ask for players first, then intent, then surface).
- When requires_clarification is false, set clarification_question to null.
- Copy player names exactly as the user wrote them (do not correct spelling).
- If surface is missing but the prediction works without it, leave surface null
  and set requires_clarification to false.

──────────────────────────────────────────
EXAMPLES — study these carefully
──────────────────────────────────────────

User: "Who is favored between Sinner and Alcaraz on clay?"
Output:
{"intent":"predict_match_winner_probability","entities":{"player_a":"Sinner","player_b":"Alcaraz","surface":"clay","tournament_category":null,"round":null,"best_of":null,"match_date":null},"requires_clarification":false,"clarification_question":null}

User: "How many aces will Zverev hit against Medvedev at Wimbledon?"
Output:
{"intent":"predict_player_aces","entities":{"player_a":"Zverev","player_b":"Medvedev","surface":"grass","tournament_category":"Grand Slam","round":null,"best_of":5,"match_date":null},"requires_clarification":false,"clarification_question":null}

User: "Total games expected in the match?"
Output:
{"intent":"predict_total_games","entities":{"player_a":null,"player_b":null,"surface":null,"tournament_category":null,"round":null,"best_of":null,"match_date":null},"requires_clarification":true,"clarification_question":"Which two players are playing?"}

User: "Will there be a tiebreak in Djokovic vs Rublev on hard?"
Output:
{"intent":"predict_tiebreak_probability","entities":{"player_a":"Djokovic","player_b":"Rublev","surface":"hard","tournament_category":null,"round":null,"best_of":null,"match_date":null},"requires_clarification":false,"clarification_question":null}

User: "What's the probability this goes to a deciding set?"
Output:
{"intent":"predict_deciding_set_probability","entities":{"player_a":null,"player_b":null,"surface":null,"tournament_category":null,"round":null,"best_of":null,"match_date":null},"requires_clarification":true,"clarification_question":"Which two players are playing?"}

User: "First set winner between Fritz and Rublev at Roland Garros SF?"
Output:
{"intent":"predict_first_set_winner_probability","entities":{"player_a":"Fritz","player_b":"Rublev","surface":"clay","tournament_category":"Grand Slam","round":"SF","best_of":5,"match_date":null},"requires_clarification":false,"clarification_question":null}

──────────────────────────────────────────
Now extract the JSON from the next user message.
Remember: respond with ONLY the JSON object, nothing else.
──────────────────────────────────────────"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class QueryParser:
    """Parse a natural-language tennis query into a structured AgentResponse."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY") or settings.agent.groq_api_key
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not set – will use heuristic fallback for all queries."
            )
            self._client = None
        else:
            self._client = Groq(api_key=api_key)
        self._model = settings.agent.groq_model
        self._max_tokens = settings.agent.max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, query: str) -> AgentResponse:
        """
        Parse *query* and return an AgentResponse.
        Falls back to regex-based heuristic parsing if the API call fails.
        """
        if self._client is None:
            logger.debug("No Groq client – using heuristic parser.")
            return self._heuristic_parse(query)
        try:
            raw = self._call_llm(query)
            return self._parse_llm_response(raw, query)
        except Exception as exc:
            logger.warning("Groq parse failed (%s) – falling back to heuristic.", exc)
            return self._heuristic_parse(query)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content.strip()

    def _temperature(self) -> float:
        return settings.agent.temperature

    def _parse_llm_response(self, raw: str, original_query: str) -> AgentResponse:
        # Strip any accidental markdown fences or leading/trailing text
        # Strategy: find the first { and the last } and extract that slice
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")

        json_str = raw[start : end + 1]
        data: dict[str, Any] = json.loads(json_str)

        entities_data = data.get("entities") or {}
        entities = MatchEntities(
            player_a=entities_data.get("player_a"),
            player_b=entities_data.get("player_b"),
            surface=entities_data.get("surface"),
            tournament_category=entities_data.get("tournament_category"),
            round=entities_data.get("round"),
            best_of=entities_data.get("best_of"),
            match_date=entities_data.get("match_date"),
        )

        raw_intent = data.get("intent", "unknown")
        # Guard against the model returning an invalid intent string
        valid_intents = {
            "predict_match_winner_probability", "predict_match_winner_label",
            "predict_player_aces", "predict_total_aces", "predict_total_games",
            "predict_deciding_set_probability", "predict_tiebreak_probability",
            "predict_first_set_winner_probability", "unknown",
        }
        intent: Intent = raw_intent if raw_intent in valid_intents else "unknown"  # type: ignore[assignment]

        return AgentResponse(
            intent=intent,
            entities=entities,
            requires_clarification=bool(data.get("requires_clarification", False)),
            clarification_question=data.get("clarification_question"),
            raw_query=original_query,
        )

    # ------------------------------------------------------------------
    # Heuristic fallback (no API call needed)
    # ------------------------------------------------------------------

    def _heuristic_parse(self, query: str) -> AgentResponse:
        q = query.lower()

        intent: Intent = "unknown"
        if any(w in q for w in ["win probability", "favored", "favourite", "favorite", "odds", "chances"]):
            intent = "predict_match_winner_probability"
        elif any(w in q for w in ["who wins", "who will win", "predict winner", "chi vince"]):
            intent = "predict_match_winner_label"
        elif "total aces" in q or ("aces" in q and "total" in q):
            intent = "predict_total_aces"
        elif "ace" in q:
            intent = "predict_player_aces"
        elif any(w in q for w in ["total games", "how many games", "quanti game"]):
            intent = "predict_total_games"
        elif any(w in q for w in ["deciding set", "3 sets", "five sets", "full distance", "set decisivo"]):
            intent = "predict_deciding_set_probability"
        elif any(w in q for w in ["tie-break", "tiebreak", "tie break", "tiebreaker"]):
            intent = "predict_tiebreak_probability"
        elif "first set" in q or "primo set" in q:
            intent = "predict_first_set_winner_probability"

        surface = None
        for s in ("clay", "terra", "hard", "grass", "erba"):
            if s in q:
                surface = "clay" if s == "terra" else ("grass" if s == "erba" else s)
                break

        player_a, player_b = None, None
        vs_match = re.search(
            r"(?:between\s+|fra\s+|)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            r"(?:\s+and\s+|\s+vs\.?\s+|\s+contro\s+|\s+against\s+)"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            query,
        )
        if vs_match:
            player_a, player_b = vs_match.group(1), vs_match.group(2)

        entities = MatchEntities(player_a=player_a, player_b=player_b, surface=surface)
        needs_clarification = (player_a is None and player_b is None) or intent == "unknown"
        clarification = None
        if player_a is None and player_b is None:
            clarification = "Which two players are you asking about?"
        elif intent == "unknown":
            clarification = (
                "What would you like to predict? "
                "(winner, aces, total games, tie-break, deciding set, or first set)"
            )

        return AgentResponse(
            intent=intent,
            entities=entities,
            requires_clarification=needs_clarification,
            clarification_question=clarification,
            raw_query=query,
        )
