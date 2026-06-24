"""Betting backtest evaluator — separate from the standard ML evaluation pipeline.

Simulates flat-stake betting using model confidence as the filter criterion.

Strategy:
  For each match the model assigns prob_a = P(player_a wins).
  - If prob_a >= (0.5 + min_edge)  → bet on player_a at their B365 odds
  - If prob_b >= (0.5 + min_edge)  → bet on player_b at their B365 odds
  - Otherwise                       → skip
  Then check the real match outcome to compute P&L.

min_edge is therefore a confidence threshold on the model's predicted
probability, NOT a comparison against bookmaker implied probability.

Usage:
    atp-eval betting
    atp-eval betting --min-edge 0.10 --stake 10
    atp-eval betting --min-edge 0.15 --odds ps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BettingResult:
    n_matches: int = 0
    n_bets: int = 0
    n_won: int = 0
    total_staked: float = 0.0
    total_return: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    avg_confidence: float = 0.0   # avg model prob on the predicted winner
    avg_odds: float = 0.0         # avg bookmaker odds collected on bets
    profit_by_year: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------

class BettingEvaluator:
    """
    Backtest flat-stake betting filtered by model confidence.

    Input DataFrame must have columns:
        year, winner_label (0/1 — 1 if player_a won),
        prob_a             (model P(player_a wins)),
        {odds_col}_a/b     (bookmaker decimal odds, e.g. b365_a, b365_b)
    """

    def __init__(
        self,
        min_edge: float = 0.10,
        unit_stake: float = 10.0,
        odds_col: str = "b365",
    ):
        self.min_edge = min_edge          # confidence threshold above 0.5
        self.unit_stake = unit_stake
        self.odds_col = odds_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> dict[str, BettingResult]:
        """Run model-confidence strategy + bookmaker-favourite baseline."""
        df = self._prepare(df)
        if df.empty:
            logger.warning("No rows with valid odds + predictions for backtest.")
            return {}

        threshold = 0.5 + self.min_edge
        n_eligible = ((df["prob_a"] >= threshold) | (df["prob_a"] <= 1 - threshold)).sum()
        logger.info(
            "Betting backtest: %d total matches, %d pass confidence threshold (prob >= %.0f%%)",
            len(df), n_eligible, threshold * 100,
        )

        return {
            "model_confidence": self._backtest_model(df),
            "bookie_favourite":  self._backtest_bookie(df),
        }

    def summary_table(self, results: dict[str, BettingResult]) -> str:
        threshold = 0.5 + self.min_edge
        r_model  = results.get("model_confidence")
        r_bookie = results.get("bookie_favourite")

        lines = [
            f"\n{'='*62}",
            f"  BETTING BACKTEST  —  confidence threshold ≥ {threshold:.0%}",
            f"  (bet on player_a if prob_a ≥ {threshold:.0%}, on player_b if prob_b ≥ {threshold:.0%})",
            f"{'='*62}",
        ]

        for label, r in [
            (f"MODEL  (min confidence {threshold:.0%})", r_model),
            ("BASELINE  (always bet bookie favourite)", r_bookie),
        ]:
            if r is None:
                continue
            pnl = r.total_return - r.total_staked
            lines += [
                f"\n  ── {label} ──",
                f"  Matches evaluated : {r.n_matches}",
                f"  Bets placed       : {r.n_bets}  ({r.n_bets / max(r.n_matches, 1):.1%} of matches)",
                f"  Bets won          : {r.n_won}  ({r.win_rate:.1%})",
                f"  Avg model prob    : {r.avg_confidence:.1%}",
                f"  Avg odds          : {r.avg_odds:.2f}",
                f"  Total staked      : €{r.total_staked:.2f}",
                f"  Total return      : €{r.total_return:.2f}",
                f"  Net P&L           : €{pnl:+.2f}",
                f"  ROI               : {r.roi:+.2%}",
            ]
            if r.profit_by_year:
                lines.append("  P&L by year:")
                for yr, p in sorted(r.profit_by_year.items()):
                    lines.append(f"    {yr}: €{p:+.2f}")

        lines.append(f"\n{'='*62}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        odds_a = f"{self.odds_col}_a"
        odds_b = f"{self.odds_col}_b"

        required = {"prob_a", "winner_label", odds_a, odds_b}
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing columns for backtest: %s", missing)
            return pd.DataFrame()

        df = df.copy()
        df = df.dropna(subset=[odds_a, odds_b, "prob_a", "winner_label"])
        df = df[(df[odds_a] > 1.0) & (df[odds_b] > 1.0)]
        return df

    def _backtest_model(self, df: pd.DataFrame) -> BettingResult:
        """Bet when model confidence >= 0.5 + min_edge on either player.

        In the odds table player_a is always the real winner (tennis-data
        convention), so b365_a/b365_b are not symmetric after role randomisation.
        We collect the odds for whichever player we bet on using winner_label
        to identify which odds column corresponds to the actual winner.
        """
        odds_a = f"{self.odds_col}_a"
        odds_b = f"{self.odds_col}_b"
        threshold = 0.5 + self.min_edge

        result = BettingResult(n_matches=len(df))
        pnl_by_year: dict[str, float] = {}

        for row in df.itertuples(index=False):
            prob_a = float(row.prob_a)
            won_a  = int(row.winner_label) == 1
            yr     = str(int(getattr(row, "year", 0)))

            if prob_a >= threshold:
                bet_on_a = True
            elif prob_a <= 1 - threshold:
                bet_on_a = False
            else:
                continue

            # Collect the odds for the player we are betting on.
            # winner_label=1 means player_a won, so b365_a is the winner's odds.
            # After role randomisation, won_a may be False even for td_* rows.
            if bet_on_a:
                odds = getattr(row, odds_a) if won_a else getattr(row, odds_b)
                won  = won_a
            else:
                odds = getattr(row, odds_b) if won_a else getattr(row, odds_a)
                won  = not won_a
            conf = prob_a if bet_on_a else 1 - prob_a

            pnl = (odds - 1) * self.unit_stake if won else -self.unit_stake
            result.n_bets          += 1
            result.n_won           += int(won)
            result.total_staked    += self.unit_stake
            result.total_return    += self.unit_stake + pnl
            result.avg_confidence  += conf
            result.avg_odds        += odds
            pnl_by_year[yr]         = pnl_by_year.get(yr, 0.0) + pnl

        if result.n_bets:
            result.roi            = (result.total_return - result.total_staked) / result.total_staked
            result.win_rate       = result.n_won / result.n_bets
            result.avg_confidence /= result.n_bets
            result.avg_odds       /= result.n_bets
        result.profit_by_year = pnl_by_year
        return result

    def _backtest_bookie(self, df: pd.DataFrame) -> BettingResult:
        """Baseline: always bet on the bookmaker's favourite (lower odds)."""
        odds_a = f"{self.odds_col}_a"
        odds_b = f"{self.odds_col}_b"

        result = BettingResult(n_matches=len(df))
        pnl_by_year: dict[str, float] = {}

        for row in df.itertuples(index=False):
            won_a  = int(row.winner_label) == 1
            yr     = str(int(getattr(row, "year", 0)))
            oa, ob = getattr(row, odds_a), getattr(row, odds_b)

            # b365_a = winner's odds, b365_b = loser's odds (tennis-data convention)
            # Bookie favourite = lower odds = b365_a (the real winner)
            bet_on_winner = oa <= ob
            if bet_on_winner:
                odds = oa
                won  = won_a   # betting on real winner side
            else:
                odds = ob
                won  = not won_a

            prob = float(row.prob_a) if won_a else 1 - float(row.prob_a)

            pnl = (odds - 1) * self.unit_stake if won else -self.unit_stake
            result.n_bets          += 1
            result.n_won           += int(won)
            result.total_staked    += self.unit_stake
            result.total_return    += self.unit_stake + pnl
            result.avg_confidence  += prob
            result.avg_odds        += odds
            pnl_by_year[yr]         = pnl_by_year.get(yr, 0.0) + pnl

        if result.n_bets:
            result.roi            = (result.total_return - result.total_staked) / result.total_staked
            result.win_rate       = result.n_won / result.n_bets
            result.avg_confidence /= result.n_bets
            result.avg_odds       /= result.n_bets
        result.profit_by_year = pnl_by_year
        return result
