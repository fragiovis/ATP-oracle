"""Tournament-level aggregate statistics.

Computes per-tournament profile features (t_* columns) from historical match
and stat data. These capture tournament-specific tendencies like average ace
counts, game length, tiebreak frequency, and serve dominance — all of which
vary significantly between surfaces and tournament types.

Output: one row per tournament_name, indexed by tournament_name.
Columns: t_avg_aces_per_match, t_avg_games_per_match, t_avg_sets_played,
         t_tiebreak_pct, t_avg_first_serve_pts_won_pct,
         t_avg_service_pts_won_pct, t_avg_return_pts_won_pct,
         t_n_matches (sample size indicator).

At feature-join time, each match row gets the t_* values for its
tournament_name. Unknown tournaments get NaN (models handle via NaN routing).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_MATCHES = 10  # minimum matches in tournament history to emit stats


class TournamentFeatureComputer:

    def compute(
        self,
        matches: pd.DataFrame,
        match_stats: pd.DataFrame,
        targets: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build tournament profiles and per-match tournament feature columns.

        Parameters
        ----------
        matches   : silver matches table (must have match_id, tournament_name,
                    match_date, best_of columns)
        match_stats : silver match_player_stats table (aces, serve pcts, …)
        targets   : optional gold targets table (total_games_target,
                    sets_played_target, tiebreak_present_label) — if provided,
                    enriches tournament stats with actual match outcomes

        Returns
        -------
        profiles_df  : DataFrame indexed by tournament_name with t_* columns
        per_match_df : DataFrame indexed by match_id with t_* columns
                       (each match gets its tournament's profile, computed
                        strictly from matches BEFORE that match — no leakage)
        """
        if matches.empty:
            return pd.DataFrame(), pd.DataFrame()

        matches = matches.copy()
        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
        matches = matches.sort_values("match_date", na_position="last").reset_index(drop=True)

        # Build stat lookup: match_id → aggregated stats (both players averaged)
        stat_lookup = self._build_stat_lookup(match_stats)

        # Optionally pull target-based stats
        target_lookup: dict[str, dict] = {}
        if targets is not None and not targets.empty:
            target_lookup = self._build_target_lookup(targets)

        # Compute per-match tournament features (no leakage: use only past matches)
        rows: list[dict] = []
        # Pre-group matches by tournament for fast slicing
        tourn_groups: dict[str, pd.DataFrame] = {
            name: grp for name, grp in matches.groupby("tournament_name")
        }

        for _, m in matches.iterrows():
            mid   = m["match_id"]
            tname = m.get("tournament_name", "")
            mdate = m["match_date"]

            feats = {"match_id": mid}
            if not tname or pd.isna(tname):
                rows.append(feats)
                continue

            past = tourn_groups.get(str(tname), pd.DataFrame())
            if not past.empty and pd.notnull(mdate):
                past = past[past["match_date"] < mdate]

            feats.update(self._tournament_feats(past, stat_lookup, target_lookup))
            rows.append(feats)

        per_match_df = pd.DataFrame(rows).set_index("match_id")

        # Build overall profiles (all-time, no leakage constraint — used at inference)
        profiles: dict[str, dict] = {}
        for tname, grp in tourn_groups.items():
            p = self._tournament_feats(grp, stat_lookup, target_lookup)
            if p:
                profiles[tname] = p

        profiles_df = pd.DataFrame.from_dict(profiles, orient="index")
        profiles_df.index.name = "tournament_name"

        logger.info(
            "Tournament profiles: %d tournaments, %d per-match rows",
            len(profiles_df), len(per_match_df),
        )
        return profiles_df, per_match_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_stat_lookup(self, stats: pd.DataFrame) -> dict[str, dict]:
        """Aggregate per-player stats to per-match level (average of both players)."""
        if stats.empty:
            return {}

        COLS = [
            "aces", "first_serve_pct", "first_serve_pts_won_pct",
            "service_pts_won_pct", "return_pts_won_pct",
        ]
        available = [c for c in COLS if c in stats.columns]
        if not available:
            return {}

        lookup: dict[str, dict] = {}
        for mid, grp in stats.groupby("match_id"):
            row: dict[str, float] = {}
            for col in available:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    row[col] = float(vals.mean())
            if row:
                lookup[str(mid)] = row
        return lookup

    def _build_target_lookup(self, targets: pd.DataFrame) -> dict[str, dict]:
        """Extract outcome columns from gold targets table keyed by match_id."""
        COLS = ["total_games_target", "sets_played_target", "tiebreak_present_label"]
        available = [c for c in COLS if c in targets.columns]
        lookup: dict[str, dict] = {}
        for mid, row in targets.iterrows():
            d: dict[str, float] = {}
            for col in available:
                v = row.get(col)
                if pd.notnull(v):
                    d[col] = float(v)
            if d:
                lookup[str(mid)] = d
        return lookup

    def _tournament_feats(
        self,
        hist: pd.DataFrame,
        stat_lookup: dict[str, dict],
        target_lookup: dict[str, dict],
    ) -> dict[str, float]:
        """Compute aggregate tournament features from a historical match DataFrame."""
        if hist.empty:
            return {}

        n = len(hist)
        if n < MIN_MATCHES:
            return {}

        f: dict[str, float] = {"t_n_matches": float(n)}

        # Collect match-level stat averages
        aces_list:      list[float] = []
        fs_pct_list:    list[float] = []
        srv_won_list:   list[float] = []
        ret_won_list:   list[float] = []

        for mid in hist["match_id"].astype(str):
            s = stat_lookup.get(mid, {})
            if "aces" in s:
                aces_list.append(s["aces"])
            if "first_serve_pts_won_pct" in s:
                fs_pct_list.append(s["first_serve_pts_won_pct"])
            if "service_pts_won_pct" in s:
                srv_won_list.append(s["service_pts_won_pct"])
            if "return_pts_won_pct" in s:
                ret_won_list.append(s["return_pts_won_pct"])

        if len(aces_list) >= MIN_MATCHES:
            f["t_avg_aces_per_match"] = float(np.mean(aces_list))
        if len(fs_pct_list) >= MIN_MATCHES:
            f["t_avg_first_serve_pts_won_pct"] = float(np.mean(fs_pct_list))
        if len(srv_won_list) >= MIN_MATCHES:
            f["t_avg_service_pts_won_pct"] = float(np.mean(srv_won_list))
        if len(ret_won_list) >= MIN_MATCHES:
            f["t_avg_return_pts_won_pct"] = float(np.mean(ret_won_list))

        # Outcome-based stats (from targets table)
        games_list: list[float] = []
        sets_list:  list[float] = []
        tb_list:    list[float] = []

        for mid in hist["match_id"].astype(str):
            t = target_lookup.get(mid, {})
            if "total_games_target" in t:
                games_list.append(t["total_games_target"])
            if "sets_played_target" in t:
                sets_list.append(t["sets_played_target"])
            if "tiebreak_present_label" in t:
                tb_list.append(t["tiebreak_present_label"])

        if len(games_list) >= MIN_MATCHES:
            f["t_avg_games_per_match"] = float(np.mean(games_list))
        if len(sets_list) >= MIN_MATCHES:
            f["t_avg_sets_played"] = float(np.mean(sets_list))
        if len(tb_list) >= MIN_MATCHES:
            f["t_tiebreak_pct"] = float(np.mean(tb_list))

        return f
