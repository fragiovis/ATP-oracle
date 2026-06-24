"""CLI entry point for model evaluation."""
from __future__ import annotations
import logging
import json
import pandas as pd
import typer
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

from models.evaluator import evaluate_all
from storage.writer import StorageWriter

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
app = typer.Typer(help="ATP Oracle – model evaluation CLI")
console = Console()


@app.command()
def run(
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON instead of table"),
):
    """Evaluate all trained models on the held-out test split."""
    writer = StorageWriter()
    logger.info("Loading gold features…")
    features = writer.read_gold_features()

    logger.info("Evaluating models…")
    results = evaluate_all(features)

    if json_out:
        typer.echo(json.dumps(results, indent=2))
        return

    # ── Classification table ──────────────────────────────────────────────
    clf_table = Table(title="Classification models (test split)", show_lines=True)
    clf_table.add_column("Model",            style="cyan")
    clf_table.add_column("N test",           justify="right")
    clf_table.add_column("Accuracy",         justify="right")
    clf_table.add_column("Baseline acc.",    justify="right")
    clf_table.add_column("Lift",             justify="right")
    clf_table.add_column("ROC-AUC",          justify="right")
    clf_table.add_column("Log-loss",         justify="right")

    # ── Regression table ──────────────────────────────────────────────────
    reg_table = Table(title="Regression models (test split)", show_lines=True)
    reg_table.add_column("Model",        style="cyan")
    reg_table.add_column("N test",       justify="right")
    reg_table.add_column("MAE",          justify="right")
    reg_table.add_column("RMSE",         justify="right")
    reg_table.add_column("R²",           justify="right")
    reg_table.add_column("Mean target",  justify="right")

    for model_id, m in results.items():
        if "accuracy" in m:
            lift = m.get("lift_vs_baseline", 0)
            lift_str = f"[green]+{lift:.4f}[/green]" if lift >= 0 else f"[red]{lift:.4f}[/red]"
            clf_table.add_row(
                model_id,
                str(m["n_test"]),
                str(m["accuracy"]),
                str(m.get("baseline_accuracy", "-")),
                lift_str,
                str(m.get("roc_auc", "-")),
                str(m.get("log_loss", "-")),
            )
        else:
            reg_table.add_row(
                model_id,
                str(m["n_test"]),
                str(m["mae"]),
                str(m["rmse"]),
                str(m["r2"]),
                str(m["mean_target"]),
            )

    console.print(clf_table)
    console.print(reg_table)


@app.command()
def betting(
    min_edge: float = typer.Option(
        0.03, "--min-edge",
        help="Minimum model edge over implied odds to place a bet (e.g. 0.05 = 5%%)",
    ),
    stake: float = typer.Option(
        10.0, "--stake",
        help="Flat stake per bet in euros",
    ),
    odds_source: str = typer.Option(
        "b365", "--odds",
        help="Odds source: b365, ps, or max",
    ),
    min_year: int = typer.Option(
        2015, "--min-year",
        help="Only include matches from this year onwards",
    ),
):
    """Betting backtest: bet when model confidence >= 0.5 + min_edge.

    For each match: if the model assigns prob >= (0.5 + min_edge) to a player,
    simulate placing a flat stake on that player at the bookmaker odds, then
    check the real outcome. Compares against always-bet-bookie-favourite baseline.

    This command is SEPARATE from the standard ML evaluation pipeline.

    Examples:
        atp-eval betting                     # confidence >= 60%
        atp-eval betting --min-edge 0.15     # confidence >= 65%
        atp-eval betting --odds ps           # use Pinnacle odds
    """
    import pandas as pd
    from models.targets import WinnerModel
    from models.trainer import ModelTrainer
    from models.betting_evaluator import BettingEvaluator
    from features.pipeline import FeaturePipeline

    writer = StorageWriter()

    # Load gold features + match odds
    logger.info("Loading gold features…")
    features = writer.read_gold_features()

    logger.info("Loading match odds from silver…")
    try:
        odds_df = writer.read_match_odds()
    except Exception as e:
        typer.echo(f"[ERROR] Could not load match odds: {e}", err=True)
        typer.echo("Run 'atp-scrape update' to download and persist odds data.", err=True)
        raise typer.Exit(1)

    odds_cols = [f"{odds_source}_a", f"{odds_source}_b"]
    missing_cols = [c for c in odds_cols if c not in odds_df.columns]
    if missing_cols:
        typer.echo(f"[ERROR] Odds columns not found: {missing_cols}. Available: {[c for c in odds_df.columns if c not in ('match_id',)]}", err=True)
        raise typer.Exit(1)

    # Load winner model and get test-set predictions
    logger.info("Loading winner model…")
    trainer = ModelTrainer()
    model = trainer.load_latest_model(WinnerModel)

    p = float(features["winner_label"].dropna().mean())
    if p > 0.85 or p < 0.15:
        features = FeaturePipeline._randomise_roles(features, seed=42)

    df = features.dropna(subset=["winner_label"]).copy()
    _, _, test = model.time_split(df)

    # Filter by min_year
    if min_year and "year" in test.columns:
        test = test[test["year"] >= min_year]
        logger.info("Test set after year filter (>=%d): %d rows", min_year, len(test))

    X_test = test[model.feature_cols]
    probs = model.model.predict_proba(X_test)[:, 1]
    test = test.copy()
    test["prob_a"] = probs

    # Join gold predictions with odds on surname+date fingerprint.
    # td_* match_ids are unstable across scrape runs; we use player names
    # (same "Surname I." format in both gold td_* rows and odds table)
    # combined with match_date for an exact, collision-free join.
    def _lastname(name) -> str:
        return str(name or "").split()[0].lower().rstrip(".")

    def _fp_date(a, b, date) -> str | None:
        a, b = _lastname(a), _lastname(b)
        if not a or not b:
            return None
        d = str(date)[:10] if date and str(date) not in ("None", "nan", "NaT", "") else None
        if not d:
            return None
        if a > b:
            a, b = b, a
        return f"{a}|{b}|{d}"

    test["_fp"] = test.apply(
        lambda r: _fp_date(r.get("player_a_name", ""), r.get("player_b_name", ""), r.get("match_date"))
        if str(r["match_id"]).startswith("td_") else None,
        axis=1,
    )

    odds_df["_fp"] = odds_df.apply(
        lambda r: _fp_date(r.get("player_a_name", ""), r.get("player_b_name", ""), r.get("match_date")),
        axis=1,
    )

    odds_keep_cols = ["_fp"] + odds_cols
    merged = test.dropna(subset=["_fp"]).merge(
        odds_df.dropna(subset=["_fp"])[odds_keep_cols], on="_fp", how="inner"
    )
    logger.info(
        "Matched %d / %d test rows with odds via name+date fingerprint (odds source: %s)",
        len(merged), len(test), odds_source,
    )

    if merged.empty:
        typer.echo("[ERROR] No test rows matched with odds. Run 'atp-scrape update' to refresh odds.", err=True)
        raise typer.Exit(1)

    # Run backtest
    evaluator = BettingEvaluator(
        min_edge=min_edge,
        unit_stake=stake,
        odds_col=odds_source,
    )
    results = evaluator.run(merged)
    typer.echo(evaluator.summary_table(results))


@app.command("betting-games")
def betting_games(
    line: float = typer.Option(
        22.5, "--line",
        help="Over/Under line in total games for bo3 (default 22.5 ≈ median)",
    ),
    min_distance: float = typer.Option(
        2.0, "--min-distance",
        help="Minimum distance from line to place a bet (e.g. 2.0 means pred <20.5 or >24.5)",
    ),
    stake: float = typer.Option(
        10.0, "--stake",
        help="Flat stake per bet in euros",
    ),
    ou_odds: float = typer.Option(
        1.85, "--odds",
        help="Assumed Over/Under odds (standard market is ~1.85 each side)",
    ),
    min_year: int = typer.Option(
        2015, "--min-year",
        help="Only include matches from this year onwards",
    ),
):
    """Over/Under backtest on total games per match (bo3).

    The model predicts total games. If the prediction is far enough from the
    line (|pred - line| >= min_distance), simulate a flat-stake bet and check
    the real outcome.

    Examples:
        atp-eval betting-games                          # line 20.5, distance 2
        atp-eval betting-games --line 20.5 --min-distance 3
        atp-eval betting-games --min-distance 1.5 --odds 1.90
    """
    from models.targets import TotalGamesModel
    from models.trainer import ModelTrainer

    writer = StorageWriter()
    logger.info("Loading gold features…")
    features = writer.read_gold_features()

    logger.info("Loading TotalGames model…")
    trainer = ModelTrainer()
    model = trainer.load_latest_model(TotalGamesModel)

    df = features.dropna(subset=["total_games_target"]).copy()
    _, _, test = model.time_split(df)

    if min_year and "year" in test.columns:
        test = test[test["year"] >= min_year]

    # bo3 only + complete matches (retirements/walkovers have <13 games)
    before = len(test)
    test = test[(test["best_of_5"] == 0) & (test["total_games_target"] >= 13)]
    logger.info("Test set: %d bo3 complete matches (filtered from %d)", len(test), before)

    preds = model.model.predict(test[model.feature_cols])
    actual = test["total_games_target"].values
    years = test["year"].values if "year" in test.columns else [0] * len(test)

    # Backtest
    n_total = len(preds)
    n_over_bets = n_under_bets = 0
    n_over_won = n_under_won = 0
    total_staked = total_return = 0.0
    pnl_by_year: dict[str, float] = {}

    for pred, real, yr in zip(preds, actual, years):
        yr_str = str(int(yr))
        distance = pred - line

        if distance >= min_distance:
            # Bet OVER
            won = real > line
            pnl = (ou_odds - 1) * stake if won else -stake
            n_over_bets += 1
            n_over_won += int(won)
        elif distance <= -min_distance:
            # Bet UNDER
            won = real < line
            pnl = (ou_odds - 1) * stake if won else -stake
            n_under_bets += 1
            n_under_won += int(won)
        else:
            continue

        total_staked += stake
        total_return += stake + pnl
        pnl_by_year[yr_str] = pnl_by_year.get(yr_str, 0.0) + pnl

    n_bets = n_over_bets + n_under_bets
    n_won = n_over_won + n_under_won
    roi = (total_return - total_staked) / total_staked if total_staked > 0 else 0.0
    win_rate = n_won / n_bets if n_bets > 0 else 0.0
    pnl_net = total_return - total_staked
    breakeven = 1 / ou_odds

    # MAE on all test rows for context
    import numpy as np
    mae = float(np.abs(preds - actual).mean())
    within_line = float((np.abs(preds - actual) <= min_distance).mean())

    typer.echo(f"""
{'='*60}
  OVER/UNDER BACKTEST  —  line={line}, min_distance=±{min_distance}
  Assumed odds: {ou_odds} each side  (break-even: {breakeven:.1%})
{'='*60}

  Model accuracy context:
    MAE on test set        : {mae:.2f} games
    Predictions within ±{min_distance:.0f}  : {within_line:.1%} (these are skipped)

  Bets:
    Total matches          : {n_total}
    Bets placed            : {n_bets}  ({n_bets/max(n_total,1):.1%} of matches)
      → OVER  bets         : {n_over_bets}  (won: {n_over_won}, {n_over_won/max(n_over_bets,1):.1%})
      → UNDER bets         : {n_under_bets}  (won: {n_under_won}, {n_under_won/max(n_under_bets,1):.1%})
    Overall win rate       : {win_rate:.1%}  (break-even: {breakeven:.1%})

  P&L:
    Total staked           : €{total_staked:.2f}
    Total return           : €{total_return:.2f}
    Net P&L                : €{pnl_net:+.2f}
    ROI                    : {roi:+.2%}

  P&L by year:""")

    for yr, pnl in sorted(pnl_by_year.items()):
        typer.echo(f"    {yr}: €{pnl:+.2f}")

    typer.echo(f"\n{'='*60}")


if __name__ == "__main__":
    app()
