# ATP Oracle - Claude Code Guide

## Project Overview
ATP Tennis Prediction System: pre-match ML pipeline with NL agent for ATP men's singles.
Predicts: match winner, aces, total games, deciding set, tiebreak, first set winner.
Stack: Python 3.10+, XGBoost/LightGBM, HDFS (Docker), FastAPI, Groq for NL agent.

## CLI Entry Points
- `atp-scrape` — scraper.cli (rankings, players, matches, match stats)
- `atp-features` — features.cli (feature pipeline, player profiles, Elo)
- `atp-train` — models.cli (train per-target models)
- `atp-eval` — models.eval_cli (evaluate models)
- `atp-predict` — serving.cli (inference)
- `atp-agent` — agent.cli (NL query agent)
- `atp-serve` — serving.api (FastAPI server)

## Data Sources
- Gold/reference data is stored on **S3**, NOT local cache. After modifying data files locally, always re-upload to S3 before running validation.
- S3 layout follows: raw/, bronze/, silver/, gold/features/, models/, evaluation/
- Format: Parquet for silver/gold layers.
- Always verify `player_id` consistency between ranking data and match history when generating player profiles or features.
- When syncing data, verify row counts match between local and S3.

## Model Training Checklist
1. Player profiles **must** include Elo ratings (overall + surface-specific). Do not omit.
2. After training, validate end-to-end on gold data before declaring success.
3. Watch for NaN mismatches between train and eval pipelines.
4. Verify `player_id` consistency across datasets before joining.
5. Use time-based splits only (no random splits) to avoid leakage.
6. All features must use data **strictly before** the match date.

## Key Modules
- `features/player_profiler.py` — generates player profile features
- `features/elo.py` — Elo rating computation (overall + surface)
- `features/pipeline.py` — orchestrates full feature pipeline
- `features/targets.py` — scoreline parsing and target derivation
- `models/targets.py` — per-target model definitions
- `models/base.py` — base model training logic
- `agent/parser.py` — NL intent/entity extraction
- `agent/router.py` — routes parsed intent to correct model
- `config/settings.py` — all configuration (scraping, features, models)

## Testing
```bash
pytest tests/           # all tests
ruff check .            # lint
```

## Common Pitfalls
- Scoreline parser can be fragile — test with edge cases (walkovers, retirements, super-tiebreaks).
- Prior drift: ensure priors/defaults are recalibrated when retraining on new data windows.
- NaN handling must be consistent: if train fills NaN with 0, eval must do the same.
