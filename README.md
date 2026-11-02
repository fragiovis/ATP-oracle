# ATP Oracle 🎾

**Pre-match prediction system for ATP men's singles** — a production-style data and ML pipeline with a natural-language interface.

*Big Data course project (A.Y. 2025/26) · Università degli Studi Roma Tre · Francesco Giovanardi*

Ask *"Who is favored between Sinner and Alcaraz on clay?"* and the system parses the question with an LLM, builds a leakage-free feature vector from decades of match history and routes it to the right model. The answer comes back with probabilities and the key drivers behind the prediction.

**Highlights**
- **Medallion data lake** (raw → silver → gold) on a swappable storage backend, **AWS S3** or the **local filesystem** (the project started on Hadoop HDFS): 130,961 matches (1986–2026), 65,989 players, 144 feature columns
- **Data acquisition** from Jeff Sackmann's `tennis_atp` dataset, tennis-data.co.uk (2025+) and the official ATP Tour site (Playwright scraper with rate limiting and checkpoint/resume)
- **Feature engineering**: overall and per-surface Elo, rolling form (last 5/10/25), head-to-head, serve-vs-return interactions, tournament context — all computed strictly before the match date
- **6 specialised models** (LightGBM / XGBoost) trained on time-based splits, evaluated with accuracy / AUC / log-loss / Brier (classifiers) and MAE / RMSE / R² (regressors), plus calibration curves and a betting backtest
- **Serving**: FastAPI + web UI, CLI tools and an LLM agent (Groq, gpt-oss-120b) for intent recognition, entity extraction and clarifying questions
- **Explainability**: SHAP feature attributions turned into a plain-language explanation of each prediction (`/api/explain`)
- **Cloud**: EC2 bootstrap script for training on S3 data

**Stack:** Python · pandas · PyArrow · LightGBM · XGBoost · scikit-learn · SHAP · Hadoop HDFS · Docker · AWS (S3, EC2) · FastAPI · Playwright · Groq API · Pydantic · Typer · pytest

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/fragiovis/ATP-oracle.git
cd ATP-oracle

# 2. Create a virtual environment and install the dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt  # or: requirements-dev.txt to include tests/lint

# 2b. REQUIRED — install the project itself, otherwise the atp-* commands
#     (atp-serve, atp-train, …) will not exist: "command not found".
pip install -e .

# 2c. Install the headless browser used by the scraper
playwright install chromium

# 2d. Check it worked — should print a path inside .venv/bin
which atp-serve

# 3. Configure — copy the template, then set GROQ_API_KEY and the storage backend
cp .env.example .env             # ATP_STORAGE_BACKEND=local needs no cloud account

# 4. Launch the web UI and open http://localhost:8000
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** the trained models and cached datasets are **not** committed to the
> repo (they live on S3 / are regenerated). To produce them locally, run the
> pipeline below in order: `atp-scrape run` → `atp-features all` →
> `atp-train`. See [Pipeline walkthrough](#pipeline-walkthrough).

---

## Prediction targets

| Model | Task | Output |
|---|---|---|
| Match winner | classification | `win_probability_player_a/b`, `predicted_winner` |
| Tie-break | classification | `prob_tiebreak` (at least one tie-break in the match) |
| Player aces | regression | `expected_aces_player_a` |
| Total aces | regression | `expected_total_aces` |
| Total games | regression | `expected_total_games` |
| Sets played | regression | `expected_sets_played` |

The winner probability is made **order-invariant**: the model is queried for both A-vs-B and B-vs-A and the two estimates are averaged.

---

## Results

Time-based split: train 1986–2013 (95,679 matches), validation 2014–2019 (17,314), test 2020–2026 (17,968). The ace models stop at 2024 because tennis-data has no serve statistics.

| Model | Test matches | Result on the test set |
|---|---|---|
| Winner | 17,968 | accuracy 64.3% (baseline 50.3%), ROC-AUC 0.70 |
| Tie-break | 17,968 | accuracy 61.5% (baseline 60.6%), ROC-AUC 0.58 |
| Player aces | 7,610 | MAE 3.2, R² 0.42 |
| Total aces | 15,238 | MAE 5.0, R² 0.39 |
| Total games | 17,968 | MAE 5.8, R² 0.32 |
| Sets played | 17,759 | MAE 0.51, R² 0.46 |

The winner model is 14 points above the baseline but only 0.5 points above the simple "higher Elo wins" rule, and the tie-break model has almost no signal: new features (fatigue, travel, odds) are the clear next step.

The full data pipeline takes about 1h 08m locally, almost all of it CPU time in the rolling features. Training the six models takes 19 s. Details are in [`benchmarks/`](benchmarks/) and in the [presentation](ATP_Oracle_Presentazione_2026.pptx) (Italian).

---

## Project layout

```
ATP-oracle/
├── agent/          # Primary NLP agent (parser → router → response)
├── scraper/        # Data acquisition: Sackmann CSVs, tennis-data.co.uk, ATP Tour scrapers
├── storage/        # S3 and local backends (plus the original HDFS client), normaliser, writer
├── features/       # Feature engineering (Elo, rolling, H2H, interaction)
├── models/         # One model per prediction target, trainer, betting backtest
├── evaluation/     # Metrics, calibration plots, residual plots
├── serving/        # Structured inference layer (MatchPredictor)
├── config/         # Pydantic settings (all tuneable parameters)
├── app/            # Web UI (served by FastAPI)
├── infra/          # docker-compose for HDFS, EC2 bootstrap, IAM setup
├── benchmarks/     # Pipeline timings and logs
├── scripts/        # Local storage seeding
└── tests/          # Unit tests
```

---

## Requirements

- Python ≥ 3.10
- A local copy of the Sackmann `tennis_atp` CSVs, set in `ATP_SCRAPER_SACKMANN_SOURCE` (the upstream GitHub repository is no longer available)
- A `GROQ_API_KEY` in `.env` (for the agent's LLM calls; without it the parser
  falls back to a heuristic and natural-language understanding is degraded)
- With `ATP_STORAGE_BACKEND=s3`: AWS credentials in the environment and `ATP_S3_BUCKET`
  (otherwise the loaders silently fall back to the local `.cache/`)

---

## Setup

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Configure .env
ATP_STORAGE_BACKEND=local                      # or s3, together with ATP_S3_BUCKET
ATP_SCRAPER_SACKMANN_SOURCE=/path/to/tennis_atp
GROQ_API_KEY=...
```

---

## Pipeline walkthrough

### 1 – Scrape ATP data

```bash
atp-scrape run
```

Collects the current Top-200 rankings and player bios from atptour.com, match
history and serve statistics from Sackmann, and 2025+ results from
tennis-data.co.uk.  Raw data is saved as JSON Lines under `/data/raw/`.  Supports checkpoint/resume – safe to re-run.

Tune scraping behaviour via environment variables:

```bash
ATP_SCRAPER_REQUESTS_PER_MINUTE=10
ATP_SCRAPER_MIN_DELAY_SECONDS=3.0
ATP_SCRAPER_MAX_DELAY_SECONDS=8.0
ATP_SCRAPER_YEARS_BACK=40
```

### 2 – Build features

```bash
atp-features all
```

Reads the silver tables and computes:
- Elo ratings (overall + per surface)
- Rolling player stats (last 5 / 10 / 25 matches, career, surface-specific)
- Head-to-head features
- Match context features
- Serve-vs-return interaction features

Writes the gold-layer `pre_match_features.parquet` under `/data/gold/features/`, plus player and tournament profiles.

### 3 – Train models

```bash
atp-train
```

Trains 6 separate models (LightGBM for the two classifiers, XGBoost for the four regressors) using time-based train/val/test splits.
Saves model artifacts and metadata under `/data/models/<run_id>/`.

### 4 – Evaluate models

```bash
atp-eval run
```

Loads the latest trained models, evaluates on the held-out test set, and saves:
- Metrics JSON (accuracy / AUC / log-loss / Brier for classifiers; MAE / RMSE / R² for regressors)
- Calibration curves, ROC curves, residual plots, feature importance plots

Results under `/data/evaluation/<run_id>/`.

### 5 – Structured inference

```bash
atp-predict \
  --player-a "Jannik Sinner" \
  --player-b "Carlos Alcaraz" \
  --surface clay \
  --category "ATP 1000" \
  --round SF \
  --best-of 3
```

Returns a JSON prediction dict with all targets.

### 6 – Web UI (recommended)

```bash
# Start the FastAPI server (serves the HTML UI + API)
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

Features of the UI:
- Chat-style interface for natural-language queries
- Surface-adaptive theming (clay / hard / grass / indoor)
- Live animated prediction dashboard:
  - Win probability dual bar with animated fill
  - Expected aces & games with number counters
  - Tie-break and deciding-set probability rings
  - First set winner probability bars
- Example queries you can click to prefill
- Auto-detects surface from query text

### 7 – Natural-language agent (CLI)

```bash
# One-shot
atp-agent "Who is favored between Sinner and Alcaraz on clay?"

# Interactive REPL
atp-agent --interactive
```

Example interaction:

```
You: How many aces will Sinner hit against Medvedev?
Agent: Jannik Sinner is expected to hit approximately 9.3 aces.

You: Will there be a tiebreak?
Agent: Probability of at least one tie-break: 38%. [Context: hard]
```

If information is missing, the agent asks the **single most important clarifying question**:

```
You: How many games will there be?
Agent: Which two players are you asking about?
```

---

## Configuration

All parameters live in `config/settings.py` and can be overridden with
environment variables (prefix `ATP_SCRAPER_`, `ATP_STORAGE_`, `ATP_S3_`, `ATP_FEATURES_`,
`ATP_MODEL_`, `ATP_AGENT_`).

Key settings:

| Variable | Default | Description |
|---|---|---|
| `ATP_SCRAPER_REQUESTS_PER_MINUTE` | 10 | Max requests/min |
| `ATP_SCRAPER_YEARS_BACK` | 40 | Years of history to scrape |
| `ATP_SCRAPER_SACKMANN_SOURCE` | upstream URL | URL or local folder of the Sackmann CSVs |
| `ATP_STORAGE_BACKEND` | `s3` | `s3` or `local` |
| `ATP_STORAGE_LOCAL_ROOT` | `.cache/storage` | Root folder of the local backend |
| `ATP_S3_BUCKET` | none | Bucket name, required with the `s3` backend |
| `ATP_MODEL_RANDOM_SEED` | 42 | Reproducibility seed |
| `ATP_AGENT_GROQ_MODEL` | `openai/gpt-oss-120b` | LLM used by the agent |

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover:
- HTML parsing (rankings, matches, match stats)
- Checkpointing and caching
- Scoreline → target parsing
- Elo computation
- H2H feature logic
- Agent intent routing and entity extraction
- Clarification behaviour

---

## Storage layout

```
/data/
  raw/
    rankings/      ← raw JSON from scraper
    players/
    matches/
    match_stats/
  silver/          ← normalised Parquet tables
    players.parquet
    matches.parquet
    match_player_stats.parquet
    head_to_head_history.parquet
    rankings_snapshot_current_top200.parquet
  gold/features/
    pre_match_features.parquet
  models/
    <run_id>/
      winner/model.pkl
      tiebreak/model.pkl
      ...
  evaluation/
    <run_id>/
      winner/metrics.json
      winner/calibration.png
      ...
```

---

## Design notes

- **No data leakage**: all features are computed strictly before the match date.
- **Graceful degradation**: missing ATP stats (pages that don't exist or parse failures) are handled without crashing; the feature pipeline falls back to a smaller feature subset.
- **Modular agent**: the NLP parser can be replaced (e.g. with a local model) without changing the serving or model layers.
- **Idempotent jobs**: scraping and feature generation use checkpointing and overwrite logic, so they are safe to re-run.

---

## Troubleshooting

### `atp-serve: command not found` (or `atp-train`, `atp-scrape`, …)

The project itself was not installed — `requirements.txt` only installs the
libraries, while the `atp-*` commands are entry points created by installing
the package:

```bash
source .venv/bin/activate
pip install -e .
which atp-serve            # must point inside .venv/bin
```

Still failing? Check these in order:

1. **Is the venv active?** The prompt should show `(.venv)`.
   `which python` must point inside `.venv/bin`.
2. **Mixed interpreters?** Install through the interpreter itself so pip and
   python cannot diverge: `python -m pip install -e .`
3. **Workaround** — the server also runs without the entry points, from the
   project directory: `python -m uvicorn serving.api:app --port 8000`

### `Address already in use` on port 8000

A previous server is still running:

```bash
lsof -ti:8000 | xargs kill
```

### Predictions look stale after retraining

The predictor is a lazy singleton cached in memory, so it does not hot-reload.
**Restart `atp-serve`** after `atp-train`.

### `Rankings snapshot empty or missing columns — rank features will be NaN`

The rankings snapshot is missing or has unexpected columns, so rank features
fall back to NaN and predictions are degraded. Regenerate it with
`atp-scrape run`, and verify with:

```bash
python -c "from storage.writer import Writer; df = Writer().read_rankings_snapshot(); print(len(df), list(df.columns))"
```

It must contain the `player_name` and `rank` columns.

### `Groq parse failed (403 …) – falling back to heuristic`

`GROQ_API_KEY` is missing, expired, or blocked. Predictions still work, but
natural-language parsing is degraded. Check that `.env` contains a valid key
and that the shell running the server actually loads it.

### Models load from the local cache instead of S3

The loaders fall back to `.cache/` **silently** when AWS credentials are
expired or absent — you only see a `WARNING` in the logs. Export valid
credentials before starting the server, and check the startup logs for
`Loaded <model> from S3 run <run_id>` versus `falling back to local cache`.
