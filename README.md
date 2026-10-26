# ATP Oracle 🎾

**Pre-match prediction system for ATP men's singles** — a production-style data and ML pipeline with a natural-language interface.

*Big Data course project (A.Y. 2025/26) · Università degli Studi Roma Tre · Francesco Giovanardi, Alessandro Cavina*

Ask *"Who is favored between Sinner and Alcaraz on clay?"* and the system parses the question with an LLM, builds a leakage-free feature vector from decades of match history and routes it to the right model. The answer comes back with probabilities and the key drivers behind the prediction.

**Highlights**
- **Medallion data lake** (raw → bronze → silver → gold) on **Hadoop HDFS** (Docker) or **AWS S3**, with ~700K historical matches and 60K+ players
- **Data acquisition** from Jeff Sackmann's `tennis_atp` dataset, tennis-data.co.uk (2025+) and the official ATP Tour site (Playwright scraper with rate limiting and checkpoint/resume)
- **Feature engineering**: overall and per-surface Elo, rolling form (last 5/10/25), head-to-head, serve-vs-return interactions, tournament context — all computed strictly before the match date
- **6 specialised models** (LightGBM / XGBoost) trained on time-based splits, evaluated with accuracy / AUC / log-loss / Brier (classifiers) and MAE / RMSE / R² (regressors), plus calibration curves and a betting backtest
- **Serving**: FastAPI + web UI, CLI tools and an LLM agent (Groq, Llama 3.3 70B) for intent recognition, entity extraction and clarifying questions
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

# 3. Configure secrets — copy the template and fill in your keys
cp .env.example .env             # then edit .env

# 4. Start Hadoop HDFS (Docker must be running)
docker compose -f infra/docker-compose.yml up -d

# 5. Launch the web UI and open http://localhost:8000
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** the trained models and cached datasets are **not** committed to the
> repo (they live on S3 / are regenerated). To produce them locally, run the
> pipeline below in order: `atp-scrape run` → `atp-features run` →
> `atp-train run`. See [Pipeline walkthrough](#pipeline-walkthrough).

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

## Project layout

```
ATP-oracle/
├── agent/          # Primary NLP agent (parser → router → response)
├── scraper/        # Data acquisition: Sackmann CSVs, tennis-data.co.uk, ATP Tour scrapers
├── storage/        # HDFS and S3 clients, normaliser, writer
├── features/       # Feature engineering (Elo, rolling, H2H, interaction)
├── models/         # One model per prediction target, trainer, betting backtest
├── evaluation/     # Metrics, calibration plots, residual plots
├── serving/        # Structured inference layer (MatchPredictor)
├── config/         # Pydantic settings (all tuneable parameters)
├── app/            # Web UI (served by FastAPI)
├── infra/          # docker-compose for HDFS, EC2 bootstrap, IAM setup
└── tests/          # Unit tests
```

---

## Requirements

- Python ≥ 3.10
- Docker + Docker Compose
- A `GROQ_API_KEY` in `.env` (for the agent's LLM calls; without it the parser
  falls back to a heuristic and natural-language understanding is degraded)
- AWS credentials in the environment, if you want to read models/features from S3
  (otherwise the loaders silently fall back to the local `.cache/`)

---

## Setup

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Start Hadoop HDFS
docker compose -f infra/docker-compose.yml up -d

# Wait ~30 seconds for the NameNode to start, then verify:
curl http://localhost:9870   # HDFS Web UI
```

---

## Pipeline walkthrough

### 1 – Scrape ATP data

```bash
atp-scrape run
```

Scrapes the current ATP Top-200 rankings, player profiles, match history
(~20 years), and per-match stats.  Raw data is saved as JSON on HDFS under
`/data/raw/`.  Supports checkpoint/resume – safe to re-run.

Tune scraping behaviour via environment variables:

```bash
ATP_SCRAPER_REQUESTS_PER_MINUTE=10
ATP_SCRAPER_MIN_DELAY_SECONDS=3.0
ATP_SCRAPER_MAX_DELAY_SECONDS=8.0
ATP_SCRAPER_YEARS_BACK=20
```

### 2 – Build features

```bash
atp-features run
```

Reads silver HDFS tables, computes:
- Elo ratings (overall + per surface)
- Rolling player stats (last 5 / 10 / 25 matches, career, surface-specific)
- Head-to-head features
- Match context features
- Serve-vs-return interaction features

Writes the gold-layer `pre_match_features.parquet` to HDFS under `/data/gold/features/`.

### 3 – Train models

```bash
atp-train run
```

Trains 6 separate models (LightGBM for the two classifiers, XGBoost for the four regressors) using time-based train/val/test splits.
Saves model artifacts and metadata to HDFS under `/data/models/<run_id>/`.

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
atp-predict predict \
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
atp-agent ask "Who is favored between Sinner and Alcaraz on clay?"

# Interactive REPL
atp-agent ask --interactive
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
environment variables (prefix `ATP_SCRAPER_`, `ATP_HDFS_`, `ATP_FEATURES_`,
`ATP_MODEL_`, `ATP_AGENT_`).

Key settings:

| Variable | Default | Description |
|---|---|---|
| `ATP_SCRAPER_REQUESTS_PER_MINUTE` | 10 | Max requests/min |
| `ATP_SCRAPER_YEARS_BACK` | 20 | Years of history to scrape |
| `ATP_HDFS_NAMENODE_URL` | `http://localhost:9870` | HDFS NameNode URL |
| `ATP_MODEL_RANDOM_SEED` | 42 | Reproducibility seed |
| `ATP_AGENT_GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM used by the agent |

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

## HDFS directory layout

```
/data/
  raw/
    rankings/      ← raw JSON from scraper
    players/
    matches/
    match_stats/
  bronze/          ← reserved for minimally cleaned data
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
      deciding_set/model.pkl
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
**Restart `atp-serve`** after `atp-train run`.

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
