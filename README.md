# ATP Oracle

Production-style data and ML pipeline for **pre-match ATP tennis predictions**.

Scrapes the official ATP Tour website, stores data on **Hadoop HDFS** running in Docker, engineers player/matchup features, trains **separate models per prediction target**, and exposes predictions through a **natural-language primary agent**.

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
pip install -e .                 # installs the atp-* CLI entry points

# 2b. Install the headless browser used by the scraper
playwright install chromium

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

| Intent | Output |
|---|---|
| Who is favored between X and Y on surface? | `win_probability_player_a/b` |
| Who wins the match? | `predicted_winner` |
| How many aces will X hit against Y? | `expected_aces_player_a` |
| How many total aces in the match? | `expected_total_aces` |
| How many total games? | `expected_total_games` |
| Probability of a deciding set? | `prob_deciding_set` |
| Probability of at least one tie-break? | `prob_tiebreak` |
| Who is favored to win the first set? | `prob_player_a_first_set` |

---

## Project layout

```
project1_atp_oracle/
├── agent/          # Primary NLP agent (parser → router → response)
├── scraper/        # ATP Tour web scrapers (rankings, matches, stats)
├── storage/        # HDFS client, normaliser, writer
├── features/       # Feature engineering (Elo, rolling, H2H, interaction)
├── models/         # One model per prediction target + trainer
├── evaluation/     # Metrics, calibration plots, residual plots
├── serving/        # Structured inference layer (MatchPredictor)
├── config/         # Pydantic settings (all tuneable parameters)
├── infra/          # docker-compose.yml for Hadoop HDFS
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

Trains 7 separate models (LightGBM / XGBoost) using time-based train/val/test splits.
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
| `ATP_AGENT_CLAUDE_MODEL` | `claude-sonnet-4-6` | LLM model for the agent |

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
