# Tennis Prediction System – PRD

## Overview

Build a production-style data and ML pipeline for an ATP tennis prediction system that answers multiple **pre-match** questions in natural language.

The system must use **web scraping from the official ATP Tour website as the primary data source**, restricted to players who are in the **current ATP Top 200 singles rankings** at execution time, and collect match and player data for approximately the **last 20 years** for those players.

Data (raw and processed) must be stored on **Hadoop HDFS running inside Docker**. The pipeline must engineer player- and matchup-level features, train **separate models for each prediction target**, and expose a clean Python codebase that includes a **primary agent** responsible for translating the user’s natural-language request into a structured query routed to one of the trained models.

---

## Goals

Support pre-match prediction for these targets:

### Result-related

- **Who is favored between player X and player Y on a given surface?**  
  - Output: win probability for each player (e.g. `win_probability_player_a`, `win_probability_player_b`).

- **Who wins the match between X and Y?**  
  - Output: hard class label (`predicted_winner`).

### Aces

- **How many aces will player X hit against player Y?**  
  - Output: expected aces for player X (`expected_aces_player_a`).

- **How many total aces will there be in the match between X and Y?**  
  - Output: expected total aces (`expected_total_aces`).

### Games / duration

- **How many total games are expected in the match between X and Y?**  
  - Output: expected total games (`expected_total_games`).

- **Is there a high/low probability that the match will go to a deciding set?**  
  - Output: probability of a deciding set (`prob_deciding_set`).

- **What is the probability that at least one tie-break will be played?**  
  - Output: probability of at least one tie-break (`prob_tiebreak`).

### First set

- **Who is favored to win the first set between X and Y?**  
  - Output: probability that player A wins the first set (`prob_player_a_first_set`) and implied class label.

All predictions must be computed **only from information available before the match starts**, based on historical data and statistics for the two players (plus context like surface/round), *not* from in-match stats of the target match.

---

## Scope

- Domain: **ATP men’s singles only**.
- Player universe: **current ATP Top 200 singles rankings**, scraped from the official ATP rankings page at runtime.
- Time horizon: approximately the **last 20 years of matches** for those players.
- Targets: all the questions listed in the Goals section.
- Models: **separate models per target**, not a single multitask model (at least in v1).
- Query interface: a **primary agent** must interpret user language and route the request to the correct trained model.

---

## Primary Agent Layer

A **primary agent** must sit in front of the trained models and act as the natural-language orchestration layer.

### Responsibilities

The primary agent must:

1. Receive the user’s natural-language query.
2. Detect the user intent.
3. Extract structured entities and context from the request.
4. Map the request to the correct trained model.
5. Build the structured inference payload.
6. Call the appropriate prediction module.
7. Return a normalized response format.

### Example intents

The agent must support queries such as:

- “Who is favored between Sinner and Alcaraz on clay?”
- “How many aces will Sinner hit against Medvedev?”
- “How many total games are expected?”
- “Will there be a tie-break?”
- “Will player X win the first set?”
- “What is the probability this goes to three sets?”

### Intent-to-model mapping

The primary agent must map user requests to one of the trained target-specific models:

- `predict_match_winner_probability` → winner probability model
- `predict_match_winner_label` → winner classification model
- `predict_player_aces` → player aces regression model
- `predict_total_aces` → total aces regression model
- `predict_total_games` → total games regression model
- `predict_deciding_set_probability` → deciding set classification model
- `predict_tiebreak_probability` → tie-break classification model
- `predict_first_set_winner_probability` → first set winner classification model

### Required extracted fields

The primary agent should extract as many of the following fields as possible from the user request:

- `player_a`
- `player_b`
- `surface`
- `tournament_category`
- `round`
- `best_of`
- `indoor`
- `match_date` (optional)
- `target_intent`

If some fields are missing, the agent must:

- infer safe defaults where appropriate,
- or return a structured clarification request.

### Clarification behavior

If the request is ambiguous, the primary agent must ask for the **single most important missing field** rather than asking many questions at once.

Examples:

- If the players are missing: ask for the players.
- If the target is unclear: ask whether the user wants winner, aces, games, tie-break, or first-set prediction.
- If the surface is missing and strongly affects the prediction: ask for the surface.

### Output contract

The primary agent must produce a normalized internal representation such as:

```json
{
  "intent": "predict_total_games",
  "entities": {
    "player_a": "Jannik Sinner",
    "player_b": "Carlos Alcaraz",
    "surface": "clay",
    "tournament_category": "ATP 1000",
    "round": "SF",
    "best_of": 3,
    "indoor": false
  },
  "requires_clarification": false,
  "clarification_question": null
}
```

The serving layer must then transform this into the exact feature/model query required by the selected model.

### Design requirements

- The primary agent does **not** train models.
- The primary agent does **not** scrape data.
- The primary agent is responsible only for **understanding the request, routing it, and formatting the output**.
- The design should allow swapping the NLP implementation later without changing the model-serving layer.
- Implement the agent as a separate module, for example `agent/` or `orchestration/`.

---

## High-Level Architecture

Implement a modular Python project with the following top-level directories:

- `agent/` or `orchestration/`  
  Natural-language primary agent that:
  - detects user intent,
  - extracts entities,
  - asks clarifying questions when needed,
  - routes the request to the correct trained model,
  - formats the final response.

- `scraper/`  
  Scrape:
  - current ATP top 200 rankings,
  - player profile/stat pages,
  - tournament and results pages,
  - match-stat pages (where available).

- `storage/`  
  - Normalize scraped data into structured tables.
  - Write raw and normalized data to **HDFS**.

- `features/`  
  - Generate historical **player features**, **head-to-head features**, **match context features**, and **interaction features** for each match.

- `models/`  
  - Train **one model per prediction target** (see Targets section).

- `evaluation/`  
  - Evaluate each model with appropriate metrics.
  - Save evaluation results and diagnostic plots/metrics to HDFS.

- `serving/`  
  - Provide a simple inference API (Python functions/CLI) that:
    - takes structured input `(player_a, player_b, surface, context)`,
    - returns the full set of predictions for that match.

- `infra/`  
  - Docker Compose for Hadoop HDFS (and optionally Spark).
  - Any other infrastructure scripts.

- `config/`  
  - Centralization of:
    - scraping settings (rate limit, retries),
    - feature windows,
    - dataset filters (years, surfaces),
    - model hyperparameters.

---

## Data Acquisition

### Player universe

1. Scrape the **current ATP Top 200 singles rankings** from the official ATP rankings page.
   - Extract: rank, player name, country, and player profile links / IDs.
   - This set defines the **player universe**.

2. For each player in the top 200:
   - Resolve to a stable ATP identifier (from profile URLs).
   - Store a mapping `player_id` → canonical name.

### Match and stats scraping

For each top-200 player:

1. Scrape historical matches for approximately the last 20 years:
   - Tournament name and ID.
   - Match date.
   - Round.
   - Surface.
   - Indoor/outdoor flag if available.
   - Best-of format if available.
   - Opponent.
   - Winner/loser.
   - Scoreline (e.g. `"7-6(5) 3-6 6-4"`).

2. For each match, attempt to scrape **match-level player stats**:
   - Aces.
   - Double faults.
   - First serve in %.
   - First serve points won %.
   - Second serve points won %.
   - Service points won / total service points.
   - Return points won / total return points.
   - Total points won.
   - Break points saved / faced.
   - Break points converted / created.
   - etc., as available on ATP match-stats pages.

3. Where available, also scrape any **player stats summary pages** to get season/career aggregates per surface.

### Scraper requirements

- Implement a **conservative, configurable rate limiting** policy:
  - Parameters via config:
    - `requests_per_minute`
    - `min_delay_seconds`
    - `max_delay_seconds`
    - `max_retries`
    - `backoff_multiplier`
    - `max_parallel_requests`
  - Use random jitter between requests.
  - Use exponential backoff on HTTP 429/403/5xx.
  - Low concurrency by default.

- Add:
  - Rotating user-agent support.
  - HTML caching (optional) to avoid repeated downloads.
  - Logging: URL, timestamp, status code, retries.
  - Checkpointing: if the scraper stops, it can resume without restarting from scratch.

- Treat ATP HTML structure as **unstable**:
  - Use robust parsing strategies (e.g. XPath + fallback).
  - Fail gracefully when some stats are missing (don’t crash the entire pipeline).

> Note: ATP does not publish official API rate limits; do not hardcode any specific limit value as an ATP guarantee. Make the limits configurable.

---

## Storage on Hadoop HDFS

Run a **Hadoop cluster inside Docker** and store all data on HDFS.

### Docker/Hadoop

- Provide `infra/docker-compose.yml` that starts:
  - HDFS NameNode
  - HDFS DataNode(s)
  - (Optional) YARN ResourceManager / NodeManager
  - Jupyter or a dev container as needed

### HDFS layout

Use an HDFS directory structure such as:

- `/data/raw/rankings/`
- `/data/raw/players/`
- `/data/raw/matches/`
- `/data/raw/match_stats/`
- `/data/bronze/`  (cleaned but minimally transformed)
- `/data/silver/`  (normalized relational tables)
- `/data/gold/features/`  (final feature sets for training/inference)
- `/data/models/`  (trained model artifacts)
- `/data/evaluation/`  (metrics, plots, logs)

Use **Parquet** as the default format in `silver` and `gold`.

---

## Dataset Design

Create at least these normalized tables (silver layer):

- `players`  
  - `player_id`, `name`, `country`, etc.

- `rankings_snapshot_current_top200`  
  - `snapshot_date`, `rank`, `player_id`

- `matches`  
  - `match_id`, `date`, `tournament_id`, `tournament_name`, `round`, `surface`, `indoor`, `best_of`, `player_a_id`, `player_b_id`, `winner_id`, `scoreline`

- `match_player_stats`  
  - `match_id`, `player_id`, and all per-player match stats (aces, dfs, first serve info, bp stats, points won, etc.)

- `player_stats_snapshots` (optional if needed)  
  - aggregated stats per player per season/surface.

- `head_to_head_history`  
  - `player_a_id`, `player_b_id`, `match_id`, `winner_id`, `surface`, `date`.

Then create a **final modeling table** (gold layer):

- `pre_match_features`  
  Each row corresponds to one match (viewed from a given player perspective, or symmetric) and contains:
  - target labels for that match,
  - pre-match features for both players,
  - matchup features (see below).

---

## Feature Engineering

**Critical rule:** all features must be computed using only data **strictly before the match date** to avoid data leakage.

### Player historical features

For each player, compute rolling, surface-aware features over several windows (e.g. last 5, 10, 25 matches; 52 weeks; career):

- **Outcome features**
  - total wins/losses
  - win rate

- **Serve features**
  - first serve in %
  - first serve points won %
  - second serve points won %
  - service points won %
  - service games won % (hold%)
  - aces per match / per service game
  - double faults per match / per service game

- **Return features**
  - return points won %
  - return games won % (break%)
  - break points converted %
  - break points created per return game

- **Clutch features**
  - break points saved %
  - break points faced per match
  - break points converted in last N matches

- **Meta features**
  - ranking
  - ranking points (if available)
  - Elo rating (overall and surface-specific)
  - recent form (e.g. Win/Loss streak, last N matches win rate)
  - indoor/outdoor splits where possible

Compute per-surface variants wherever possible:
- overall
- clay
- hard
- grass
- indoor

### Head-to-head features

For `(player_a, player_b)` computed before the match:

- total number of H2H matches
- wins by player A vs player B
- H2H win rate for A vs B
- H2H win rate for A vs B on current surface
- H2H results in the last N years
- time elapsed since last H2H meeting

### Match context features

Include:

- surface
- indoor/outdoor
- tournament category (e.g. 250/500/1000/GS) if derivable
- round (e.g. R32, QF, SF, F)
- best-of (3 vs 5)
- season/month
- days since last match for each player (if feasible)

### Interaction features (matchup-specific)

Design **interpretable matchup features** that combine the profiles of both players:

Examples:

- Serve vs return:
  - `p1_return_1st_pts_won - p2_1st_serve_pts_won`
  - `p1_return_2nd_pts_won - p2_2nd_serve_pts_won`
  - `p1_1st_serve_pts_won - p2_return_1st_pts_won`
  - `p1_2nd_serve_pts_won - p2_return_2nd_pts_won`

- Hold vs break:
  - `p1_service_games_won - p2_return_games_won`
  - `p1_return_games_won - p2_service_games_won`

- Break-point dynamics:
  - `p1_bp_converted - p2_bp_saved`
  - `p1_bp_saved - p2_bp_converted`

- Aces vs return:
  - `p1_ace_rate - p2_ace_suppression_proxy`
  - `p1_ace_to_df_ratio - p2_ace_to_df_ratio`

- Surface-adjusted:
  - surface-specific versions of the above, e.g. `p1_service_games_won_clay - p2_return_games_won_clay`.

Include symmetric/mirrored versions for both players or define a consistently ordered perspective (e.g. player with smaller ID is always P1).

---

## Target Construction

For each match, derive the following targets.

### Classification targets

- `winner_label`  
  - 1 if player A wins, 0 otherwise (or multiclass if needed).

- `deciding_set_label`  
  - 1 if the match goes to a deciding set (3rd in Bo3, 5th in Bo5), 0 otherwise.

- `tiebreak_present_label`  
  - 1 if at least one tie-break occurs in the match, 0 otherwise.

- `first_set_winner_label`  
  - 1 if player A wins the first set, 0 otherwise.

### Regression targets

- `player_aces_target`  
  - Number of aces by player A in the match.

- `total_aces_target`  
  - Total number of aces in the match.

- `total_games_target`  
  - Total number of games in the match.

Parsing rules must be defined to compute these from the scoreline (e.g. parse sets, games per set, tie-break notation).

---

## Modeling Approach

Use **separate models per target**, not multitask, for clarity and easier debugging.

### Model families

Prefer tree-based gradient boosting for tabular data:

- XGBoost
- LightGBM
- CatBoost (optional)

### Per-target choice

- Classification targets:
  - winner (probability + class)
  - deciding set (probability)
  - tie-break present (probability)
  - first set winner (probability + class)

- Regression targets:
  - player A aces
  - total aces
  - total games

Configure each model with:

- feature list (documented),
- hyperparameters (documented),
- appropriate regularization to avoid overfitting.

---

## Evaluation

Use **time-based splits** to simulate future prediction and avoid leakage:

- Train: older years (e.g. first ~70% of temporal range)
- Validation: middle range
- Test: most recent years (e.g. last 2–3 seasons)

### Metrics

- For classification:
  - Accuracy
  - ROC AUC
  - Log loss
  - Brier score
  - Calibration curves

- For regression:
  - MAE
  - RMSE
  - R²
  - Residual plots

Also compute breakdowns by:

- surface (clay, hard, grass, indoor),
- tournament category (250/500/1000/GS),
- ranking-difference buckets (favorite vs underdog).

Store evaluation results and plots under `/data/evaluation/`.

---

## Inference Contract

Design an inference interface that takes structured input such as:

```json
{
  "player_a": "Jannik Sinner",
  "player_b": "Carlos Alcaraz",
  "surface": "clay",
  "tournament_category": "ATP 1000",
  "round": "SF",
  "best_of": 3,
  "indoor": false,
  "match_date": "2026-05-20"
}
```

And returns:

```json
{
  "win_probability_player_a": 0.54,
  "win_probability_player_b": 0.46,
  "predicted_winner": "Jannik Sinner",
  "expected_aces_player_a": 7.2,
  "expected_total_aces": 11.8,
  "expected_total_games": 24.6,
  "prob_deciding_set": 0.41,
  "prob_tiebreak": 0.33,
  "prob_player_a_first_set": 0.49
}
```

The inference layer should:

- load the appropriate models,
- fetch the latest features from HDFS (or pre-computed feature store),
- construct the feature vector from the two players’ profiles and match context,
- output the prediction dictionary.

---

## NLP Readiness

The current scope does **not** require a full conversational agent with memory or multi-turn dialogue management, but it **does require a primary agent** that translates natural language into a structured model query.

The architecture must therefore keep the following separation:

- **Primary agent layer**: understands user requests and routes them.
- **Inference layer**: receives structured parameters and calls the correct model.
- **Model layer**: produces predictions.

This separation is mandatory so the NLP/orchestration logic can evolve independently from the ML models.

---

## Code Quality & Infra Requirements

Claude Code should:

- Use a clean Python package structure with modules for:
  - primary agent / orchestration,
  - scraping,
  - normalization,
  - feature engineering,
  - modeling,
  - evaluation,
  - inference,
  - configuration.
- Use type hints where practical.
- Provide `docker-compose.yml` to run Hadoop HDFS (and optionally Spark).
- Provide a README with:
  - setup instructions,
  - how to run scraping,
  - how to build features,
  - how to train models,
  - how to run inference,
  - how to run the primary agent router.

Additional requirements:

- Idempotent scraping and feature generation jobs (safe to re-run).
- Checkpointing / resume for scraping.
- Reproducible training scripts with fixed randomness seeds.
- Saving models and configs in `/data/models/`.
- Reasonable unit tests around:
  - HTML parsing,
  - feature computations,
  - scoreline → target parsing,
  - primary agent intent routing,
  - entity extraction,
  - clarification behavior.

---

## Extra Implementation Note (for Claude Code)

> Build the project for robustness first, not scraping speed. Treat ATP scraping as unstable and partially incomplete over a 20-year horizon. Make the code resilient to HTML changes, missing match-stat pages, and partial data coverage. Design the feature pipeline so that missing advanced ATP stats do not break model training; instead, degrade gracefully to a smaller feature subset.

> Also ensure the primary agent is modular and decoupled from the trained models, so the natural-language understanding layer can be improved or replaced later without changing the prediction APIs.
