---
description: Retrain ATP prediction models end-to-end with validation
user_invocable: true
---

# Retrain Models

Follow these steps in order. Do not skip validation steps.

## 1. Pre-flight checks
- Verify S3 access is configured (boto3 credentials)
- Check that gold feature data exists on S3 at the expected paths
- If features are stale or missing, run the feature pipeline first (`atp-features`)
- Sync gold data from S3 to local cache and verify row counts match

## 2. Regenerate player profiles
- Run `features/player_profiler.py` to regenerate player profiles
- **Verify** player_id consistency between ranking data and match history (compare ID sets, flag mismatches)
- **Verify** Elo ratings (overall + surface-specific) are included in profiles — do NOT omit Elo
- Check for NaN values and ensure consistent handling

## 3. Run feature pipeline
- Execute the full feature pipeline via `atp-features`
- Verify output row counts and feature completeness
- **Re-upload updated features to S3** if you modified any data locally

## 4. Train models
- Train all per-target models via `atp-train`
- Models: winner probability, winner label, player aces, total aces, total games, deciding set, tiebreak, first set winner
- Use time-based train/val/test splits (no random splits)
- Save model artifacts to S3 under the models path

## 5. Evaluate
- Run `atp-eval` on the held-out test set
- Report metrics per target: accuracy, ROC AUC, log loss, Brier score (classification); MAE, RMSE, R² (regression)
- Report breakdowns by surface and tournament category
- Compare against previous baseline if available
- Save evaluation results to S3

## 6. Validate end-to-end
- Run a sample inference via `atp-predict` for a known matchup (e.g. Sinner vs Alcaraz on clay)
- Verify predictions are reasonable (probabilities in [0,1], aces/games in plausible ranges)
- If anything looks off, diagnose before declaring success

## 7. Report
- Produce a summary with:
  - Before/after metrics (if baseline exists)
  - Any data issues found and fixed
  - Model artifacts saved
- Ask user to review before committing changes
