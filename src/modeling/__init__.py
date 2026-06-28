"""modeling — time-aware CV, threshold tuning, evaluation, persistence.

Responsibilities:
    * Time-aware cross-validation (train earlier periods, test later periods).
    * A single imbalance mechanism per run (resampling XOR class weights).
    * Decision-threshold tuning on validation (never a fixed 0.5).
    * Evaluation reporting PR-AUC / average precision alongside F1.
    * Persist model + fitted scaler + chosen threshold to artifacts/.

See CLAUDE.md for the non-negotiable methodology rules this module must enforce.
"""

# TODO: implement CV, threshold tuning, evaluation, and artifact persistence.
