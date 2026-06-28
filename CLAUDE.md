# The Self-Auditing Ledger

ERP fraud-detection research project. This file is the source of truth for *how*
work is done here. Read it before writing any analysis code.

## Goal — two-phase fraud detection

**Phase 1 — Shadow Graph (topology).** Build a *temporal* Neo4j graph from ERP
transaction data and extract structural / topology features for **each
transaction**. The graph is "temporal" because edges and reachability are
constrained by time.

**Phase 2 — Modeling.** Train ML models on the per-transaction feature table
produced by Phase 1, with rigorous, leakage-free evaluation.

**The actual claim we must prove.** The real-time platform in the project PDF is
the *vision*, not the *proof*. The defensible scientific claim is narrower:
*topology features add predictive value over a strong tabular baseline, measured
without leakage.* The decisive test is a within-dataset ablation (tabular vs
tabular+topology) — see `experiments/ablation.py` and `docs/DESIGN.md`. We commit
to reporting the result either way, including a negative one.

## Locked decisions (2026-06-24) — see `docs/DESIGN.md` for full reasoning

- **Deliverable:** both a rigorous result *and* an explainable demo.
- **Feature engine:** Pandas + NetworkX/igraph is the **compute path** (reproducible,
  scales, leakage-safe). Neo4j is the **demo/explanation path** only (the visual
  shadow graph + detected-path audit alerts). The old version OOM'd on Neo4j at
  scale and silently disabled topology — do not repeat that.
- **Datasets (tiered):** Tier 1 = IBM AML + BankSim; Tier 2 = SAP Würzburg + PaySim;
  Tier 3 = FIFAR (optional). **SAP IDES is dropped** (trial-watermark left 7 frauds).
- **Synthetic ERP generator:** deferred to Phase 6 (avoid the circularity trap).

## Non-negotiable methodology rules

These exist because the earlier version (see `reference/`) violated them. Do not
regress.

1. **One imbalance mechanism, not two.** Use resampling (e.g. SMOTE) *or*
   class weights — **never both at once**. Pick one per experiment and record it.
   **Default to class weights** for graph/topology features: SMOTE interpolates
   between samples, producing non-physical topology vectors ("half a cycle"). Use
   SMOTE only as a deliberate, logged comparison.
2. **Tune the decision threshold on validation data.** Never report metrics at a
   fixed `0.5` threshold. The chosen threshold is selected on a validation split
   and then frozen for the test set.
3. **Time-aware cross-validation.** Train on earlier periods, test on later
   periods. **Never** use randomly shuffled folds — that leaks the future into
   the past.
4. **No look-ahead through the graph.** Topology features for a transaction may
   only depend on transactions with `timestamp <= its own`. No information from
   later transactions may flow into an earlier transaction's features. Implement
   this as *as-of* / streaming computation: walk transactions in time order,
   compute features against the graph state at time T (optionally a trailing
   window), then insert. **No whole-dataset normalization** — every statistic
   (mean, z-score, entity average) must be expanding/rolling over the past only.
   Attribute a structure's signal to the transaction that *completes* it, never
   retroactively to its earlier members. (The old engine violated all of this:
   `time_to_next` was a top feature and z-scores used whole-dataset stats.)
5. **Report PR-AUC / average precision alongside F1.** On heavily imbalanced
   fraud data, precision-recall metrics are primary; ROC-AUC alone is misleading.
   PR-AUC is also the **model-selection** metric — never select on F1-at-0.5. Also
   report **precision@k / recall@k**: an auditor only reviews the top-N alerts/day.
6. **Persist artifacts.** Every trained model must be saved together with its
   fitted scaler and its chosen decision threshold (see `artifacts/`).

## The `reference/` folder

`reference/shadowGraphs_old/` is an earlier version of this project (a flat
`step1…step9` script pile covering 5 datasets). Treat it as **read-only**.

- **DO** mine it for proven Cypher queries and hard-won ETL fixes — see its
  `step*_format_*.py`, `step*_load_into_neo4j.py`, and `step*_topology_engine.py`
  scripts.
- **DO NOT** copy its training / evaluation code (`step*_model_training*.py`).
  That code contains exactly the flaws the rules above forbid (SMOTE + class
  weights together, fixed 0.5 threshold, shuffled CV).
- **Do not trust its reported metrics.** Its headline claim ("graph topology is
  the differentiator") is contradicted by its own feature-importance artifacts —
  the topology detectors scored 0.0 importance; the signal was `total_amount` +
  amount z-scores. All metrics are leakage-contaminated. See `docs/DESIGN.md` §1–2.

## Source layout (`src/`)

| Package          | Responsibility                                                   |
|------------------|------------------------------------------------------------------|
| `common`         | Config loader, logging, Neo4j connection.                        |
| `etl`            | Raw CSV → canonical node / edge files.                           |
| `graph`          | Neo4j load + schema constraints.                                 |
| `topology`       | The 5 detectors + structural features (the **single** copy).     |
| `features`       | Assemble the per-transaction feature table.                      |
| `modeling`       | Time-aware CV, threshold tuning, evaluation, persistence.        |

`experiments/` holds runnable entry points (`run_dataset.py`, `ablation.py`).
Intermediate ETL outputs live in `data/processed/`; saved models/scalers/
thresholds in `artifacts/`; metrics and plots in `results/`.

## Setup

`pip install -e .` makes the `src/` packages importable without `sys.path` hacks.
Per-dataset configuration lives in `config/datasets.yaml`.
