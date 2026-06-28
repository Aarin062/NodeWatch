# Design & Plan — The Self-Auditing Ledger (v2)

> This document is the reasoning behind the rebuild. `CLAUDE.md` holds the terse
> rules; this holds the *why*. Written 2026-06-24 after a full audit of the old
> version in `reference/shadowGraphs_old/`.

---

## 1. The finding that reshaped the project

The old version's headline claim — *"graph topology is the differentiator for
fraud detection"* — **is not supported by its own results.**

On the flagship SAP ERP dataset, the model's feature-importance file shows the
signal came almost entirely from `total_amount` (62%) plus a few amount-anomaly
z-scores and timing gaps. **Every deep topology feature — `cycle_count`,
`in_cycle`, lapping, density, rapid-chain, degree — scored 0.0 importance.** The
five "fraud shapes" the entire Phase-1 engine exists to find contributed nothing.

This is not a reason to abandon the idea. It is the reason for v2: **the
hypothesis was never given a fair test.** Three things sabotaged it, all fixable
(Sections 2–4).

## 2. Why the old numbers can't be trusted (the leakage chain)

Two independent leaks stacked on top of each other. Every impressive number
(e.g. BankSim ROC-AUC ≈ 0.997) is contaminated.

**Leak A — the feature engine looked into the future.** Every detector ran over
the *entire* graph (all timestamps) and credited each structure back to *all* its
members, including the earliest. `time_to_next` (seconds until the *next*
transaction) was a top-5 feature. Z-scores used whole-dataset statistics. No
`timestamp <= T` restriction existed anywhere. → Violates Rule #4.

**Leak B — the modeling protocol leaked too** (confirmed in all 5 training
scripts): SMOTE **and** class weights together; fixed 0.5 threshold (`.predict()`
everywhere, never tuned); shuffled `StratifiedKFold` with `timestamp` dropped;
nothing persisted; "best model" chosen by F1-at-0.5 instead of PR-AUC.

**Consequence:** we are not improving the old metrics — we are discarding them and
producing the first trustworthy ones.

## 3. The reframed thesis (what we actually have to prove)

The PDF describes a real-time production platform. That is the *vision*; it is not
the *proof*. The defensible scientific claim is narrower:

> **Topology features from a temporal financial graph add predictive value over a
> strong tabular baseline — measured without leakage.**

We keep the two separate:
- **The proof** = a leakage-free within-dataset ablation (Section 7). The deliverable.
- **The vision** = the streaming + explainable-alert architecture, which we also
  demonstrate — and which, conveniently, is the cleanest way to *prevent* leakage.

## 4. Locked decisions (2026-06-24)

| Decision | Choice | Rationale |
|---|---|---|
| Primary deliverable | **Both** rigor + demo | Build for proof *and* presentation. |
| Feature engine | **Pandas + NetworkX/igraph (primary); Neo4j (demo/explanation)** | Old version OOM'd on Neo4j at scale and disabled topology. Python computes features reproducibly and leakage-safe; Neo4j powers the visual shadow-graph & path explanations. |
| Synthetic ERP generator | **Deferred to Phase 6** | Real datasets first; avoid the circularity trap (Section 8). |
| Dataset roster | **Tiered** (Section 6) | Fail fast/cheap on the decisive datasets before broadening. |

## 5. Architecture

```
raw CSV ─▶ etl ─▶ (unified entity graph) ─▶ topology (streaming, as-of) ─▶ features
                                                                              │
                                          modeling (time-aware CV, threshold) ◀┘
                                                     │
                                  artifacts (model+scaler+threshold)  +  results
                                                     │
                          explainability: SHAP + detected-path audit alert (Neo4j demo)
```

### 5.1 Data layer (`etl/`) — fix this first; everything sits on it
- **Unify the account namespace (highest-leverage fix).** In IBM AML & PaySim the
  old ETL split sender→`Account` and receiver→`Vendor`, so account *X*-as-sender
  and *X*-as-receiver became different nodes — making cycles structurally
  undetectable. v2: one `Account` node type with directed `TRANSFERS_TO` edges.
- **Preserve transaction `type`** in PaySim (TRANSFER→CASH_OUT is *the* fraud pattern; old ETL dropped it).
- **Honest time axis.** No dataset has true wall-clock time (DS1 = time-of-day,
  dates fabricated per filename; DS3/4/5 = integer step/day counters). Use the
  coarse step/day for ordering & splitting; **never** fabricate within-tie order
  from row order (the old version did — it leaked the source file's pre-sort).
- **Reuse the good old ETL bits:** source-file ID-prefixing, BankSim quote-strip,
  SAP `bkpf ⋈ bseg` join SQL (exact locations noted in the audit).

### 5.2 Leakage-free temporal feature engine (`topology/`, `features/`) — the heart
Implement as **incremental/streaming**: walk transactions in time order, maintain
graph state, compute features for transaction *t* against the graph **as of its
timestamp T** (optionally a trailing window, e.g. last 24h/7d), then insert *t*.
The future literally cannot leak in. Two hard rules:
1. **No global normalization** — all stats (means, z-scores, vendor averages) are
   *expanding or rolling* over the past only.
2. **Causal attribution** — a structure's signal attaches to the transaction that
   *completes* it, never retroactively to earlier members.

### 5.3 Detectors + real graph features
| Detector / feature | Keep | Notes |
|---|---|---|
| Circular payments | ✅ | Real directed cycles in unified account graph, trailing window. |
| Rapid transfer chains | ✅ | Short inter-hop gaps; fix old hop-cap bug. |
| Density / burst | ✅ | Direct in/out edge counts in window (not capped path expansion). |
| Lapping | ✅ | Near-equal sequential amounts on same entity within window. |
| Segregation of Duties | ⚠️ | Needs Employee + workflow (create/approve/pay) data; only SAP could support it. **Old version never implemented it** (silently swapped in amount-anomalies). Don't fake it. |
| Real centrality (PageRank, betweenness, degree) | ✅ NEW | Compute properly on the as-of snapshot (old "centrality" was just edge counts). |

### 5.4 Modeling (`modeling/`)
- **Time-aware split:** train early → validate middle → test latest; optional
  walk-forward (expanding window). Never shuffle.
- **One imbalance mechanism — default class weights, not SMOTE.** SMOTE
  interpolates between samples; interpolating topology vectors yields non-physical
  points ("half a cycle"). Use SMOTE only as a deliberate, logged comparison.
- **Tune threshold on validation**, freeze, apply to test.
- **Metrics:** PR-AUC / average precision is **primary** (and the model-selection
  metric). Add **precision@k / recall@k** (auditors review only top-N alerts/day).
  Report ROC-AUC but de-emphasize it. Never select on F1-at-0.5.
- **Models:** XGBoost/LightGBM workhorse (interpretable via SHAP) + logistic
  baseline. GNN/GraphSAGE = optional "advanced comparison" only — a black box
  works against our explainability selling point.
- **Persist** model + scaler + threshold + feature list + metadata per run.

### 5.5 Explainability — the real differentiator
On a flag, surface (a) which detector(s) fired + the concrete subgraph/path (from
Neo4j) and (b) SHAP attributions. This is the "audit alert" the old version never
actually built, and it's what separates this from "yet another XGBoost."

## 6. Dataset roster (tiered)

Datasets are points on a **graph-structure-richness spectrum** — the independent
variable of the thesis. Each gets its own within-dataset ablation; the
cross-dataset trend is *secondary* confirmation (the old project made that trend
its *primary*, confounded, evidence).

| Tier | Dataset | Real fraud | Structure | Role |
|---|---|---|---|---|
| 1 | **IBM AML** | 1,719 (0.13%) | Rich account↔account, cycles | Topology's best shot — decisive testbed |
| 1 | **BankSim** | 7,200 (1.21%) | Bipartite, no cycles | Sparse control ("topology should help less") |
| 2 | **SAP Würzburg** | 49 (0.08%) | Thin SAP star | ERP domain + audit-alert demo (weak stats) |
| 2 | **PaySim** | 8,213 (0.13%) | Rich P2P @ 6.3M rows | Scale + transfer→cash-out pattern |
| 3 | **FIFAR** | 5,705 (1.13%) | None (tabular) | Optional floor anchor |
| — | ~~SAP IDES~~ | 7 | Broken (trial watermark) | **Dropped** |

## 7. The experiment that decides everything

> Same dataset, same time-aware splits, same model, same single imbalance
> mechanism — trained twice: **(A) tabular only** vs **(B) tabular + topology**.
> Compare PR-AUC. Repeat per dataset (`experiments/ablation.py`).

If (B) > (A) on AML with leakage-free features → the thesis is real and properly
proven. If not → an honest, publishable **negative result** ("explicit topology
features add nothing beyond tabular + amount-anomaly signal on these datasets"),
far more credible than the old 0.997 AUC. **We commit now to reporting whichever
way it lands.** Designing the fair test is the contribution.

## 8. Risks
- **Negative result is possible** — accepted (Section 7).
- **Synthetic-generator circularity** (Phase 6): if we inject fraud as the exact
  pattern a detector seeks, we "prove" nothing. Inject by realistic *behavior /
  intent*; let detectors rediscover structure blind. (The old DS2 fell into this:
  it injected fraud literally labeled `"Circular Payment"` / `"Lapping"`.)
- **Scale** (PaySim 6.3M): the streaming/windowed Python engine must chunk; this
  is why Neo4j is demo-only, not the compute path.

## 9. Roadmap (mapped to the scaffold)
- **Phase 0** — Data audit & honest re-baseline; lock fraud rates & time semantics.
- **Phase 1** — ETL fixes (`etl/`): unified accounts, preserve `type`, honest time. Tier-1 datasets first.
- **Phase 2** — Leakage-free temporal feature engine (`topology/` + `features/`).
- **Phase 3** — Modeling harness (`modeling/`): time-aware CV, one imbalance mechanism, threshold tuning, PR-AUC + precision@k, persistence.
- **Phase 4** — The ablation (`experiments/ablation.py`). The money experiment.
- **Phase 5** — Explainability: SHAP + detected-path audit alerts; Neo4j demo.
- **Phase 6 (optional)** — Synthetic generator, GNN comparison, Tier-3 dataset.

## 10. Phase-0 audit findings (Tier-1, 2026-06-24)

- **Raw data location:** `C:\Users\Aarin Bhatta\OneDrive\Desktop\Projects\erp datasets\`
  (external to the repo; referenced via `raw_root` in `config/datasets.yaml`, not
  copied — `data/` is gitignored).
- **IBM AML — the namespace fix is CONFIRMED valid.** `transactions.csv` = 1,323,234
  rows, fraud 0.13% (1,719). Senders (9,999) and receivers (9,926) overlap **99.3%**
  → one shared account namespace, exactly what cycle detection needs. `TX_TYPE` is
  100% `TRANSFER` (type adds nothing here — unlike PaySim). `TIMESTAMP` = integer day
  0–199 (200 buckets) → clean axis for time-aware splits.
- **Bonus files the old version ignored (upgrades):**
  - `accounts.csv` (10,000): per-account attributes — `CUSTOMER_ID`, `INIT_BALANCE`,
    `COUNTRY`, `ACCOUNT_TYPE`, `IS_FRAUD` (bad-actor flag), `TX_BEHAVIOR_ID` → node features.
  - `alerts.csv` (1,719): **ground-truth fraud typology per transaction —
    `ALERT_TYPE` ∈ {cycle, fan_in, …}.** Use it to (a) check detectors against the
    *true* structure, (b) stratify the ablation by typology, (c) power audit-alert
    explanations. **Evaluation metadata only — never a feature** (it's label-derived).
- **BankSim — bipartite confirmed.** 594,643 rows, fraud 1.21% (7,200). customer
  (`C…`) → merchant (`M…`) are disjoint namespaces → no account-to-account cycles
  possible → correct as the sparse control. Values are single-quote-wrapped (strip
  on load). `bsNET…csv` is a ready-made edge list.
- **Canonical v2 schema (transfer/bipartite datasets):** `Account` nodes +
  `(:Account)-[:TRANSACTS {amount, timestamp, label, …}]->(:Account)` edges; the
  prediction unit is one transaction (one edge). Replaces the old Document-hub +
  synthetic `TEMPORAL_NEXT` spine.
