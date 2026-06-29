"""features — assemble the per-transaction model table (ordinary + topology).

Box (4) of the architecture. Joins two clearly-separated feature families so the
ablation (experiments/ablation.py) can switch topology on and off:

  * ORDINARY  — a strong, leakage-free tabular baseline: the transaction's own
    amount plus **expanding** (past-only) per-account statistics, plus static
    account attributes. Every statistic is computed over an account's *prior*
    transactions only — the legitimate cousin of the old version's #1 leak,
    ``time_to_next`` (we use ``time_since_prev``; the future is never touched).
  * TOPOLOGY  — the 6 columns produced by ``topology.engine`` (cycle / fan-in /
    degrees), already leakage-free by construction.

Leakage discipline here (CLAUDE.md rules 3-4):
  * Expanding stats use ``cumsum``/``cumcount`` shifted to EXCLUDE the current row,
    grouped per account, in (timestamp, transaction_id) order — strictly past-only,
    consistent with the topology engine's "<= T, id-order tie-break" rule.
  * **No whole-dataset normalisation.** We do NOT z-score against global means; the
    z-score is relative to each account's own past. Any train-only fitting (e.g. a
    scaler for logistic regression) happens later in the modeling split, never here.
  * Zero-variance columns (IBM: tx_type=TRANSFER, COUNTRY=US, ACCOUNT_TYPE=I) and
    label-derived columns (``is_fraud_account``) are dropped — they are useless or
    cheating, respectively.

Output: ``data/processed/<name>/features_model.parquet`` + ``features_manifest.json``
listing which columns are ordinary vs topology (and the label / keys).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_dataset

log = logging.getLogger(__name__)

# Columns that must NEVER become features (ids, raw time, labels, eval-only metadata).
_NEVER_FEATURE = {
    "transaction_id", "source_account", "dest_account", "timestamp", "label",
    "alert_type", "alert_id", "run_id", "is_fraud_account", "CUSTOMER_ID",
    "account_id", "TX_BEHAVIOR_ID",  # simulator-internal / id-like -> exclude to stay honest
    "position", "vendor",            # SAP line-index / high-card vendor id -> exclude
}

# Edge-level categoricals with cardinality in this range become one-hot baseline
# features (BankSim `category`, PaySim/SAP `tx_type`, SAP posting semantics). This
# keeps the tabular baseline fair across datasets — omitting them would unfairly
# flatter topology.
_CAT_MIN, _CAT_MAX = 2, 30


def _expanding_account_stats(edges: pd.DataFrame) -> pd.DataFrame:
    """Past-only per-account features. Assumes ``edges`` is in (timestamp, id) order.

    For the SOURCE account of each transaction, using only that account's earlier
    transactions:
      * ``src_count_prior``      — how many txns this account sent before (expanding count)
      * ``src_amt_mean_prior``   — mean amount of those prior txns
      * ``amount_zscore_src``    — (amount - prior mean) / prior std  (NaN if no history)
      * ``time_since_prev_src``  — gap to this account's previous txn (NaN if first)
    For the DEST account:
      * ``dest_count_prior``     — how many txns this account received before
    NaNs (no history yet) are left as-is — XGBoost handles missing values natively.
    """
    out = pd.DataFrame(index=edges.index)
    amt = edges["amount"].astype(float)

    gsrc = edges.groupby("source_account", sort=False)
    cnt_prior = gsrc.cumcount()                              # # of prior src txns (0-based)
    cum_amt = gsrc["amount"].cumsum() - amt                  # sum of prior amounts
    cum_sq = gsrc["amount"].transform(lambda s: (s * s).cumsum()) - amt * amt
    mean_prior = cum_amt / cnt_prior.where(cnt_prior > 0)    # NaN when no prior
    var_prior = (cum_sq / cnt_prior.where(cnt_prior > 0)) - mean_prior ** 2
    std_prior = np.sqrt(var_prior.clip(lower=0))
    prev_ts_src = gsrc["timestamp"].shift(1)                 # PAST only (shift +1, never -1)

    out["src_count_prior"] = cnt_prior.astype("float32")
    out["src_amt_mean_prior"] = mean_prior.astype("float32")
    out["amount_zscore_src"] = ((amt - mean_prior) / std_prior.replace(0, np.nan)).astype("float32")
    out["time_since_prev_src"] = (edges["timestamp"] - prev_ts_src).astype("float32")
    out["dest_count_prior"] = edges.groupby("dest_account", sort=False).cumcount().astype("float32")
    return out


def build_features(name: str, config_path: str | None = None) -> dict:
    """Assemble the model feature table for one dataset. Returns a manifest dict."""
    ds = get_dataset(name, config_path)
    proc = Path(ds["processed_dir"])
    edges = pd.read_csv(proc / "edges_transactions.csv", low_memory=False)
    topo = pd.read_csv(proc / "features_topology.csv")
    if not edges["timestamp"].is_monotonic_increasing:
        edges = edges.sort_values("timestamp", kind="stable").reset_index(drop=True)

    pcol = ds.get("partition_col")  # e.g. run_id (SAP) — carried for the split, never a feature
    base_keys = ["transaction_id", "timestamp", "label"]
    if pcol and pcol in edges.columns:
        base_keys.append(pcol)
    feats = edges[base_keys].copy()

    # --- ordinary: amount + expanding account stats --------------------------
    feats["log_amount"] = np.log1p(edges["amount"].clip(lower=0)).astype("float32")
    feats = pd.concat([feats, _expanding_account_stats(edges)], axis=1)

    # --- ordinary: static account attributes (leakage-safe; drop zero-variance) ---
    nodes = pd.read_csv(proc / "nodes_accounts.csv", low_memory=False)
    nodes["account_id"] = nodes["account_id"].astype(str)
    attr_cols = [c for c in nodes.columns if c not in _NEVER_FEATURE]
    # keep only attributes with real variance (IBM's COUNTRY/ACCOUNT_TYPE are constant)
    attr_cols = [c for c in attr_cols if pd.api.types.is_numeric_dtype(nodes[c]) and nodes[c].nunique() > 1]
    if attr_cols:
        amap = nodes.set_index("account_id")[attr_cols]
        for side, key in (("src", "source_account"), ("dest", "dest_account")):
            joined = edges[key].astype(str).map(amap.to_dict(orient="index"))
            jdf = pd.json_normalize(joined).add_prefix(f"{side}_")
            jdf.index = feats.index
            feats = pd.concat([feats, jdf.astype("float32")], axis=1)

    # --- ordinary: low-cardinality edge categoricals (one-hot) ---------------
    used = {"amount", "source_account", "dest_account"}
    cat_cols = [
        c for c in edges.columns
        if c not in _NEVER_FEATURE and c not in used
        and not pd.api.types.is_numeric_dtype(edges[c])
        and _CAT_MIN <= edges[c].nunique(dropna=True) <= _CAT_MAX
    ]
    for c in cat_cols:
        dummies = pd.get_dummies(edges[c].astype("string").fillna("NA"), prefix=c).astype("float32")
        dummies.index = feats.index
        feats = pd.concat([feats, dummies], axis=1)
    if cat_cols:
        log.info("[%s] one-hot edge categoricals: %s", name, cat_cols)

    ordinary_cols = [c for c in feats.columns if c not in _NEVER_FEATURE]

    # --- topology: join the 6 leakage-free columns ---------------------------
    topo_cols = [c for c in topo.columns if c != "transaction_id"]
    feats = feats.merge(topo, on="transaction_id", how="left")

    # --- write ----------------------------------------------------------------
    out_path = proc / "features_model.parquet"
    try:
        feats.to_parquet(out_path, index=False)
    except Exception:  # pragma: no cover - fallback if no parquet engine
        out_path = proc / "features_model.csv"
        feats.to_csv(out_path, index=False)

    manifest = {
        "dataset": name,
        "n_rows": int(len(feats)),
        "n_fraud": int(feats["label"].sum()),
        "keys": ["transaction_id", "timestamp"],
        "label": "label",
        "ordinary_cols": ordinary_cols,
        "topology_cols": topo_cols,
        "output": out_path.name,
    }
    with open(proc / "features_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("[%s] features: %d ordinary + %d topology cols -> %s",
             name, len(ordinary_cols), len(topo_cols), out_path.name)
    return manifest


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "ibm_aml"
    print(json.dumps(build_features(target), indent=2))
