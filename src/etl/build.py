"""Raw CSV -> canonical Account graph (nodes + transaction edges).

Canonical outputs (written to ``<processed_root>/<dataset>/``):
  * ``nodes_accounts.csv``     — one row per account (unified namespace).
  * ``edges_transactions.csv`` — one row per transaction (the prediction unit),
    with ``source_account``, ``dest_account``, ``amount``, ``timestamp``,
    ``label`` (0/1) and any extra columns (``tx_type``, ``category``,
    ``alert_type`` ...).
  * ``audit.json``             — row counts and the observed-vs-expected fraud rate.

The key correctness fix vs. the old version: senders and receivers share ONE
``account_id`` namespace, so account-to-account structure (cycles, chains) is
preserved. ``alert_type`` is carried as EVALUATION metadata only — it is
label-derived and must never be used as a model feature.

No topology / look-ahead logic lives here (that is the ``topology`` package).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from common.config import get_dataset

log = logging.getLogger(__name__)

# Canonical extra columns we carry through when present in a dataset's schema.
_OPTIONAL_COLS = ("tx_type", "category", "alert_id")


def _strip_quotes(df: pd.DataFrame, quote: str) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip(quote)
    return df


def _to_binary_label(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy).astype("int8")


def build_dataset(name: str, config_path: str | None = None) -> dict:
    """Build the canonical node/edge files for one dataset. Returns an audit dict."""
    ds = get_dataset(name, config_path)
    schema = ds["schema"]
    out_dir = Path(ds["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- transactions -> canonical edges ------------------------------------
    log.info("[%s] reading transactions", name)
    tx = pd.read_csv(ds["raw"]["transactions"])
    if ds.get("quote_char"):
        tx = _strip_quotes(tx, ds["quote_char"])

    rename = {
        schema["source"]: "source_account",
        schema["dest"]: "dest_account",
        schema["amount"]: "amount",
        schema["timestamp"]: "timestamp",
        schema["label"]: "label",
    }
    if "tx_id" in schema:
        rename[schema["tx_id"]] = "transaction_id"
    for opt in _OPTIONAL_COLS:
        if opt in schema:
            rename[schema[opt]] = opt
    edges = tx.rename(columns=rename)

    # Unified string account namespace (senders & receivers share it).
    edges["source_account"] = edges["source_account"].astype(str)
    edges["dest_account"] = edges["dest_account"].astype(str)
    edges["label"] = _to_binary_label(edges["label"])
    if "transaction_id" not in edges.columns:
        edges.insert(0, "transaction_id", range(len(edges)))

    # ---- attach fraud typology (EVALUATION METADATA ONLY) -------------------
    if "alerts" in ds["raw"]:
        alerts = pd.read_csv(ds["raw"]["alerts"])
        amap = alerts.rename(columns={"TX_ID": "transaction_id", "ALERT_TYPE": "alert_type"})
        edges = edges.merge(amap[["transaction_id", "alert_type"]], on="transaction_id", how="left")

    keep = ["transaction_id", "source_account", "dest_account", "amount", "timestamp", "label"]
    keep += [c for c in (*_OPTIONAL_COLS, "alert_type") if c in edges.columns and c != "alert_id"]
    if "alert_id" in edges.columns:
        keep.append("alert_id")
    edges = edges[keep].sort_values("timestamp", kind="stable").reset_index(drop=True)

    # ---- accounts -> canonical nodes ----------------------------------------
    if "accounts" in ds["raw"]:
        acc = pd.read_csv(ds["raw"]["accounts"])
        nodes = acc.rename(columns={"ACCOUNT_ID": "account_id", "IS_FRAUD": "is_fraud_account"})
        nodes["account_id"] = nodes["account_id"].astype(str)
    else:
        ids = pd.unique(pd.concat([edges["source_account"], edges["dest_account"]], ignore_index=True))
        nodes = pd.DataFrame({"account_id": ids})

    # ---- write + audit ------------------------------------------------------
    edges.to_csv(out_dir / "edges_transactions.csv", index=False)
    nodes.to_csv(out_dir / "nodes_accounts.csv", index=False)

    n_fraud = int(edges["label"].sum())
    observed_rate = n_fraud / len(edges) if len(edges) else 0.0
    s, r = set(edges["source_account"]), set(edges["dest_account"])
    audit = {
        "dataset": name,
        "kind": ds.get("kind"),
        "n_transactions": int(len(edges)),
        "n_accounts": int(len(nodes)),
        "n_fraud": n_fraud,
        "observed_fraud_rate": round(observed_rate, 6),
        "expected_fraud_rate": ds.get("expected_fraud_rate"),
        "namespace_overlap": round(len(s & r) / max(len(s | r), 1), 4),
        "timestamp_range": [int(edges["timestamp"].min()), int(edges["timestamp"].max())],
        "alert_types": (
            edges["alert_type"].value_counts(dropna=True).to_dict()
            if "alert_type" in edges.columns else None
        ),
    }
    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    log.info("[%s] wrote %d edges, %d nodes -> %s", name, len(edges), len(nodes), out_dir)
    return audit


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "ibm_aml"
    print(json.dumps(build_dataset(target), indent=2))
