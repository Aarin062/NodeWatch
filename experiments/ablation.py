"""The ablation — the experiment that decides everything (DESIGN §9).

Train the SAME model with the SAME time-aware split and the SAME single imbalance
mechanism, twice:

    (A) baseline           — ordinary tabular features only
    (B) baseline+topology  — ordinary + the graph-shape features

The only difference is the feature set, so any gap in PR-AUC / precision@k is
attributable to topology. We report whichever way it lands (a null result is a
real result). This is a SMOKE TEST on the current 6-feature topology set — a
directional read, to be re-run as detectors are added.

Run:  python -m experiments.ablation ibm_aml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from common.config import get_dataset
from modeling.harness import train_and_evaluate

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_features(name: str) -> tuple[pd.DataFrame, dict]:
    proc = Path(get_dataset(name)["processed_dir"])
    with open(proc / "features_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    path = proc / manifest["output"]
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df, manifest


def run_ablation(name: str, seed: int = 42, partition_aware: bool | None = None) -> dict:
    df, manifest = _load_features(name)
    ordinary = manifest["ordinary_cols"]
    topology = manifest["topology_cols"]
    amount_cols = [c for c in ordinary if "amount" in c.lower()]
    no_amount = [c for c in ordinary if c not in amount_cols]
    # SAP's synthetic time axis isn't a real cross-run chronology -> split within run.
    if partition_aware is None:
        partition_aware = "run_id" in df.columns
    kw = {"seed": seed, "partition": df["run_id"] if partition_aware and "run_id" in df else None}

    # Five models. Only the two deployable ones (baseline, +topology) are persisted.
    log.info("[%s] fitting 5 models (baseline, +topo, no-amount, no-amount+topo, topo-only)", name)
    base = train_and_evaluate(df, ordinary, persist_as=f"{name}/baseline", **kw)
    topo = train_and_evaluate(df, ordinary + topology, persist_as=f"{name}/topology", **kw)
    base_na = train_and_evaluate(df, no_amount, **kw)
    topo_na = train_and_evaluate(df, no_amount + topology, **kw)
    topo_only = train_and_evaluate(df, topology, **kw)

    pr = lambda r: r["test"]["pr_auc"]
    marginal = round(pr(topo) - pr(base), 5)               # the headline ablation
    headroom = round(pr(topo_na) - pr(base_na), 5)         # topology's value when amount removed

    summary = {
        "dataset": name,
        "partition_aware_split": bool(kw["partition"] is not None),
        "split_sizes": base["split_sizes"],
        "split_frauds": base["split_frauds"],
        "pr_auc": {
            "baseline": pr(base), "baseline_plus_topology": pr(topo),
            "no_amount": pr(base_na), "no_amount_plus_topology": pr(topo_na),
            "topology_only": pr(topo_only),
        },
        "marginal_topology_lift": marginal,
        "amount_blind_topology_lift": headroom,
        "baseline_test": base["test"],
        "baseline_plus_topology_test": topo["test"],
        "topology_importance": {k: round(v, 4) for k, v in topo["feature_importance"].items()
                                if k in topology},
        "baseline_top_features": list(base["feature_importance"])[:6],
        "verdict": _verdict(marginal, pr(base), headroom, pr(topo_only),
                            base["split_frauds"]["test"]),
    }
    out_dir = REPO_ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "baseline_full": base, "topology_full": topo,
                   "no_amount_full": base_na, "no_amount_topo_full": topo_na,
                   "topo_only_full": topo_only}, f, indent=2)
    return summary


def _verdict(marginal: float, base_prauc: float, headroom: float,
             topo_only: float, test_frauds: int) -> str:
    if test_frauds < 20:
        return f"INCONCLUSIVE — only {test_frauds} test frauds; metrics unstable."
    parts = []
    if base_prauc > 0.9 and marginal <= 0.01:
        parts.append(f"baseline SATURATED (PR-AUC={base_prauc:.3f}) -> marginal test uninformative")
    if marginal > 0.01:
        rel = marginal / base_prauc * 100 if base_prauc else 0
        parts.append(f"topology adds +{marginal:.4f} PR-AUC ({rel:+.1f}% rel)")
    elif marginal <= 0.001:
        parts.append(f"topology adds ~nothing marginally (+{marginal:.4f})")
    parts.append(f"topology-only PR-AUC={topo_only:.3f}")
    parts.append(f"amount-blind lift=+{headroom:.4f}")
    return "; ".join(parts) + "."


def _print(summary: dict) -> None:
    pa = summary["pr_auc"]
    print("\n" + "=" * 66)
    print(f"ABLATION — {summary['dataset']}   "
          f"(test frauds: {summary['split_frauds']['test']}, "
          f"partition-split: {summary['partition_aware_split']})")
    print("=" * 66)
    print("PR-AUC by feature set:")
    for k in ("baseline", "baseline_plus_topology", "no_amount",
              "no_amount_plus_topology", "topology_only"):
        print(f"    {k:<26}{pa[k]:>10.4f}")
    print("-" * 66)
    print(f"  marginal topology lift (full baseline) : {summary['marginal_topology_lift']:+.4f}")
    print(f"  amount-blind topology lift (headroom)  : {summary['amount_blind_topology_lift']:+.4f}")
    print("-" * 66)
    print("topology feature importance (in baseline+topology):")
    for k, v in summary["topology_importance"].items():
        print(f"    {k:<18}{v:>8}")
    print("-" * 66)
    print("VERDICT:", summary["verdict"])
    print("=" * 66 + "\n")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "ibm_aml"
    s = run_ablation(target)
    _print(s)
