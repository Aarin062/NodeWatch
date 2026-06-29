"""Automated leakage guard for the topology engine (CLAUDE.md rule 4).

The engine claims a transaction's features depend only on the past. This test
*proves* it empirically on real data, two independent ways:

1. **Truncation invariance.** Features for the first ``k`` transactions must be
   identical whether or not the later transactions even exist. We compute
   features on the full table and on the table truncated at ``k`` and assert the
   first ``k`` rows match exactly. If anything later influenced an earlier row,
   they would differ.

2. **Future-corruption invariance.** We keep the first ``k`` rows intact but
   replace every later row's source / dest / amount with garbage (timestamps and
   row positions unchanged, so streaming order is preserved). Past features must
   be byte-identical. This catches leaks that truncation alone might miss (e.g. a
   statistic that secretly folds in future *values*).

A leak makes either check fail and must fail the build. Run directly:

    python -m topology.leakage_test            # default: banksim (fast, bipartite)
    python -m topology.leakage_test ibm_aml    # sampled for speed
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from common.config import get_dataset
from topology.engine import FEATURE_COLS, extract_features, resolve_params
from pathlib import Path

log = logging.getLogger(__name__)


def _features(edges: pd.DataFrame, params: dict) -> pd.DataFrame:
    return extract_features(
        edges,
        window=params["window"],
        max_cycle_len=params["max_cycle_len"],
        search_budget=params["search_budget"],
        partition_col=params["partition_col"],
    )


def check_no_leakage(edges: pd.DataFrame, params: dict, cut_frac: float = 0.5) -> dict:
    """Run both invariance checks on a time-sorted edge table. Returns a report
    dict with ``passed`` (bool). Raises nothing — the caller decides what to do."""
    edges = edges.reset_index(drop=True)
    n = len(edges)
    k = max(1, int(n * cut_frac))

    full = _features(edges, params)[list(FEATURE_COLS)].to_numpy()

    # (1) truncation invariance ------------------------------------------------
    trunc = _features(edges.iloc[:k].copy(), params)[list(FEATURE_COLS)].to_numpy()
    trunc_ok = bool(np.array_equal(full[:k], trunc))

    # (2) future-corruption invariance ----------------------------------------
    corrupted = edges.copy()
    fut = slice(k, n)
    rng = np.random.default_rng(0)
    # Shuffle the future's endpoints and blow up its amounts; keep timestamps/order.
    corrupted.loc[corrupted.index[fut], "source_account"] = (
        rng.permutation(corrupted.loc[corrupted.index[fut], "source_account"].to_numpy())
    )
    corrupted.loc[corrupted.index[fut], "dest_account"] = (
        rng.permutation(corrupted.loc[corrupted.index[fut], "dest_account"].to_numpy())
    )
    corrupted.loc[corrupted.index[fut], "amount"] = -123456789.0
    corr = _features(corrupted, params)[list(FEATURE_COLS)].to_numpy()
    corrupt_ok = bool(np.array_equal(full[:k], corr[:k]))

    report = {
        "n_rows": n,
        "cut_at": k,
        "truncation_invariance": trunc_ok,
        "future_corruption_invariance": corrupt_ok,
        "passed": trunc_ok and corrupt_ok,
    }
    if not trunc_ok:
        # Surface the first offending row to make debugging concrete.
        diff = np.where(~(full[:k] == trunc).all(axis=1))[0]
        report["first_truncation_mismatch_row"] = int(diff[0]) if diff.size else None
    return report


def run_leakage_test(name: str, sample: int | None = None, cut_frac: float = 0.5) -> dict:
    """Load a built dataset and run the leakage checks. ``sample`` caps rows
    (head, preserving time order) for speed on the large datasets."""
    ds = get_dataset(name)
    params = resolve_params(ds)
    edges_path = Path(ds["processed_dir"]) / "edges_transactions.csv"
    edges = pd.read_csv(edges_path, low_memory=False)
    if sample and len(edges) > sample:
        edges = edges.iloc[:sample].copy()
    report = check_no_leakage(edges, params, cut_frac=cut_frac)
    report["dataset"] = name
    report["sampled_rows"] = len(edges)
    return report


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    name = sys.argv[1] if len(sys.argv) > 1 else "banksim"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else (None if name == "banksim" else 150_000)
    rep = run_leakage_test(name, sample=cap)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["passed"] else 1)
