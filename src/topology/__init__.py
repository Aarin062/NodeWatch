"""topology — the temporal topology engine (the SINGLE copy of the shape logic).

This is the one and only home for the topology/structural feature logic. Do not
duplicate it elsewhere.

Leakage-safety (CLAUDE.md rule 4) is structural: features are computed by
*streaming* transactions in time order and reading each transaction's features
from a past-only window graph **before** inserting it (see ``engine`` and
``window_graph``). The ``leakage_test`` module proves this empirically and must
pass.

Public API:
    * ``extract_features(edges, ...)`` — as-of per-transaction topology features.
    * ``build_topology(name)``         — run the engine for a configured dataset.
    * ``WindowGraph``                  — the incremental windowed multigraph.
    * ``run_leakage_test(name)``       — the automated no-look-ahead guard.

Current increment: cycle + fan-in detectors and structural degrees (the shapes
IBM AML labels as fraud). Density / rapid-chain / lapping / real centrality next.
"""
from __future__ import annotations

from topology.engine import (
    FEATURE_COLS,
    build_topology,
    extract_features,
    resolve_params,
)
from topology.leakage_test import check_no_leakage, run_leakage_test
from topology.window_graph import WindowGraph

__all__ = [
    "FEATURE_COLS",
    "WindowGraph",
    "build_topology",
    "check_no_leakage",
    "extract_features",
    "resolve_params",
    "run_leakage_test",
]
