"""Config loader for the Self-Auditing Ledger.

Reads ``config/datasets.yaml`` and resolves paths. ``raw_root`` is taken as-is
(it points outside the repo); ``processed_root`` is resolved relative to the
repo root so the pipeline works regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# config.py -> common -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "datasets.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the full datasets config as a dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """Return one dataset's config, with absolute ``raw`` paths and a resolved
    ``processed_dir``. Raises KeyError with the available names if not found."""
    cfg = load_config(path)
    datasets = cfg.get("datasets", {})
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not in config; available: {sorted(datasets)}")

    ds = dict(datasets[name])
    raw_root = Path(cfg["raw_root"])
    ds["raw"] = {key: str(raw_root / rel) for key, rel in ds.get("raw", {}).items()}

    processed_root = Path(cfg.get("processed_root", "data/processed"))
    if not processed_root.is_absolute():
        processed_root = REPO_ROOT / processed_root
    ds["processed_dir"] = str(processed_root / name)
    return ds
