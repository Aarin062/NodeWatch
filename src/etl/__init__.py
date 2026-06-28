"""etl — raw CSV to canonical node / edge files.

Responsibilities:
    * Read a dataset's raw CSV (path + rename_map from config/datasets.yaml).
    * Apply the canonical column schema and basic cleaning.
    * Emit node and edge files into data/processed/ for the graph loader.

No analysis logic lives here.
"""

# TODO: implement raw -> node/edge transformation.
