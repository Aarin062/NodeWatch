"""topology — the 5 detectors + structural features (the SINGLE copy).

This is the one and only home for the topology/structural feature logic. Do not
duplicate it elsewhere.

Responsibilities:
    * Implement the 5 fraud detectors and per-transaction structural features.

Constraints:
    * No look-ahead: a transaction's features may depend only on transactions
      with timestamp <= its own (enforce the temporal window in every query).
    * Must support BOTH:
        - a Cypher / Neo4j graph path, and
        - a Pandas fallback for datasets too large to fit in Neo4j.
      Both paths must yield identical feature semantics.

No modeling logic lives here.
"""

# TODO: implement the 5 detectors + structural features (Cypher + Pandas paths).
