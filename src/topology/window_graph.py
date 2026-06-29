"""WindowGraph — an incremental directed multigraph over a trailing time window.

This is the data structure the streaming topology engine walks. It holds only the
edges currently inside the trailing window ``[now - window, now]`` and supports the
two operations the engine needs in time order:

    * cheap O(1) degree / fan queries against the *current* (past-only) state, and
    * a bounded shortest-cycle search (does adding ``u -> v`` close a loop ``v ⇝ u``).

Leakage-safety is a property of *how the engine uses this object*, not of the
object itself: the engine reads a transaction's features **before** calling
``add`` for that transaction, so a transaction can never see itself or anything
later. The window only ever evicts edges that are too *old*; it never holds a
future edge. See ``engine.extract_features``.

All node ids are integers (the engine factorizes account strings up front for
speed). Amounts are deliberately not tracked here yet — the first feature set is
purely structural; amount-aware cycle features arrive with the later detectors.
"""
from __future__ import annotations

from collections import defaultdict, deque


class WindowGraph:
    """A directed multigraph holding only edges within a trailing time window.

    Parameters
    ----------
    window:
        Trailing window length in the dataset's native time units. Edges with
        ``timestamp < now - window`` are evicted. ``None`` means *unbounded*
        (keep every edge seen so far in the current partition).
    """

    __slots__ = ("window", "_edges", "out", "in_", "indeg", "outdeg")

    def __init__(self, window: int | float | None):
        self.window = window
        self._edges: deque[tuple[int, int, int]] = deque()      # (ts, u, v), time order
        self.out: dict[int, dict[int, int]] = defaultdict(dict)  # out[u][v] = multiplicity
        self.in_: dict[int, dict[int, int]] = defaultdict(dict)  # in_[v][u] = multiplicity
        self.indeg: dict[int, int] = defaultdict(int)            # total in-edges  (multiplicity)
        self.outdeg: dict[int, int] = defaultdict(int)           # total out-edges (multiplicity)

    # ---- mutation -----------------------------------------------------------
    def evict(self, now: int | float) -> None:
        """Drop every edge older than the trailing window relative to ``now``."""
        if self.window is None:
            return
        cutoff = now - self.window
        edges = self._edges
        while edges and edges[0][0] < cutoff:
            ts, u, v = edges.popleft()
            self._dec(self.out, u, v)
            self._dec(self.in_, v, u)
            self.outdeg[u] -= 1
            self.indeg[v] -= 1

    def add(self, ts: int | float, u: int, v: int) -> None:
        """Insert edge ``u -> v`` stamped ``ts`` (call AFTER reading its features)."""
        self._edges.append((ts, u, v))
        ou = self.out[u]
        ou[v] = ou.get(v, 0) + 1
        iv = self.in_[v]
        iv[u] = iv.get(u, 0) + 1
        self.outdeg[u] += 1
        self.indeg[v] += 1

    @staticmethod
    def _dec(side: dict[int, dict[int, int]], a: int, b: int) -> None:
        inner = side[a]
        c = inner[b] - 1
        if c <= 0:
            del inner[b]
            if not inner:
                del side[a]
        else:
            inner[b] = c

    # ---- O(1) structural queries (current, past-only state) -----------------
    def fan_in(self, v: int) -> int:
        """Number of *distinct* accounts that have sent to ``v`` in the window."""
        d = self.in_.get(v)
        return len(d) if d else 0

    def fan_out(self, u: int) -> int:
        """Number of *distinct* accounts ``u`` has sent to in the window."""
        d = self.out.get(u)
        return len(d) if d else 0

    def in_degree(self, v: int) -> int:
        """Total in-edges to ``v`` in the window (counting parallel edges)."""
        return self.indeg.get(v, 0)

    def out_degree(self, u: int) -> int:
        """Total out-edges from ``u`` in the window (counting parallel edges)."""
        return self.outdeg.get(u, 0)

    # ---- bounded cycle search ----------------------------------------------
    def shortest_cycle_len(self, v: int, u: int, max_len: int, budget: int) -> int:
        """Length of the shortest cycle that adding ``u -> v`` would close, or 0.

        Searches for a directed path ``v ⇝ u`` already present in the window using
        breadth-first search, so the first time ``u`` is reached gives the shortest
        such path. The closed cycle is ``u -> v ⇝ u``; its length in edges is
        ``(edges on v ⇝ u) + 1``.

        Bounds keep the per-transaction cost flat on dense/hub-heavy graphs:
            * ``max_len``  — only cycles up to this many edges are detectable;
            * ``budget``   — abandon the search after visiting this many nodes
                             (returns 0; a conservative miss, never a false cycle).
        """
        if v == u:
            return 1  # self-loop: u -> v(==u) is a length-1 cycle
        out = self.out
        if v not in out:
            return 0  # v has no outgoing edges -> cannot reach u
        dist = {v: 0}
        frontier = [v]
        visited = 1
        depth = 0
        max_path = max_len - 1  # edges allowed on the v ⇝ u path
        while frontier and depth < max_path:
            depth += 1
            nxt: list[int] = []
            for x in frontier:
                neigh = out.get(x)
                if not neigh:
                    continue
                for w in neigh:
                    if w == u:
                        return depth + 1  # closed cycle: depth edges (v⇝u) + 1 (u->v)
                    if w not in dist:
                        dist[w] = depth
                        visited += 1
                        if visited > budget:
                            return 0
                        nxt.append(w)
            frontier = nxt
        return 0
