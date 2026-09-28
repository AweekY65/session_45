"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index.

Pure-Python, in-memory + local-file snapshots only. Deterministic for a fixed
seed: level assignment and all graph operations depend only on the index's own
seeded RNG and the insertion sequence.

Result ordering contract: every search returns (id, distance) pairs sorted by
(distance, id) — ascending distance, ties broken by ascending id — so results
are stable and directly comparable with the brute-force baseline.
"""

import heapq
import json
import math
import random

from .distances import get_distance_fn

SNAPSHOT_VERSION = 1


class _Node:
    __slots__ = ("id", "vector", "level", "neighbors", "deleted")

    def __init__(self, nid, vector, level):
        self.id = nid
        self.vector = vector
        self.level = level
        # neighbors[l] = list of neighbor ids at layer l (0 <= l <= level)
        self.neighbors = [[] for _ in range(level + 1)]
        self.deleted = False


def _validate_vector(vector, dim):
    """Coerce and validate a query/document vector. Returns a list of floats."""
    if isinstance(vector, (str, bytes)) or not hasattr(vector, "__iter__"):
        raise TypeError("vector must be a sequence of real numbers")
    vec = list(vector)
    if len(vec) != dim:
        raise ValueError(
            "dimension mismatch: expected %d components, got %d" % (dim, len(vec))
        )
    out = []
    for x in vec:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("vector components must be real numbers, got %r" % (x,))
        x = float(x)
        if math.isnan(x):
            raise ValueError("vector contains NaN")
        if math.isinf(x):
            raise ValueError("vector contains infinity")
        out.append(x)
    return out


class HNSWIndex:
    """Fixed-dimension float vector ANN index over a hierarchical graph.

    Parameters
    ----------
    dim : int
        Vector dimension; every inserted/queried vector must match it.
    metric : str
        'cosine' (cosine distance) or 'l2' (Euclidean distance).
    M : int
        Max connections per node per layer (layer 0 uses 2*M). Higher M
        gives better recall at higher build cost and memory.
    ef_construction : int
        Beam width while building the graph. Higher = better graph quality.
    ef_search : int
        Default beam width at query time. Higher = better recall, slower.
    seed : int
        Seed for the internal RNG (level assignment). Same seed + same
        insertion order => identical graph and identical query results.
    """

    def __init__(self, dim, metric="cosine", M=16, ef_construction=100,
                 ef_search=50, seed=42):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer, got %r" % (dim,))
        if isinstance(M, bool) or not isinstance(M, int) or M < 2:
            raise ValueError("M must be an integer >= 2, got %r" % (M,))
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be >= 1")
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._dist_fn = get_distance_fn(metric)  # raises on bad metric
        self._ml = 1.0 / math.log(M)  # level multiplier
        self._reset_state()

    # ------------------------------------------------------------------ #
    # internal state                                                      #
    # ------------------------------------------------------------------ #

    def _reset_state(self):
        self._rng = random.Random(self.seed)
        self._nodes = {}
        self._enter_point = None
        self._max_layer = -1
        self._next_id = 0
        self._deleted_count = 0
        # Number of distance evaluations since the last reset of this counter;
        # used by tests to prove ANN queries do not degenerate to full scans.
        self.dist_computations = 0

    def _distance(self, a, b):
        self.dist_computations += 1
        return self._dist_fn(a, b)

    def _sample_level(self):
        # Exponential decay: P(level >= l) drops by a factor of exp(-1/ml) per layer.
        u = self._rng.random()
        return int(-math.log(u + 1e-300) * self._ml)

    # ------------------------------------------------------------------ #
    # graph search primitives                                             #
    # ------------------------------------------------------------------ #

    def _search_layer(self, entry_ids, query, ef, layer):
        """Beam search within one layer.

        Returns a list of (distance, id) for the ef closest visited nodes,
        sorted by (distance, id). Deterministic: all heap keys and the final
        sort use (distance, id) tuples.
        """
        visited = set(entry_ids)
        candidates = []  # min-heap of (dist, id): next node to expand
        results = []     # max-heap of (-dist, -id): current ef-best
        for ep in entry_ids:
            d = self._distance(query, self._nodes[ep].vector)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, -ep))
        while candidates:
            d, current = heapq.heappop(candidates)
            worst = -results[0][0]
            if d > worst:
                break
            for nb in self._nodes[current].neighbors[layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = self._distance(query, self._nodes[nb].vector)
                if len(results) < ef or d_nb < -results[0][0]:
                    heapq.heappush(candidates, (d_nb, nb))
                    heapq.heappush(results, (-d_nb, -nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-neg_d, -neg_id) for neg_d, neg_id in results)

    def _greedy_entry(self, entry_id, query, layer):
        """Descend one layer greedily (ef=1); returns the closest node id."""
        return self._search_layer([entry_id], query, 1, layer)[0][1]

    def _connect(self, uid, vid, layer):
        """Add edge uid -> vid at the given layer, pruning to the layer's max."""
        node = self._nodes[uid]
        nbs = node.neighbors[layer]
        if vid in nbs:
            return
        nbs.append(vid)
        m_max = self.M_max0 if layer == 0 else self.M
        if len(nbs) > m_max:
            scored = sorted(
                (self._distance(node.vector, self._nodes[w].vector), w) for w in nbs
            )
            node.neighbors[layer] = [w for _, w in scored[:m_max]]

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #

    def insert(self, vector):
        """Insert a vector; returns its integer id (monotonically assigned)."""
        return self._insert_with_id(self._next_id, vector)

    def _insert_with_id(self, nid, vector):
        vec = _validate_vector(vector, self.dim)
        if nid in self._nodes:
            raise ValueError("id %d already exists in the index" % nid)
        level = self._sample_level()
        node = _Node(nid, vec, level)
        self._nodes[nid] = node
        self._next_id = max(self._next_id, nid + 1)

        if self._enter_point is None:
            self._enter_point = nid
            self._max_layer = level
            return nid

        ep = self._enter_point
        # Zoom in from the top layer down to the new node's own top layer.
        for l in range(self._max_layer, level, -1):
            ep = self._greedy_entry(ep, vec, l)
        # Connect at every layer the new node participates in.
        for l in range(min(level, self._max_layer), -1, -1):
            cands = self._search_layer([ep], vec, self.ef_construction, l)
            selected = [cid for _, cid in cands[: self.M]]
            node.neighbors[l] = list(selected)
            for cid in selected:
                self._connect(cid, nid, l)
            ep = cands[0][1]
        if level > self._max_layer:
            self._enter_point = nid
            self._max_layer = level
        return nid

    def delete(self, nid):
        """Tombstone a vector. Deleted ids never appear in search results.

        Edges are left in place (the graph stays navigable); call compact()
        to physically remove tombstoned nodes and rebuild a clean graph.
        """
        node = self._nodes.get(nid)
        if node is None:
            raise KeyError("no vector with id %r" % (nid,))
        if node.deleted:
            raise KeyError("vector %r is already deleted" % (nid,))
        node.deleted = True
        self._deleted_count += 1

    def compact(self):
        """Rebuild the index from live vectors only, dropping tombstones.

        Ids are preserved. The rebuild is deterministic (same seed, ids
        re-inserted in ascending order), so repeated compacts of the same
        live set produce the same graph.
        """
        live = [
            (nid, node.vector)
            for nid, node in sorted(self._nodes.items())
            if not node.deleted
        ]
        self._reset_state()
        for nid, vec in live:
            self._insert_with_id(nid, vec)

    def search(self, query, k=10, ef_search=None):
        """Approximate top-k. Returns [(id, distance)] sorted by (distance, id).

        ef_search overrides the index default for this call; it is raised to
        at least k. Deleted vectors are never returned.
        """
        vec = _validate_vector(query, self.dim)
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer, got %r" % (k,))
        if self._enter_point is None:
            return []
        ef = ef_search if ef_search is not None else self.ef_search
        if ef < 1:
            raise ValueError("ef_search must be >= 1, got %r" % (ef,))
        ef = max(ef, k)

        ep = self._enter_point
        for l in range(self._max_layer, 0, -1):
            ep = self._greedy_entry(ep, vec, l)
        cands = self._search_layer([ep], vec, ef, 0)
        out = []
        for d, nid in cands:
            if not self._nodes[nid].deleted:
                out.append((nid, d))
                if len(out) == k:
                    break
        return out

    def brute_force_search(self, query, k=10):
        """Exact top-k over live vectors — the recall/correctness baseline."""
        vec = _validate_vector(query, self.dim)
        if k <= 0:
            raise ValueError("k must be positive, got %r" % (k,))
        scored = sorted(
            (self._dist_fn(vec, node.vector), nid)
            for nid, node in self._nodes.items()
            if not node.deleted
        )
        return [(nid, d) for d, nid in scored[:k]]

    def items(self):
        """Iterate over (id, vector) for all live vectors."""
        for nid, node in self._nodes.items():
            if not node.deleted:
                yield nid, node.vector

    def __len__(self):
        return len(self._nodes) - self._deleted_count

    def __contains__(self, nid):
        node = self._nodes.get(nid)
        return node is not None and not node.deleted

    @property
    def deleted_count(self):
        return self._deleted_count

    # ------------------------------------------------------------------ #
    # persistence (local snapshot files only)                             #
    # ------------------------------------------------------------------ #

    def save(self, path):
        """Serialize the full index (vectors, graph, metadata) to a local file.

        JSON floats round-trip exactly, so a loaded index returns query
        results identical to the saved one.
        """
        data = {
            "version": SNAPSHOT_VERSION,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self.seed,
            "max_layer": self._max_layer,
            "enter_point": self._enter_point,
            "next_id": self._next_id,
            "deleted_count": self._deleted_count,
            "nodes": [
                {
                    "id": node.id,
                    "vector": node.vector,
                    "level": node.level,
                    "neighbors": node.neighbors,
                    "deleted": node.deleted,
                }
                for node in sorted(self._nodes.values(), key=lambda n: n.id)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load an index previously written by save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != SNAPSHOT_VERSION:
            raise ValueError(
                "unsupported snapshot version %r" % (data.get("version"),)
            )
        idx = cls(
            dim=data["dim"],
            metric=data["metric"],
            M=data["M"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
            seed=data["seed"],
        )
        idx._max_layer = data["max_layer"]
        idx._enter_point = data["enter_point"]
        idx._next_id = data["next_id"]
        idx._deleted_count = data["deleted_count"]
        for nd in data["nodes"]:
            node = _Node(nd["id"], [float(x) for x in nd["vector"]], nd["level"])
            node.neighbors = [list(layer) for layer in nd["neighbors"]]
            node.deleted = bool(nd["deleted"])
            idx._nodes[node.id] = node
        return idx
