"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index.

Everything here is local and in-process:

* vectors and the layered graph live in Python memory;
* :meth:`HNSWIndex.save` writes a single self-contained binary snapshot to a
  local file, which :meth:`HNSWIndex.load` reads back.

There are no external services, no embedding calls and no third-party
dependencies.

Deletion model
--------------
Deleting a vector marks its node as a *tombstone*.  Tombstones immediately
disappear from results, but stay in the graph as routing nodes so that the
remaining graph need not be repaired on every delete.  :meth:`compact`
physically rebuilds the index and drops tombstones; the rebuilt graph is
functionally equivalent to a fresh index populated with the live vectors.
"""

import heapq
import math
import os
import struct

from .distances import distance_bucket, prepare_vector
from .rng import Rng

_MAGIC = b"LVANNH01"
_VERSION = 1
_METRIC_CODES = {"l2": 0, "cosine": 1}


class _Node:
    __slots__ = ("id", "label", "vector", "level", "deleted", "neighbors")

    def __init__(self, node_id, label, vector, level):
        self.id = node_id
        self.label = label
        self.vector = vector
        self.level = level
        self.deleted = False
        # neighbors[lc] is a list of node ids connected at layer lc.
        self.neighbors = [[] for _ in range(level + 1)]


class HNSWIndex:
    """In-memory HNSW index with local-file persistence.

    Parameters
    ----------
    dim:
        Fixed vector dimensionality.  Every inserted/searched vector must have
        exactly this many components.
    metric:
        ``"l2"`` (squared Euclidean distance) or ``"cosine"``
        (``1 - cosine_similarity``).
    M:
        Construction parameter: target number of neighbors per node at layers
        above the base layer.  At layer 0 nodes may keep up to ``2 * M``
        neighbors (standard HNSW).
    ef_construction:
        Search width used while building the graph.  Larger values produce a
        better-connected graph at higher build cost.
    ef_search:
        Default query-time search width.  Can be overridden per query.
    seed:
        Integer seed for the built-in deterministic RNG that assigns node
        levels.  Same data + same seed => same graph.
    """

    def __init__(
        self,
        dim,
        metric="cosine",
        M=16,
        ef_construction=100,
        ef_search=50,
        seed=42,
    ):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if metric not in _METRIC_CODES:
            raise ValueError(f"metric must be one of {sorted(_METRIC_CODES)}")
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        if not isinstance(ef_construction, int) or ef_construction < 1:
            raise ValueError("ef_construction must be a positive integer")
        if not isinstance(ef_search, int) or ef_search < 1:
            raise ValueError("ef_search must be a positive integer")

        self.dim = dim
        self.metric = metric
        self.M = M
        # Max degree: 2*M at the base layer, M above it.
        self.maxM0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # Level sampling normalization: 1 / ln(M).
        self.ml = 1.0 / math.log(M) if M > 1 else 1.0
        self.seed = seed
        self._rng = Rng(seed)

        self._nodes = {}          # id -> _Node (includes tombstones)
        self._label_to_id = {}    # live label -> id
        self._next_id = 0
        self._entry_point = -1
        # Number of distance evaluations since the last reset.  Tests use it
        # to prove the graph search does not degenerate into a full scan.
        self.distance_count = 0

    # ------------------------------------------------------------------ #
    # Distance plumbing
    # ------------------------------------------------------------------ #
    def _dist(self, a, b):
        self.distance_count += 1
        if self.metric == "l2":
            s = 0.0
            for x, y in zip(a, b):
                d = x - y
                s += d * d
            return s
        # cosine on unit vectors: distance = 1 - dot product.
        dot = 0.0
        for x, y in zip(a, b):
            dot += x * y
        d = 1.0 - dot
        # Guard against tiny negative values from rounding.
        return d if d > 0.0 else 0.0

    def reset_distance_count(self):
        self.distance_count = 0

    # ------------------------------------------------------------------ #
    # Basic API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self._label_to_id)

    def labels(self):
        """Return the set of currently live labels."""
        return set(self._label_to_id)

    def insert(self, label, vector):
        """Insert one vector under ``label``.

        Labels must be ints or strings and unique among live entries.
        Re-inserting a label that is currently present raises ``ValueError``;
        delete it first to replace a vector.
        """
        self._validate_label(label)
        if label in self._label_to_id:
            raise ValueError(f"label {label!r} is already indexed")
        v = prepare_vector(vector, self.dim, self.metric)

        node_id = self._next_id
        self._next_id += 1
        level = self._random_level()
        node = _Node(node_id, label, v, level)
        self._nodes[node_id] = node
        self._label_to_id[label] = node_id

        if self._entry_point == -1:
            # First node becomes the entry point at every layer it occupies.
            self._entry_point = node_id
            return node_id

        ep = self._entry_point
        ep_dist = self._dist(v, self._nodes[ep].vector)

        # Phase 1: greedily descend through layers above the new node's level.
        top = self._nodes[self._entry_point].level
        for lc in range(top, level, -1):
            ep, ep_dist = self._greedy_descend(lc, ep, ep_dist, v)

        # Phase 2: connect from min(top, level) down to layer 0.
        for lc in range(min(top, level), -1, -1):
            candidates = self._search_layer(
                lc, [(ep_dist, ep)], self.ef_construction, v
            )
            neighbors = self._select_neighbors_heuristic(candidates, self.M)
            node.neighbors[lc] = list(neighbors)
            for nb_id in neighbors:
                nb = self._nodes[nb_id]
                conns = nb.neighbors[lc]
                if node_id not in conns:
                    conns.append(node_id)
                max_degree = self.maxM0 if lc == 0 else self.M
                if len(conns) > max_degree:
                    # Re-prune the neighbor's connection list heuristically.
                    cand = [
                        (self._dist(nb.vector, self._nodes[c].vector), c)
                        for c in conns
                    ]
                    nb.neighbors[lc] = self._select_neighbors_heuristic(
                        cand, max_degree
                    )
            if lc > 0:
                # Keep a fresh entry point for the next layer down.
                ep, ep_dist = candidates[0][1], candidates[0][0]

        # The highest node in the graph is the entry point.
        if level > top:
            self._entry_point = node_id
        return node_id

    def delete(self, label):
        """Mark ``label`` as deleted.  Returns True if it existed."""
        node_id = self._label_to_id.pop(label, None)
        if node_id is None:
            return False
        self._nodes[node_id].deleted = True
        return True

    def compact(self):
        """Physically remove tombstones by rebuilding a fresh graph.

        The RNG is reset to the original seed and live nodes are re-inserted
        in ascending id order (i.e. original insertion order), so the rebuilt
        graph is bit-for-bit the same as one built by inserting only the live
        vectors in the first place.  Search results are unchanged.
        """
        live = [
            (n.label, n.vector)
            for n in sorted(self._nodes.values(), key=lambda n: n.id)
            if not n.deleted
        ]
        self._nodes = {}
        self._label_to_id = {}
        self._next_id = 0
        self._entry_point = -1
        self._rng = Rng(self.seed)
        for label, vector in live:
            # Vectors are already prepared (normalized for cosine); insert
            # through the internal path to skip re-normalization.
            self._insert_prepared(label, vector)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, vector, k=10, ef_search=None):
        """Return the ``k`` nearest live labels as ``(label, distance)``.

        Results are ordered by ascending distance with a deterministic
        tie-break: exact/near-exact distance ties (within ``TIE_EPS``) are
        broken by label (numeric order, then lexicographic for strings of
        different types).

        The returned distance is squared Euclidean distance for ``l2`` and
        ``1 - cosine_similarity`` for ``cosine``.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if ef_search is None:
            ef_search = self.ef_search
        if not isinstance(ef_search, int) or ef_search < 1:
            raise ValueError("ef_search must be a positive integer")

        v = prepare_vector(vector, self.dim, self.metric)
        if self._entry_point == -1 or k > len(self._label_to_id):
            k = min(k, len(self._label_to_id))
        if k == 0:
            return []
        ef = max(ef_search, k)

        ep = self._entry_point
        ep_dist = self._dist(v, self._nodes[ep].vector)
        top = self._nodes[ep].level
        for lc in range(top, 0, -1):
            ep, ep_dist = self._greedy_descend(lc, ep, ep_dist, v)

        candidates = self._search_layer(0, [(ep_dist, ep)], ef, v)
        results = [(d, nid) for (d, nid) in candidates
                   if not self._nodes[nid].deleted]
        results.sort(key=lambda dn: (distance_bucket(dn[0]),
                                     _label_sort_key(self._nodes[dn[1]].label)))
        results = results[:k]
        return [(self._nodes[nid].label, self._finalize_distance(d))
                for d, nid in results]

    def _finalize_distance(self, d):
        if self.metric == "cosine":
            # Cosine distance is in [0, 2]; clamp tiny round-off.
            if d < 0.0:
                return 0.0
        return d

    def _greedy_descend(self, lc, ep, ep_dist, query):
        """Greedy search on a single layer: walk to the closest neighbor."""
        changed = True
        while changed:
            changed = False
            node = self._nodes[ep]
            for nb_id in node.neighbors[lc]:
                d = self._dist(query, self._nodes[nb_id].vector)
                if d < ep_dist:
                    ep_dist = d
                    ep = nb_id
                    changed = True
        return ep, ep_dist

    def _search_layer(self, lc, entry_points, ef, query):
        """HNSW layer search.

        ``entry_points`` is a list of ``(dist, id)`` seeds.  Returns up to
        ``ef`` closest nodes (deleted nodes are included here -- callers
        filter them -- because tombstones still route the graph) as a list of
        ``(dist, id)``.
        """
        visited = set()
        # worklist: min-heap of candidates to expand, (dist, id).
        worklist = []
        # found: max-heap of current best, (-dist, id).
        found = []
        for d, ep in entry_points:
            if ep in visited:
                continue
            visited.add(ep)
            heapq.heappush(worklist, (d, ep))
            heapq.heappush(found, (-d, ep))

        while worklist:
            d_c, c = heapq.heappop(worklist)
            worst_found = -found[0][0]
            if d_c > worst_found and len(found) >= ef:
                break
            node = self._nodes[c]
            for nb_id in node.neighbors[lc]:
                if nb_id in visited:
                    continue
                visited.add(nb_id)
                worst_found = -found[0][0]
                if len(found) >= ef:
                    d_nb = self._dist(query, self._nodes[nb_id].vector)
                    if d_nb >= worst_found:
                        continue
                else:
                    d_nb = self._dist(query, self._nodes[nb_id].vector)
                heapq.heappush(worklist, (d_nb, nb_id))
                heapq.heappush(found, (-d_nb, nb_id))
                if len(found) > ef:
                    heapq.heappop(found)

        return [(-neg_d, nid) for neg_d, nid in found]

    def _select_neighbors_heuristic(self, candidates, m):
        """Heuristic neighbor selection (Algorithm 4 in the HNSW paper).

        ``candidates`` is a list of ``(distance, id)``.  Returns up to ``m``
        node ids, preferring nodes that are close to the query themselves over
        nodes whose closeness is "explained" by an even closer node.
        """
        if len(candidates) <= m:
            return [nid for _, nid in sorted(candidates)]

        # Min-heap by distance.
        cand = sorted(candidates, key=lambda dn: dn[0])
        chosen = []
        chosen_set = set()
        while cand and len(chosen) < m:
            d_e, e = cand.pop(0)
            good = True
            for f_id in chosen:
                # Reject e when it is closer to an already-chosen node than
                # to the query: that link is "explained" by the closer node.
                # This extra distance call is charged to the build.
                d_ef = self._dist(self._nodes[e].vector,
                                  self._nodes[f_id].vector)
                if d_ef < d_e:
                    good = False
                    break
            if good and e not in chosen_set:
                chosen.append(e)
                chosen_set.add(e)

        # Fallback (normally unreachable): fill with closest remaining.
        if len(chosen) < m:
            for d_e, e in sorted(candidates, key=lambda dn: dn[0]):
                if e not in chosen_set:
                    chosen.append(e)
                    chosen_set.add(e)
                    if len(chosen) >= m:
                        break
        return chosen

    def _random_level(self):
        r = self._rng.random()
        if r == 0.0:
            r = 1e-16
        return int(math.floor(-math.log(r) * self.ml))

    # ------------------------------------------------------------------ #
    # Internal insert used by compact (vectors already normalized)
    # ------------------------------------------------------------------ #
    def _insert_prepared(self, label, v):
        self._validate_label(label)
        node_id = self._next_id
        self._next_id += 1
        level = self._random_level()
        node = _Node(node_id, label, v, level)
        self._nodes[node_id] = node
        self._label_to_id[label] = node_id

        if self._entry_point == -1:
            self._entry_point = node_id
            return node_id

        ep = self._entry_point
        ep_dist = self._dist(v, self._nodes[ep].vector)
        top = self._nodes[self._entry_point].level
        for lc in range(top, level, -1):
            ep, ep_dist = self._greedy_descend(lc, ep, ep_dist, v)

        for lc in range(min(top, level), -1, -1):
            candidates = self._search_layer(
                lc, [(ep_dist, ep)], self.ef_construction, v
            )
            neighbors = self._select_neighbors_heuristic(candidates, self.M)
            node.neighbors[lc] = list(neighbors)
            for nb_id in neighbors:
                nb = self._nodes[nb_id]
                conns = nb.neighbors[lc]
                if node_id not in conns:
                    conns.append(node_id)
                max_degree = self.maxM0 if lc == 0 else self.M
                if len(conns) > max_degree:
                    cand = [
                        (self._dist(nb.vector, self._nodes[c].vector), c)
                        for c in conns
                    ]
                    nb.neighbors[lc] = self._select_neighbors_heuristic(
                        cand, max_degree
                    )
            if lc > 0:
                ep, ep_dist = candidates[0][1], candidates[0][0]

        if level > top:
            self._entry_point = node_id
        return node_id

    @staticmethod
    def _validate_label(label):
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise TypeError("label must be an int or str")
        if isinstance(label, int) and not (-(1 << 63) <= label < (1 << 63)):
            raise ValueError("integer labels must fit in signed 64 bits")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path):
        """Write a binary snapshot of vectors, graph and metadata to ``path``.

        The file is written atomically (temp file + ``os.replace``).
        """
        header = struct.pack(
            "<8sBBHIII",
            _MAGIC,
            _VERSION,
            _METRIC_CODES[self.metric],
            self.dim,
            self.M,
            self.ef_construction,
            self.ef_search,
        )
        with open(path + ".tmp", "wb") as f:
            f.write(header)
            f.write(struct.pack("<dqQqqQ",
                                self.ml,
                                self.seed & 0xFFFFFFFFFFFFFFFF,
                                self._rng.state,
                                self._next_id,
                                self._entry_point,
                                len(self._nodes)))
            vec_pack = struct.Struct(f"<{self.dim}d")
            for node in self._nodes.values():
                f.write(struct.pack("<iqh", node.id,
                                    1 if node.deleted else 0, node.level))
                if isinstance(node.label, int):
                    f.write(struct.pack("<Bq", 0, node.label))
                else:
                    raw = node.label.encode("utf-8")
                    f.write(struct.pack("<BI", 1, len(raw)))
                    f.write(raw)
                f.write(vec_pack.pack(*node.vector))
                for lc in range(node.level + 1):
                    conns = node.neighbors[lc]
                    f.write(struct.pack("<I", len(conns)))
                    if conns:
                        f.write(struct.pack(f"<{len(conns)}q", *conns))
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        """Load an index previously written by :meth:`save`."""
        with open(path, "rb") as f:
            data = f.read()
        off = 0
        magic, version, metric_code, dim, M, ef_c, ef_s = struct.unpack_from(
            "<8sBBHIII", data, off)
        off += struct.calcsize("<8sBBHIII")
        if magic != _MAGIC:
            raise ValueError("not a local_vector_ann snapshot (bad magic)")
        if version != _VERSION:
            raise ValueError(f"unsupported snapshot version {version}")

        metric = {v: k for k, v in _METRIC_CODES.items()}[metric_code]
        ml, seed, rng_state, next_id, ep, n_nodes = struct.unpack_from(
            "<dqQqqQ", data, off)
        off += struct.calcsize("<dqQqqQ")

        idx = cls(dim, metric=metric, M=M, ef_construction=ef_c,
                  ef_search=ef_s, seed=seed)
        idx.ml = ml
        idx._rng.state = rng_state
        idx._next_id = next_id
        idx._entry_point = ep

        vec_size = dim * 8
        for _ in range(n_nodes):
            node_id, deleted, level = struct.unpack_from("<iqh", data, off)
            off += 14
            (label_kind,) = struct.unpack_from("<B", data, off)
            off += 1
            if label_kind == 0:
                (label,) = struct.unpack_from("<q", data, off)
                off += 8
            else:
                (slen,) = struct.unpack_from("<I", data, off)
                off += 4
                label = data[off:off + slen].decode("utf-8")
                off += slen
            vector = list(struct.unpack_from(f"<{dim}d", data, off))
            off += vec_size
            node = _Node(node_id, label, vector, level)
            node.deleted = bool(deleted)
            for lc in range(level + 1):
                (cnt,) = struct.unpack_from("<I", data, off)
                off += 4
                if cnt:
                    node.neighbors[lc] = list(
                        struct.unpack_from(f"<{cnt}q", data, off))
                    off += cnt * 8
            idx._nodes[node_id] = node
            if not node.deleted:
                idx._label_to_id[label] = node_id
        return idx


def _label_sort_key(label):
    # Deterministic cross-type ordering: ints before strings.
    if isinstance(label, int):
        return (0, label, "")
    return (1, 0, label)
