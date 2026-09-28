"""Exact brute-force nearest-neighbor index, used as the ground-truth oracle.

It deliberately shares the same validation, storage convention (unit vectors
for cosine) and deterministic result ordering as :class:`HNSWIndex`, so the
only thing being measured in recall tests is graph search quality.
"""

from .distances import distance_bucket, prepare_vector
from .hnsw import _label_sort_key


class BruteForceIndex:
    """Exact scan over every stored vector.  Local and dependency-free."""

    def __init__(self, dim, metric="cosine"):
        if metric not in ("l2", "cosine"):
            raise ValueError("metric must be 'l2' or 'cosine'")
        self.dim = dim
        self.metric = metric
        self._vectors = {}  # label -> prepared vector
        self.distance_count = 0

    def __len__(self):
        return len(self._vectors)

    def reset_distance_count(self):
        self.distance_count = 0

    def insert(self, label, vector):
        self._vectors[label] = prepare_vector(vector, self.dim, self.metric)

    def delete(self, label):
        return self._vectors.pop(label, None) is not None

    def _dist(self, a, b):
        self.distance_count += 1
        if self.metric == "l2":
            s = 0.0
            for x, y in zip(a, b):
                d = x - y
                s += d * d
            return s
        dot = 0.0
        for x, y in zip(a, b):
            dot += x * y
        d = 1.0 - dot
        return d if d > 0.0 else 0.0

    def search(self, vector, k=10):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        v = prepare_vector(vector, self.dim, self.metric)
        scored = [(self._dist(v, stored), label)
                  for label, stored in self._vectors.items()]
        scored.sort(key=lambda dl: (distance_bucket(dl[0]),
                                   _label_sort_key(dl[1])))
        return [(label, d) for d, label in scored[:k]]
