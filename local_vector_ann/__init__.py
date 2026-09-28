"""local_vector_ann: a fully local, dependency-free HNSW vector index.

No vector database, no embedding API and no network calls are involved:
vectors, graph structure, metadata and snapshots live only in process memory
or in local files produced by :meth:`HNSWIndex.save`.
"""

from .bruteforce import BruteForceIndex
from .distances import cosine_distance, l2_distance
from .hnsw import HNSWIndex

__all__ = [
    "HNSWIndex",
    "BruteForceIndex",
    "cosine_distance",
    "l2_distance",
]
