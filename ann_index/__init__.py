"""Local approximate nearest-neighbor index (pure Python, no external services).

Everything lives in memory or in local snapshot files. No vector database,
no embedding API, no network access of any kind.
"""

from .distances import cosine_distance, l2_distance, get_distance_fn
from .hnsw import HNSWIndex
from .brute_force import brute_force_search

__all__ = [
    "HNSWIndex",
    "brute_force_search",
    "cosine_distance",
    "l2_distance",
    "get_distance_fn",
]
