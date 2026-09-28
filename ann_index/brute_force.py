"""Brute-force exact k-NN search, used as the correctness/recall baseline."""

from .distances import get_distance_fn


def brute_force_search(items, query, k, metric="cosine"):
    """Exact top-k over an iterable of (id, vector) pairs.

    Returns a list of (id, distance) sorted by (distance, id) — the same
    tie-break rule the ANN index uses, so results are directly comparable.
    """
    if k <= 0:
        raise ValueError("k must be positive, got %r" % (k,))
    dist_fn = get_distance_fn(metric)
    scored = sorted((dist_fn(query, vec), vid) for vid, vec in items)
    return [(vid, d) for d, vid in scored[:k]]
