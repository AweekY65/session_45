"""Distance functions. All distances are "smaller is closer"."""

import math


def l2_distance(a, b):
    """Euclidean (L2) distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def cosine_distance(a, b):
    """Cosine distance: 1 - cosine_similarity(a, b), clamped to [0, 2].

    A zero-norm vector has undefined direction; by convention its distance
    to anything is 1.0.
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    d = 1.0 - dot / math.sqrt(na * nb)
    # Guard against tiny floating-point overshoot outside [0, 2].
    if d < 0.0:
        return 0.0
    if d > 2.0:
        return 2.0
    return d


_METRICS = {
    "cosine": cosine_distance,
    "l2": l2_distance,
}


def get_distance_fn(metric):
    """Return the distance function for a metric name ('cosine' or 'l2')."""
    try:
        return _METRICS[metric]
    except KeyError:
        raise ValueError(
            "unknown metric %r; expected one of %s" % (metric, sorted(_METRICS))
        ) from None
