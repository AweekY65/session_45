"""Distance functions, input validation and vector preparation.

All distance metrics are converted to a *"smaller is better"* convention so
that the rest of the index can treat them uniformly:

* ``l2``     : squared Euclidean distance  ``sum((a-b)^2)``.
* ``cosine`` : ``1 - cosine_similarity``   (range [0, 2] for unit vectors).

Tie handling note: floating point rounding can make two mathematically equal
distances differ in the last bit.  Search results are therefore sorted by
``(distance rounded to a tolerance bucket, label)`` which gives a stable order
independent of dict / heap iteration (see :data:`TIE_EPS`).
"""

import math

TIE_EPS = 1e-12


def l2_distance(a, b):
    """Squared Euclidean distance between two float sequences."""
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def cosine_distance(a, b):
    """``1 - cosine similarity``; inputs must be non-zero and finite."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("cosine distance is undefined for zero vectors")
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    # Clamp to [-1, 1] to absorb rounding error.
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return 1.0 - sim


def _is_real_number(x):
    # bool is a subclass of int; reject it explicitly to avoid silently
    # treating True as 1.0.
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float))


def validate_vector(vec, dim):
    """Validate an incoming vector and return it as ``list[float]``.

    Raises :class:`TypeError` for non-vector / non-numeric input and
    :class:`ValueError` for wrong length, NaN or infinite components.
    """
    if isinstance(vec, (str, bytes)):
        raise TypeError("vector must be a sequence of real numbers, not str/bytes")
    # Generators / iterators are rejected: the dimension cannot be checked
    # once and the values would be consumed.
    if not hasattr(vec, "__len__") or not hasattr(vec, "__getitem__"):
        raise TypeError("vector must be a sized, indexable sequence of real numbers")
    if len(vec) != dim:
        raise ValueError(
            f"vector dimension mismatch: expected {dim}, got {len(vec)}"
        )
    out = []
    for x in vec:
        if not _is_real_number(x):
            raise TypeError("every vector component must be a real number (int/float)")
        fx = float(x)
        if math.isnan(fx):
            raise ValueError("vector must not contain NaN")
        if math.isinf(fx):
            raise ValueError("vector must not contain infinite values")
        out.append(fx)
    return out


def prepare_vector(vec, dim, metric):
    """Validate a vector and normalize it when using cosine distance.

    For cosine search the index stores unit vectors, which makes the dot
    product exactly equivalent to cosine similarity (and l2-on-unit-vectors
    monotonic with it).
    """
    v = validate_vector(vec, dim)
    if metric == "cosine":
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            raise ValueError("cannot index a zero vector with cosine distance")
        v = [x / norm for x in v]
    return v


def distance_bucket(d):
    """Quantize a distance for stable tie-break ordering."""
    if d == 0.0:
        return 0.0
    return round(d / TIE_EPS)
