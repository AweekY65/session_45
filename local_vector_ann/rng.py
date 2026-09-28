"""Deterministic pseudo-random generator used for HNSW level assignment.

The index deliberately avoids the global ``random`` module so that:

* insertion order is reproducible for a given seed, regardless of other code
  consuming the global RNG;
* the generator state is part of the saved snapshot, so an index can be
  saved, reloaded and keep producing exactly the same levels it would have
  produced without the save/reload round trip.

xorshift64* (Marsaglia) plus splitmix64 seeding, all in pure Python.
"""


def splitmix64(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


class Rng:
    """xorshift64* generator with an explicit 64-bit state."""

    def __init__(self, seed=42):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        # splitmix64(0) is non-zero, so zero is a perfectly good seed.
        self.state = splitmix64(seed)

    def next_u64(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return (self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def random(self):
        """Uniform float in [0, 1)."""
        return self.next_u64() / float(1 << 64)

    def to_dict(self):
        return {"state": self.state}

    @classmethod
    def from_dict(cls, d):
        r = cls(0)
        r.state = d["state"] & 0xFFFFFFFFFFFFFFFF
        return r
