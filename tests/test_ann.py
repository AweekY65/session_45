"""Automated tests for local_vector_ann.

Run directly in a terminal:

    python tests/test_ann.py

or:

    python -m tests.test_ann

Every test prints its result; recall@k statistics are printed for the ANN
tests.  Exit code is non-zero if any test fails.
"""

import math
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_vector_ann import BruteForceIndex, HNSWIndex  # noqa: E402

_PASS = 0
_FAIL = 0
_FAILURES = []


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {msg}")
    else:
        _FAIL += 1
        _FAILURES.append(msg)
        print(f"  [FAIL] {msg}")


def section(name):
    print(f"\n=== {name} ===")


# --------------------------------------------------------------------- #
# 1. Input validation
# --------------------------------------------------------------------- #
def test_dimension_and_input_errors():
    section("dimension / NaN / illegal input")
    idx = HNSWIndex(4, metric="l2")
    idx.insert(0, [0.0, 0.0, 0.0, 0.0])

    try:
        idx.insert(1, [1.0, 2.0, 3.0])
        check(False, "wrong-length vector raises ValueError")
    except ValueError:
        check(True, "wrong-length vector raises ValueError")

    try:
        idx.insert(1, [1.0, 2.0, 3.0, 4.0, 5.0])
        check(False, "too-long vector raises ValueError")
    except ValueError:
        check(True, "too-long vector raises ValueError")

    try:
        idx.insert(1, [1.0, float("nan"), 3.0, 4.0])
        check(False, "NaN vector raises ValueError")
    except ValueError:
        check(True, "NaN vector raises ValueError")

    try:
        idx.insert(1, [1.0, float("inf"), 3.0, 4.0])
        check(False, "inf vector raises ValueError")
    except ValueError:
        check(True, "inf vector raises ValueError")

    try:
        idx.insert(1, [1.0, "x", 3.0, 4.0])
        check(False, "non-numeric component raises TypeError")
    except TypeError:
        check(True, "non-numeric component raises TypeError")

    try:
        idx.insert(1, "abcd")
        check(False, "string vector raises TypeError")
    except TypeError:
        check(True, "string vector raises TypeError")

    try:
        idx.insert(1, (x for x in range(4)))
        check(False, "generator vector raises TypeError")
    except TypeError:
        check(True, "generator vector raises TypeError")

    # Queries must be validated too.
    try:
        idx.search([1.0, 2.0, 3.0], k=1)
        check(False, "wrong-length query raises ValueError")
    except ValueError:
        check(True, "wrong-length query raises ValueError")
    try:
        idx.search([1.0, float("nan"), 0.0, 0.0], k=1)
        check(False, "NaN query raises ValueError")
    except ValueError:
        check(True, "NaN query raises ValueError")

    # Zero vector with cosine is illegal; with l2 it is fine.
    cidx = HNSWIndex(3, metric="cosine")
    try:
        cidx.insert(0, [0.0, 0.0, 0.0])
        check(False, "zero vector under cosine raises ValueError")
    except ValueError:
        check(True, "zero vector under cosine raises ValueError")

    # Bad labels / constructor args.
    try:
        idx.insert(1.5, [1.0, 2.0, 3.0, 4.0])
        check(False, "float label raises TypeError")
    except TypeError:
        check(True, "float label raises TypeError")
    try:
        HNSWIndex(0)
        check(False, "dim=0 raises ValueError")
    except ValueError:
        check(True, "dim=0 raises ValueError")
    try:
        HNSWIndex(4, metric="manhattan")
        check(False, "unknown metric raises ValueError")
    except ValueError:
        check(True, "unknown metric raises ValueError")
    try:
        idx.search([0.0] * 4, k=0)
        check(False, "k=0 raises ValueError")
    except ValueError:
        check(True, "k=0 raises ValueError")

    # Duplicate label.
    try:
        idx.insert(0, [1.0, 1.0, 1.0, 1.0])
        check(False, "duplicate label raises ValueError")
    except ValueError:
        check(True, "duplicate label raises ValueError")


# --------------------------------------------------------------------- #
# 2. Exactness on a small index + various k
# --------------------------------------------------------------------- #
def test_small_exact_and_k():
    section("small index: HNSW matches brute force for several k")
    rng = random.Random(7)
    n, dim = 80, 16
    data = {i: [rng.gauss(0, 1) for _ in range(dim)] for i in range(n)}

    for metric in ("cosine", "l2"):
        hnsw = HNSWIndex(dim, metric=metric, M=12,
                         ef_construction=200, ef_search=200, seed=1)
        bf = BruteForceIndex(dim, metric=metric)
        for label, vec in data.items():
            hnsw.insert(label, vec)
            bf.insert(label, vec)
        ok_all = True
        for k in (1, 3, 10, 25):
            for q_id in (0, 17, 42, 79):
                q = data[q_id]
                got = [label for label, _ in hnsw.search(q, k=k)]
                want = [label for label, _ in bf.search(q, k=k)]
                if got != want:
                    ok_all = False
                    print(f"    mismatch metric={metric} k={k} q={q_id}: "
                          f"{got[:5]} vs {want[:5]}")
        check(ok_all, f"HNSW exact vs brute force ({metric}, k in 1/3/10/25)")

    # k larger than the index returns everything; empty index returns [].
    hnsw = HNSWIndex(3, metric="l2")
    check(hnsw.search([1.0, 0.0, 0.0], k=5) == [], "search on empty index -> []")
    hnsw.insert(0, [1.0, 0.0, 0.0])
    hnsw.insert(1, [0.0, 1.0, 0.0])
    r = hnsw.search([1.0, 0.0, 0.0], k=10)
    check([lab for lab, _ in r] == [0, 1], "k > n returns all live vectors")


# --------------------------------------------------------------------- #
# 3. Duplicate vectors and stable tie-break
# --------------------------------------------------------------------- #
def test_duplicate_vectors_tie_break():
    section("duplicate vectors: stable tie-break by label")
    dim = 5
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    other = [0.9, 0.0, 0.0, 0.0, 0.1]

    hnsw = HNSWIndex(dim, metric="cosine", M=8, ef_construction=64, seed=3)
    bf = BruteForceIndex(dim, metric="cosine")
    for label in (5, 3, 1, 4, 2):
        hnsw.insert(label, v if label != 4 else other)
        bf.insert(label, v if label != 4 else other)

    got = hnsw.search(v, k=4)
    labels = [lab for lab, _ in got]
    check(labels == [1, 2, 3, 5],
          f"identical vectors ordered by label, got {labels}")

    want = bf.search(v, k=4)
    check([l for l, _ in got] == [l for l, _ in want],
          "tie ordering matches brute force")

    # Distances among duplicates must be exactly equal.
    dups = [d for lab, d in got if lab != 4]
    check(all(dups[0] == d for d in dups),
          f"duplicate distances identical: {dups}")

    # Repeated queries must be deterministic.
    again = [lab for lab, _ in hnsw.search(v, k=4)]
    check(again == labels, "repeat query gives identical order")


# --------------------------------------------------------------------- #
# 4. Deletion and compaction
# --------------------------------------------------------------------- #
def test_delete_and_compact():
    section("delete + compact")
    rng = random.Random(99)
    n, dim = 300, 16
    data = [(i, [rng.gauss(0, 1) for _ in range(dim)]) for i in range(n)]
    hnsw = HNSWIndex(dim, metric="cosine", M=12,
                     ef_construction=100, ef_search=100, seed=5)
    for label, vec in data:
        hnsw.insert(label, vec)

    deleted = set(range(0, n, 7))
    for label in deleted:
        check_ = hnsw.delete(label)
        assert check_
    check(not hnsw.delete(999999), "deleting missing label returns False")
    check(len(hnsw) == n - len(deleted), "len() excludes deleted nodes")

    queries = [data[i][1] for i in (1, 50, 150, 299)]
    before = [[lab for lab, _ in hnsw.search(q, k=10)] for q in queries]
    check(all(all(lab not in deleted for lab in labs) for labs in before),
          "no deleted label appears in search results")

    # Rebuild the matching brute-force oracle over live vectors only.
    bf = BruteForceIndex(dim, metric="cosine")
    for label, vec in data:
        if label not in deleted:
            bf.insert(label, vec)
    oracle = [[lab for lab, _ in bf.search(q, k=10)] for q in queries]
    check(before == oracle, "post-delete results match live-only brute force")

    # Tombstones still physically present before compact.
    tombstones = [n for n in hnsw._nodes.values() if n.deleted]
    check(len(tombstones) == len(deleted),
          f"tombstones retained before compact ({len(tombstones)})")

    hnsw.compact()
    tombstones = [n for n in hnsw._nodes.values() if n.deleted]
    check(tombstones == [], "compact physically removes tombstones")
    check(len(hnsw) == n - len(deleted), "len unchanged by compact")

    after = [[lab for lab, _ in hnsw.search(q, k=10)] for q in queries]
    check(after == before, "search results identical after compact")
    check(after == oracle, "post-compact results still match brute force")

    # Compacted index must behave like a fresh index built from live data.
    fresh = HNSWIndex(dim, metric="cosine", M=12,
                      ef_construction=100, ef_search=100, seed=5)
    for label, vec in data:
        if label not in deleted:
            fresh.insert(label, vec)
    levels_old = sorted(n.level for n in hnsw._nodes.values())
    levels_new = sorted(n.level for n in fresh._nodes.values())
    check(levels_old == levels_new,
          "compact rebuild reproduces fresh graph levels (deterministic)")
    fresh_res = [[lab for lab, _ in fresh.search(q, k=10)] for q in queries]
    check(fresh_res == after, "compact rebuild results equal fresh index")


# --------------------------------------------------------------------- #
# 5. Persistence
# --------------------------------------------------------------------- #
def test_persistence():
    section("save / reload")
    rng = random.Random(2024)
    n, dim = 250, 20
    data = [(i, [rng.gauss(0, 1) for _ in range(dim)]) for i in range(n)]

    idx = HNSWIndex(dim, metric="cosine", M=12,
                    ef_construction=100, ef_search=64, seed=11)
    for label, vec in data[:150]:
        idx.insert(label, vec)
    # Delete a few so tombstone serialization is exercised.
    for label in (2, 10, 77):
        idx.delete(label)

    queries = [data[i][1] for i in (3, 33, 99, 140)]
    before = [idx.search(q, k=15) for q in queries]

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.bin")
        idx.save(path)
        size = os.path.getsize(path)
        check(size > 0, f"snapshot written ({size} bytes)")

        # Magic header check.
        with open(path, "rb") as f:
            check(f.read(8) == b"LVANNH01", "snapshot magic bytes present")

        loaded = HNSWIndex.load(path)

    check(loaded.dim == dim and loaded.metric == "cosine",
          "metadata (dim, metric) restored")
    check(loaded.M == 12 and loaded.ef_construction == 100
          and loaded.ef_search == 64, "build/search params restored")
    check(len(loaded) == len(idx), "live count restored")

    # Delete a missing label in the loaded index must stay missing.
    check(not loaded.delete(10), "tombstone remains a tombstone after reload")

    after = [loaded.search(q, k=15) for q in queries]
    same = all(
        [l for l, _ in b] == [l for l, _ in a]
        and all(math.isclose(db, da, rel_tol=1e-12, abs_tol=1e-12)
                for (_, db), (_, da) in zip(b, a))
        for b, a in zip(before, after)
    )
    check(same, "query results identical after save/reload")

    # Continue inserting on both sides: persisted RNG state must make the
    # continued graph identical to an uninterrupted run.
    ref = HNSWIndex(dim, metric="cosine", M=12,
                    ef_construction=100, ef_search=64, seed=11)
    for label, vec in data[:150]:
        ref.insert(label, vec)
    for label in (2, 10, 77):
        ref.delete(label)
    for label, vec in data[150:]:
        idx.insert(label, vec)
        loaded.insert(label, vec)
        ref.insert(label, vec)

    q2 = [data[i][1] for i in (160, 200, 249)]
    r_loaded = [loaded.search(q, k=12) for q in q2]
    r_ref = [ref.search(q, k=12) for q in q2]
    check(r_loaded == r_ref,
          "post-reload inserts reproduce uninterrupted-index results (RNG state)")

    # Loading garbage must fail cleanly.
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.bin")
        with open(bad, "wb") as f:
            f.write(b"not an index" + b"\x00" * 40)
        try:
            HNSWIndex.load(bad)
            check(False, "loading non-snapshot file raises ValueError")
        except ValueError:
            check(True, "loading non-snapshot file raises ValueError")


# --------------------------------------------------------------------- #
# 6. ANN recall + no-full-scan guarantee
# --------------------------------------------------------------------- #
def recall_at_k(approx, exact):
    a = set(lab for lab, _ in approx)
    b = set(lab for lab, _ in exact)
    if not b:
        return 1.0
    return len(a & b) / len(b)


def test_recall_and_no_full_scan():
    section("ANN recall@k vs brute force, both metrics")
    n, dim = 2000, 32
    rng = random.Random(4242)
    data = [(i, [rng.gauss(0, 1) for _ in range(dim)]) for i in range(n)]
    nq = 30
    query_ids = rng.sample(range(n), nq)

    for metric in ("cosine", "l2"):
        t0 = time.time()
        hnsw = HNSWIndex(dim, metric=metric, M=16,
                         ef_construction=100, ef_search=100, seed=42)
        bf = BruteForceIndex(dim, metric=metric)
        for label, vec in data:
            hnsw.insert(label, vec)
            bf.insert(label, vec)
        build_s = time.time() - t0

        for k in (1, 10, 50):
            rs = []
            for qi in query_ids:
                q = data[qi][1]
                rs.append(recall_at_k(hnsw.search(q, k=k), bf.search(q, k=k)))
            mean = sum(rs) / len(rs)
            threshold = {1: 0.95, 10: 0.90, 50: 0.85}[k]
            print(f"  metric={metric:6s} n={n} dim={dim} "
                  f"ef_search=100  recall@{k:<2d} = {mean:.4f} "
                  f"(min {min(rs):.3f}, threshold {threshold})")
            check(mean >= threshold,
                  f"recall@{k} ({metric}) {mean:.3f} >= {threshold}")

        # Distance-budget check: graph search must touch far fewer than all
        # nodes.  Measured at the default ef_search=50 on a separate instance.
        small = HNSWIndex(dim, metric=metric, M=16,
                          ef_construction=100, ef_search=50, seed=42)
        for label, vec in data:
            small.insert(label, vec)
        budgets = []
        for qi in query_ids:
            small.reset_distance_count()
            small.search(data[qi][1], k=10)
            budgets.append(small.distance_count)
        avg_budget = sum(budgets) / len(budgets)
        fraction = avg_budget / n
        print(f"  metric={metric:6s} avg distance evals/query = "
              f"{avg_budget:.0f} / {n} = {fraction:.1%} of a full scan "
              f"(build {build_s:.1f}s)")
        check(fraction < 0.45,
              f"ANN query touches {fraction:.1%} of nodes (< 45%, "
              f"not a full scan; sublinear scaling verified separately)")

    # Increasing ef_search must not decrease recall (monotonic in aggregate).
    hnsw = HNSWIndex(dim, metric="cosine", M=16,
                     ef_construction=100, ef_search=10, seed=42)
    bf = BruteForceIndex(dim, metric="cosine")
    for label, vec in data:
        hnsw.insert(label, vec)
        bf.insert(label, vec)
    low = sum(recall_at_k(hnsw.search(data[qi][1], k=10, ef_search=10),
                          bf.search(data[qi][1], k=10))
              for qi in query_ids) / nq
    high = sum(recall_at_k(hnsw.search(data[qi][1], k=10, ef_search=200),
                           bf.search(data[qi][1], k=10))
               for qi in query_ids) / nq
    print(f"  recall@10 ef=10  -> {low:.4f}")
    print(f"  recall@10 ef=200 -> {high:.4f}")
    check(high >= low, "larger ef_search gives >= recall")


def test_sublinear_scaling():
    """The defining 'not a full scan' guarantee: distance evaluations per
    query must grow sub-linearly with index size."""
    section("sublinear distance-budget scaling")
    dim = 32
    sizes = (400, 1600, 4000)
    nq = 25
    fractions = {}
    totals = {}
    for n in sizes:
        rng = random.Random(31337)
        data = [(i, [rng.gauss(0, 1) for _ in range(dim)]) for i in range(n)]
        idx = HNSWIndex(dim, metric="cosine", M=8,
                        ef_construction=64, ef_search=32, seed=7)
        for label, vec in data:
            idx.insert(label, vec)
        qids = [int(i * (n - 1) / (nq - 1)) for i in range(nq)]
        budgets = []
        for qi in qids:
            idx.reset_distance_count()
            idx.search(data[qi][1], k=10)
            budgets.append(idx.distance_count)
        avg = sum(budgets) / len(budgets)
        fractions[n] = avg / n
        totals[n] = avg
        print(f"  n={n:5d}  evals/query={avg:7.0f}  "
              f"fraction of full scan={fractions[n]:6.2%}")

    check(fractions[4000] < fractions[400],
          "fraction scanned decreases as the index grows "
          f"({fractions[400]:.1%} -> {fractions[4000]:.1%})")
    check(fractions[4000] < 0.30,
          f"at n=4000 a query scans < 30% of nodes "
          f"({fractions[4000]:.1%})")
    # 10x more data must cost far less than 10x the distance work.
    ratio = totals[4000] / totals[400]
    check(ratio < 6.0,
          f"10x data costs only {ratio:.1f}x distance evals (< 6x, "
          f"sublinear)")


# --------------------------------------------------------------------- #
# 7. Deterministic seed
# --------------------------------------------------------------------- #
def test_deterministic_seed():
    section("deterministic seed")
    rng = random.Random(555)
    n, dim = 200, 12
    data = [(i, [rng.gauss(0, 1) for _ in range(dim)]) for i in range(n)]

    def build():
        idx = HNSWIndex(dim, metric="l2", M=10,
                        ef_construction=64, ef_search=32, seed=42)
        for label, vec in data:
            idx.insert(label, vec)
        return idx

    a, b = build(), build()
    la = {nid: n.level for nid, n in a._nodes.items()}
    lb = {nid: n.level for nid, n in b._nodes.items()}
    check(la == lb, "same seed -> identical level assignment")
    ea = {nid: [list(c) for c in n.neighbors]
          for nid, n in a._nodes.items()}
    eb = {nid: [list(c) for c in n.neighbors]
          for nid, n in b._nodes.items()}
    check(ea == eb, "same seed -> identical graph edges")

    q = data[123][1]
    check(a.search(q, k=20) == b.search(q, k=20),
          "same seed -> identical query results")

    # Different seed: still valid results, allowed to differ.
    c = HNSWIndex(dim, metric="l2", M=10,
                  ef_construction=64, ef_search=32, seed=999)
    for label, vec in data:
        c.insert(label, vec)
    bf = BruteForceIndex(dim, metric="l2")
    for label, vec in data:
        bf.insert(label, vec)
    got = {l for l, _ in c.search(q, k=20)}
    want = {l for l, _ in bf.search(q, k=20)}
    check(len(got & want) >= 18,
          f"different seed still finds good results "
          f"({len(got & want)}/20 overlap)")


# --------------------------------------------------------------------- #
# 8. Ordering contract: ascending distance, ties by label
# --------------------------------------------------------------------- #
def test_result_ordering():
    section("result ordering contract")
    rng = random.Random(8)
    idx = HNSWIndex(8, metric="l2", M=8, ef_construction=64, seed=2)
    for i in range(120):
        idx.insert(i, [rng.gauss(0, 1) for _ in range(8)])
    q = [rng.gauss(0, 1) for _ in range(8)]
    res = idx.search(q, k=30)
    dists = [d for _, d in res]
    check(all(dists[i] <= dists[i + 1] + 1e-9
              for i in range(len(dists) - 1)),
          "distances non-decreasing")
    # Within a tie bucket, labels must be sorted.
    from local_vector_ann.distances import distance_bucket
    from local_vector_ann.hnsw import _label_sort_key
    keys = [(distance_bucket(d), _label_sort_key(l)) for l, d in res]
    check(keys == sorted(keys), "tie buckets ordered by label")
    check(all(d >= 0.0 for d in dists), "all distances non-negative")
    labs = [l for l, _ in res]
    check(len(labs) == len(set(labs)), "no duplicate labels in results")


def main():
    print("local_vector_ann test suite")
    print(f"python {sys.version.split()[0]}")
    t0 = time.time()
    test_dimension_and_input_errors()
    test_small_exact_and_k()
    test_duplicate_vectors_tie_break()
    test_delete_and_compact()
    test_persistence()
    test_recall_and_no_full_scan()
    test_sublinear_scaling()
    test_deterministic_seed()
    test_result_ordering()
    dt = time.time() - t0

    print("\n" + "=" * 60)
    print(f"TOTAL: {_PASS} passed, {_FAIL} failed, "
          f"{_PASS + _FAIL} checks, {dt:.1f}s")
    if _FAIL:
        print("FAILURES:")
        for m in _FAILURES:
            print(f"  - {m}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
