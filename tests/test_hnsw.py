"""Automated tests for the local HNSW ANN index.

Run directly in a terminal:

    python -m unittest discover -s tests -v

All data is randomly generated in-code with fixed seeds — no external
services, no network, no datasets. Recall@k results are printed to stdout.
"""

import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann_index import HNSWIndex, brute_force_search, cosine_distance, l2_distance


def random_vectors(n, dim, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n)]


def build_index(vectors, dim, metric="cosine", seed=42, **kw):
    idx = HNSWIndex(dim=dim, metric=metric, seed=seed, **kw)
    ids = [idx.insert(v) for v in vectors]
    return idx, ids


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.idx = HNSWIndex(dim=4, metric="cosine", seed=1)

    def test_dimension_mismatch_on_insert(self):
        with self.assertRaises(ValueError):
            self.idx.insert([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.idx.insert([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_dimension_mismatch_on_query(self):
        self.idx.insert([1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.idx.search([1.0, 2.0], k=1)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.idx.insert([1.0, float("nan"), 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.idx.search([float("nan")] * 4, k=1)

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            self.idx.insert([1.0, float("inf"), 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.idx.insert([1.0, float("-inf"), 0.0, 0.0])

    def test_invalid_types_rejected(self):
        for bad in ("abcd", None, 3.14, [1.0, "x", 0.0, 0.0], [1.0, None, 0.0, 0.0]):
            with self.assertRaises((TypeError, ValueError)):
                self.idx.insert(bad)

    def test_bad_constructor_params(self):
        with self.assertRaises(ValueError):
            HNSWIndex(dim=0)
        with self.assertRaises(ValueError):
            HNSWIndex(dim=-3)
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, metric="manhattan")
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, M=1)

    def test_bad_k_rejected(self):
        self.idx.insert([1.0, 0.0, 0.0, 0.0])
        for bad_k in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                self.idx.search([1.0, 0.0, 0.0, 0.0], k=bad_k)


class TestBasicSearch(unittest.TestCase):
    def test_exact_match_is_closest(self):
        idx, ids = build_index(random_vectors(200, 16, 7), 16)
        probe = [0.5] * 16
        pid = idx.insert(probe)
        res = idx.search(probe, k=1)
        self.assertEqual(res[0][0], pid)
        self.assertAlmostEqual(res[0][1], 0.0, places=9)

    def test_results_sorted_with_stable_tiebreak(self):
        idx = HNSWIndex(dim=2, metric="l2", seed=3)
        a = idx.insert([1.0, 0.0])
        b = idx.insert([1.0, 0.0])  # duplicate: identical distance
        c = idx.insert([0.0, 1.0])
        res = idx.search([1.0, 0.0], k=3)
        # distances must be non-decreasing; equal distances ordered by id
        keys = [(d, i) for i, d in res]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual([i for i, _ in res[:2]], [a, b])

    def test_duplicate_vectors_all_returned(self):
        idx = HNSWIndex(dim=3, metric="cosine", seed=5)
        ids = [idx.insert([1.0, 2.0, 3.0]) for _ in range(5)]
        res = idx.search([1.0, 2.0, 3.0], k=5)
        self.assertEqual([i for i, _ in res], ids)
        for _, d in res:
            self.assertAlmostEqual(d, 0.0, places=9)

    def test_various_k(self):
        vecs = random_vectors(100, 8, 11)
        idx, _ = build_index(vecs, 8)
        q = random_vectors(1, 8, 12)[0]
        for k in (1, 3, 10, 50):
            res = idx.search(q, k=k)
            self.assertEqual(len(res), k)
        # k larger than the live set returns everything live
        res = idx.search(q, k=1000)
        self.assertEqual(len(res), 100)

    def test_empty_index(self):
        idx = HNSWIndex(dim=4, seed=1)
        self.assertEqual(idx.search([1.0] * 4, k=5), [])
        self.assertEqual(len(idx), 0)

    def test_l2_metric_matches_hand_computation(self):
        idx = HNSWIndex(dim=2, metric="l2", seed=1)
        idx.insert([3.0, 4.0])   # distance 5 from origin
        idx.insert([1.0, 0.0])   # distance 1
        idx.insert([0.0, -2.0])  # distance 2
        res = idx.search([0.0, 0.0], k=3, ef_search=10)
        self.assertEqual([i for i, _ in res], [1, 2, 0])
        self.assertAlmostEqual(res[0][1], 1.0)
        self.assertAlmostEqual(res[1][1], 2.0)
        self.assertAlmostEqual(res[2][1], 5.0)

    def test_cosine_metric_matches_hand_computation(self):
        idx = HNSWIndex(dim=2, metric="cosine", seed=1)
        idx.insert([1.0, 0.0])
        idx.insert([0.0, 1.0])   # orthogonal: cosine distance 1
        res = idx.search([2.0, 0.0], k=2, ef_search=10)
        self.assertEqual(res[0][0], 0)
        self.assertAlmostEqual(res[0][1], 0.0, places=9)
        self.assertAlmostEqual(res[1][1], 1.0, places=9)


class TestDeleteAndCompact(unittest.TestCase):
    def test_deleted_vectors_never_returned(self):
        vecs = random_vectors(300, 16, 21)
        idx, ids = build_index(vecs, 16)
        deleted = set(ids[::10])
        for i in deleted:
            idx.delete(i)
        self.assertEqual(len(idx), 300 - len(deleted))
        for q in random_vectors(20, 16, 22):
            for nid, _ in idx.search(q, k=10):
                self.assertNotIn(nid, deleted)

    def test_delete_errors(self):
        idx, ids = build_index(random_vectors(10, 4, 23), 4)
        idx.delete(ids[0])
        with self.assertRaises(KeyError):
            idx.delete(ids[0])       # already deleted
        with self.assertRaises(KeyError):
            idx.delete(999999)       # never existed

    def test_delete_then_reinsert_gets_new_id(self):
        idx = HNSWIndex(dim=2, seed=1)
        a = idx.insert([1.0, 1.0])
        idx.delete(a)
        b = idx.insert([1.0, 1.0])
        self.assertNotEqual(a, b)
        res = idx.search([1.0, 1.0], k=1)
        self.assertEqual(res[0][0], b)

    def test_compact_drops_tombstones_and_preserves_ids(self):
        vecs = random_vectors(400, 16, 31)
        idx, ids = build_index(vecs, 16)
        deleted = set(ids[::4])
        for i in deleted:
            idx.delete(i)
        self.assertEqual(idx.deleted_count, len(deleted))
        live_before = dict(idx.items())
        idx.compact()
        self.assertEqual(idx.deleted_count, 0)
        self.assertEqual(len(idx), len(live_before))
        # ids and vectors preserved
        self.assertEqual(dict(idx.items()), live_before)
        # queries after compact still exclude deleted ids and stay sane
        for q in random_vectors(10, 16, 32):
            res = idx.search(q, k=10)
            self.assertTrue(res)
            for nid, _ in res:
                self.assertNotIn(nid, deleted)
        # a second compact of the same live set is a fixed point
        idx.compact()
        self.assertEqual(dict(idx.items()), live_before)

    def test_search_after_compact_matches_brute_force_closely(self):
        vecs = random_vectors(500, 24, 33)
        idx, ids = build_index(vecs, 24, ef_construction=100)
        for i in ids[::3]:
            idx.delete(i)
        idx.compact()
        hits = 0
        total = 0
        for q in random_vectors(20, 24, 34):
            ann = {i for i, _ in idx.search(q, k=10, ef_search=100)}
            bf = {i for i, _ in idx.brute_force_search(q, k=10)}
            hits += len(ann & bf)
            total += 10
        recall = hits / total
        print("\n[compact] recall@10 after delete+compact: %.4f" % recall)
        self.assertGreaterEqual(recall, 0.9)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip_identical_results(self):
        vecs = random_vectors(500, 24, 41)
        idx, ids = build_index(vecs, 24, metric="cosine", ef_construction=120)
        for i in ids[::7]:  # include tombstones in the snapshot
            idx.delete(i)
        queries = random_vectors(15, 24, 42)
        before = [idx.search(q, k=10) for q in queries]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index_snapshot.json")
            idx.save(path)
            self.assertTrue(os.path.getsize(path) > 0)
            loaded = HNSWIndex.load(path)

        self.assertEqual(len(loaded), len(idx))
        self.assertEqual(loaded.deleted_count, idx.deleted_count)
        after = [loaded.search(q, k=10) for q in queries]
        # Serialized-then-reloaded queries must match pre-save results exactly.
        self.assertEqual(before, after)

    def test_save_load_l2(self):
        vecs = random_vectors(200, 12, 43)
        idx, _ = build_index(vecs, 12, metric="l2")
        q = random_vectors(1, 12, 44)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.json")
            idx.save(path)
            loaded = HNSWIndex.load(path)
        self.assertEqual(idx.search(q, k=5), loaded.search(q, k=5))

    def test_load_rejects_bad_version(self):
        idx, _ = build_index(random_vectors(5, 4, 45), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.json")
            idx.save(path)
            with open(path, "r") as f:
                content = f.read().replace('"version": 1', '"version": 999', 1)
            with open(path, "w") as f:
                f.write(content)
            with self.assertRaises(ValueError):
                HNSWIndex.load(path)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_results(self):
        vecs = random_vectors(400, 16, 51)
        idx1, _ = build_index(vecs, 16, seed=99)
        idx2, _ = build_index(vecs, 16, seed=99)
        for q in random_vectors(10, 16, 52):
            self.assertEqual(idx1.search(q, k=10), idx2.search(q, k=10))

    def test_reproducible_random_data(self):
        self.assertEqual(random_vectors(5, 4, 123), random_vectors(5, 4, 123))


class TestRecallAndNoDegeneration(unittest.TestCase):
    def test_recall_at_k_vs_brute_force(self):
        n, dim, k, n_queries = 2000, 32, 10, 50
        vecs = random_vectors(n, dim, 61)
        idx, _ = build_index(vecs, dim, metric="cosine",
                             M=16, ef_construction=100, ef_search=100)
        queries = random_vectors(n_queries, dim, 62)
        hits = 0
        for q in queries:
            ann = {i for i, _ in idx.search(q, k=k)}
            bf = {i for i, _ in brute_force_search(
                ((i, vecs[i]) for i in range(n)), q, k, metric="cosine")}
            hits += len(ann & bf)
        recall = hits / (n_queries * k)
        print("\n[recall] cosine recall@%d on %d vectors, %d queries: %.4f"
              % (k, n, n_queries, recall))
        self.assertGreaterEqual(recall, 0.90)

    def test_recall_l2(self):
        n, dim, k, n_queries = 1500, 16, 10, 30
        vecs = random_vectors(n, dim, 63)
        idx, _ = build_index(vecs, dim, metric="l2",
                             M=16, ef_construction=100, ef_search=100)
        hits = 0
        for q in random_vectors(n_queries, dim, 64):
            ann = {i for i, _ in idx.search(q, k=k)}
            bf = {i for i, _ in idx.brute_force_search(q, k=k)}
            hits += len(ann & bf)
        recall = hits / (n_queries * k)
        print("\n[recall] l2 recall@%d on %d vectors, %d queries: %.4f"
              % (k, n, n_queries, recall))
        self.assertGreaterEqual(recall, 0.90)

    def test_ann_does_not_degenerate_to_full_scan(self):
        n, dim = 2000, 32
        vecs = random_vectors(n, dim, 71)
        idx, _ = build_index(vecs, dim, ef_search=50)
        q = random_vectors(1, dim, 72)[0]
        idx.dist_computations = 0
        res = idx.search(q, k=10)
        used = idx.dist_computations
        print("\n[scan] ANN query used %d distance computations "
              "vs %d for a full scan" % (used, n))
        self.assertEqual(len(res), 10)
        # A brute-force scan would cost exactly n distance computations;
        # the graph search must be well below that.
        self.assertLess(used, n // 2)

    def test_ef_search_widens_search(self):
        # ef_search is honored: a larger beam must not reduce the candidate pool.
        vecs = random_vectors(500, 16, 73)
        idx, _ = build_index(vecs, 16)
        q = random_vectors(1, 16, 74)[0]
        idx.dist_computations = 0
        idx.search(q, k=10, ef_search=10)
        small = idx.dist_computations
        idx.dist_computations = 0
        idx.search(q, k=10, ef_search=200)
        large = idx.dist_computations
        self.assertLessEqual(small, large)


class TestDistanceFunctions(unittest.TestCase):
    def test_cosine_distance_properties(self):
        self.assertAlmostEqual(cosine_distance([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(cosine_distance([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(cosine_distance([1, 0], [-1, 0]), 2.0)
        # scale invariance
        self.assertAlmostEqual(cosine_distance([1, 2], [2, 4]), 0.0)
        # zero vector convention
        self.assertEqual(cosine_distance([0, 0], [1, 1]), 1.0)

    def test_l2_distance_properties(self):
        self.assertAlmostEqual(l2_distance([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(l2_distance([1, 1], [1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
