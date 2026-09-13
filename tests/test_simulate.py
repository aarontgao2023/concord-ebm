"""Lightweight, independently checked invariants for the new simulation design.

Run: pytest tests/test_simulate.py
No DEBM fits, SciPy, downloads, or legacy results are required.
"""

from collections import Counter
from itertools import permutations
import json
from math import factorial
import unittest

import numpy as np
import pandas as pd

import concord.simulate as D


def brute_distance(left, right):
    """Independent definition using pairwise biomarker precedence."""
    a = {item: rank for rank, item in enumerate(left)}
    b = {item: rank for rank, item in enumerate(right)}
    return sum((a[i] < a[j]) != (b[i] < b[j]) for i in a for j in a if i < j)


class ExactAlternativeTests(unittest.TestCase):
    def test_mahonian_counts_match_enumeration(self):
        for n in range(2, 8):
            counts = Counter(brute_distance(range(n), order) for order in permutations(range(n)))
            row = D._mahonian_table(n)[n]
            self.assertEqual(tuple(counts[k] for k in range(len(row))), row)
            self.assertEqual(sum(row), factorial(n))

    def test_every_k_is_exact_for_all_small_spaces_and_fourteen(self):
        rng = np.random.default_rng(7249)
        for n in (2, 3, 5, 8, 14):
            for _ in range(4):
                base = rng.permutation(n)
                for k in range(n * (n - 1) // 2 + 1):
                    out = D.ordering_at_inversions(base, k, rng)
                    self.assertEqual(brute_distance(base, out), k)
                    self.assertEqual(sorted(out.tolist()), list(range(n)))

    def test_k6_geometries_and_block_swaps(self):
        rng = np.random.default_rng(328)
        for _ in range(50):
            base = rng.permutation(14)
            single = D.ordering_at_inversions(base, 6, rng, "single_displacement", moved_index=0)
            self.assertEqual(abs(list(base).index(0) - list(single).index(0)), 6)
            self.assertEqual([x for x in base if x != 0], [x for x in single if x != 0])
            disjoint = D.ordering_at_inversions(base, 6, rng, "disjoint_adjacent")
            ranks = np.argsort(base)[disjoint]
            self.assertTrue(np.all(np.abs(ranks - np.arange(14)) <= 1))
            self.assertEqual(np.count_nonzero(ranks != np.arange(14)), 12)
            for k in (14, 27):
                block = D.ordering_at_inversions(base, k, rng, "block_swap")
                self.assertEqual(brute_distance(base, block), k)

    def test_invalid_requests_fail_instead_of_returning_wrong_alternative(self):
        rng = np.random.default_rng(8)
        for k, geometry in ((14, "single_displacement"), (8, "disjoint_adjacent"),
                            (55, "block_swap"), (-1, "exact_k"), (92, "exact_k")):
            with self.assertRaises(ValueError):
                D.ordering_at_inversions(np.arange(14), k, rng, geometry)
        with self.assertRaises(ValueError):
            D.ordering_at_inversions(np.arange(14), 2.5, rng)
        with self.assertRaises(ValueError):
            D.ordering_at_inversions(np.array([0, 0, 2]), 1, rng)


class SimulationTests(unittest.TestCase):
    def test_fixed_composition_preserves_both_correct_margins(self):
        reference = np.asarray(D.REFERENCE_COUNTS)
        for scale in (1, 2, 4, 16):
            for lam in (0., .1, .5, .9, 1.):
                counts = D.fixed_group_dx(scale, lam)
                table = np.array([[counts[g][dx] for dx in D.DX_ORDER] for g in D.GROUP_ORDER])
                np.testing.assert_array_equal(table.sum(0), scale * np.array([411, 228, 332]))
                np.testing.assert_array_equal(table.sum(1), scale * reference.sum(1))
                self.assertTrue(np.all(table >= 0))
                if lam == 1:
                    np.testing.assert_array_equal(table, scale * reference)

    def test_seed_and_serialized_config_reproduce_everything(self):
        cfg = D.resolve("PWR_E2_K55", stage_eta=.5)
        serialized = json.loads(json.dumps(cfg.to_dict(), allow_nan=False))
        one, truth1 = D.simulate(cfg, 6540)
        two, truth2 = D.simulate(serialized, 6540)
        pd.testing.assert_frame_equal(one, two, check_exact=True)
        self.assertEqual(truth1, truth2)
        self.assertEqual(set(truth1["manifest"]["streams"]), set(D.STREAM_NAMES))
        self.assertEqual(truth1["realized_inversions"], {"e2": 55, "e33": 0, "e4": 0})
        self.assertTrue(truth1["pair_truth"]["e33_vs_e4"]["null"])
        self.assertFalse(truth1["common_order_null"])
        json.dumps(truth1, allow_nan=False)

    def test_reference_truth_aligns_with_rows_and_has_common_order(self):
        df, truth = D.simulate(D.resolve("REF_H0"), 1024)
        self.assertEqual(len(df), 971)
        self.assertEqual(df.PTID.tolist(), truth["latent_row_ptid"])
        self.assertEqual(len(truth["latent_stage"]), len(df))
        np.testing.assert_array_equal(np.rint(14 * np.array(truth["latent_stage_fraction"])).astype(int),
                                      truth["latent_stage"])
        self.assertTrue(truth["common_order_null"])
        self.assertTrue(truth["conditional_exchangeability_by_design"])
        self.assertFalse(truth["unrestricted_exchangeability_by_design"])
        self.assertEqual(truth["group_dx_counts"], D.fixed_group_dx())
        for group in D.GROUP_ORDER:
            self.assertEqual(truth["group_orderings"][group], truth["base_order"])

    def test_iid_control_randomizes_diagnosis_counts_and_groups(self):
        counts = []
        for seed in (204, 205, 206):
            df, truth = D.simulate(D.resolve("IID_H0"), seed)
            self.assertTrue(truth["unrestricted_exchangeability_by_design"])
            self.assertEqual(df.APOE.value_counts().sort_index().tolist(), [75, 411, 485])
            self.assertGreater(np.count_nonzero(np.diff(df.APOE.to_numpy())), 100)
            counts.append(truth["group_dx_counts"])
        self.assertNotEqual(counts[0], counts[1])
        self.assertNotEqual(counts[1], counts[2])

    def test_eta_zero_equals_reference_and_stage_stress_changes_only_affected_rows(self):
        baseline, bt = D.simulate(D.resolve("REF_H0", missing=False), 338)
        zero, zt = D.simulate(D.resolve("STAGE_H0", stage_eta=0, missing=False), 338)
        pd.testing.assert_frame_equal(baseline, zero, check_exact=True)
        stressed, st = D.simulate(D.resolve("STAGE_H0", missing=False), 338)
        unaffected = baseline.APOE != D.GROUP_CODE["e4"]
        pd.testing.assert_frame_equal(baseline[unaffected], stressed[unaffected], check_exact=True)
        self.assertEqual(bt["latent_stage"][:486], st["latent_stage"][:486])
        self.assertNotEqual(bt["latent_stage"][486:], st["latent_stage"][486:])
        self.assertTrue(st["common_order_null"])
        self.assertFalse(st["conditional_exchangeability_by_design"])
        self.assertEqual(bt["group_orderings"], st["group_orderings"])

    def test_csf_stress_does_not_change_other_modality_masks(self):
        base, bt = D.simulate(D.resolve("REF_H0"), 7420)
        csf, ct = D.simulate(D.resolve("MISS_CSF_H0"), 7420)
        all_modal, at = D.simulate(D.resolve("MISS_ALL_H0"), 7420)
        unaffected = [bm.name for bm in D.BIOMARKERS if bm.modality != "CSF"]
        pd.testing.assert_frame_equal(base[unaffected], csf[unaffected], check_exact=True)
        self.assertTrue(csf.MMSE.notna().all())
        e4_cn = (all_modal.APOE == 2) & (all_modal.Diagnosis == "CN")
        self.assertGreater(int(all_modal.loc[e4_cn, "MMSE"].isna().sum()), 0)
        self.assertEqual(ct["availability_by_group_dx"]["e4"]["CN"]["MMSE"]["expected"], 1.)
        self.assertAlmostEqual(at["availability_by_group_dx"]["e4"]["CN"]["MMSE"]["expected"], .7)
        self.assertEqual(bt["latent_stage"], ct["latent_stage"])

    def test_component_auc_and_observed_diagnosis_auc_are_distinct(self):
        _, truth = D.simulate(D.resolve("REF_H0"), 887)
        component = truth["component_parameters"]["e4"]["ABETA"]["component_auc"]
        diagnosis = truth["realized_diagnosis_auc"]["e4"]["ABETA"]["complete"]
        self.assertAlmostEqual(component, .88, places=11)
        self.assertNotEqual(component, diagnosis)
        self.assertEqual(D._auc(np.array([1., 1.]), np.array([1., 1.])), .5)
        self.assertEqual(D._auc(np.array([0., 1.]), np.array([2., 3.])), 1.)
        self.assertIsNone(D._auc(np.array([np.nan]), np.array([1.])))

    def test_small_i_and_invalid_configs(self):
        cfg = D.resolve("PWR_E4_K6", biomarker_names=tuple(b.name for b in D.BIOMARKERS[:6]))
        df, truth = D.simulate(cfg, 913)
        self.assertEqual(len(df.columns), 9)
        self.assertEqual(truth["realized_inversions"]["e4"], 6)
        for overrides in ({"sample_scale": .5}, {"composition_lambda": 1.2},
                          {"missing_scope": "maybe"}, {"component_auc_shift": .2},
                          {"missing": False, "missing_stress_eta": 1.}):
            with self.assertRaises(ValueError):
                D.resolve("REF_H0", **overrides)
        with self.assertRaises(ValueError):
            D.resolve("IID_H0", composition_lambda=1.)
        with self.assertRaises(ValueError):
            D.simulate(D.resolve("REF_H0"), -1)


if __name__ == "__main__":
    unittest.main()
