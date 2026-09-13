"""Checks for concord.compare; run with pytest.

The end-to-end tests need the pinned pyebm wheel (small fits, seconds each). The rule test
enumerates every completion of the permutation budget explicitly.
"""
from __future__ import annotations

from fractions import Fraction
import json
import unittest

import numpy as np
import pandas as pd

import concord.core as ec
from _data import small_dataset


def _spec(frame, group_values, names, **kw):
    return ec.CompareSpec(group_column="APOE", group_values=tuple(group_values),
                          labels=("CN", "MCI", "AD"), biomarkers=tuple(names), **kw)


class RuleTests(unittest.TestCase):
    def test_exact_decision_matches_explicit_enumeration(self):
        # every completion of the unknown fits: reject iff (1 + final count) <= alpha * (B + 1)
        rng = np.random.default_rng(3)
        for _ in range(300):
            budget = int(rng.choice([7, 19, 39]))
            done = int(rng.integers(0, budget + 1))
            count = int(rng.integers(0, done + 1))
            alpha = Fraction(1, 20)
            for rule in ("le", "strict"):
                outcomes = set()
                for extra in range(budget - done + 1):
                    final = 1 + count + extra
                    ok = final * alpha.denominator <= (budget + 1) * alpha.numerator if rule == "le" \
                        else final * alpha.denominator < (budget + 1) * alpha.numerator
                    outcomes.add(ok)
                expected = True if outcomes == {True} else False if outcomes == {False} else None
                self.assertEqual(ec.exact_decision(count, done, budget, alpha, rule), expected)

    def test_frozen_b599_boundaries(self):
        # pair: reject iff G+E <= 9 at B=599, alpha/3, le; max: iff G+E <= 29
        self.assertTrue(ec.exact_decision(9, 599, 599, Fraction("0.05") / 3))
        self.assertFalse(ec.exact_decision(10, 599, 599, Fraction("0.05") / 3))
        self.assertTrue(ec.exact_decision(29, 599, 599, 0.05))
        self.assertFalse(ec.exact_decision(30, 599, 599, 0.05))
        # undetermined while unknown fits could still change the answer
        self.assertIsNone(ec.exact_decision(5, 300, 599, Fraction("0.05") / 3))
        self.assertFalse(ec.exact_decision(10, 300, 599, Fraction("0.05") / 3))
        self.assertTrue(ec.exact_decision(0, 590, 599, Fraction("0.05") / 3))


class OperatorTests(unittest.TestCase):
    def setUp(self):
        raw = small_dataset()
        self.frame, self.groups, self.names = ec.prepare_data(raw, "APOE", ("CN", "MCI", "AD"))
        self.spec = _spec(self.frame, self.groups, self.names)
        self.defs = ec.scheme_definitions(self.spec, ec.SCHEMES)

    def test_scheme_expansion(self):
        self.assertEqual(list(self.defs), ["unrestricted", "diagnosis", "diagnosis_pair_0-1",
                                           "diagnosis_pair_0-2", "diagnosis_pair_1-2"])
        two = ec.CompareSpec("APOE", (0, 1), ("CN", "MCI", "AD"), self.names)
        self.assertEqual(list(ec.scheme_definitions(two, ec.SCHEMES)), ["unrestricted", "diagnosis"])

    def test_stratified_permutations_preserve_strata_and_pair_leaves_third_fixed(self):
        f = self.frame
        table = lambda d: pd.crosstab(d["Diagnosis"], d["APOE"])
        for name, definition in self.defs.items():
            for index in range(3):
                permuted = ec.permute_labels(f, self.spec, name, definition, index)
                self.assertTrue((permuted.drop(columns="APOE") == f.drop(columns="APOE")).all().all())
                self.assertEqual(sorted(permuted["APOE"]), sorted(f["APOE"]))
                if definition["stratify"] == "diagnosis":
                    pd.testing.assert_frame_equal(table(permuted), table(f))
                fixed = [g for g in range(3) if g not in definition["groups"]]
                for g in fixed:
                    self.assertTrue(((permuted["APOE"] == g) == (f["APOE"] == g)).all())
        a = ec.permute_labels(f, self.spec, "diagnosis", self.defs["diagnosis"], 5)
        b = ec.permute_labels(f, self.spec, "diagnosis", self.defs["diagnosis"], 5)
        c = ec.permute_labels(f, self.spec, "diagnosis", self.defs["diagnosis"], 6)
        self.assertTrue((a["APOE"] == b["APOE"]).all())
        self.assertFalse((a["APOE"] == c["APOE"]).all())

    def test_resampling_keeps_composition(self):
        f = self.frame
        r = ec.resample_within_strata(f, self.spec, 1)
        pd.testing.assert_frame_equal(pd.crosstab(r["Diagnosis"], r["APOE"]),
                                      pd.crosstab(f["Diagnosis"], f["APOE"]))
        self.assertEqual(len(set(r["PTID"])), len(r))
        self.assertLess(len(set(map(tuple, r[list(self.names)].round(9).to_numpy()))), len(r))


class PreparationTests(unittest.TestCase):
    def test_rejects_unknown_diagnosis_reserved_names_and_missing_groups(self):
        raw = small_dataset()
        bad = raw.copy(); bad.loc[0, "Diagnosis"] = "SMC"
        with self.assertRaisesRegex(ValueError, "outside labels"):
            ec.prepare_data(bad, "APOE", ("CN", "MCI", "AD"))
        bad = raw.rename(columns={"b0": "PTID_score"})
        with self.assertRaisesRegex(ValueError, "reserved token"):
            ec.prepare_data(bad, "APOE", ("CN", "MCI", "AD"))
        bad = raw.copy(); bad.loc[3, "APOE"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values in group"):
            ec.prepare_data(bad, "APOE", ("CN", "MCI", "AD"))

    def test_group_encoding_and_order(self):
        raw = small_dataset()
        raw["APOE"] = raw["APOE"].map({0: "e2", 1: "e33", 2: "e4"})
        frame, groups, names = ec.prepare_data(raw, "APOE", ("CN", "MCI", "AD"), group_order=("e4", "e33", "e2"))
        self.assertEqual(groups, ("e4", "e33", "e2"))
        self.assertEqual(sorted(frame["APOE"].unique()), [0, 1, 2])
        self.assertTrue(((raw["APOE"] == "e4").to_numpy() == (frame["APOE"] == 0).to_numpy()).all())
        self.assertEqual(names, ("b0", "b1", "b2", "b3"))


class EndToEndTests(unittest.TestCase):
    """Real pyebm fits on the 72-subject toy set; identical group compositions by construction."""

    @classmethod
    def setUpClass(cls):
        from concord._pyebm import verify_pyebm
        verify_pyebm()
        cls.raw = small_dataset()

    def check_result(self, result, n_pairs=3):
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.groups, ["0", "1", "2"])
        for order in result.orderings.values():
            self.assertEqual(sorted(order), sorted(result.biomarkers))
        self.assertEqual(set(result.tests), {"unrestricted", "diagnosis", "diagnosis_pair_0-1",
                                             "diagnosis_pair_0-2", "diagnosis_pair_1-2"})
        for name, test in result.tests.items():
            self.assertEqual(test["completed"] + test["failed"], test["budget"])
            for cell in test["pairs"].values():
                self.assertTrue(0 < cell["p"] <= 1)
                self.assertIn(cell["reject"], (True, False))
            if test["definition"]["pair_index"] is None:
                self.assertTrue(0 < test["max"]["p"] <= 1)
            else:
                self.assertEqual(len(test["pairs"]), 1)
        self.assertEqual(result.composition["counts"]["0"], {"CN": 8, "MCI": 8, "AD": 8})
        self.assertIn("pairs", result.paired_difference)
        json.dumps(result.to_dict(), allow_nan=True)
        self.assertIn("orderings:", result.summary())

    def test_standard_estimator_inline(self):
        result = ec.compare(self.raw, B=3, stability_resamples=2, workers=1,
                                      estimator="standard", verbose=False)
        self.check_result(result)
        self.assertEqual(result.effective_sample_size, {"0": 24.0, "1": 24.0, "2": 24.0})
        self.assertEqual(result.stability["completed"], 2)
        self.assertEqual(result.fits, {"ok": 1 + 5 * 3 + 2, "error": 0, "timeout": 0})

    def test_invariant_equals_shared_under_identical_compositions(self):
        # weights are all one when the compositions coincide, so the min-reference estimator must
        # reproduce the pooled-mixture fit exactly, with ESS equal to the group size
        shared = ec.compare(self.raw, B=2, stability_resamples=0, workers=1,
                                      estimator="shared", schemes=("diagnosis",), verbose=False)
        invariant = ec.compare(self.raw, B=2, stability_resamples=0, workers=1,
                                         estimator="invariant_min", schemes=("diagnosis",), verbose=False)
        self.assertEqual(shared.orderings, invariant.orderings)
        self.assertEqual(shared.tests["diagnosis"]["pairs"], invariant.tests["diagnosis"]["pairs"])
        self.assertEqual(invariant.effective_sample_size, {"0": 24.0, "1": 24.0, "2": 24.0})
        self.assertEqual(shared.status, "ok")

    def test_pool_matches_inline(self):
        inline = ec.compare(self.raw, B=2, stability_resamples=0, workers=1,
                                      schemes=("diagnosis",), verbose=False)
        pooled = ec.compare(self.raw, B=2, stability_resamples=0, workers=2,
                                      schemes=("diagnosis",), verbose=False)
        self.assertEqual(inline.orderings, pooled.orderings)
        self.assertEqual(inline.tests["diagnosis"]["pairs"], pooled.tests["diagnosis"]["pairs"])
        self.assertEqual(pooled.timing["workers"], 2)

    def test_early_stopping_reports_bounds(self):
        # with B=3 and alpha/3 nothing can ever be rejected, so every reference is decided at once
        result = ec.compare(self.raw, B=3, stability_resamples=0, workers=1,
                                      schemes=("diagnosis",), stop_when_decided=True, verbose=False)
        test = result.tests["diagnosis"]
        self.assertEqual(test["completed"], 0)
        for cell in test["pairs"].values():
            self.assertIsNone(cell["p"])
            self.assertFalse(cell["reject"])
            self.assertEqual(cell["p_bounds"], [0.25, 1.0])


if __name__ == "__main__":
    unittest.main()
