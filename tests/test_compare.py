"""Checks for concord.compare; run with pytest.

The end-to-end tests need the pinned pyebm wheel (small fits, seconds each). The rule test
enumerates every completion of the permutation budget explicitly.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import concord
import concord.core as ec
from _data import small_dataset

LABELS = ("CN", "MCI", "AD")


def _spec(group_values, names, **kw):
    return ec.CompareSpec(group_column="APOE", group_values=tuple(group_values),
                          labels=LABELS, biomarkers=tuple(names), **kw)


def _without(raw, group, diagnosis):
    """The toy data set without the participants of one diagnosis in one group."""
    return raw[~((raw["APOE"] == group) & (raw["Diagnosis"] == diagnosis))].reset_index(drop=True)


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

    def test_b599_boundaries(self):
        # three comparisons: reject iff E <= 9 at B=599, alpha/3, le; two groups (alpha/1): iff E <= 29
        self.assertTrue(ec.exact_decision(9, 599, 599, Fraction("0.05") / 3))
        self.assertFalse(ec.exact_decision(10, 599, 599, Fraction("0.05") / 3))
        self.assertTrue(ec.exact_decision(29, 599, 599, 0.05))
        self.assertFalse(ec.exact_decision(30, 599, 599, 0.05))
        # undetermined while unknown fits could still change the answer
        self.assertIsNone(ec.exact_decision(5, 300, 599, Fraction("0.05") / 3))
        self.assertFalse(ec.exact_decision(10, 300, 599, Fraction("0.05") / 3))
        self.assertTrue(ec.exact_decision(0, 590, 599, Fraction("0.05") / 3))


class NameTests(unittest.TestCase):
    names = ("b0", "b1", "b2", "b3")

    def test_estimator_and_search_aliases(self):
        for old, new in (("invariant_min", "concord"), ("shared", "pooled_score"), ("standard", "separate"),
                         ("concord", "concord"), ("pooled_score", "pooled_score"), ("separate", "separate")):
            self.assertEqual(_spec((0, 1, 2), self.names, estimator=old).estimator, new)
        self.assertEqual(_spec((0, 1, 2), self.names).estimator, "concord")
        self.assertEqual(_spec((0, 1, 2), self.names).search, "continued")
        self.assertEqual(_spec((0, 1, 2), self.names, search="pyebm").search, "pyebm")
        self.assertEqual(_spec((0, 1, 2), self.names, consensus="repaired").search, "continued")
        self.assertEqual(_spec((0, 1, 2), self.names, consensus="original").search, "pyebm")
        self.assertEqual(_spec((0, 1, 2), self.names, search="original").search, "pyebm")
        with self.assertRaisesRegex(ValueError, "unknown estimator"):
            _spec((0, 1, 2), self.names, estimator="kde_ebm")
        with self.assertRaisesRegex(ValueError, "unknown search"):
            _spec((0, 1, 2), self.names, search="greedy")
        self.assertEqual(ec.SCHEMES, ("unrestricted", "within_diagnosis", "pairwise_within_diagnosis"))
        self.assertEqual(concord.ESTIMATORS, ("concord", "pooled_score", "separate", "saebm"))

    def test_saebm_defaults(self):
        spec = _spec((0, 1, 2), self.names, estimator="saebm")
        self.assertEqual((spec.saebm_iterations, spec.saebm_burn_in), (10000, 2500))
        self.assertIsNone(spec.common_proportions)


class CommonProportionTests(unittest.TestCase):
    names = ("b0", "b1", "b2", "b3")

    def test_spec_rules_and_validation(self):
        self.assertEqual(_spec((0, 1), self.names).common_proportions, "min")
        self.assertEqual(_spec((0, 1), self.names, common_proportions="min").common_proportions, "min")
        self.assertIsNone(_spec((0, 1), self.names, estimator="pooled_score").common_proportions)
        fixed = _spec((0, 1), self.names, common_proportions={"CN": 0.5, "MCI": 0.25, "AD": 0.25})
        self.assertEqual(fixed.common_proportions, (0.5, 0.25, 0.25))
        self.assertEqual(_spec((0, 1), self.names, common_proportions=[0.5, 0.25, 0.25]).common_proportions,
                         (0.5, 0.25, 0.25))
        self.assertEqual(_spec((0, 1), self.names, common_proportions={"CN": 0.6, "AD": 0.4}).common_proportions,
                         (0.6, 0.0, 0.4))
        rounded = _spec((0, 1), self.names, common_proportions=(0.4893, 0.2859, 0.2249)).common_proportions
        self.assertAlmostEqual(sum(rounded), 1.0, places=12)
        for bad in ((0.5, 0.5), (0.6, 0.6, -0.2), (48.9, 28.6, 22.5), (0.5, 0.25, 0.2), {"CN": 0.5, "SMC": 0.5},
                    (0.5, float("nan"), 0.5), "pooled"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                _spec((0, 1), self.names, common_proportions=bad)
        for estimator in ("pooled_score", "separate", "saebm"):
            with self.assertRaisesRegex(ValueError, "concord"):
                _spec((0, 1), self.names, estimator=estimator, common_proportions=(0.5, 0.25, 0.25))

    def test_helper_minimum_rule(self):
        # the simulation design counts (CN-heavy, intermediate, AD-heavy): 48.6/17.1/34.3%, largest weight 2.14
        counts = np.array([[57, 6, 12], [244, 66, 101], [110, 156, 219]], float)
        props = concord.common_proportions(*counts.tolist())
        self.assertEqual(list(props), list(LABELS))
        np.testing.assert_allclose(list(props.values()), [0.4859, 0.1714, 0.3428], atol=1e-4)
        largest = (np.array(list(props.values())) / (counts / counts.sum(axis=1, keepdims=True))).max()
        self.assertAlmostEqual(largest, 2.1422, places=4)
        self.assertAlmostEqual(largest, 1 / (counts / counts.sum(axis=1, keepdims=True)).min(axis=0).sum())
        # the same result from a count table, from mappings, from a 2-D array and from frames
        table = pd.DataFrame(counts, columns=list(LABELS), index=["g1", "g2", "g3"])
        self.assertEqual(concord.common_proportions(table), props)
        self.assertEqual(concord.common_proportions(counts), props)
        self.assertEqual(concord.common_proportions(*[dict(zip(LABELS, row)) for row in counts]), props)
        frame = pd.DataFrame([{"Diagnosis": d, "APOE": g} for g, row in enumerate(counts)
                              for d, n in zip(LABELS, row) for _ in range(int(n))])
        self.assertEqual(concord.common_proportions(frame, group_column="APOE"), props)
        # applied jointly across separate data sets (e.g. two cohorts): same as one set of all their groups
        first, second = frame[frame["APOE"] < 2], frame[frame["APOE"] == 2]
        self.assertEqual(concord.common_proportions(first, second, group_column="APOE"), props)
        # one frame without group_column is one group: its own proportions
        own = concord.common_proportions(frame)
        np.testing.assert_allclose(list(own.values()), counts.sum(axis=0) / counts.sum())

    def test_helper_validation(self):
        props = concord.common_proportions([10, 0, 10], [5, 5, 10])
        self.assertEqual(props["MCI"], 0.0)
        with self.assertRaisesRegex(ValueError, "no diagnosis is present in every group"):
            concord.common_proportions([10, 0, 0], [0, 10, 0])
        with self.assertRaisesRegex(ValueError, "at least one participant"):
            concord.common_proportions([10, 5, 5], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "outside labels"):
            concord.common_proportions(pd.DataFrame({"Diagnosis": ["CN", "SMC"]}))
        with self.assertRaisesRegex(ValueError, "one value per"):
            concord.common_proportions([10, 5])
        with self.assertRaises(ValueError):
            concord.common_proportions()


class OperatorTests(unittest.TestCase):
    def setUp(self):
        raw = small_dataset()
        self.frame, self.groups, self.names = ec.prepare_data(raw, "APOE", LABELS)
        self.spec = _spec(self.groups, self.names)
        self.defs = ec.scheme_definitions(self.spec, ec.SCHEMES)

    def test_scheme_expansion(self):
        self.assertEqual(list(self.defs), ["unrestricted", "within_diagnosis", "pairwise_within_diagnosis_0-1",
                                           "pairwise_within_diagnosis_0-2", "pairwise_within_diagnosis_1-2"])
        self.assertEqual(self.defs["pairwise_within_diagnosis_0-2"]["groups"], [0, 2])
        # earlier names give the same definitions
        self.assertEqual(ec.scheme_definitions(self.spec, ("unrestricted", "diagnosis", "diagnosis_pair")), self.defs)
        two = ec.CompareSpec("APOE", (0, 1), LABELS, self.names)
        self.assertEqual(list(ec.scheme_definitions(two, ec.SCHEMES)), ["unrestricted", "within_diagnosis"])
        self.assertEqual(list(ec.scheme_definitions(two, ("pairwise_within_diagnosis",))), ["within_diagnosis"])
        for removed in ("diagnosis_count", "stratified"):
            with self.assertRaisesRegex(ValueError, "unknown permutation scheme"):
                ec.scheme_definitions(self.spec, (removed,))

    def test_default_schemes(self):
        # schemes=None: pairwise within-diagnosis permutation for three or more groups, within-diagnosis
        # permutation for two; unrestricted permutation only when requested
        self.assertEqual(self.spec.default_schemes, ("pairwise_within_diagnosis",))
        self.assertEqual(list(ec.scheme_definitions(self.spec, None)),
                         ["pairwise_within_diagnosis_0-1", "pairwise_within_diagnosis_0-2",
                          "pairwise_within_diagnosis_1-2"])
        two = ec.CompareSpec("APOE", (0, 1), LABELS, self.names)
        self.assertEqual(two.default_schemes, ("within_diagnosis",))
        self.assertEqual(list(ec.scheme_definitions(two, None)), ["within_diagnosis"])
        four = ec.CompareSpec("APOE", (0, 1, 2, 3), LABELS, self.names)
        self.assertEqual(len(ec.scheme_definitions(four, None)), 6)
        import inspect
        self.assertIsNone(inspect.signature(ec.compare).parameters["schemes"].default)

    def test_relabelings_are_those_of_version_01(self):
        # the random streams keep their earlier names, so a seed gives the same relabelings as before
        for name, old_name in (("within_diagnosis", "diagnosis"),
                               ("pairwise_within_diagnosis_0-2", "diagnosis_pair_0-2")):
            old_definition = {k: v for k, v in self.defs[name].items() if k != "stream"}
            for index in range(3):
                new = ec.permute_labels(self.frame, self.spec, name, self.defs[name], index)
                old = ec.permute_labels(self.frame, self.spec, old_name, old_definition, index)
                self.assertTrue((new["APOE"] == old["APOE"]).all())

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
        a = ec.permute_labels(f, self.spec, "within_diagnosis", self.defs["within_diagnosis"], 5)
        b = ec.permute_labels(f, self.spec, "within_diagnosis", self.defs["within_diagnosis"], 5)
        c = ec.permute_labels(f, self.spec, "within_diagnosis", self.defs["within_diagnosis"], 6)
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
            ec.prepare_data(bad, "APOE", LABELS)
        bad = raw.rename(columns={"b0": "PTID_score"})
        with self.assertRaisesRegex(ValueError, "reserved token"):
            ec.prepare_data(bad, "APOE", LABELS)
        bad = raw.copy(); bad.loc[3, "APOE"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values in group"):
            ec.prepare_data(bad, "APOE", LABELS)

    def test_group_encoding_and_order(self):
        raw = small_dataset()
        raw["APOE"] = raw["APOE"].map({0: "e2", 1: "e33", 2: "e4"})
        frame, groups, names = ec.prepare_data(raw, "APOE", LABELS, group_order=("e4", "e33", "e2"))
        self.assertEqual(groups, ("e4", "e33", "e2"))
        self.assertEqual(sorted(frame["APOE"].unique()), [0, 1, 2])
        self.assertTrue(((raw["APOE"] == "e4").to_numpy() == (frame["APOE"] == 0).to_numpy()).all())
        self.assertEqual(names, ("b0", "b1", "b2", "b3"))

    def test_command_line_common_proportions(self):
        self.assertEqual(ec._parse_common_proportions("min"), "min")
        self.assertIsNone(ec._parse_common_proportions(None))
        self.assertEqual(ec._parse_common_proportions("0.5,0.25,0.25"), [0.5, 0.25, 0.25])
        self.assertEqual(ec._parse_common_proportions("CN=0.5, MCI=0.25, AD=0.25"),
                         {"CN": 0.5, "MCI": 0.25, "AD": 0.25})


class EndToEndTests(unittest.TestCase):
    """Real pyebm fits on the 72-participant toy set; identical group compositions by construction."""

    @classmethod
    def setUpClass(cls):
        from concord._pyebm import verify_pyebm
        verify_pyebm()
        cls.raw = small_dataset()

    def check_result(self, result):
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.groups, ["0", "1", "2"])
        for order in result.orderings.values():
            self.assertEqual(sorted(order), sorted(result.biomarkers))
        self.assertEqual(set(result.tests), {"unrestricted", "within_diagnosis", "pairwise_within_diagnosis_0-1",
                                             "pairwise_within_diagnosis_0-2", "pairwise_within_diagnosis_1-2"})
        for name, test in result.tests.items():
            self.assertNotIn("max", test)
            self.assertEqual(test["completed"] + test["failed"], test["budget"])
            for cell in test["pairs"].values():
                self.assertTrue(0 < cell["p"] <= 1)
                self.assertEqual(cell["p_adjusted"], min(1.0, 3 * cell["p"]))
                self.assertEqual(cell["p_adjusted_bounds"], [min(1.0, 3 * b) for b in cell["p_bounds"]])
                self.assertIn(cell["reject"], (True, False))
            if test["definition"]["pair_index"] is not None:
                self.assertEqual(len(test["pairs"]), 1)
                self.assertEqual(test["scheme"], "pairwise_within_diagnosis")
        # decisions are the pairwise Bonferroni decisions only
        self.assertEqual(set(result.decisions), {"rule", "alpha", "n_pairs", "per_pair_threshold", "convention",
                                                 "reject", "reject_any_pair", "stop_when_decided"})
        self.assertEqual(result.decisions["n_pairs"], 3)
        self.assertEqual(set(result.decisions["reject"]), set(ec.SCHEMES))
        for scheme in ec.SCHEMES:
            self.assertEqual(set(result.decisions["reject"][scheme]), {"0-1", "0-2", "1-2"})
        self.assertEqual(set(result.distances), {"pairs", "statistic"})
        self.assertEqual(result.composition["counts"]["0"], {"CN": 8, "MCI": 8, "AD": 8})
        self.assertIn("pairs", result.paired_difference)
        json.dumps(result.to_dict(), allow_nan=True)
        text = result.summary()
        self.assertIn("orderings", text)
        self.assertIn("adjusted P=", text)
        self.assertIn("largest weight", text)
        self.assertIn("note: unrestricted permutation", text)
        self.assertNotIn("max", text)

    def test_separate_estimator_inline(self):
        result = ec.compare(self.raw, B=3, stability_resamples=2, workers=1, schemes=ec.SCHEMES,
                            estimator="separate", verbose=False)
        self.check_result(result)
        self.assertEqual(result.spec["estimator"], "separate")
        self.assertEqual(result.effective_sample_size, {"0": 24.0, "1": 24.0, "2": 24.0})
        self.assertIsNone(result.common_proportions)
        self.assertEqual(result.weights["1"], {"CN": 1.0, "MCI": 1.0, "AD": 1.0})
        self.assertEqual(result.largest_weight, 1.0)
        self.assertEqual(result.stability["completed"], 2)
        self.assertEqual(result.fits, {"ok": 1 + 5 * 3 + 2, "error": 0, "timeout": 0})

    def test_default_schemes_end_to_end(self):
        result = ec.compare(self.raw, B=1, stability_resamples=0, workers=1, verbose=False)
        self.assertEqual(list(result.tests), ["pairwise_within_diagnosis_0-1", "pairwise_within_diagnosis_0-2",
                                              "pairwise_within_diagnosis_1-2"])
        self.assertEqual(result.spec["schemes"], list(result.tests))
        self.assertEqual(set(result.decisions["reject"]), {"pairwise_within_diagnosis"})
        self.assertEqual(result.paired_difference["pairs"], {})
        self.assertNotIn("unrestricted", result.summary())
        self.assertEqual(result.fits["ok"] + result.fits["error"] + result.fits["timeout"], 1 + 3 * 1)

    def test_concord_equals_pooled_score_under_identical_compositions(self):
        # weights are all one when the compositions coincide, so CONCORD (minimum rule, or fixed
        # proportions equal to the common composition) reproduces pooled-score DEBM exactly
        kw = dict(B=2, stability_resamples=0, workers=1, schemes=("within_diagnosis",), verbose=False)
        pooled = ec.compare(self.raw, estimator="pooled_score", **kw)
        weighted = ec.compare(self.raw, estimator="concord", **kw)
        fixed = ec.compare(self.raw, estimator="concord", common_proportions=(1 / 3, 1 / 3, 1 / 3), **kw)
        for result in (weighted, fixed):
            self.assertEqual(pooled.orderings, result.orderings)
            self.assertEqual(pooled.tests["within_diagnosis"]["pairs"], result.tests["within_diagnosis"]["pairs"])
            self.assertEqual(result.effective_sample_size, {"0": 24.0, "1": 24.0, "2": 24.0})
            for value in result.common_proportions.values():
                self.assertAlmostEqual(value, 1 / 3)
            self.assertAlmostEqual(result.largest_weight, 1.0)
        self.assertEqual(weighted.spec["common_proportions"], "min")
        self.assertEqual(fixed.spec["common_proportions"], (1 / 3, 1 / 3, 1 / 3))
        self.assertIsNone(pooled.common_proportions)

    def test_earlier_names_give_identical_results(self):
        kw = dict(B=2, stability_resamples=0, workers=1, verbose=False)
        new = ec.compare(self.raw, estimator="concord", search="continued", schemes=("within_diagnosis",), **kw)
        old = ec.compare(self.raw, estimator="invariant_min", consensus="repaired", schemes=("diagnosis",), **kw)
        self.assertEqual(old.spec["estimator"], "concord")
        self.assertEqual(old.spec["search"], "continued")
        self.assertEqual(list(old.tests), ["within_diagnosis"])
        self.assertEqual(new.orderings, old.orderings)
        self.assertEqual(new.tests["within_diagnosis"]["pairs"], old.tests["within_diagnosis"]["pairs"])
        self.assertEqual(new.tests["within_diagnosis"]["definition"], old.tests["within_diagnosis"]["definition"])

    def test_fixed_common_proportions_weights_and_support(self):
        result = ec.compare(self.raw, B=2, stability_resamples=0, workers=1, schemes=("unrestricted",),
                            common_proportions={"CN": 0.5, "MCI": 0.25, "AD": 0.25}, verbose=False)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.common_proportions, {"CN": 0.5, "MCI": 0.25, "AD": 0.25})
        for group in result.groups:
            self.assertEqual(result.weights[group], {"CN": 1.5, "MCI": 0.75, "AD": 0.75})
            # Kish: n / sum_d pi(d)^2 / pihat(d) = 24 / 1.125
            self.assertAlmostEqual(result.effective_sample_size[group], 24 / 1.125)
        self.assertEqual(result.largest_weight, 1.5)
        self.assertIn("common proportions (fixed)", result.summary())
        # a diagnosis with a positive proportion must be present in every group
        lacking = _without(self.raw, 2, "MCI")
        with self.assertRaisesRegex(ValueError, "MCI has a positive common proportion but no participants in group 2"):
            ec.compare(lacking, B=0, stability_resamples=0, common_proportions=(0.4, 0.3, 0.3), verbose=False)
        zero = ec.compare(lacking, B=0, stability_resamples=0, schemes=("within_diagnosis",),
                          common_proportions=(0.5, 0.0, 0.5), verbose=False)
        self.assertEqual(zero.status, "ok")
        self.assertIsNone(zero.weights["2"]["MCI"])
        self.assertEqual(zero.weights["0"], {"CN": 1.5, "MCI": 0.0, "AD": 1.5})
        # the minimum rule gives MCI proportion zero when one group has no MCI participants
        minimum = ec.compare(lacking, B=0, stability_resamples=0, schemes=("within_diagnosis",), verbose=False)
        self.assertEqual(minimum.common_proportions, concord.common_proportions(lacking, group_column="APOE"))
        self.assertEqual(minimum.common_proportions["MCI"], 0.0)

    def test_fixed_proportions_held_fixed_after_unrestricted_relabeling(self):
        frame, groups, names = ec.prepare_data(self.raw, "APOE", LABELS)
        minimum_spec = _spec(groups, names)
        fixed_spec = _spec(groups, names, common_proportions=(0.5, 0.25, 0.25))
        definition = ec.scheme_definitions(minimum_spec, ("unrestricted",))["unrestricted"]
        relabeled = ec.permute_labels(frame, minimum_spec, "unrestricted", definition, 0)
        counts = pd.crosstab(relabeled["APOE"], relabeled["Diagnosis"])[list(LABELS)]
        self.assertGreater(counts.to_numpy().std(), 0)      # the relabeled groups differ in composition
        minimum = ec.fit_once(relabeled, minimum_spec)
        fixed = ec.fit_once(relabeled, fixed_spec)
        self.assertTrue(minimum.ok and fixed.ok)
        expected = concord.common_proportions(relabeled, group_column="APOE")
        np.testing.assert_allclose(minimum.diagnostics["common_proportions"], [expected[d] for d in LABELS])
        np.testing.assert_allclose(fixed.diagnostics["common_proportions"], [0.5, 0.25, 0.25])
        self.assertEqual(fixed.diagnostics["common_proportions_rule"], "fixed")
        share = counts.div(counts.sum(axis=1), axis=0)
        for g in range(3):
            np.testing.assert_allclose(fixed.diagnostics["group_weights"][str(g)],
                                       [0.5 / share.loc[g, "CN"], 0.25 / share.loc[g, "MCI"],
                                        0.25 / share.loc[g, "AD"]])

    def test_two_groups(self):
        two = self.raw[self.raw["APOE"] < 2].reset_index(drop=True)
        result = ec.compare(two, B=2, stability_resamples=0, workers=1, schemes=("pairwise_within_diagnosis",),
                            verbose=False)
        self.assertEqual(list(result.tests), ["within_diagnosis"])
        self.assertEqual(result.decisions["n_pairs"], 1)
        cell = result.tests["within_diagnosis"]["pairs"]["0-1"]
        self.assertEqual(cell["p_adjusted"], cell["p"])

    def test_pool_matches_inline(self):
        kw = dict(B=2, stability_resamples=0, schemes=("within_diagnosis",), verbose=False)
        inline = ec.compare(self.raw, workers=1, **kw)
        pooled = ec.compare(self.raw, workers=2, **kw)
        self.assertEqual(inline.orderings, pooled.orderings)
        self.assertEqual(inline.tests["within_diagnosis"]["pairs"], pooled.tests["within_diagnosis"]["pairs"])
        self.assertEqual(pooled.timing["workers"], 2)

    def test_early_stopping_reports_bounds(self):
        # with B=3 and alpha/3 nothing can ever be rejected, so every scheme is decided at once
        result = ec.compare(self.raw, B=3, stability_resamples=0, workers=1,
                            schemes=("within_diagnosis",), stop_when_decided=True, verbose=False)
        test = result.tests["within_diagnosis"]
        self.assertEqual(test["completed"], 0)
        for cell in test["pairs"].values():
            self.assertIsNone(cell["p"])
            self.assertIsNone(cell["p_adjusted"])
            self.assertFalse(cell["reject"])
            self.assertEqual(cell["p_bounds"], [0.25, 1.0])
            self.assertEqual(cell["p_adjusted_bounds"], [0.75, 1.0])
        self.assertIn("P in [0.250, 1.000]", result.summary())

    def test_early_stopping_ignores_the_maximum_distance(self):
        # B=39: no comparison can reach P <= 0.05/3, so the pairwise decisions are fixed before any
        # relabeling; a maximum-distance test at 0.05 (E <= 1) would still be open, and must not matter
        result = ec.compare(self.raw, B=39, stability_resamples=0, workers=1,
                            schemes=("within_diagnosis",), stop_when_decided=True, verbose=False)
        self.assertEqual(result.tests["within_diagnosis"]["completed"], 0)
        self.assertEqual(result.decisions["reject"]["within_diagnosis"], {"0-1": False, "0-2": False, "1-2": False})

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.csv", Path(tmp) / "result.json"
            self.raw.to_csv(source, index=False)
            code = ec.main([str(source), "--B", "0", "--stability", "0", "--schemes", "within_diagnosis",
                            "--common-proportions", "CN=0.5,MCI=0.25,AD=0.25", "--output", str(output), "--quiet"])
            self.assertEqual(code, 0)
            saved = json.loads(output.read_text())
        self.assertEqual(saved["common_proportions"], {"CN": 0.5, "MCI": 0.25, "AD": 0.25})
        self.assertEqual(saved["largest_weight"], 1.5)
        self.assertEqual(saved["spec"]["estimator"], "concord")

    def test_command_line_default_schemes(self):
        two = self.raw[self.raw["APOE"] < 2].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            for data, expected in ((self.raw, ["pairwise_within_diagnosis_0-1", "pairwise_within_diagnosis_0-2",
                                               "pairwise_within_diagnosis_1-2"]),
                                   (two, ["within_diagnosis"])):
                source, output = Path(tmp) / "input.csv", Path(tmp) / "result.json"
                data.to_csv(source, index=False)
                code = ec.main([str(source), "--B", "0", "--stability", "0", "--output", str(output), "--quiet"])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output.read_text())["spec"]["schemes"], expected)

    def test_command_line_rejects_invalid_options_before_reading_the_input(self):
        import contextlib
        import io
        missing = str(Path(tempfile.gettempdir()) / "concord-test-input-that-does-not-exist.csv")
        for argv in (["--estimator", "separate", "--common-proportions", "0.5,0.25,0.25"],
                     ["--estimator", "pooled_score", "--common-proportions", "min"],
                     ["--common-proportions", "48.9,28.6,22.5"],
                     ["--common-proportions", "CN=0.5,SMC=0.5"],
                     ["--common-proportions", "a,b,c"],
                     ["--schemes", "diagnosis_count"]):
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(stderr):
                ec.main([missing, *argv])
            self.assertEqual(caught.exception.code, 2, argv)
            self.assertIn("error:", stderr.getvalue())

    def test_compare_inside_a_worker_process(self):
        # workers=1 starts no processes, so compare() may run inside a worker; workers > 1 there is refused
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        from _data import compare_in_worker
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            self.assertEqual(pool.submit(compare_in_worker, 1).result(), "ok")
            self.assertTrue(pool.submit(compare_in_worker, 2).result().startswith("RuntimeError"))


class FixedProportionFitTests(unittest.TestCase):
    def setUp(self):
        from concord.engine import EngineConfig
        self.frame, groups, self.names = ec.prepare_data(small_dataset(), "APOE", LABELS)
        self.config = EngineConfig(mode="repaired", expected_events=4, group_column="APOE", group_values=(0, 1, 2),
                                   labels=LABELS, biomarker_names=tuple(self.names))

    def test_explicit_proportions_equal_named_rule(self):
        # fixed proportions equal to the pooled mix must reproduce invariant_pooled exactly
        from concord.invariant import fit_invariant_orderings
        pooled = [(self.frame["Diagnosis"] == d).mean() for d in LABELS]
        a = fit_invariant_orderings(self.frame, self.config, variant="invariant_pooled", use_cache=False)
        b = fit_invariant_orderings(self.frame, self.config, variant="invariant_min", reference=pooled, use_cache=False)
        self.assertEqual([o.tolist() for o in a.orderings], [o.tolist() for o in b.orderings])
        np.testing.assert_allclose(b.diagnostics["common_proportions"], pooled)
        self.assertEqual(b.diagnostics["common_proportions_rule"], "fixed")
        self.assertEqual(a.diagnostics["common_proportions_rule"], "pooled")
        with self.assertRaises(ValueError):
            fit_invariant_orderings(self.frame, self.config, reference=(1, 0), use_cache=False)

    def test_support_check_inside_the_fit(self):
        from concord.invariant import fit_invariant_orderings
        lacking = _without(self.frame, 2, "MCI")
        result = fit_invariant_orderings(lacking, self.config, reference=(0.4, 0.3, 0.3), use_cache=False)
        self.assertEqual(result.status, "error")
        self.assertIn("MCI has a positive common proportion but no participants in group 2", result.diagnostics["error"])
        allowed = fit_invariant_orderings(lacking, self.config, reference=(0.5, 0.0, 0.5), use_cache=False)
        self.assertTrue(allowed.ok, allowed.diagnostics.get("error"))
        self.assertIsNone(allowed.diagnostics["group_weights"]["2"][1])


class PooledMixtureCacheTests(unittest.TestCase):
    """The pooled mixture depends on the measurements and the diagnoses, not on the group labels."""

    def setUp(self):
        from concord import invariant
        from concord.engine import EngineConfig
        self.invariant = invariant
        invariant._CACHE.clear()
        self.frame, groups, self.names = ec.prepare_data(small_dataset(), "APOE", LABELS)
        self.config = EngineConfig(mode="repaired", expected_events=4, group_column="APOE", group_values=(0, 1, 2),
                                   labels=LABELS, biomarker_names=tuple(self.names))

    def tearDown(self):
        self.invariant._CACHE.clear()

    def test_same_measurements_different_diagnoses_do_not_reuse_the_mixture(self):
        fit = self.invariant.fit_invariant_orderings
        first = fit(self.frame, self.config)
        self.assertTrue(first.ok)
        self.assertFalse(first.diagnostics["pooled_mixture_cached"])
        # a relabeling changes only the group column: the mixture is reused
        spec = _spec((0, 1, 2), self.names)
        definition = ec.scheme_definitions(spec, ("within_diagnosis",))["within_diagnosis"]
        relabeled = ec.permute_labels(self.frame, spec, "within_diagnosis", definition, 0)
        self.assertTrue(fit(relabeled, self.config).diagnostics["pooled_mixture_cached"])
        # the same measurements with other diagnoses: CN and AD participants initialize the mixture,
        # so it must be fitted again, and the result must equal a fit without the cache
        changed = self.frame.copy()
        cn = changed.index[changed["Diagnosis"] == "CN"][:4]
        ad = changed.index[changed["Diagnosis"] == "AD"][:4]
        changed.loc[cn, "Diagnosis"] = "AD"
        changed.loc[ad, "Diagnosis"] = "CN"
        self.assertTrue((changed[list(self.names)] == self.frame[list(self.names)]).all().all())
        self.assertNotEqual(self.invariant._matrix_key(changed, self.names, LABELS),
                            self.invariant._matrix_key(self.frame, self.names, LABELS))
        second = fit(changed, self.config)
        self.assertTrue(second.ok)
        self.assertFalse(second.diagnostics["pooled_mixture_cached"])
        uncached = fit(changed, self.config, use_cache=False)
        self.assertEqual([o.tolist() for o in second.orderings], [o.tolist() for o in uncached.orderings])
        # the original data still find their own mixture
        self.assertTrue(fit(self.frame, self.config).diagnostics["pooled_mixture_cached"])


if __name__ == "__main__":
    unittest.main()
