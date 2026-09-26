"""Checks of the DEBM fits and the continued consensus search; run with pytest.

Requires the pinned pyebm wheel and its scientific dependencies. The tests use
pyebm's own ranking loss and a small complete DEBM fit on synthetic data.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from concord._pyebm import verify_pyebm
from concord.engine import (EngineConfig, _neighbors, _repaired_consensus,
                       _temporary_engine, fit_orderings, map_orderings,
                       validate_orderings)


from _data import small_dataset  # noqa: E402


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_pyebm()
        from pyebm.central_ordering.generalized_mallows import weighted_mallows
        from pyebm.inputs_outputs.prepare_outputs import Prob2ListAndWeights
        cls.mallows = weighted_mallows
        cls.data, cls.weights = Prob2ListAndWeights(
            np.tile([0.1, 0.3, 0.7, 0.9], (3, 1)))

    def test_actual_objective_descends_to_adjacent_local_optimum(self):
        original = self.mallows.consensus(4, self.data, self.weights, [0, 1, 2, 3])
        records = []
        repaired = _repaired_consensus(self.mallows, 4, self.data, self.weights,
                                        [0, 1, 2, 3], records=records)
        self.assertAlmostEqual(original[1], 2.4)
        self.assertLess(repaired[1], original[1])
        self.assertGreater(records[0]["accepted_swaps"], 1)
        neighbors = _neighbors(self.mallows, repaired[0], self.data, self.weights)
        self.assertTrue(all(item[0] >= repaired[1] - 1e-12 for item in neighbors))
        np.testing.assert_allclose(repaired[2], [item[0] for item in neighbors])
        self.assertTrue(records[0]["local_optimal"])

    def test_initialization_only_and_existing_optimum_are_unchanged(self):
        for initial, flag in (([0, 1, 2, 3], 1), ([3, 2, 1, 0], None)):
            upstream = self.mallows.consensus(4, self.data, self.weights, initial, flag)
            repaired = _repaired_consensus(self.mallows, 4, self.data, self.weights,
                                            initial, flag)
            self.assertEqual(upstream[0], repaired[0])
            np.testing.assert_allclose(upstream[1], repaired[1])
            np.testing.assert_allclose(upstream[2], repaired[2])

    def test_original_context_is_exact_and_restores_on_exception(self):
        from pyebm.mixture_model import gaussian_mixture_model as gmm
        descriptor, optimizer = self.mallows.__dict__["consensus"], gmm.opt
        before = self.mallows.consensus(4, self.data, self.weights, [0, 1, 2, 3])
        diagnostics = {}
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with _temporary_engine(EngineConfig(audit_original_neighbors=True), diagnostics):
                during = self.mallows.consensus(4, self.data, self.weights, [0, 1, 2, 3])
                self.assertEqual(before[0], during[0])
                for left, right in zip(before[1:], during[1:]):
                    np.testing.assert_equal(left, right)
                raise RuntimeError("intentional")
        self.assertIs(self.mallows.__dict__["consensus"], descriptor)
        self.assertIs(gmm.opt, optimizer)
        self.assertFalse(diagnostics["consensus"][0]["local_optimal"])

    def test_invalid_permutations_are_rejected(self):
        for values in ([0, 1, 1, 3], [0, 1, 2, 4], [0, 1, 2, np.nan],
                       [0, 1, 2, 2.5], [0, 1, 2]):
            with self.assertRaises(ValueError):
                validate_orderings([values], 4, 1)

    def test_ordering_indices_follow_requested_biomarker_names(self):
        mapped = map_orderings([[0, 1, 2]], ["b2", "b0", "b1"],
                               ["b0", "b1", "b2"], 1)
        np.testing.assert_array_equal(mapped[0], [2, 0, 1])
        with self.assertRaises(ValueError):
            map_orderings([[0, 1]], ["b0", "b1"], ["b0", "extra"], 1)

    def test_separate_fit_matches_pyebm_and_continued_search_is_isolated(self):
        from pyebm import debm
        frame = small_dataset()
        direct, _, _ = debm.fit(frame, Factors=[], Labels=["CN", "MCI", "AD"],
                                Groups=["APOE"])
        original = fit_orderings(frame, EngineConfig(expected_events=4))
        self.assertTrue(original.ok, original.diagnostics)
        np.testing.assert_array_equal(original.orderings, direct.MeanCentralOrdering)
        self.assertGreater(original.diagnostics["optimizer_calls"], 0)
        repaired = fit_orderings(frame, EngineConfig(mode="repaired", expected_events=4))
        self.assertTrue(repaired.ok, repaired.diagnostics)
        self.assertEqual(len(repaired.diagnostics["consensus"]), 3)
        for record in repaired.diagnostics["consensus"]:
            self.assertTrue(record["local_optimal"])
            self.assertLessEqual(record["final_score"], record["initial_score"])
        # A continued-search call cannot leave a patched optimizer or consensus behind.
        original_again = fit_orderings(frame, EngineConfig(expected_events=4))
        self.assertTrue(original_again.ok, original_again.diagnostics)
        np.testing.assert_array_equal(original_again.orderings, original.orderings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
