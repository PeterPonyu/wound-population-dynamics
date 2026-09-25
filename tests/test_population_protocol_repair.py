"""Behavioral checks for information exclusion and a common scoring geometry."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "population_protocol_repair", Path(__file__).resolve().parents[1] / "scripts/repair_population_protocol.py")
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


class PopulationProtocolRepairTests(unittest.TestCase):
    def test_disallowed_reference_or_test_cells_cannot_change_training_scaler(self):
        values = np.array([[1., 2.], [3., 6.], [11., 3.], [19., 4.], [100., -30.]], np.float32)
        permitted = np.array([True, True, False, False, False])
        mean, scale = REPAIR.fit_scaler(values, permitted)
        before = REPAIR.transform(values, mean, scale)[permitted]
        values[~permitted] *= 1e5
        after_mean, after_scale = REPAIR.fit_scaler(values, permitted)
        np.testing.assert_array_equal(mean, after_mean)
        np.testing.assert_array_equal(scale, after_scale)
        np.testing.assert_array_equal(before, REPAIR.transform(values, after_mean, after_scale)[permitted])

    def test_identical_raw_prediction_has_identical_common_score_from_different_training_scalers(self):
        prediction = np.array([[1., 8.], [4., -3.], [2., 6.]])
        target = np.array([[2., 5.], [3., 2.]])
        reference_mean, reference_scale = np.array([[2., 1.]]), np.array([[3., 5.]])
        results = []
        for mean, scale in [(np.array([[0., 0.]]), np.array([[1., 1.]])),
                            (np.array([[30., -7.]]), np.array([[.5, 12.]]))]:
            train_z = (prediction-mean)/scale
            raw, reference = REPAIR.reference_prediction(train_z, mean, scale, reference_mean, reference_scale)
            np.testing.assert_allclose(raw, prediction)
            results.append(REPAIR.weighted_energy(reference, (target-reference_mean)/reference_scale))
        self.assertAlmostEqual(results[0], results[1], places=12)

    def test_marginal_has_exact_equal_donor_mass_with_odd_particle_count(self):
        endpoints = {"donor_a": np.arange(10), "donor_b": np.arange(20, 37)}
        for draw in (0, 1, 29):
            rows, weights, counts = REPAIR.marginal_draw(endpoints, 7, draw)
            self.assertEqual(len(rows), 7)
            self.assertEqual(len(np.unique(rows)), 7)
            self.assertAlmostEqual(weights[rows < 10].sum(), .5)
            self.assertAlmostEqual(weights[rows >= 20].sum(), .5)
            self.assertEqual(sum(counts.values()), 7)
            self.assertTrue(set(rows) <= set(np.r_[endpoints['donor_a'], endpoints['donor_b']]))
        a = REPAIR.marginal_draw(endpoints, 7, 0)[2]
        b = REPAIR.marginal_draw(endpoints, 7, 1)[2]
        self.assertEqual(a["donor_a"], b["donor_b"])

    def test_exact_target_marginal_is_unchanged_by_replication_within_one_donor(self):
        cells = np.array([[0.], [2.], [8.], [12.]])
        endpoints = {"a": np.array([0, 1]), "b": np.array([2, 3])}
        rows, weights = REPAIR.exact_marginal(endpoints)
        original = REPAIR.weighted_energy(cells[rows], np.array([[5.]]), weights)
        replicated = np.vstack([cells[:2], cells[2:], cells[2:]])
        rows, weights = REPAIR.exact_marginal({"a": np.array([0, 1]), "b": np.array([2, 3, 4, 5])})
        self.assertAlmostEqual(original, REPAIR.weighted_energy(replicated[rows], np.array([[5.]]), weights))

    def test_target_indices_preserve_original_benchmark_samples(self):
        np.testing.assert_array_equal(REPAIR.original_target_indices(920), np.arange(920))
        np.testing.assert_array_equal(REPAIR.original_target_indices(1992),
                                      np.random.default_rng(0).choice(1992, 1200, replace=False))


if __name__ == "__main__":
    unittest.main()
