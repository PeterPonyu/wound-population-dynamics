"""Analytic and exclusion tests for the saved-prediction mixture audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("population_mixture", ROOT / "scripts/audit_population_mixtures.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PopulationMixtureAuditTests(unittest.TestCase):
    def test_analytic_point_mass_identity_is_target_independent(self):
        # ED(delta_0, delta_2)=4, so the gain must be exactly one for any Q.
        for target in (np.array([[1.]]), np.array([[-5.], [2.], [13.]])):
            result = AUDIT.mixture_decomposition(np.array([[0.]]), np.array([[2.]]), target)
            self.assertAlmostEqual(result["component_pair_ed"], 4.)
            self.assertAlmostEqual(result["convexity_gain"], 1.)
            self.assertAlmostEqual(result["quarter_component_pair_ed"], 1.)
            self.assertAlmostEqual(result["identity_residual"], 0.)

    def test_unequal_empirical_sizes_still_have_equal_donor_mass(self):
        rng = np.random.default_rng(123)
        p1, p2 = rng.normal(size=(3, 4)), rng.normal(size=(8, 4)) + 2
        for target in (rng.normal(size=(7, 4)), rng.normal(size=(9, 4)) + 100):
            result = AUDIT.mixture_decomposition(p1, p2, target)
            self.assertLess(abs(result["identity_residual"]), 1e-11)
            self.assertGreater(result["convexity_gain"], 0)

    def identity(self):
        return pd.DataFrame(dict(latent_row=range(8),
            donor=["held", "held", "a", "a", "b", "b", "a", "b"],
            cond=["Wound1", "Wound7", "Wound1", "Wound7", "Wound1", "Wound7", "Wound1", "Wound1"]))

    def test_wrong_source_donor_time_and_row_order_exclusions(self):
        identity = self.identity()
        good = np.array([2, 6, 4, 7])
        rows = AUDIT.validate_wrong_rows(identity, good, "held", ["a", "b"])
        np.testing.assert_array_equal(rows["a"], [2, 6])
        for bad in ([0, 6, 4, 7], [3, 6, 4, 7], [2, 2, 4, 7], [6, 2, 4, 7], [2, 6, 4, 99], [2.5, 6, 4, 7]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                AUDIT.validate_wrong_rows(identity, bad, "held", ["a", "b"])
        for permitted in (["held", "a"], ["a", "a"], ["a", "missing"]):
            with self.assertRaises(ValueError):
                AUDIT.source_rows(identity, "held", permitted)

    def test_fixed_count_fallback_odd_exact_mass_reproducibility(self):
        for draw in range(4):
            individual, parts, weights, counts = AUDIT.balanced_draw([9, 5, 8], 200, draw)
            again = AUDIT.balanced_draw([9, 5, 8], 200, draw)
            self.assertEqual(counts, [3, 2] if draw % 2 == 0 else [2, 3])
            self.assertEqual(sum(counts), 5)
            self.assertAlmostEqual(weights[:counts[0]].sum(), .5)
            self.assertAlmostEqual(weights[counts[0]:].sum(), .5)
            for i, indices in enumerate(individual):
                self.assertEqual(len(indices), 5)
                self.assertEqual(len(np.unique(indices)), 5)
                np.testing.assert_array_equal(indices, again[0][i])
            for i in range(2):
                np.testing.assert_array_equal(parts[i], individual[i + 1][:counts[i]])
        for sizes, request, draw in (([2, 1, 3], 200, 0), ([3, 3, 3], 1, 0), ([3, 3, 3], 2, -1)):
            with self.assertRaises(ValueError):
                AUDIT.balanced_draw(sizes, request, draw)

    def test_even_count_rng_order_matches_declared_protocol(self):
        sizes = [255, 502, 255]
        individual, parts, weights, counts = AUDIT.balanced_draw(sizes, 200, 17)
        rng = np.random.default_rng(17)
        for i, size in enumerate(sizes):
            np.testing.assert_array_equal(individual[i], rng.choice(size, 200, replace=False))
        self.assertEqual(counts, [100, 100])
        np.testing.assert_array_equal(weights, np.full(200, .005))

    def test_cached_metric_and_actual_component_identity(self):
        rng = np.random.default_rng(8)
        prediction, target = rng.normal(size=(23, 3)), rng.normal(size=(12, 3))
        cache = AUDIT.CachedEnergy(prediction, target)
        a, b = np.arange(3), np.arange(3, 8)
        indices = np.concatenate([a, b])
        weights = np.r_[np.full(3, .5 / 3), np.full(5, .5 / 5)]
        self.assertAlmostEqual(cache.score(indices, weights), AUDIT.weighted_energy(prediction[indices], target, weights))
        self.assertAlmostEqual(.5 * (cache.score(a) + cache.score(b)) - cache.score(indices, weights),
                               .25 * cache.pair(a, b))
        with self.assertRaises(ValueError):
            cache.score(indices, np.ones(8))

    def test_hierarchy_does_not_count_draws_as_equal_weight_donors(self):
        records = []
        for held, seed, values in [("a", 0, [1., 3.]), ("a", 1, [10.]), ("b", 0, [0., 0., 0.])]:
            records.extend(dict(held_donor=held, seed=seed, draw=d, real_ed=v) for d, v in enumerate(values))
        frame = pd.DataFrame(records)
        per_fit, donor, macro = AUDIT.hierarchical_summary(frame, ["real_ed"])
        self.assertEqual(len(per_fit), 3)
        self.assertEqual(donor.real_ed.tolist(), [6., 0.])
        self.assertEqual(macro["real_ed"], 3.)
        with self.assertRaises(ValueError):
            AUDIT.hierarchical_summary(pd.concat([frame, frame.iloc[:1]]), ["real_ed"])

    def test_manifest_requires_coverage_integrity_safe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "input.json"
            path.write_text('{"value": 1}')
            manifest = {path.name: AUDIT.digest(path)}
            manifest_path = folder / "output_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            observed = {}
            self.assertEqual(AUDIT.verified_manifest(folder, [path.name], observed), manifest)
            self.assertIn(str(path), observed)
            with self.assertRaises(ValueError):
                AUDIT.verified_manifest(folder, ["missing.json"], {})
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                AUDIT.verified_manifest(folder, [path.name], {})
            manifest_path.write_text(json.dumps({"../outside": "0" * 64}))
            with self.assertRaises(ValueError):
                AUDIT.verified_manifest(folder, [], {})

    def test_scaler_excludes_held_and_recomputes_parameters(self):
        identity = self.identity()
        mu = np.arange(16, dtype=np.float32).reshape(8, 2)
        rows = np.arange(2, 8)
        scaler = dict(fitting_latent_rows=rows, mean=mu[rows].mean(0, keepdims=True),
                      scale=mu[rows].std(0, keepdims=True))
        AUDIT.validate_scaler(scaler, mu, identity, ["a", "b"], "test")
        with self.assertRaises(ValueError):
            AUDIT.validate_scaler(dict(scaler, fitting_latent_rows=np.arange(8)), mu, identity, ["a", "b"], "test")
        with self.assertRaises(ValueError):
            AUDIT.validate_scaler(dict(scaler, mean=scaler["mean"] + 1), mu, identity, ["a", "b"], "test")


if __name__ == "__main__":
    unittest.main()
