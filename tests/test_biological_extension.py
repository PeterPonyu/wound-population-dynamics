"""Analytic checks for the added estimands, independent of fitted outcomes."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
spec = importlib.util.spec_from_file_location('extension', Path(__file__).resolve().parents[1] / 'scripts/expand_biological_evidence.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

class BiologicalExtensionTests(unittest.TestCase):

    def test_mean_interpolation_preserves_centered_source_geometry(self):
        source = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]])
        anchor = np.array([[10.0, 4.0], [8.0, 8.0]])
        for alpha in [0, 0.5, 1]:
            control = extension.interpolation_controls(source, anchor, alpha, np.random.default_rng(17))
            translated = control['Centroid translation']
            np.testing.assert_allclose(translated.mean(0), (1 - alpha) * source.mean(0) + alpha * anchor.mean(0))
            np.testing.assert_allclose(pdist(source), pdist(translated))
        with self.assertRaises(ValueError):
            extension.interpolation_controls(source, anchor, 1.5, np.random.default_rng(17))

    def test_full_empirical_energy_matches_analytic_singletons_and_replication(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        self.assertAlmostEqual(extension.energy(a, b), 10.0)
        self.assertAlmostEqual(extension.energy(np.repeat(a, 3, axis=0), np.repeat(b, 7, axis=0)), 10.0)
        self.assertAlmostEqual(extension.energy(b, b), 0.0)

    def test_mean_donor_distance_differs_from_pooled_donor_mass(self):
        source = [np.array([[0.0]]), np.array([[10.0]])]
        target = source[::-1]
        macro = np.mean([extension.energy(a, b) for a, b in zip(source, target)])
        pooled = extension.energy(np.vstack(source), np.vstack(target))
        self.assertAlmostEqual(macro, 20.0)
        self.assertAlmostEqual(pooled, 0.0)
if __name__ == '__main__':
    unittest.main()
