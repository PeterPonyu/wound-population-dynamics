"""Analytic invariants for patient measures and endpoint-only predictors."""
import unittest
import numpy as np
from scripts.strengthen_computational_evidence import geometry_metrics, location_scale

class ComputationalExtensionTests(unittest.TestCase):

    def test_location_scale_recovers_known_diagonal_affine_endpoints(self):
        source = np.array([[0.0, 1.0], [1.0, 4.0], [3.0, 2.0], [5.0, 6.0]])
        anchor = source * np.array([2.0, 0.5]) + np.array([6.0, -3.0])
        np.testing.assert_allclose(location_scale(source, anchor, 0), source)
        np.testing.assert_allclose(location_scale(source, anchor, 1), anchor)
        np.testing.assert_allclose(location_scale(source, anchor, 0.5), (source + anchor) / 2)
        with self.assertRaises(ValueError):
            location_scale(source, anchor, -0.1)
        constant = location_scale(np.ones((4, 2)), anchor, 0.5)
        np.testing.assert_allclose(constant, np.tile((1 + anchor.mean(0)) / 2, (4, 1)))

    def test_centered_distance_separates_translation_from_shape_diagnostic(self):
        target = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
        translated = target + np.array([3.0, 4.0])
        metric = geometry_metrics(translated, target)
        self.assertAlmostEqual(metric['centered_energy'], 0.0)
        self.assertAlmostEqual(metric['centroid_error'], 5.0)
        self.assertAlmostEqual(metric['variance_ratio'], 1.0)
        self.assertGreater(metric['energy_distance'], 0.0)
        self.assertGreater(geometry_metrics(target * 2, target)['centered_energy'], 0.0)
if __name__ == '__main__':
    unittest.main()
