"""Analytic checks of path units, masking, weighting and integration."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wound_models.population_flow import compute_cfm_loss, compute_conditional_cfm_loss, FlowMatchingIntegrator
from wound_models.human_wound_data import fold_standardize
from scripts.audit_population_math import donor_weights, weighted_energy


class PopulationMathematics(unittest.TestCase):
    def test_interval_velocity_recovers_analytic_translation(self):
        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        for a, b in ((0., 1.), (0., 1 / 30), (1 / 3, 2 / 3)):
            target = source + 3 * (b - a)
            loss = compute_cfm_loss(lambda z, time: torch.full_like(z, 3), target, source, a, b)
            self.assertLess(loss.item(), 1e-9)

    def test_context_mask_is_per_example_and_whole_vector(self):
        observed = []
        def field(z, time, context):
            observed.append(context.detach().clone())
            return torch.zeros_like(z)
        endpoint = torch.zeros(4, 2)
        with patch('torch.rand', side_effect=[torch.tensor([0.1, 0.3, 0.6, 0.9]),
                                             torch.tensor([[0.1], [0.4], [0.19], [0.9]])]):
            compute_conditional_cfm_loss(field, endpoint, endpoint, torch.tensor([2., 7.]), context_dropout=0.2)
        torch.testing.assert_close(observed[0], torch.tensor([[0., 0.], [2., 7.], [0., 0.], [2., 7.]]))

    def test_invalid_domains_are_rejected(self):
        z = torch.zeros(3, 2)
        for end in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compute_cfm_loss(lambda z, time: z, z, z, 0., end)
        for dropout in (-0.01, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                compute_conditional_cfm_loss(lambda z, time, c: z, z, z, torch.zeros(2), context_dropout=dropout)
        with self.assertRaises(ValueError):
            compute_cfm_loss(lambda z, time: z, z, torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            FlowMatchingIntegrator(lambda z, time: z).solve_rk4(z, 0)

    def test_rk4_translation_and_fourth_order_linear_ode(self):
        z = torch.tensor([[1., -2.]], dtype=torch.float64)
        translated = FlowMatchingIntegrator(lambda z, time: torch.full_like(z, 3)).solve_rk4(z, 4)
        torch.testing.assert_close(translated[-1], z + 3)
        errors = [torch.abs(FlowMatchingIntegrator(lambda z, time: z).solve_rk4(z, n)[-1] - z * np.e).max().item()
                  for n in (4, 8, 16)]
        self.assertGreater(errors[0] / errors[1], 14)
        self.assertGreater(errors[1] / errors[2], 14)

    def test_weighted_distance_matches_replicated_empirical_measure(self):
        x, y = np.array([[0., 0.], [1., 2.]]), np.array([[3., 1.], [-1., 2.]])
        a, b = np.array([0.25, 0.75]), np.array([0.5, 0.5])
        result = weighted_energy(x, y, a, b)
        expanded = weighted_energy(np.repeat(x, [1, 3], axis=0), np.repeat(y, [2, 2], axis=0))
        self.assertAlmostEqual(result, expanded)
        self.assertAlmostEqual(result, weighted_energy(y, x, b, a))
        self.assertGreaterEqual(result, 0)
        self.assertAlmostEqual(weighted_energy(x, x, a, a), 0)
        self.assertAlmostEqual(weighted_energy([[0., 0.]], [[3., 4.]]), 10)

    def test_donor_mass_and_training_only_scaling(self):
        donors = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
        weights = donor_weights(donors)
        for name in np.unique(donors):
            self.assertAlmostEqual(weights[donors == name].sum(), 1 / 3)
        x = np.array([[0., 3.], [2., 3.], [1000., -400.]])
        selected = np.array([True, True, False])
        original, _ = fold_standardize(x, selected)
        x[2] = [1e7, 1e7]
        changed, _ = fold_standardize(x, selected)
        np.testing.assert_array_equal(original[:2], changed[:2])
        np.testing.assert_allclose(original[:2], [[-1, 0], [1, 0]])


if __name__ == '__main__':
    unittest.main()
