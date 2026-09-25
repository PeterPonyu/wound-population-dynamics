"""Clock audit checks target exclusion, paired randomness and exact integration."""
import importlib.util
from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

SPEC = importlib.util.spec_from_file_location("paired_clock", Path(__file__).resolve().parents[1] / "scripts/assess_paired_clock_sensitivity.py")
CLOCK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLOCK)


class PairedClockAuditTests(unittest.TestCase):
    def setUp(self):
        self.obs = pd.DataFrame([{"donor": d, "cond": c} for d in ["A", "B", "C"]
                                 for c in CLOCK.COND_ORDER for _ in range(24)])
        self.mu = np.random.default_rng(3).normal(size=(len(self.obs), 3)).astype(np.float32)
        torch.set_num_threads(1)

    def test_target_values_cannot_change_scaler_or_training(self):
        a = CLOCK.prepare(self.mu, self.obs, "Wound7")
        changed = self.mu.copy()
        changed[self.obs.cond.eq("Wound7")] *= 10000
        b = CLOCK.prepare(changed, self.obs, "Wound7")
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        for z in (a[0], b[0]):
            field, _ = CLOCK.train_axis(z, self.obs, a[4], CLOCK.time_axes()["rank"],
                                        seed=1, steps=2, batch_size=4, device="cpu")
            weights = [v.detach().clone() for v in field.parameters()]
            if z is a[0]:
                first = weights
            else:
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(first, weights)))

    def test_clock_axes_use_identical_training_random_streams(self):
        z, _, _, _, pairs = CLOCK.prepare(self.mu, self.obs, "Wound7")
        streams = []
        for clock in CLOCK.time_axes().values():
            _, record = CLOCK.train_axis(z, self.obs, pairs, clock, seed=7, steps=3, batch_size=4, device="cpu")
            streams.append({k: v for k, v in record.items() if k.endswith("sha256")})
        self.assertTrue(all(record == streams[0] for record in streams))

    def test_integrator_stops_at_exact_non_grid_target(self):
        class Constant(torch.nn.Module):
            def forward(self, z, t):
                return torch.ones_like(z) * 2
        value = CLOCK.INTEGRATOR.integrate(Constant(), torch.zeros(2, 3), 1 / 30, 7 / 30, 7)
        np.testing.assert_allclose(value, .4, atol=1e-7)

    def arguments(self, root):
        np.save(root / "mu.npy", self.mu)
        self.obs.to_csv(root / "obs.csv", index=False)
        return Namespace(latent=root / "mu.npy", obs=root / "obs.csv", output_dir=root / "clock",
                         holdout="Wound7", seeds="0", steps=1, batch_size=4, rk4_steps=2,
                         evaluation_cap=4, device="cpu", resume=False)

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): (CLOCK.sha(p), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}

    def test_later_axis_corruption_causes_no_resume_writes(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(tmp))
            CLOCK.run(args)
            (args.output_dir / "seed0/days/field.pt").write_bytes(b"corrupt later axis")
            before = self.snapshot(args.output_dir)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                CLOCK.run(args)
            self.assertEqual(self.snapshot(args.output_dir), before)

    def test_completed_resume_is_a_byte_and_mtime_preserving_noop(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(tmp))
            CLOCK.run(args)
            before = self.snapshot(args.output_dir)
            args.resume = True
            with patch.object(CLOCK, "train_axis", side_effect=AssertionError("must not fit")), \
                 patch.object(CLOCK.INTEGRATOR, "integrate", side_effect=AssertionError("must not integrate")):
                CLOCK.run(args)
            self.assertEqual(self.snapshot(args.output_dir), before)

    def test_partial_resume_reuses_completed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(tmp))
            actual = CLOCK.train_axis
            calls = []
            def interrupt_after_first(*a, **k):
                calls.append(k)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return actual(*a, **k)
            with patch.object(CLOCK, "train_axis", side_effect=interrupt_after_first):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    CLOCK.run(args)
            completed = args.output_dir / "seed0/rank"
            before = self.snapshot(completed)
            args.resume = True
            with patch.object(CLOCK, "train_axis", wraps=actual) as trained:
                CLOCK.run(args)
            self.assertEqual(trained.call_count, 3)
            self.assertEqual(self.snapshot(completed), before)
            self.assertEqual(json.loads((args.output_dir / "report.json").read_text())["fits"], 4)

    def test_finite_float64_target_that_overflows_float32_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.arguments(Path(tmp))
            changed = self.mu.astype(np.float64)
            changed[self.obs.donor.eq("A") & self.obs.cond.eq("Wound7")] = 1e40
            np.save(args.latent, changed)
            with self.assertRaises(FloatingPointError):
                CLOCK.run(args)
            self.assertFalse(args.output_dir.exists())

    def test_nonfinite_score_cannot_be_dropped_into_successful_summary(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(tmp))
            with patch.object(CLOCK, "weighted_energy", return_value=float("nan")):
                with self.assertRaisesRegex(FloatingPointError, "computed scores"):
                    CLOCK.run(args)
            self.assertFalse((args.output_dir / "report.json").exists())
            self.assertFalse((args.output_dir / "seed0/rank/fit.json").exists())

    def test_tampered_complete_summary_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments(Path(tmp))
            CLOCK.run(args)
            (args.output_dir / "summary.csv").write_text("axis,flow_ed\nrank,nan\n")
            before = self.snapshot(args.output_dir)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "manifest checksum differs"):
                CLOCK.run(args)
            self.assertEqual(self.snapshot(args.output_dir), before)


if __name__ == "__main__":
    unittest.main()
