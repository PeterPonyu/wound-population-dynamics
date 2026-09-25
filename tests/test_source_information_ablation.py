"""Contract checks for the retrospective source-replacement analysis."""
import importlib.util
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_ablation", ROOT / "scripts/assess_source_information.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceAblationContract(unittest.TestCase):
    def contract_fixture(self, root, seeds):
        """A small artifact fixture exercises contracts without fitting models."""
        identity = pd.DataFrame([dict(donor=d, cond=c, cell_id=f"{d}:{c}", latent_row=i)
                                 for i, (d, c) in enumerate(itertools.product("abc", ["Wound1", "Wound7"]))])
        latent = root / "mu.npy"
        np.save(latent, np.ones((6, 2), dtype=np.float32))
        inputs = {"mu.npy": MODULE.digest(latent)}
        count = 9 * len(seeds)
        verification = dict(status="passed", saved_fit_count=count, unique_cell_ids=True,
                            training_donor_exclusion=True, reference_excludes_held_donor=True,
                            source_target_ids_checked=True, maximum_inverse_transform_difference=0.,
                            maximum_score_reproduction_difference=0., maximum_cpu_checkpoint_prediction_difference=0.)
        report = dict(status="completed", schema_version=1, experiment="fixed_reference_donor_curve_and_target_marginal",
                      biological_donors=3, independent_new_data=False, fits=count, seeds=seeds,
                      verification=verification, provenance={"inputs": inputs})
        contract = dict(schema_version=1, seeds=seeds, steps=8000, batch_size=256, rk4_steps=50,
                        evaluation_seed=0, marginal_draws=50, target_cap=1200, inputs=inputs)
        (root / "run_contract.json").write_text(json.dumps(contract))
        (root / "verification.json").write_text(json.dumps(verification))
        fits, manifest = [], {}
        for held in "abc":
            for k in (1, 2):
                for subset in itertools.combinations([d for d in "abc" if d != held], k):
                    for seed in seeds:
                        relative = f"fits/{held}/k{k}_{'_'.join(subset)}/seed{seed}"
                        folder = root / relative
                        folder.mkdir(parents=True)
                        artifacts = {n: "a" * 64 for n in ["field.pt", "prediction_training_z.npy",
                                                          "prediction_mu.npy", "prediction_reference_z.npy"]}
                        sr = identity.loc[identity.donor.eq(held) & identity.cond.eq("Wound1"), "latent_row"].tolist()
                        tr = identity.loc[identity.donor.eq(held) & identity.cond.eq("Wound7"), "latent_row"].tolist()
                        fit = dict(held_donor=held, training_donors=list(subset), k=k, seed=seed, path=relative,
                                   training_steps=8000, batch_size=256, rk4_steps=50,
                                   reference_scaler_used_by_training=False, artifact_sha256=artifacts,
                                   source_count=1, target_evaluation_count=1, source_latent_rows=sr,
                                   target_evaluation_latent_rows=tr)
                        (folder / "fit.json").write_text(json.dumps(fit))
                        for name in ["fit.json", *artifacts]:
                            manifest[f"{relative}/{name}"] = "a" * 64
                        manifest[f"fits/{held}/reference.npz"] = "a" * 64
                        manifest[f"{Path(relative).parent}/training_scaler.npz"] = "a" * 64
                        fits.append(fit)
        return report, fits, identity, manifest

    def test_full_grid_accepts_old_and_additional_seed_sets(self):
        for seeds in ([0, 1, 2], [3, 4, 5, 6, 7]):
            with self.subTest(seeds=seeds), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                values = self.contract_fixture(root, seeds)
                with patch.object(MODULE, "ROOT", root), patch.object(MODULE.REPAIR, "LATENT", root / "mu.npy"):
                    selected = MODULE.validate_fit_contract(root, *values)
                self.assertEqual(len(selected), 3 * len(seeds))
                self.assertEqual({fit["seed"] for fit in selected}, set(seeds))

    def test_generalized_grid_rejects_invalid_reports_and_fit_manifests(self):
        changes = {
            "duplicate configuration": lambda r, f, m: f.__setitem__(1, copy.deepcopy(f[0])),
            "missing fit": lambda r, f, m: f.pop(),
            "undeclared seed": lambda r, f, m: f[0].update(seed=0),
            "duplicate report seed": lambda r, f, m: r["seeds"].append(3),
            "boolean seed": lambda r, f, m: r["seeds"].__setitem__(0, True),
            "wrong count": lambda r, f, m: r.update(fits=27),
            "unverified report": lambda r, f, m: r["verification"].update(status="failed"),
            "new donors claim": lambda r, f, m: r.update(independent_new_data=True),
            "held donor in training": lambda r, f, m: f[0].update(training_donors=[f[0]["held_donor"]]),
            "wrong k": lambda r, f, m: f[0].update(k=2),
            "altered steps": lambda r, f, m: f[0].update(training_steps=9000),
            "unsafe path": lambda r, f, m: f[0].update(path="../outside"),
            "hash not in output manifest": lambda r, f, m: m.__setitem__(f[0]["path"] + "/field.pt", "b" * 64),
            "missing artifact coverage": lambda r, f, m: m.pop(f[0]["path"] + "/fit.json"),
            "wrong saved fit": lambda r, f, m: f[0].update(source_count=999),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, fits, identity, manifest = self.contract_fixture(root, [3, 4, 5, 6, 7])
            with patch.object(MODULE, "ROOT", root), patch.object(MODULE.REPAIR, "LATENT", root / "mu.npy"):
                for label, change in changes.items():
                    r, f, m = copy.deepcopy((report, fits, manifest))
                    change(r, f, m)
                    with self.subTest(label=label), self.assertRaises(ValueError):
                        MODULE.validate_fit_contract(root, r, f, identity, m)
                (root / "mu.npy").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "latent coordinates"):
                    MODULE.validate_fit_contract(root, report, fits, identity, manifest)

    def test_candidate_pool_excludes_held_donor_and_target_time(self):
        identity = pd.DataFrame({"latent_row": range(8),
                                 "donor": ["held", "held", "a", "a", "b", "b", "a", "b"],
                                 "cond": ["Wound1", "Wound7", "Wound1", "Wound7",
                                          "Wound1", "Wound7", "Wound1", "Wound1"]})
        pool = MODULE.source_rows(identity, "held", ["a", "b"])
        np.testing.assert_array_equal(pool["a"], [2, 6])
        np.testing.assert_array_equal(pool["b"], [4, 7])
        with self.assertRaises(ValueError):
            MODULE.source_rows(identity, "held", ["a", "held"])

    def test_donor_summary_averages_draws_then_seeds_not_pseudoreplication(self):
        draws = pd.DataFrame([dict(held_donor="d", seed=seed, draw=draw,
                                   real_ed=real, wrong_source_ed=wrong,
                                   target_marginal_ed=0.5, wrong_minus_real=wrong-real,
                                   wrong_minus_marginal=wrong-0.5)
                              for seed, real, values in [(0, 1., [2., 4.]), (1, 3., [2., 2.])]
                              for draw, wrong in enumerate(values)])
        grouped = MODULE.paired_summary(draws)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["wrong_worse_draws"], 2)
        self.assertEqual(grouped[1]["wrong_worse_draws"], 0)
        self.assertAlmostEqual(grouped[0]["wrong_minus_real"], 2.)
        self.assertAlmostEqual(grouped[1]["wrong_minus_real"], -1.)


if __name__ == "__main__":
    unittest.main()
