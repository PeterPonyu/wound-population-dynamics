"""Mathematics, raw identity, assay coverage, and information-exclusion checks."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("population_observable", ROOT / "scripts/audit_population_observable.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ObservableMathematicsTests(unittest.TestCase):
    def test_projection_rejects_nonfinite_checkpoints_saved_mu_and_batches(self):
        import torch
        from wound_models.topic_model import TopicModel, TopicEncoder
        model = TopicModel(3, 2).eval()
        counts = sparse.csr_matrix([[2, 1, 1], [1, 1, 2]], dtype=np.int32)
        values = np.asarray(AUDIT.library_normalize(counts).todense(), dtype=np.float32)
        with torch.no_grad():
            mu = model.encoder(torch.from_numpy(values))[0].numpy()
            beta = model.decoder.beta.numpy()
        protocol = {"checks": {"checkpoint_beta_max_abs": 1e-6, "cpu_encoder_mu_max_abs": 5e-5}}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            valid = AUDIT.check_projection(protocol, counts, np.arange(3), beta, mu, checkpoint)
            self.assertTrue(valid["checkpoint_inputs_and_batch_outputs_explicitly_finite"])
            for bad in [np.nan, np.inf, -np.inf]:
                with self.subTest(bad=bad):
                    state = {key: value.clone() for key, value in model.state_dict().items()}
                    state["encoder.fc_mu.weight"][0, 0] = bad
                    torch.save(state, checkpoint)
                    with self.assertRaisesRegex(ValueError, "checkpoint state"):
                        AUDIT.check_projection(protocol, counts, np.arange(3), beta, mu, checkpoint)
                    torch.save(model.state_dict(), checkpoint)
                    corrupted_mu = mu.copy()
                    corrupted_mu[0, 0] = bad
                    with self.assertRaisesRegex(ValueError, "saved mu"):
                        AUDIT.check_projection(protocol, counts, np.arange(3), beta, corrupted_mu, checkpoint)
                    with patch.object(TopicEncoder, "forward", return_value=(torch.full((2, 2), bad), torch.zeros((2, 2)))):
                        with self.assertRaisesRegex(ValueError, "batch output"):
                            AUDIT.check_projection(protocol, counts, np.arange(3), beta, mu, checkpoint)
            with patch.object(TopicEncoder, "forward", return_value=(torch.zeros((2, 1)), torch.zeros((2, 1)))):
                with self.assertRaisesRegex(ValueError, "batch output"):
                    AUDIT.check_projection(protocol, counts, np.arange(3), beta, mu, checkpoint)

    def test_softmax_stable_shift_invariance(self):
        mu = np.array([[1000., 1001., -1000.], [-1000., -999., -999.]])
        np.testing.assert_allclose(AUDIT.softmax(mu), AUDIT.softmax(mu + 2345), atol=1e-15)
        np.testing.assert_allclose(AUDIT.softmax(mu).sum(1), 1)

    def test_inverse_training_scaler_is_required(self):
        mu = np.array([[2., -1.], [0., 3.]])
        beta = np.array([[.8, .1, .1], [.1, .2, .7]])
        expected, _ = AUDIT.decode_cells(mu, beta, np.arange(3))
        for mean, scale in [(np.array([[20., -4.]]), np.array([[.1, 7.]])),
                            (np.zeros((1, 2)), np.ones((1, 2)))]:
            z = (mu - mean) / scale
            actual, _ = AUDIT.decode_cells(AUDIT.inverse_scaler(z, mean, scale), beta, np.arange(3))
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        wrong, _ = AUDIT.decode_cells((mu - 20) / .1, beta, np.arange(3))
        self.assertGreater(np.max(np.abs(wrong - expected)), .01)

    def test_common_panel_conditioning_is_after_topic_mixture(self):
        beta = np.array([[.1, .1, .8], [.7, .2, .1]])
        actual, missing = AUDIT.decode_cells(np.zeros((1, 2)), beta, np.array([0, 1]))
        np.testing.assert_allclose(actual, [[.4 / .55, .15 / .55]])
        np.testing.assert_allclose(missing, [.45])
        incorrect = np.mean(beta[:, :2] / beta[:, :2].sum(1, keepdims=True), axis=0)
        self.assertGreater(np.max(np.abs(actual[0] - incorrect)), .05)

    def test_mean_is_after_per_cell_conditioning(self):
        beta = np.array([[.05, .05, .9], [.72, .18, .1]])
        mu = np.array([[12., -12.], [-12., 12.]])
        actual, _ = AUDIT.decode_cells(mu, beta, np.array([0, 1]))
        np.testing.assert_allclose(actual.mean(0), [.65, .35], atol=1e-9)
        wrong = (AUDIT.softmax(mu) @ beta)[:, :2].mean(0)
        wrong /= wrong.sum()
        self.assertGreater(np.max(np.abs(actual.mean(0) - wrong)), .1)

    def test_all_panel_mass_zero_excluded_and_absent_not_scored_as_zero(self):
        beta = np.array([[.6, .2, .2], [.1, .7, .2]])
        full, missing = AUDIT.decode_cells([[0., 0.]], beta, np.arange(3))
        np.testing.assert_allclose(missing, [0], atol=1e-15)
        common, missing = AUDIT.decode_cells([[0., 0.]], beta, np.array([0, 1]))
        np.testing.assert_allclose(missing, [.2])
        self.assertAlmostEqual(AUDIT.profile_metrics(common[0], common[0])["total_variation"], 0)
        self.assertGreater(AUDIT.profile_metrics(full[0], np.r_[common[0], 0])["total_variation"], .19)

    def test_observed_equal_cells_not_library_weighted_pseudobulk(self):
        counts = sparse.csr_matrix([[1, 0], [0, 100]])
        normalized = AUDIT.normalized_counts(counts)
        np.testing.assert_allclose(AUDIT.mean_profile(normalized), [.5, .5])
        self.assertFalse(np.allclose(AUDIT.mean_profile(normalized), [1 / 101, 100 / 101]))

    def test_metrics_known_values_and_zeros_without_pseudocount(self):
        result = AUDIT.profile_metrics([1., 0, 0], [0, 1., 0])
        self.assertEqual(result["total_variation"], 1)
        self.assertAlmostEqual(result["js_divergence_nats"], np.log(2), places=14)
        self.assertAlmostEqual(result["gene_rmse"], np.sqrt(2 / 3), places=14)
        same = AUDIT.profile_metrics([.2, .8, 0], [.2, .8, 0])
        self.assertTrue(all(value == 0 for value in same.values()))

    def test_invalid_numeric_inputs_fail_closed(self):
        for counts in [sparse.csr_matrix([[0, 0]]), sparse.csr_matrix([[-1, 2]])]:
            with self.assertRaises(ValueError):
                AUDIT.normalized_counts(counts)
        for p in [[.1, .2], [-1, 2], [np.nan, 0]]:
            with self.assertRaises(ValueError):
                AUDIT.profile_metrics(p, [.5, .5])
        for common in [np.array([0, 0]), np.array([-1]), np.array([2]), np.array([.5])]:
            with self.assertRaises(ValueError):
                AUDIT.decode_cells([[0., 0.]], [[.1, .9], [.8, .2]], common)
        with self.assertRaises(ValueError):
            AUDIT.inverse_scaler([[1., 2.]], [[0., 0.]], [[0., 1.]])
        with self.assertRaises(ValueError):
            AUDIT.weights_for(2, [.2, .3])

    def test_decoder_reconstruction_not_proven_floor(self):
        # A different latent prediction can be closer to observed expression
        # than its encoded-then-decoded reconstruction; no floor is implied.
        beta = np.array([[.9, .1], [.1, .9]])
        target = np.array([.8, .2])
        reconstruction, _ = AUDIT.decode_cells([[0, 0]], beta, np.arange(2))
        candidate, _ = AUDIT.decode_cells([[np.log(7), 0]], beta, np.arange(2))
        self.assertLess(AUDIT.profile_metrics(candidate[0], target)["total_variation"],
                        AUDIT.profile_metrics(reconstruction[0], target)["total_variation"])


class ObservableExclusionTests(unittest.TestCase):
    def identity(self):
        return pd.DataFrame(dict(latent_row=range(8), donor=["held", "held", "a", "a", "b", "b", "b", "b"],
                                 cond=["Wound1", "Wound7", "Wound1", "Wound7", "Wound1", "Wound7", "Wound7", "Wound1"]))

    def proportions(self):
        return sparse.csr_matrix([[.25, .25, .25, .25], [.4, .3, .2, .1],
            [.1, .2, .3, .4], [.4, .3, .2, .1], [.1, .2, .3, .4],
            [.4, .3, .2, .1], [.4, .3, .2, .1], [.1, .2, .3, .4]])

    def test_training_signatures_ignore_held_and_other_excluded_donor(self):
        identity, proportions = self.identity(), self.proportions()
        original = AUDIT.training_signature(proportions, identity, ["g0", "g1", "g2", "g3"], ["a"], "held", n=1)
        changed = proportions.toarray()
        changed[~identity.donor.eq("a")] = [0, 0, 0, 1]
        actual = AUDIT.training_signature(sparse.csr_matrix(changed), identity,
                                         ["g0", "g1", "g2", "g3"], ["a"], "held", n=1)
        self.assertEqual(original, actual)
        self.assertEqual(original["up_genes"], ["g0"])
        self.assertEqual(original["down_genes"], ["g3"])
        with self.assertRaises(ValueError):
            AUDIT.training_signature(proportions, identity, ["g0", "g1", "g2", "g3"], ["held"], "held", n=1)

    def test_signature_ties_have_gene_name_order_not_held_outcome(self):
        identity = pd.DataFrame(dict(donor=["a", "a", "held", "held"], cond=["Wound1", "Wound7"] * 2))
        matrix = sparse.csr_matrix([[.1, .1, .4, .4], [.4, .4, .1, .1], [.25] * 4, [.25] * 4])
        result = AUDIT.training_signature(matrix, identity, ["b", "a", "d", "c"], ["a"], "held", n=1)
        self.assertEqual(result["up_genes"], ["a"])
        self.assertEqual(result["down_genes"], ["c"])

    def test_signature_score_uses_original_proportions_not_set_renormalization(self):
        result = AUDIT.signature_scores(np.array([.1, .2, .3, .4]), dict(up_indices=[0, 1], down_indices=[2]))
        self.assertAlmostEqual(result["up_mass"], .3)
        self.assertAlmostEqual(result["down_mass"], .3)
        self.assertAlmostEqual(result["signed_mass"], 0)

    def test_marginal_equal_donor_weights_with_unequal_cells(self):
        rows, weights = AUDIT.exact_marginal(self.identity(), ["a", "b"], "held")
        np.testing.assert_array_equal(rows, [3, 5, 6])
        np.testing.assert_array_equal(weights, [.5, .25, .25])
        changed = self.proportions().toarray()
        changed[3] = [1, 0, 0, 0]
        changed[5:7] = [0, 1, 0, 0]
        np.testing.assert_allclose(AUDIT.mean_profile(changed, rows, weights), [.5, .5, 0, 0])
        for permitted in [["held"], ["a", "a"], []]:
            with self.assertRaises(ValueError):
                AUDIT.exact_marginal(self.identity(), permitted, "held")

    def test_invalid_row_donor_time_duplicate_or_fraction_rejected(self):
        for rows in [[0], [3], [2, 2], [2.5], [-1], [99]]:
            with self.assertRaises(ValueError):
                AUDIT.checked_rows(self.identity(), rows, ["a"], "Wound1")
        np.testing.assert_array_equal(AUDIT.checked_rows(self.identity(), [2], ["a"], "Wound1"), [2])

    def test_scaler_matches_only_permitted_rows(self):
        identity = self.identity()
        mu = np.arange(16, dtype=np.float32).reshape(8, 2)
        rows = np.array([2, 3])
        scaler = dict(fitting_latent_rows=rows, mean=mu[rows].mean(0, keepdims=True), scale=mu[rows].std(0, keepdims=True))
        AUDIT.validate_scaler(scaler, mu, identity, ["a"], "held")
        mu[identity.donor.ne("a")] *= 1e5
        AUDIT.validate_scaler(scaler, mu, identity, ["a"], "held")
        with self.assertRaises(ValueError):
            AUDIT.validate_scaler({**scaler, "fitting_latent_rows": np.array([0, 2, 3])}, mu, identity, ["a"], "held")
        with self.assertRaises(ValueError):
            AUDIT.validate_scaler({**scaler, "mean": scaler["mean"] + 1}, mu, identity, ["a"], "held")

    def test_hierarchy_does_not_promote_seeds_or_subsets_to_donors(self):
        rows = []
        # Mean seeds within configuration, mean configurations within donor,
        # then equal donor mean = ((0+10)/2 + 20 + 30)/3 = 55/3.
        for held, subset, values in [("a", "b", [0, 0, 0]), ("a", "c", [10]),
                                     ("b", "a", [20, 20]), ("c", "a", [30])]:
            for seed, value in enumerate(values):
                rows.append(dict(held_donor=held, k=1, training_donors=subset, seed=seed,
                                 target_set="original_evaluation", method="test", **{m: value for m in AUDIT.METRICS}))
        frame = pd.DataFrame(rows)
        _, blocks, macro = AUDIT.hierarchical_summary(frame)
        self.assertEqual(len(blocks), 3)
        self.assertAlmostEqual(macro.total_variation.iloc[0], 55 / 3)
        self.assertFalse(macro.biological_ci_available.iloc[0])
        two = frame.loc[frame.held_donor.ne("c")]
        self.assertEqual(AUDIT.hierarchical_summary(two)[2].biological_donors.iloc[0], 2)
        with self.assertRaisesRegex(ValueError, "donor grid"):
            AUDIT.hierarchical_summary(two, ["a", "b", "c"])
        AUDIT.hierarchical_summary(frame, ["a", "b", "c"])
        with self.assertRaises(ValueError):
            AUDIT.hierarchical_summary(pd.concat([frame, frame.iloc[:1]]))


class ObservableIdentityAndFreezeTests(unittest.TestCase):
    def test_barcode_author_fibroblast_and_matrix_row_identity(self):
        barcodes = np.array(["a-1", "b-1", "c-1"])
        identity = pd.DataFrame(dict(gsm=["GSM1", "GSM1"], latent_row=[2, 3], archive_cell_row=[0, 2],
            cell_id=["GSM1:a-1", "GSM1:c-1"], author_key=["PD7_a-1", "PD7_c-1"], raw_matrix_row=[10, 12],
            donor=["P", "P"], cond=["Wound7", "Wound7"]))
        meta = pd.DataFrame(dict(newMainCellTypes=["Fibroblast", "Other", "Fibroblast"]), index=["PD7_a-1", "PD7_b-1", "PD7_c-1"])
        latent, sample = AUDIT.align_archive(identity, "GSM1", "PD7", "P", "Wound7", barcodes, 10, meta)
        np.testing.assert_array_equal(latent, [2, 3])
        np.testing.assert_array_equal(sample, [0, 2])
        for column, values in [("cell_id", ["GSM1:c-1", "GSM1:a-1"]), ("raw_matrix_row", [0, 2]),
                               ("donor", ["X", "P"]), ("archive_cell_row", [2, 0])]:
            changed = identity.copy()
            changed[column] = values
            with self.assertRaises(ValueError):
                AUDIT.align_archive(changed, "GSM1", "PD7", "P", "Wound7", barcodes, 10, meta)

    def test_duplicate_gene_symbols_are_summed_not_selected(self):
        features = io.BytesIO(b"id1\tA\nid2\tB\nid3\tA\n")
        barcodes = io.BytesIO(b"cell1\ncell2\n")
        matrix = io.BytesIO(b"%%MatrixMarket matrix coordinate integer general\n3 2 4\n1 1 1\n2 1 2\n3 1 3\n3 2 5\n")
        counts, genes, _ = AUDIT._read_10x_triplet(features, barcodes, matrix)
        np.testing.assert_array_equal(genes, ["A", "B"])
        np.testing.assert_array_equal(counts.toarray(), [[4, 2], [5, 0]])

    def test_requested_model_not_assumed_runtime_verified(self):
        result = AUDIT.runtime_evidence(None, "gpt-6-astra")
        self.assertIsNone(result["runtime_reported_model"])
        self.assertFalse(result["runtime_metadata_available"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            path.write_text(json.dumps(dict(type="turn_context", timestamp="2026-09-25T00:00:00Z",
                payload=dict(cwd=str(ROOT), model="gpt-6-astra"))) + "\n")
            result = AUDIT.runtime_evidence(path, "gpt-6-astra")
            self.assertTrue(result["matches_request"])
            with self.assertRaises(ValueError):
                AUDIT.runtime_evidence(path, "gpt-6-sol")

    def test_manifest_binding_detects_input_changes_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "file.txt"
            path.write_text("frozen")
            inputs = {}
            checksum = AUDIT.bind_file(path, inputs)
            self.assertEqual(inputs[str(path.resolve())], checksum)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                AUDIT.bind_file(path, {}, checksum)
            for relative in ["../outside", "/absolute", "./file.txt"]:
                with self.assertRaises(ValueError):
                    AUDIT.safe_path(folder, relative)
            with self.assertRaises(FileExistsError):
                AUDIT.write_json(path, {"overwrite": "forbidden"})

    def test_protocol_requires_actual_observed_target_all_genes_and_no_ci(self):
        protocol = AUDIT.read_json(AUDIT.DEFAULT_PROTOCOL)
        self.assertEqual(protocol["expected"]["common_genes"], 5383)
        self.assertEqual(protocol["expected"]["panel_genes"], 7002)
        self.assertEqual(protocol["methods"], AUDIT.METHODS)
        self.assertEqual(protocol["metrics"]["all"], AUDIT.METRICS)
        self.assertIn("real count", protocol["normalization"]["observed"])
        self.assertIn("None", protocol["inference"]["confidence_intervals"])
        self.assertIn("NOT a proven", protocol["reconstruction_reference"])

    def test_frozen_preparation_rejects_changed_feature_input_and_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = folder / "audit"
            out.mkdir()
            protocol = folder / "protocol.json"
            protocol.write_text(json.dumps({"expected": {"common_genes": 2, "panel_genes": 3,
                "fits": 1, "latent_cells": 2, "donors": ["a"]},
                "checks": {"checkpoint_beta_max_abs": 1e-6, "cpu_encoder_mu_max_abs": 5e-5}}))
            prepared_files = {
                **{name: '{}' for name in AUDIT.PREPARATION_FILES},
                "prepared.json": json.dumps(dict(status="prepared", no_predictive_scores_computed=True,
                    no_new_training=True, common_genes=2, panel_genes=3, fit_count=1,
                    raw_count_cells=2, biological_donors=1, input_files=2, training_only_signatures=1)),
                "protocol.json": protocol.read_text(),
                "features.csv": "gene\nA\nB\n",
                "cell_identity.csv": "cell_id,donor\nc1,a\nc2,a\n",
                "split_verification.json": '[{"path": "fits/a/k1_b/seed0"}]',
                "training_signatures.json": '{"fits/a/k1_b": {}}',
                "projection_verification.json": json.dumps(dict(checked_cells=2,
                    checkpoint_inputs_and_batch_outputs_explicitly_finite=True,
                    checkpoint_beta_max_abs_error=0, all_cells_encoder_mu_max_abs_error=0)),
                "input_sha256.json": json.dumps({str(protocol.resolve()): AUDIT.sha(protocol),
                                                  str(Path(AUDIT.__file__).resolve()): AUDIT.sha(AUDIT.__file__)})
            }
            for name, value in prepared_files.items():
                (out / name).write_text(value)
            sparse.save_npz(out / "observed_common_counts.npz", sparse.csr_matrix([[1, 2], [2, 1]]))
            manifest = {name: AUDIT.sha(out / name) for name in prepared_files}
            (out / "preparation_manifest.json").write_text(json.dumps(manifest))
            AUDIT.validate_preparation(protocol, out)
            # Even self-consistent reduced manifests must fail required closure.
            incomplete = dict(manifest)
            del incomplete["split_identity_rows.npz"]
            (out / "preparation_manifest.json").write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, "closure"):
                AUDIT.validate_preparation(protocol, out)
            for updates in [{"status": "failed"}, {"no_new_training": False}, {"fit_count": 0}]:
                broken = json.loads(prepared_files["prepared.json"])
                broken.update(updates)
                (out / "prepared.json").write_text(json.dumps(broken))
                (out / "preparation_manifest.json").write_text(json.dumps({**manifest, "prepared.json": AUDIT.sha(out / "prepared.json")}))
                with self.assertRaises(ValueError):
                    AUDIT.validate_preparation(protocol, out)
            (out / "prepared.json").write_text(prepared_files["prepared.json"])
            (out / "preparation_manifest.json").write_text(json.dumps(manifest))
            (out / "features.csv").write_text("gene\nB\nA\n")
            with self.assertRaises(ValueError):
                AUDIT.validate_preparation(protocol, out)
            (out / "features.csv").write_text(prepared_files["features.csv"])
            protocol.write_text('{"frozen": false}')
            with self.assertRaises(ValueError):
                AUDIT.validate_preparation(protocol, out)
            protocol.write_text(prepared_files["protocol.json"])
            (out / "report.json").write_text('{"status": "completed"}')
            with self.assertRaises(ValueError):
                AUDIT.validate_preparation(protocol, out)

    def test_js_matches_scipy_distance_squared_not_distance(self):
        from scipy.spatial.distance import jensenshannon
        p, q = np.array([.1, .2, .7, 0]), np.array([.7, .2, 0, .1])
        result = AUDIT.profile_metrics(p, q)["js_divergence_nats"]
        self.assertAlmostEqual(result, jensenshannon(p, q, base=np.e) ** 2, places=14)
        self.assertNotAlmostEqual(result, jensenshannon(p, q, base=np.e), places=4)


if __name__ == "__main__":
    unittest.main()
