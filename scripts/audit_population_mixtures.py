#!/usr/bin/env python3
"""Separate empirical-mixture convexity from held-source replacement.

Read the saved script-49/51 numeric artifacts. Never train or select a model.
The metric is the energy V-statistic (NOT its square root), in the original
held-fold reference coordinates. Cell draws and fitted seeds are computational
sensitivity checks on the same three donors, not new biological replicates.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path, PurePosixPath
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_population_math import weighted_energy
POPULATION = ROOT / 'outputs/scientific_revision_20260922/population'
ABLATION = ROOT / 'outputs/scientific_revision_20260922/source_ablation'
LATENT = ROOT / 'outputs/analysis/human_temporal_flow/mu.npy'
OUTPUT = ROOT / 'outputs/scientific_revision_20260923/population_mixture'
SEEDS = [0, 1, 2]
CHECKPOINT_TOLERANCE = 5e-05
IDENTITY_TOLERANCE = 1e-10
SCORES = ['real_ed', 'wrong_a_ed', 'wrong_b_ed', 'individual_wrong_mean_ed', 'mixed_wrong_ed']
METRICS = SCORES + ['wrong_a_minus_real', 'wrong_b_minus_real', 'individual_wrong_minus_real', 'mixed_wrong_minus_real', 'individual_wrong_mean_minus_mixed', 'mixture_component_a_ed', 'mixture_component_b_ed', 'mixture_component_mean_ed', 'mixture_component_pair_ed', 'mixture_identity_gain', 'mixture_identity_rhs', 'mixture_identity_residual']

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def verified_manifest(folder: Path, required, inputs: dict) -> dict:
    """Verify only explicitly required numerical artifacts; never read prose."""
    manifest_path = folder / 'output_manifest.json'
    if any((p.is_symlink() for p in [manifest_path, *manifest_path.parents])):
        raise ValueError('Input manifest must not be a symlink')
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Invalid input manifest')
    for relative, sha in manifest.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts or str(name) != relative or (not isinstance(sha, str)) or (len(sha) != 64) or any((c not in '0123456789abcdef' for c in sha)):
            raise ValueError(f'Invalid manifest entry: {relative}')
    if not set(required) <= set(manifest):
        raise ValueError(f'Required input absent from manifest: {set(required) - set(manifest)}')
    inputs[str(manifest_path)] = digest(manifest_path)
    for relative in sorted(required):
        path = folder / relative
        if not path.is_file() or any((p.is_symlink() for p in [path, *path.parents])) or digest(path) != manifest[relative]:
            raise ValueError(f'Input checksum mismatch: {path}')
        inputs[str(path)] = manifest[relative]
    return manifest

def checked_rows(identity, rows, permitted, condition=None, label='rows'):
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or (not len(rows)) or np.any(rows < 0) or np.any(rows >= len(identity)) or (len(np.unique(rows)) != len(rows)):
        raise ValueError(f'Invalid or repeated latent indices: {label}')
    selected = identity.iloc[rows]
    if not selected.donor.isin(permitted).all():
        raise ValueError(f'Excluded donor in {label}')
    if condition is not None and (not selected.cond.eq(condition).all()):
        raise ValueError(f'Excluded time in {label}')
    return rows.astype(np.int64, copy=False)

def source_rows(identity, held, permitted):
    """Only the two non-held donors' Wound1 cells can be wrong sources."""
    if len(permitted) != 2 or len(set(permitted)) != 2 or held in permitted or (set(permitted) != set(identity.donor.unique()) - {held}):
        raise ValueError('Expected exactly two distinct non-held donors')
    return {d: checked_rows(identity, identity.loc[identity.donor.eq(d) & identity.cond.eq('Wound1'), 'latent_row'].to_numpy(), [d], 'Wound1', f'wrong source {d}') for d in sorted(permitted)}

def validate_wrong_rows(identity, rows, held, permitted):
    expected = source_rows(identity, held, permitted)
    rows = checked_rows(identity, rows, permitted, 'Wound1', 'saved wrong source')
    if not np.array_equal(rows, np.concatenate(list(expected.values()))):
        raise ValueError('Saved wrong-source row order or coverage differs')
    return expected

def check_points(points, shape=None):
    points = np.asarray(points)
    if points.ndim != 2 or not len(points) or (not np.isfinite(points).all()) or (shape is not None and points.shape != shape):
        raise ValueError('Invalid prediction/coordinate dimensions or nonfinite values')
    return points

def mixture_decomposition(p1, p2, target):
    """Exact identity for two equally weighted empirical probability measures."""
    p1, p2, target = (check_points(p) for p in (p1, p2, target))
    first, second = (weighted_energy(p1, target), weighted_energy(p2, target))
    mixture = np.concatenate([p1, p2])
    weights = np.concatenate([np.full(len(p1), 0.5 / len(p1)), np.full(len(p2), 0.5 / len(p2))])
    mixed = weighted_energy(mixture, target, weights)
    pair = weighted_energy(p1, p2)
    gain = 0.5 * (first + second) - mixed
    return dict(p1_ed=first, p2_ed=second, individual_mean_ed=0.5 * (first + second), mixed_ed=mixed, component_pair_ed=pair, convexity_gain=gain, quarter_component_pair_ed=0.25 * pair, identity_residual=gain - 0.25 * pair)

def balanced_draw(pool_sizes, requested, draw):
    """Same n for real, each individual, and mixed; exact half donor masses.

    RNG order is real, wrong A, wrong B. The mixture takes prefixes of each
    individual draw, without replacement. The extra particle alternates for
    odd n; donor weights, not particle counts, remain exactly one half.
    """
    if len(pool_sizes) != 3 or requested < 2 or min(pool_sizes) < 2 or (draw < 0):
        raise ValueError('Need three pools with at least two cells and nonnegative draw')
    n = min(requested, *pool_sizes)
    rng = np.random.default_rng(draw)
    individual = [rng.choice(size, n, replace=False) for size in pool_sizes]
    counts = [n // 2, n // 2]
    if n % 2:
        counts[draw % 2] += 1
    parts = [individual[i + 1][:counts[i]] for i in range(2)]
    weights = np.concatenate([np.full(c, 0.5 / c) for c in counts])
    return (individual, parts, weights, counts)

class CachedEnergy:
    """Same V-statistic, with distances cached for repeated cell subsets."""

    def __init__(self, prediction, target):
        prediction, target = (check_points(prediction), check_points(target))
        self.distances = cdist(prediction, prediction)
        self.to_target = cdist(prediction, target).mean(axis=1)
        self.target_self = float(cdist(target, target).mean())

    def score(self, indices, weights=None):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(self.to_target)) or (len(np.unique(indices)) != len(indices)):
            raise ValueError('Invalid sampled prediction indices')
        w = np.full(len(indices), 1 / len(indices)) if weights is None else np.asarray(weights, float)
        if w.shape != (len(indices),) or not np.isfinite(w).all() or np.any(w < 0) or (not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError('Invalid empirical probability weights')
        return float(2 * w @ self.to_target[indices] - w @ self.distances[np.ix_(indices, indices)] @ w - self.target_self)

    def pair(self, a, b):
        d = self.distances
        return float(2 * d[np.ix_(a, b)].mean() - d[np.ix_(a, a)].mean() - d[np.ix_(b, b)].mean())

def hierarchical_summary(draws, metrics=METRICS):
    """Equal draw weight within fit, seed within donor, then donor within macro."""
    if draws.empty or draws.duplicated(['held_donor', 'seed', 'draw']).any() or (not np.isfinite(draws[list(metrics)].to_numpy()).all()):
        raise ValueError('Invalid draw table')
    per_fit = draws.groupby(['held_donor', 'seed'], sort=True)[list(metrics)].mean().reset_index()
    per_donor = per_fit.groupby('held_donor', sort=True)[list(metrics)].mean().reset_index()
    macro = {name: float(per_donor[name].mean()) for name in metrics}
    return (per_fit, per_donor, macro)

def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}

def validate_grid(report, contract, fits, identity):
    donors = sorted(identity.donor.unique())
    if len(donors) != 3 or report.get('status') != 'completed' or report.get('schema_version') != 1 or (report.get('experiment') != 'fixed_reference_donor_curve_and_target_marginal') or (report.get('independent_new_data') is not False) or (report.get('biological_donors') != 3) or (report.get('seeds') != SEEDS) or (contract.get('seeds') != SEEDS) or (report.get('fits') != 27) or (not isinstance(fits, list)) or (len(fits) != 27) or (contract.get('rk4_steps') != 50) or (contract.get('evaluation_seed') != 0) or (contract.get('target_cap') != 1200) or (report.get('provenance', {}).get('inputs') != contract.get('inputs')):
        raise ValueError('Saved experiment, seed grid or run contract differs')
    expected = {(h, tuple(s), seed) for h in donors for k in (1, 2) for s in itertools.combinations([d for d in donors if d != h], k) for seed in SEEDS}
    seen = set()
    for fit in fits:
        held, subset, seed = (fit.get(k) for k in ('held_donor', 'training_donors', 'seed'))
        if not isinstance(subset, list) or subset != sorted(set(subset)) or type(seed) is not int:
            raise ValueError('Invalid fit subset or seed')
        key = (held, tuple(subset), seed)
        if key not in expected or key in seen or type(fit.get('k')) is not int or (fit['k'] != len(subset)) or (fit.get('path') != f"fits/{held}/k{len(subset)}_{'_'.join(subset)}/seed{seed}") or (fit.get('training_steps') != contract.get('steps')) or (fit.get('batch_size') != contract.get('batch_size')) or (fit.get('rk4_steps') != 50) or (fit.get('reference_scaler_used_by_training') is not False):
            raise ValueError('Duplicate, incomplete or invalid fit grid/protocol')
        seen.add(key)
    if seen != expected:
        raise ValueError('Incomplete fit grid')
    return sorted([f for f in fits if f['k'] == 2], key=lambda f: (f['held_donor'], f['seed']))

def validate_scaler(scaler, mu, identity, permitted, label):
    rows = checked_rows(identity, scaler['fitting_latent_rows'], permitted, label=label)
    if not np.array_equal(rows, np.flatnonzero(identity.donor.isin(permitted))):
        raise ValueError(f'Scaler fitting identities differ: {label}')
    mean, scale = (scaler['mean'], scaler['scale'])
    if mean.shape != (1, mu.shape[1]) or scale.shape != mean.shape or (not np.isfinite(mean).all()) or (not np.isfinite(scale).all()) or (scale <= 0).any():
        raise ValueError(f'Invalid scaler parameters: {label}')
    expected_mean, expected_scale = (mu[rows].mean(0, keepdims=True), mu[rows].std(0, keepdims=True))
    expected_scale[expected_scale == 0] = 1
    if not np.array_equal(mean, expected_mean) or not np.array_equal(scale, expected_scale):
        raise ValueError(f'Scaler parameters do not reproduce: {label}')

def verify_predictions(fit, folder, reference, training, wrong, mu, identity, curve):
    held, permitted = (fit['held_donor'], fit['training_donors'])
    source = checked_rows(identity, reference['source_latent_rows'], [held], 'Wound1', 'real source')
    target = checked_rows(identity, reference['target_evaluation_latent_rows'], [held], 'Wound7', 'target')
    candidate_rows = wrong['source_latent_rows']
    validate_wrong_rows(identity, candidate_rows, held, permitted)
    if not np.array_equal(source, fit['source_latent_rows']) or not np.array_equal(target, fit['target_evaluation_latent_rows']) or (not np.array_equal(wrong['target_latent_rows'], target)) or (fit['source_count'] != len(source)) or (fit['target_evaluation_count'] != len(target)) or (fit['target_all_count'] != len(reference['target_all_latent_rows'])) or (fit['n_training_cells'] != len(training['fitting_latent_rows'])):
        raise ValueError('Saved source/target counts or identities disagree')
    mean, scale = (training['mean'], training['scale'])
    ref_mean, ref_scale = (reference['mean'], reference['scale'])
    z = ((mu - mean) / scale).astype(np.float32)
    target_z = ((mu[target] - ref_mean) / ref_scale).astype(np.float32)
    pred_z = check_points(np.load(folder / 'prediction_training_z.npy'), z[source].shape)
    pred_mu = check_points(np.load(folder / 'prediction_mu.npy'), z[source].shape)
    real = check_points(np.load(folder / 'prediction_reference_z.npy'), z[source].shape)
    wrong_pred = check_points(wrong['predicted_reference_z'], z[candidate_rows].shape)
    expected_mu = np.asarray(pred_z, float) * scale + mean
    expected_ref = (expected_mu - ref_mean) / ref_scale
    inverse_error = max(float(np.max(np.abs(expected_mu - pred_mu))), float(np.max(np.abs(expected_ref - real))))
    score_error = abs(weighted_energy(real, target_z) - fit['reference_energy_distance'])
    field = curve.LatentFlowField(mu.shape[1])
    field.load_state_dict(torch.load(folder / 'field.pt', map_location='cpu', weights_only=True))
    field.eval()
    real_reload = curve.integrate(field, torch.from_numpy(z[source]), 1 / 3, 2 / 3, 50)
    wrong_reload = curve.integrate(field, torch.from_numpy(z[candidate_rows]), 1 / 3, 2 / 3, 50)
    wrong_reload_ref = (np.asarray(wrong_reload, float) * scale + mean - ref_mean) / ref_scale
    real_error = float(np.max(np.abs(real_reload - pred_z)))
    wrong_error = float(np.max(np.abs(wrong_reload_ref - wrong_pred)))
    if inverse_error > 1e-12 or score_error > 1e-10 or max(real_error, wrong_error) > CHECKPOINT_TOLERANCE:
        raise ValueError(f"Saved prediction/checkpoint reproduction failed: {held}, {fit['seed']}")
    return (real, wrong_pred, target_z, dict(held_donor=held, seed=fit['seed'], checkpoint_sha256=digest(folder / 'field.pt'), inverse_transform_max_abs_error=inverse_error, saved_score_abs_error=score_error, real_checkpoint_max_abs_error=real_error, wrong_checkpoint_max_abs_error=wrong_error))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population-dir', type=Path, default=POPULATION)
    parser.add_argument('--ablation-dir', type=Path, default=ABLATION)
    parser.add_argument('--latent', type=Path, default=LATENT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--cells', type=int, default=200)
    parser.add_argument('--draws', type=int, default=50)
    args = parser.parse_args()
    if args.cells < 2 or args.draws < 1:
        parser.error('Need --cells >= 2 and --draws >= 1')
    population, ablation, latent, out = [p.resolve() for p in (args.population_dir, args.ablation_dir, args.latent, args.output_dir)]
    for inp in (population, ablation, latent.parent):
        if out == inp or out in inp.parents or inp in out.parents:
            raise ValueError('Input and output trees must not overlap')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {out}')
    inputs = {}
    required_population = {'report.json', 'fit_manifest.json', 'run_contract.json', 'verification.json', 'cell_identity.csv', 'curve_donor_blocks.csv', 'curve_summary.csv'}
    manifest = verified_manifest(population, required_population, inputs)
    report = json.loads((population / 'report.json').read_text())
    contract = json.loads((population / 'run_contract.json').read_text())
    previous_verification = json.loads((population / 'verification.json').read_text())
    if report.get('verification') != previous_verification or previous_verification.get('status') != 'passed':
        raise ValueError('Original verification differs or did not pass')
    fits = json.loads((population / 'fit_manifest.json').read_text())
    identity = pd.read_csv(population / 'cell_identity.csv')
    mu_sha = digest(latent)
    if mu_sha != contract.get('inputs', {}).get(LATENT.relative_to(ROOT).as_posix()):
        raise ValueError('Latent coordinates differ from original run contract')
    inputs[str(latent)] = mu_sha
    mu = check_points(np.load(latent, allow_pickle=False))
    if len(mu) != len(identity) or not identity.cell_id.is_unique or identity[['donor', 'cond', 'cell_id']].isna().any().any() or (not np.array_equal(identity.latent_row, np.arange(len(mu)))):
        raise ValueError('Cell identity and latent rows disagree')
    selected = validate_grid(report, contract, fits, identity)
    artifacts = {'field.pt', 'prediction_training_z.npy', 'prediction_mu.npy', 'prediction_reference_z.npy'}
    for fit in selected:
        prefix = fit['path']
        required_population.update((f'{prefix}/{name}' for name in artifacts | {'fit.json'}))
        required_population.update({f'{PurePosixPath(prefix).parent}/training_scaler.npz', f"fits/{fit['held_donor']}/reference.npz"})
        if set(fit.get('artifact_sha256', {})) != artifacts or any((manifest.get(f'{prefix}/{n}') != h for n, h in fit['artifact_sha256'].items())):
            raise ValueError('Fit artifact hashes differ from manifest')
    verified_manifest(population, required_population, inputs)
    required_ablation = {'report.json'} | {f"{f['held_donor']}_seed{f['seed']}_wrong_source.npz" for f in selected}
    verified_manifest(ablation, required_ablation, inputs)
    ablation_report = json.loads((ablation / 'report.json').read_text())
    if ablation_report.get('status') != 'completed' or ablation_report.get('fits') != 9 or ablation_report.get('source_sha256', {}).get('mu') != mu_sha or (ablation_report.get('source_sha256', {}).get('verified_input_manifest') != producer_manifest_digest(population / 'output_manifest.json', inputs[str(population / 'output_manifest.json')])):
        raise ValueError('Source-ablation provenance does not bind the selected population/latent inputs')
    spec = importlib.util.spec_from_file_location('mixture_saved_curve', ROOT / 'scripts/evaluate_training_donor_count.py')
    curve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curve)
    dependencies = [Path(__file__), ROOT / 'scripts/evaluate_training_donor_count.py', ROOT / 'scripts/audit_population_math.py', ROOT / 'wound_models/population_flow.py', ROOT / 'wound_models/human_wound_data.py']
    for path in dependencies:
        inputs[str(path)] = digest(path)
    torch.set_num_threads(1)
    original_blocks = pd.read_csv(population / 'curve_donor_blocks.csv')
    original_blocks = original_blocks.loc[original_blocks.method.eq('Training-target marginal (exact)')]
    if original_blocks.held_donor.duplicated().any() or set(original_blocks.held_donor) != set(identity.donor):
        raise ValueError('Original exact marginal folds differ')
    original_blocks = original_blocks.set_index('held_donor')
    marginal_rows, draw_rows, reload_checks, sampled, prepared = ([], [], [], {}, {})
    for held in sorted(identity.donor.unique()):
        permitted = sorted(set(identity.donor) - {held})
        reference = load_npz(population / 'fits' / held / 'reference.npz')
        validate_scaler(reference, mu, identity, permitted, 'reference')
        source = checked_rows(identity, reference['source_latent_rows'], [held], 'Wound1', 'real source')
        target_all = checked_rows(identity, reference['target_all_latent_rows'], [held], 'Wound7', 'all target')
        expected_source = np.flatnonzero(identity.donor.eq(held) & identity.cond.eq('Wound1'))
        expected_target = np.flatnonzero(identity.donor.eq(held) & identity.cond.eq('Wound7'))
        evaluation = target_all[np.random.default_rng(0).choice(len(target_all), 1200, replace=False)] if len(target_all) > 1200 else target_all
        if not np.array_equal(source, expected_source) or not np.array_equal(target_all, expected_target) or (not np.array_equal(evaluation, reference['target_evaluation_latent_rows'])):
            raise ValueError('Reference source/target coverage or evaluation draw differs')
        z = ((mu - reference['mean']) / reference['scale']).astype(np.float32)
        endpoint_rows = [np.flatnonzero(identity.donor.eq(d) & identity.cond.eq('Wound7')) for d in permitted]
        decomposition = mixture_decomposition(z[endpoint_rows[0]], z[endpoint_rows[1]], z[evaluation])
        saved = original_blocks.loc[held]
        errors = [abs(decomposition[a] - float(saved[b])) for a, b in [('individual_mean_ed', 'k1_ed'), ('mixed_ed', 'k2_ed'), ('convexity_gain', 'reduction')]]
        if max(errors + [abs(decomposition['identity_residual'])]) > IDENTITY_TOLERANCE:
            raise ValueError('Exact marginal identity or original scores failed reproduction')
        marginal_rows.append(dict(held_donor=held, donor_a=permitted[0], donor_b=permitted[1], n_p1=len(endpoint_rows[0]), n_p2=len(endpoint_rows[1]), n_target=len(evaluation), donor_a_mass=0.5, donor_b_mass=0.5, reported_k1_ed=float(saved.k1_ed), reported_k2_ed=float(saved.k2_ed), reported_reduction=float(saved.reduction), reported_score_max_abs_error=max(errors), **decomposition))
        candidates = source_rows(identity, held, permitted)
        sizes = [len(source)] + [len(candidates[d]) for d in permitted]
        sampled[held] = dict(target_latent_rows=evaluation, source_pool_sizes=np.asarray(sizes), wrong_donors=np.asarray(permitted))
        draws = [balanced_draw(sizes, args.cells, draw) for draw in range(args.draws)]
        for i, name in enumerate(['real', 'wrong_a', 'wrong_b']):
            positions = np.stack([d[0][i] for d in draws])
            pool = source if i == 0 else candidates[permitted[i - 1]]
            sampled[held][f'{name}_positions'] = positions
            sampled[held][f'{name}_latent_rows'] = pool[positions]
        offsets = np.cumsum([0, *sizes[:-1]])
        sampled[held]['mixture_counts'] = np.asarray([d[3] for d in draws])
        sampled[held]['mixture_weights'] = np.stack([d[2] for d in draws])
        sampled[held]['mixture_positions'] = np.stack([np.concatenate([d[1][i] + offsets[i + 1] for i in range(2)]) for d in draws])
        all_rows = np.concatenate([source, *candidates.values()])
        sampled[held]['mixture_latent_rows'] = all_rows[sampled[held]['mixture_positions']]
        sampled[held]['prediction_pool_latent_rows'] = all_rows
        sampled[held]['marginal_a_latent_rows'] = endpoint_rows[0]
        sampled[held]['marginal_b_latent_rows'] = endpoint_rows[1]
        prepared[held] = (reference, candidates, sizes, draws, offsets)
    print('Exact marginal decomposition verified for all three held folds', flush=True)
    for fit in selected:
        held, seed = (fit['held_donor'], fit['seed'])
        folder = population / fit['path']
        if json.loads((folder / 'fit.json').read_text()) != fit:
            raise ValueError('Individual fit.json differs from fit_manifest')
        reference, candidates, sizes, draws, offsets = prepared[held]
        training = load_npz(folder.parent / 'training_scaler.npz')
        validate_scaler(training, mu, identity, fit['training_donors'], 'training')
        wrong = load_npz(ablation / f'{held}_seed{seed}_wrong_source.npz')
        real, wrong_prediction, target, check = verify_predictions(fit, folder, reference, training, wrong, mu, identity, curve)
        reload_checks.append(check)
        predictions = np.concatenate([real, wrong_prediction])
        cached = CachedEnergy(predictions, target)
        for draw, (individual, parts, weights, counts) in enumerate(draws):
            indices = [individual[i] + offsets[i] for i in range(3)]
            components = [parts[i] + offsets[i + 1] for i in range(2)]
            mixture = np.concatenate(components)
            ed_real, ed_a, ed_b = [cached.score(i) for i in indices]
            ed_mix = cached.score(mixture, weights)
            component_a, component_b = [cached.score(i) for i in components]
            component_pair = cached.pair(*components)
            component_mean = 0.5 * (component_a + component_b)
            residual = component_mean - ed_mix - 0.25 * component_pair
            mass_error = max(abs(weights[:counts[0]].sum() - 0.5), abs(weights[counts[0]:].sum() - 0.5))
            if abs(residual) > IDENTITY_TOLERANCE or mass_error > 1e-14:
                raise ValueError('Sampled mixture identity or equal donor mass check failed')
            if draw == 0:
                check['cached_metric_max_abs_error'] = max(abs(ed_mix - weighted_energy(predictions[mixture], target, weights)), abs(ed_real - weighted_energy(predictions[indices[0]], target)))
                if check['cached_metric_max_abs_error'] > IDENTITY_TOLERANCE:
                    raise ValueError('Cached scorer differs from direct weighted energy')
            draw_rows.append(dict(held_donor=held, seed=seed, draw=draw, wrong_a_donor=fit['training_donors'][0], wrong_b_donor=fit['training_donors'][1], n_prediction=len(individual[0]), n_target=len(target), mixture_a_count=counts[0], mixture_b_count=counts[1], mixture_a_mass=float(weights[:counts[0]].sum()), mixture_b_mass=float(weights[counts[0]:].sum()), real_ed=ed_real, wrong_a_ed=ed_a, wrong_b_ed=ed_b, individual_wrong_mean_ed=0.5 * (ed_a + ed_b), mixed_wrong_ed=ed_mix, wrong_a_minus_real=ed_a - ed_real, wrong_b_minus_real=ed_b - ed_real, individual_wrong_minus_real=0.5 * (ed_a + ed_b) - ed_real, mixed_wrong_minus_real=ed_mix - ed_real, individual_wrong_mean_minus_mixed=0.5 * (ed_a + ed_b) - ed_mix, mixture_component_a_ed=component_a, mixture_component_b_ed=component_b, mixture_component_mean_ed=component_mean, mixture_component_pair_ed=component_pair, mixture_identity_gain=component_mean - ed_mix, mixture_identity_rhs=0.25 * component_pair, mixture_identity_residual=residual))
        print(f'{held} seed={seed}: saved checkpoint/predictions verified; {args.draws} paired draws', flush=True)
    frame = pd.DataFrame(draw_rows)
    per_fit, per_donor, macro = hierarchical_summary(frame)
    metadata = ['wrong_a_donor', 'wrong_b_donor', 'n_prediction', 'n_target']
    per_fit = per_fit.merge(frame.groupby(['held_donor', 'seed'])[metadata].first().reset_index(), on=['held_donor', 'seed'], validate='one_to_one')
    per_donor = per_donor.merge(frame.groupby('held_donor')[metadata].first().reset_index(), on='held_donor', validate='one_to_one')
    per_fit['draws'] = args.draws
    per_donor['seeds'] = len(SEEDS)
    for metric in SCORES:
        limits = frame.groupby(['held_donor', 'seed'])[metric].agg(['min', 'max']).reset_index()
        per_fit = per_fit.merge(limits.rename(columns={'min': metric + '_draw_min', 'max': metric + '_draw_max'}), on=['held_donor', 'seed'], validate='one_to_one')
    exact_macro = {name: float(np.mean([r[name] for r in marginal_rows])) for name in ['individual_mean_ed', 'mixed_ed', 'component_pair_ed', 'convexity_gain', 'quarter_component_pair_ed']}
    exact_macro['maximum_absolute_identity_residual'] = max((abs(r['identity_residual']) for r in marginal_rows))
    exact_macro['held_donors'] = len(marginal_rows)
    macro.update(held_donors=len(per_donor), saved_fits=len(per_fit), draws_per_fit=args.draws)
    verification = dict(status='passed', relevant_input_manifests_verified=True, verified_population_files=len(required_population), verified_ablation_files=len(required_ablation), manifest_scope='Only named inputs consumed by this audit; other artifacts and prose were not read or certified', fit_grid_verified=True, donor_and_time_exclusions_verified=True, scalers_recalculated=True, shared_target_and_cross_seed_draw_indices=True, checkpoint_reload_tolerance=CHECKPOINT_TOLERANCE, checkpoint_checks=reload_checks, no_new_training=True, maximum_marginal_identity_residual=exact_macro['maximum_absolute_identity_residual'], maximum_sampled_identity_residual=float(frame.mixture_identity_residual.abs().max()), maximum_donor_mass_error=float(max((frame.mixture_a_mass - 0.5).abs().max(), (frame.mixture_b_mass - 0.5).abs().max())))
    protocol = dict(requested_prediction_cells=args.cells, draws_per_fit=args.draws, seeds=SEEDS, count_adaptation='Within each held fold use min(requested, real pool, wrong A pool, wrong B pool) for every method; fail below two', target='All methods and draws use the same original held Wound7 evaluation indices; no target resampling', source_sampling='default_rng(draw), draw real then lexicographic wrong A then B without replacement; reuse identical indices across model seeds', mixture="Take each wrong draw's prefix; floor/ceil counts alternate for odd n. Assign each donor mass exactly 1/2 using .5/count weights", aggregation='draw mean within saved fit, equal seed mean within held donor, equal held-donor macro mean', metric="Float64 empirical energy V-statistic: 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||, self zeros included; no square root", coordinates='Original float32 reference standardization, then float64 distances. Both non-held donors define each scoring geometry', exact_identity='(ED(P1,Q)+ED(P2,Q))/2 - ED((P1+P2)/2,Q) = ED(P1,P2)/4', identity_scope='Full empirical training-Wound7 target marginals for script49; additionally the actual mixture-component source subsamples for each saved field/draw', count_contrast_warning='individual_wrong_mean_minus_mixed uses n-cell individual distributions vs an n-total-cell mixture. It is NOT the exact convexity identity; mixture_identity_* uses its actual component subsets', uncertainty='Draw minima/maxima are descriptive computational ranges, not confidence intervals. No p values or new biological replicates', inference='Mixture gains alone do not identify held-source information or biological trajectories. This identity does not decompose benefits of jointly trained flow fields', fitted_models='Nine existing k2 fields; CPU checkpoint verification only, no optimizer or model training')
    result = dict(schema_version=1, status='completed', experiment='population_mixture_audit', biological_donors=3, new_biological_donors=0, independent_new_data=False, new_model_training=False, protocol=protocol, exact_marginal=dict(per_fold=marginal_rows, macro=exact_macro), source_comparison=dict(per_donor=per_donor.to_dict('records'), macro=macro, per_fit=per_fit.to_dict('records')), verification=verification, input_sha256=inputs, historical_generator_sha256=dict(population=contract['inputs'].get('scripts/repair_population_protocol.py'), ablation=ablation_report['source_sha256'].get('script')), historical_code_note='Historical generator SHA is preserved, not asserted equal to the current edited generators. All numeric inputs are manifest-bound and nine checkpoints are re-integrated', runtime=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, torch=torch.__version__, checkpoint_device='cpu', torch_threads=1))
    if any((digest(Path(path)) != sha for path, sha in inputs.items())):
        raise ValueError('An input or execution dependency changed while this audit was running')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Output became nonempty during computation')
    pd.DataFrame(marginal_rows).to_csv(out / 'exact_marginal_decomposition.csv', index=False)
    frame.to_csv(out / 'source_per_draw.csv', index=False)
    per_fit.to_csv(out / 'source_per_fit.csv', index=False)
    per_donor.to_csv(out / 'source_per_donor.csv', index=False)
    pd.DataFrame([macro]).to_csv(out / 'source_macro.csv', index=False)
    for held, arrays in sampled.items():
        np.savez_compressed(out / f'{held}_sampled_indices.npz', **arrays)
    save_json(out / 'input_sha256.json', inputs)
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'verification.json', verification)
    save_json(out / 'report.json', result)
    save_json(out / 'output_manifest.json', {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(exact_marginal=exact_macro, source_comparison=macro), indent=2), flush=True)

def producer_manifest_digest(path, current_digest):
    """Resolve an exported manifest to its byte-exact producing snapshot."""
    provenance = ROOT / 'EXPORT_PROVENANCE.json'
    if not provenance.is_file() or not path.is_relative_to(ROOT):
        return current_digest
    relative = path.relative_to(ROOT).as_posix()
    records = json.loads(provenance.read_text())['files']
    record = next((row for row in records if row['exported'] == relative), None)
    if record is None or 'original' not in record:
        return current_digest
    original = ROOT / record['original']
    if current_digest != record['exported_sha256'] or digest(original) != record['source_sha256']:
        raise ValueError('Exported/producing manifest provenance mismatch')
    return record['source_sha256']

if __name__ == '__main__':
    main()
