#!/usr/bin/env python3
"""Audit held-source information using the saved two-donor flow checkpoints.

No model is trained. Each fitted field integrates the real held-donor Wound1
population and an equal-donor, source-count-matched draw from the two training
donors' Wound1 populations. Both are scored against the *same* held Wound7
cells in the fixed reference geometry from the controlled donor-curve rerun.
This is a retrospective three-donor sensitivity analysis, not validation on
new people or a test of cell-specific counterfactual trajectories.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import torch
from importlib.util import module_from_spec, spec_from_file_location
ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('population_protocol_repair', ROOT / 'scripts/repair_population_protocol.py')
REPAIR = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPAIR)
INPUT = ROOT / 'outputs/scientific_revision_20260922/population'
OUTPUT = ROOT / 'outputs/scientific_revision_20260922/source_ablation'

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def source_rows(identity: pd.DataFrame, held: str, permitted: list[str]) -> dict[str, np.ndarray]:
    """Return only Wound1 cells of the permitted training donors."""
    if held in permitted or len(permitted) != 2 or len(set(permitted)) != 2:
        raise ValueError('Expected exactly two distinct non-held training donors')
    rows = {}
    for donor in sorted(permitted):
        match = identity.donor.eq(donor) & identity.cond.eq('Wound1')
        rows[donor] = identity.loc[match, 'latent_row'].to_numpy(dtype=np.int64)
        if len(rows[donor]) == 0:
            raise ValueError(f'No permitted Wound1 cells for {donor}')
    return rows

def paired_summary(draws: pd.DataFrame) -> list[dict]:
    """Average draws within each field, then seeds within each donor."""
    grouped = draws.groupby(['held_donor', 'seed'], sort=True).agg(real_ed=('real_ed', 'first'), wrong_mean_ed=('wrong_source_ed', 'mean'), wrong_min_ed=('wrong_source_ed', 'min'), wrong_max_ed=('wrong_source_ed', 'max'), marginal_mean_ed=('target_marginal_ed', 'mean'), wrong_worse_draws=('wrong_minus_real', lambda x: int((x > 0).sum())), wrong_better_than_marginal_draws=('wrong_minus_marginal', lambda x: int((x < 0).sum())), draws=('draw', 'size')).reset_index()
    grouped['wrong_minus_real'] = grouped.wrong_mean_ed - grouped.real_ed
    grouped['real_minus_marginal'] = grouped.real_ed - grouped.marginal_mean_ed
    return grouped.to_dict('records')

def verified_manifest(inp: Path) -> dict[str, str]:
    """Require complete coverage, safe relative paths, and intact artifacts."""
    manifest = json.loads((inp / 'output_manifest.json').read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Empty or invalid controlled output manifest')
    for relative, expected in manifest.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts or str(name) != relative or (not isinstance(expected, str)) or (len(expected) != 64) or any((c not in '0123456789abcdef' for c in expected)):
            raise ValueError(f'Invalid controlled manifest entry: {relative}')
        path = inp / relative
        if not path.is_file() or any((p.is_symlink() for p in [path, *path.parents])) or digest(path) != expected:
            raise ValueError(f'Controlled input checksum mismatch: {relative}')
    actual = {p.relative_to(inp).as_posix() for p in inp.rglob('*') if p.is_file() and p != inp / 'output_manifest.json'}
    required = {'report.json', 'fit_manifest.json', 'run_contract.json', 'verification.json', 'cell_identity.csv', 'curve_marginal_draw_scores.csv'}
    if set(manifest) != actual or not required <= set(manifest):
        raise ValueError('Controlled manifest does not cover all required artifacts')
    return manifest

def validate_fit_contract(inp: Path, report: dict, fits: list, identity: pd.DataFrame, manifest: dict[str, str]) -> list[dict]:
    """Validate the complete held-donor/subset/seed grid, not just k=2 counts."""
    contract = json.loads((inp / 'run_contract.json').read_text())
    verification = json.loads((inp / 'verification.json').read_text())
    seeds = report.get('seeds')
    if not isinstance(seeds, list) or not seeds or any((type(seed) is not int or seed < 0 for seed in seeds)) or (len(set(seeds)) != len(seeds)):
        raise ValueError('Report seeds must be distinct nonnegative integers')
    donors = sorted(identity.donor.unique())
    expected = {(held, tuple(subset), seed) for held in donors for k in (1, 2) for subset in itertools.combinations([d for d in donors if d != held], k) for seed in seeds}
    if report.get('status') != 'completed' or report.get('schema_version') != 1 or report.get('experiment') != 'fixed_reference_donor_curve_and_target_marginal' or (report.get('biological_donors') != 3) or (len(donors) != 3) or (report.get('independent_new_data') is not False) or (type(report.get('fits')) is not int) or (report['fits'] != len(expected)) or (not isinstance(fits, list)) or (len(fits) != len(expected)) or (report.get('verification') != verification) or (verification.get('status') != 'passed') or (verification.get('saved_fit_count') != len(expected)):
        raise ValueError('Report/fit-manifest/verification count or scope mismatch')
    for key in ('unique_cell_ids', 'training_donor_exclusion', 'reference_excludes_held_donor', 'source_target_ids_checked'):
        if verification.get(key) is not True:
            raise ValueError(f'Unverified controlled input: {key}')
    for key, limit in (('maximum_inverse_transform_difference', 1e-12), ('maximum_score_reproduction_difference', 1e-10), ('maximum_cpu_checkpoint_prediction_difference', 5e-05)):
        value = verification.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or (not 0 <= value <= limit):
            raise ValueError(f'Controlled verification tolerance failed: {key}')
    if contract.get('schema_version') != 1 or contract.get('seeds') != seeds or any((type(contract.get(k)) is not int or contract[k] < 1 for k in ('steps', 'batch_size'))) or any((contract.get(k) != v for k, v in {'rk4_steps': 50, 'evaluation_seed': 0, 'marginal_draws': 50, 'target_cap': 1200}.items())) or (report.get('provenance', {}).get('inputs') != contract.get('inputs')) or (contract.get('inputs', {}).get(REPAIR.LATENT.relative_to(ROOT).as_posix()) != digest(REPAIR.LATENT)):
        raise ValueError('Controlled run contract differs from report or latent coordinates')
    if not identity.cell_id.is_unique or not np.array_equal(identity.latent_row, np.arange(len(identity))) or identity.donor.isna().any():
        raise ValueError('Controlled cell identity contract failed')
    seen = set()
    artifacts = {'field.pt', 'prediction_training_z.npy', 'prediction_mu.npy', 'prediction_reference_z.npy'}
    for fit in fits:
        held, subset, seed, k = (fit.get(name) for name in ('held_donor', 'training_donors', 'seed', 'k'))
        if held not in donors or not isinstance(subset, list) or (not subset) or any((d not in donors for d in subset)) or (subset != sorted(set(subset))) or (held in subset) or (type(k) is not int) or (k != len(subset)) or (type(seed) is not int) or (seed not in seeds):
            raise ValueError('Invalid held donor, training subset, k, or seed in fit manifest')
        key = (held, tuple(subset), seed)
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected fit configuration')
        seen.add(key)
        relative = f"fits/{held}/k{k}_{'_'.join(subset)}/seed{seed}"
        if fit.get('path') != relative or fit.get('training_steps') != contract['steps'] or fit.get('batch_size') != contract['batch_size'] or (fit.get('rk4_steps') != contract['rk4_steps']) or (fit.get('reference_scaler_used_by_training') is not False) or (set(fit.get('artifact_sha256', {})) != artifacts):
            raise ValueError('Fit protocol or artifact contract mismatch')
        required = {f'{relative}/fit.json', f'fits/{held}/reference.npz', f'{PurePosixPath(relative).parent}/training_scaler.npz'}
        if not required <= set(manifest) or json.loads((inp / relative / 'fit.json').read_text()) != fit:
            raise ValueError('Fit manifest differs from saved fit or scaler coverage')
        for name, expected_hash in fit['artifact_sha256'].items():
            if manifest.get(f'{relative}/{name}') != expected_hash:
                raise ValueError('Fit artifact hashes differ from controlled output manifest')
        for name, count, cond in (('source_latent_rows', 'source_count', 'Wound1'), ('target_evaluation_latent_rows', 'target_evaluation_count', 'Wound7')):
            rows = fit.get(name)
            if not isinstance(rows, list) or not rows or len(rows) != fit.get(count) or any((type(r) is not int or not 0 <= r < len(identity) for r in rows)) or (len(set(rows)) != len(rows)) or (not identity.iloc[rows].donor.eq(held).all()) or (not identity.iloc[rows].cond.eq(cond).all()):
                raise ValueError('Fit source/target identity mismatch')
    if seen != expected:
        raise ValueError('Incomplete controlled fit grid')
    return [fit for fit in fits if fit['k'] == 2]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--draws', type=int, default=50)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    inp, out = (args.input_dir.resolve(), args.output_dir.resolve())
    if inp == out or inp in out.parents or out in inp.parents:
        raise ValueError('Input and output trees must not overlap')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing ablation: {out}')
    original_manifest = verified_manifest(inp)
    report = json.loads((inp / 'report.json').read_text())
    identity = pd.read_csv(inp / 'cell_identity.csv')
    mu = np.load(REPAIR.LATENT)
    if len(identity) != len(mu) or not identity.cell_id.is_unique:
        raise ValueError('Latent-coordinate identity contract failed')
    fit_manifest = json.loads((inp / 'fit_manifest.json').read_text())
    selected = validate_fit_contract(inp, report, fit_manifest, identity, original_manifest)
    if args.draws > 50:
        raise ValueError('Requested draws exceed the 50 saved target-marginal draws')
    existing = pd.read_csv(inp / 'curve_marginal_draw_scores.csv')
    existing = existing.loc[existing.k.eq(2)].set_index(['held_donor', 'training_donors', 'draw'])
    expected_draws = {(f['held_donor'], '|'.join(f['training_donors']), d) for f in selected for d in range(50)}
    if not existing.index.is_unique or set(existing.index) != expected_draws or (not np.isfinite(existing.equal_donor_ed.to_numpy()).all()):
        raise ValueError('Invalid matched target-marginal draw coverage')
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    rows, source_index = ([], {})
    for fit in selected:
        held, seed = (fit['held_donor'], fit['seed'])
        folder = inp / fit['path']
        training = np.load(folder.parent / 'training_scaler.npz')
        reference = np.load(folder.parent.parent / 'reference.npz')
        source = reference['source_latent_rows']
        target = reference['target_evaluation_latent_rows']
        if not np.array_equal(source, np.asarray(fit['source_latent_rows'])) or not np.array_equal(target, np.asarray(fit['target_evaluation_latent_rows'])):
            raise ValueError('Saved source/target indices disagree')
        if not identity.iloc[source].donor.eq(held).all() or not identity.iloc[source].cond.eq('Wound1').all():
            raise ValueError('Actual source identities disagree')
        if not identity.iloc[target].donor.eq(held).all() or not identity.iloc[target].cond.eq('Wound7').all():
            raise ValueError('Held target identities disagree')
        if set(identity.iloc[training['fitting_latent_rows']].donor) != set(fit['training_donors']):
            raise ValueError('Training scaler contains an excluded donor')
        if held in set(identity.iloc[reference['fitting_latent_rows']].donor):
            raise ValueError('Reference scaler contains the held donor')
        candidates = source_rows(identity, held, fit['training_donors'])
        if min(map(len, candidates.values())) < (len(source) + 1) // 2:
            raise ValueError('Not enough training Wound1 cells for without-replacement matched draws')
        candidate_rows = np.concatenate([candidates[d] for d in sorted(candidates)])
        field = REPAIR.CURVE.LatentFlowField(mu.shape[1])
        checkpoint = folder / 'field.pt'
        if digest(checkpoint) != fit['artifact_sha256']['field.pt']:
            raise ValueError('Checkpoint hash differs from verified fit')
        field.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        candidate_z = REPAIR.transform(mu[candidate_rows], training['mean'], training['scale'])
        candidate_pred_z = REPAIR.CURVE.integrate(field, torch.from_numpy(candidate_z), REPAIR.COND_TIME['Wound1'], REPAIR.COND_TIME['Wound7'], n_steps=fit['rk4_steps'])
        _, candidate_ref = REPAIR.reference_prediction(candidate_pred_z, training['mean'], training['scale'], reference['mean'], reference['scale'])
        real_prediction = np.load(folder / 'prediction_reference_z.npy')
        target_ref = REPAIR.transform(mu[target], reference['mean'], reference['scale'])
        real_ed = REPAIR.weighted_energy(real_prediction, target_ref)
        if abs(real_ed - fit['reference_energy_distance']) > 1e-10:
            raise ValueError('Real-source score not reproduced from saved prediction')
        np.savez(out / f'{held}_seed{seed}_wrong_source.npz', source_latent_rows=candidate_rows, predicted_reference_z=candidate_ref, target_latent_rows=target)
        key = (held, '|'.join(fit['training_donors']))
        if held not in source_index:
            source_index[held] = []
            for draw in range(args.draws):
                sampled, weights, counts = REPAIR.marginal_draw(candidates, len(source), draw)
                source_index[held].append((sampled, weights, counts))
        lookup = {int(row): pos for pos, row in enumerate(candidate_rows)}
        for draw, (sampled, weights, counts) in enumerate(source_index[held]):
            positions = np.asarray([lookup[int(row)] for row in sampled])
            wrong_ed = REPAIR.weighted_energy(candidate_ref[positions], target_ref, weights)
            try:
                marginal_ed = float(existing.loc[(*key, draw), 'equal_donor_ed'])
            except KeyError as error:
                raise ValueError('Missing matched target-marginal draw') from error
            rows.append(dict(held_donor=held, seed=seed, draw=draw, source_count=len(source), target_count=len(target), wrong_source_donor_counts=counts, real_ed=real_ed, wrong_source_ed=wrong_ed, target_marginal_ed=marginal_ed, wrong_minus_real=wrong_ed - real_ed, wrong_minus_marginal=wrong_ed - marginal_ed))
        print(f"{held} seed={seed}: real={real_ed:.4f}, wrong mean={np.mean([r['wrong_source_ed'] for r in rows[-args.draws:]]):.4f}", flush=True)
    pd.DataFrame(rows).drop(columns='wrong_source_donor_counts').to_csv(out / 'draw_scores.csv', index=False)
    for held, indices in source_index.items():
        np.savez(out / f'{held}_draw_indices.npz', source_latent_rows=np.stack([v[0] for v in indices]), equal_donor_weights=np.stack([v[1] for v in indices]))
    per_fit = paired_summary(pd.DataFrame(rows))
    pd.DataFrame(per_fit).to_csv(out / 'per_fit.csv', index=False)
    donor = pd.DataFrame(per_fit).groupby('held_donor', sort=True).agg(real_ed=('real_ed', 'mean'), wrong_ed=('wrong_mean_ed', 'mean'), marginal_ed=('marginal_mean_ed', 'mean'), wrong_minus_real=('wrong_minus_real', 'mean'), real_minus_marginal=('real_minus_marginal', 'mean')).reset_index()
    donor.to_csv(out / 'per_donor.csv', index=False)
    result = {'status': 'completed', 'scope': 'Retrospective source-replacement audit; three healthy acute-wound donors only', 'protocol': {'model': f"Existing k=2 fields for {len(report['seeds'])} declared seeds; no training or target-dependent selection", 'sources': f'Real held Wound1 versus source-count-matched equal-donor Wound1 draws from permitted training donors; {args.draws} draws per field, without replacement within each draw; draws can overlap and the same held-fold draw indices are reused across seeds', 'target': 'Identical saved held-donor Wound7 indices in every comparison', 'metric': 'Float64 weighted empirical energy distance in the same held-fold reference coordinates', 'marginal': 'Existing matched training Wound7 target-marginal draw scores for the same held folds; its draw IDs are not paired cell samples with wrong-source draws', 'inference': 'Seeds and cell draws are dependent computational repetitions, not independent donors, hypothesis-test p values, or confirmation of source-specific biological trajectories'}, 'draws_per_fit': args.draws, 'fits': len(selected), 'seeds': report['seeds'], 'per_donor': donor.to_dict('records'), 'per_fit': per_fit, 'macro_equal_donor': {name: float(donor[name].mean()) for name in ['real_ed', 'wrong_ed', 'marginal_ed', 'wrong_minus_real', 'real_minus_marginal']}, 'checks': {'input_manifest_verified': True, 'fit_checkpoint_hashes_verified': True, 'source_target_identity_verified': True, 'real_score_reproduced': True, 'no_new_fit': True, 'n_independent_biological_donors': len(donor)}, 'source_sha256': {'mu': digest(REPAIR.LATENT), 'script': digest(Path(__file__)), 'verified_input_manifest': digest(inp / 'output_manifest.json')}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'output_manifest.json').write_text(json.dumps({str(p.relative_to(out)): digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'output_manifest.json'}, indent=2) + '\n')
    print(json.dumps(result['macro_equal_donor'], indent=2))
if __name__ == '__main__':
    main()
