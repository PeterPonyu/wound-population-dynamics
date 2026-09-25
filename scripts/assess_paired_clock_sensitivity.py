#!/usr/bin/env python3
"""Paired, exact-target clock sensitivity in the existing acute-wound cohort.

This prospective computational protocol addresses a retrospective comparison
defect. It neither adds donors nor selects a biological clock on held targets.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.population_flow import LatentFlowField, compute_cfm_loss
from wound_models.human_wound_data import COND_ORDER, paired_endpoint_pairs
from scripts.audit_population_math import weighted_energy
SPEC = importlib.util.spec_from_file_location('clock_integrator', ROOT / 'scripts/evaluate_training_donor_count.py')
INTEGRATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTEGRATOR)
LATENT = ROOT / 'outputs/analysis/human_temporal_flow/mu.npy'
OBS = ROOT / 'outputs/analysis/human_temporal_flow/obs_fibroblast.csv'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def finite(value, label):
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(f'Non-finite {label}; no successful result emitted')

def checked_member(folder, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
        raise ValueError('Unsafe output manifest path')
    path = folder / name
    if not path.is_file() or any((p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Missing or symlinked output artifact')
    return path

def validate_existing(out, seeds, donors):
    """Read and validate ALL completed outputs before any resume-side write."""
    expected = {(seed, axis) for seed in seeds for axis in time_axes()}
    fits, streams = ({}, {})
    for path in sorted(out.glob('seed*/*/fit.json')):
        checked_member(out, path.relative_to(out).as_posix())
        fit = json.loads(path.read_text())
        key = (fit.get('seed'), fit.get('axis'))
        if key not in expected or key in fits or path != out / f'seed{key[0]}' / key[1] / 'fit.json':
            raise ValueError('Completed fit identity or grid differs')
        artifacts = fit.get('artifacts', {})
        if set(artifacts) != {'field.pt', *(f'{donor}.npz' for donor in donors)}:
            raise ValueError('Completed fit artifact inventory differs')
        for name, digest in artifacts.items():
            member = checked_member(path.parent, name)
            if sha(member) != digest:
                raise ValueError('Completed fit artifact differs')
            if name.endswith('.npz'):
                with np.load(member, allow_pickle=False) as arrays:
                    if set(arrays.files) != {'source_rows', 'target_rows', 'prediction', 'refined_prediction', 'centroid_prediction'}:
                        raise ValueError('Completed prediction schema differs')
                    for label in arrays.files:
                        finite(arrays[label], label)
        weights = torch.load(path.parent / 'field.pt', map_location='cpu', weights_only=True)
        if any((not torch.isfinite(value).all() for value in weights.values())):
            raise FloatingPointError('Non-finite completed model weights')
        random = fit.get('random_streams', {})
        finite(random.get('final_training_loss', float('nan')), 'completed training loss')
        stream = {name: random.get(name) for name in ('endpoint_draws_sha256', 'initial_weights_sha256', 'initial_interpolation_rng_sha256')}
        if any((not isinstance(d, str) or len(d) != 64 for d in stream.values())):
            raise ValueError('Missing paired random stream fingerprints')
        if key[0] in streams and stream != streams[key[0]]:
            raise ValueError('Completed clocks have unpaired random streams')
        streams[key[0]] = stream
        fits[key] = fit
    manifest_path = out / 'output_manifest.json'
    if manifest_path.exists():
        checked_member(out, 'output_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p != manifest_path}
        required = {'report.json', 'run_contract.json', 'training_scaler.npz', 'summary.csv', 'per_donor_seed.csv'}
        if set(fits) != expected or set(manifest) != actual or (not required <= set(manifest)):
            raise ValueError('Completed output manifest or fit grid is incomplete')
        for name, digest in manifest.items():
            if sha(checked_member(out, name)) != digest:
                raise ValueError('Completed output manifest checksum differs')
        report = json.loads((out / 'report.json').read_text())
        if report.get('status') != 'completed' or report.get('fits') != len(expected):
            raise ValueError('Completed report disagrees with fit grid')
        validate_records(pd.read_csv(out / 'per_donor_seed.csv'), seeds, donors)
        finite(pd.read_csv(out / 'summary.csv').select_dtypes(include='number'), 'completed summary')
    elif (out / 'report.json').exists():
        raise ValueError('Completion report lacks its manifest; preserve for manual inspection')
    return (fits, manifest_path.exists())

def validate_records(frame, seeds, donors):
    expected = {(s, axis, d) for s in seeds for axis in time_axes() for d in donors}
    keys = list(frame[['seed', 'axis', 'donor']].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicated donor/seed/axis score grid')
    finite(frame[['flow_ed', 'refined_flow_ed', 'unchanged_source_ed', 'centroid_ed', 'integration_max_difference', 'target_time']].to_numpy(), 'per-donor scores')

def time_axes():
    days = np.array([0.0, 1.0, 7.0, 30.0])
    return {name: dict(zip(COND_ORDER, values.tolist())) for name, values in {'rank': np.linspace(0.0, 1.0, 4), 'days': days / 30, 'sqrt_days': np.sqrt(days / 30), 'log_days': np.log1p(days) / np.log(31)}.items()}

def prepare(mu, obs, holdout):
    if mu.ndim != 2 or not mu.shape[1] or len(mu) != len(obs) or (not np.isfinite(mu).all()):
        raise ValueError('Finite latent matrix must match observation rows')
    if holdout not in ('Wound1', 'Wound7'):
        raise ValueError('Only internal timepoint holdouts are supported')
    donors, conditions = (obs.donor.to_numpy(), obs.cond.to_numpy())
    if obs[['donor', 'cond']].isna().any().any():
        raise ValueError('Missing donor or condition identity')
    names = sorted(set(donors))
    if any((not isinstance(d, str) or Path(d).name != d or d in ('.', '..') for d in names)):
        raise ValueError('Unsafe donor identity')
    if len(names) != 3 or set(conditions) != set(COND_ORDER):
        raise ValueError('Expected the existing three-donor, four-time cohort')
    if any((not np.any((donors == d) & (conditions == c)) for d in names for c in COND_ORDER)):
        raise ValueError('Every donor must supply each observed time')
    training = conditions != holdout
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        mean, scale = (mu[training].mean(0), mu[training].std(0))
        finite(mean, 'training mean')
        finite(scale, 'training scale')
        scale[scale == 0] = 1
        z = ((mu - mean) / scale).astype(np.float32)
        finite(z, 'float32 standardized coordinates')
    observed = [c for c in COND_ORDER if c != holdout]
    segments = list(zip(observed[:-1], observed[1:]))
    pairs = paired_endpoint_pairs(donors, conditions, names, segments)
    return (z, mean, scale, training, pairs)

def train_axis(z, obs, pairs, clock, *, seed, steps, batch_size, device):
    """Reset initialization, endpoint RNG and Torch interpolation RNG per axis."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    field = LatentFlowField(z.shape[1]).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=0.001)
    state = torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else torch.get_rng_state()
    interpolation_state = hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    initial = hashlib.sha256(b''.join((v.detach().cpu().numpy().tobytes() for v in field.state_dict().values()))).hexdigest()
    cache = {(d, c): torch.from_numpy(z[obs.donor.eq(d) & obs.cond.eq(c)]).to(device) for d, a, b in pairs for c in (a, b)}
    schedule = hashlib.sha256()
    for step in range(steps):
        donor, source, target = pairs[step % len(pairs)]
        a, b = (cache[donor, source], cache[donor, target])
        ia = rng.integers(0, len(a), batch_size, dtype=np.int64)
        ib = rng.integers(0, len(b), batch_size, dtype=np.int64)
        schedule.update(ia.tobytes() + ib.tobytes())
        opt.zero_grad()
        loss = compute_cfm_loss(field, b[torch.from_numpy(ib).to(device)], a[torch.from_numpy(ia).to(device)], t_0=clock[source], t_1=clock[target])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; no successful result emitted')
        loss.backward()
        opt.step()
    return (field.eval(), {'endpoint_draws_sha256': schedule.hexdigest(), 'initial_weights_sha256': initial, 'initial_interpolation_rng_sha256': interpolation_state, 'final_training_loss': float(loss.detach().cpu())})

def selected_rows(obs, donor, condition, cap):
    rows = np.flatnonzero(obs.donor.eq(donor) & obs.cond.eq(condition))
    return rows[np.random.default_rng(0).choice(len(rows), cap, replace=False)] if len(rows) > cap else rows

def run(args):
    if min(args.steps, args.batch_size, args.rk4_steps, args.evaluation_cap) < 1:
        raise ValueError('All computation counts must be positive')
    seeds = [int(value) for value in args.seeds.split(',')]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError('Use distinct nonnegative training seeds')
    if args.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(4)
    if args.output_dir.is_symlink():
        raise ValueError('Output directory cannot be a symlink')
    out = args.output_dir.resolve()
    sources = [args.latent.resolve(), args.obs.resolve(), Path(__file__), ROOT / 'wound_models/population_flow.py', ROOT / 'wound_models/human_wound_data.py', ROOT / 'scripts/evaluate_training_donor_count.py', ROOT / 'scripts/audit_population_math.py']
    if any((path == out or out in path.parents for path in sources)):
        raise ValueError('Output directory cannot contain experiment inputs')
    contract = {'schema_version': 1, 'inputs': {str(path): sha(path) for path in sources}, 'holdout': args.holdout, 'seeds': seeds, 'steps': args.steps, 'batch_size': args.batch_size, 'device': args.device, 'rk4_steps': args.rk4_steps, 'evaluation_cap': args.evaluation_cap, 'axes': time_axes(), 'target_used_for_model_selection': False}
    if out.exists() and any(out.iterdir()):
        if not args.resume or not (out / 'run_contract.json').exists():
            raise FileExistsError('Use a fresh output directory, or resume an identical protocol')
        if json.loads((out / 'run_contract.json').read_text()) != contract:
            raise ValueError('Resume inputs or protocol differ')
    mu, obs = (np.load(args.latent, allow_pickle=False), pd.read_csv(args.obs))
    z, mean, scale, train, pairs = prepare(mu, obs, args.holdout)
    donors = sorted(set(obs.donor))
    existing, complete_run = validate_existing(out, seeds, donors)
    if (out / 'training_scaler.npz').exists():
        with np.load(checked_member(out, 'training_scaler.npz'), allow_pickle=False) as scaler:
            for key, value in (('mean', mean), ('scale', scale), ('training_rows', np.flatnonzero(train))):
                if not np.array_equal(scaler[key], value):
                    raise ValueError('Existing training scaler differs')
    if complete_run:
        print('Completed clock run verified; no files changed', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'run_contract.json').exists():
        save_json(out / 'run_contract.json', contract)
    if not (out / 'training_scaler.npz').exists():
        np.savez(out / 'training_scaler.npz', mean=mean, scale=scale, training_rows=np.flatnonzero(train))
    source = COND_ORDER[COND_ORDER.index(args.holdout) - 1]
    anchor = COND_ORDER[COND_ORDER.index(args.holdout) + 1]
    records, fits, streams = ([], [], {})
    for seed in seeds:
        for name, clock in time_axes().items():
            folder = out / f'seed{seed}' / name
            folder.mkdir(parents=True, exist_ok=True)
            completed = folder / 'fit.json'
            if (seed, name) in existing:
                fit = existing[seed, name]
                random = fit['random_streams']
            else:
                field, random = train_axis(z, obs, pairs, clock, seed=seed, steps=args.steps, batch_size=args.batch_size, device=args.device)
                torch.save(field.state_dict(), folder / 'field.pt')
            key = {k: v for k, v in random.items() if k.endswith('sha256')}
            if seed in streams and streams[seed] != key:
                raise ValueError('Clock axes do not share initialization and random draws')
            streams[seed] = key
            for donor in donors:
                sr = selected_rows(obs, donor, source, args.evaluation_cap)
                tr = selected_rows(obs, donor, args.holdout, args.evaluation_cap)
                ar = np.flatnonzero(obs.donor.eq(donor) & obs.cond.eq(anchor))
                if (seed, name) in existing:
                    with np.load(folder / f'{donor}.npz', allow_pickle=False) as saved:
                        if not np.array_equal(saved['source_rows'], sr) or not np.array_equal(saved['target_rows'], tr):
                            raise ValueError('Completed source/target indices differ')
                        prediction, refined, translated = (saved[k] for k in ('prediction', 'refined_prediction', 'centroid_prediction'))
                else:
                    initial = torch.from_numpy(z[sr]).to(args.device)
                    prediction = INTEGRATOR.integrate(field, initial, clock[source], clock[args.holdout], args.rk4_steps)
                    refined = INTEGRATOR.integrate(field, initial, clock[source], clock[args.holdout], args.rk4_steps * 4)
                    alpha = (clock[args.holdout] - clock[source]) / (clock[anchor] - clock[source])
                    full_source = z[obs.donor.eq(donor) & obs.cond.eq(source)]
                    with np.errstate(over='raise', invalid='raise'):
                        translated = z[sr] + alpha * (z[ar].mean(0) - full_source.mean(0))
                for value in (prediction, refined, translated):
                    finite(value, 'integration or centroid prediction')
                    if value.shape != z[sr].shape:
                        raise ValueError('Prediction shape differs from source')
                record = {'seed': seed, 'axis': name, 'donor': donor, 'source': source, 'target': args.holdout, 'target_time': clock[args.holdout], 'flow_ed': weighted_energy(prediction, z[tr]), 'refined_flow_ed': weighted_energy(refined, z[tr]), 'unchanged_source_ed': weighted_energy(z[sr], z[tr]), 'centroid_ed': weighted_energy(translated, z[tr]), 'integration_max_difference': float(np.max(np.abs(prediction - refined)))}
                finite([value for value in record.values() if isinstance(value, (float, int))], 'computed scores')
                records.append(record)
                if (seed, name) not in existing:
                    np.savez(folder / f'{donor}.npz', source_rows=sr, target_rows=tr, prediction=prediction, refined_prediction=refined, centroid_prediction=translated)
            if (seed, name) not in existing:
                fit = {'seed': seed, 'axis': name, 'random_streams': random, 'artifacts': {p.name: sha(p) for p in sorted(folder.iterdir()) if p.name != 'fit.json'}}
                save_json(completed, fit)
            fits.append(fit)
            print(f'Completed clock={name} seed={seed} at exact {args.holdout} time', flush=True)
    frame = pd.DataFrame(records)
    validate_records(frame, seeds, donors)
    frame.to_csv(out / 'per_donor_seed.csv', index=False)
    summary = frame.groupby('axis', sort=False)[['flow_ed', 'refined_flow_ed', 'unchanged_source_ed', 'centroid_ed']].mean()
    finite(summary.to_numpy(), 'summary')
    summary.to_csv(out / 'summary.csv')
    save_json(out / 'report.json', {'status': 'completed', 'biological_donors': 3, 'independent_new_data': False, 'fits': len(fits), 'paired_random_streams_verified': True, 'exact_target_time': True, 'holdout_excluded_from_scaling_and_fitting': True, 'summary': summary.reset_index().to_dict('records'), 'scope': 'Retrospective same-cohort clock sensitivity; no independent clock selection or clinical validation', 'evaluation': 'Equal donor and seed averaging, fixed per-donor source/target rows across axes; energy V-statistic', 'limitations': ['Three donors', 'Fixed discovery representation', 'Optimization targets change with clock', 'Numerical refinement must be examined for each clock', 'No target-guided best clock is selected'], 'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': args.device}})
    save_json(out / 'output_manifest.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent', type=Path, default=LATENT)
    parser.add_argument('--obs', type=Path, default=OBS)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--holdout', choices=['Wound1', 'Wound7'], default='Wound7')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rk4-steps', type=int, default=50)
    parser.add_argument('--evaluation-cap', type=int, default=1200)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
if __name__ == '__main__':
    main()
