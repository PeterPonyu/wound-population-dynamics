#!/usr/bin/env python3
"""Persist donor-count fits and evaluate them in a fixed reference geometry.

The reference scaler sees both available training donors, solely for scoring.
Every fitted field and its own scaler see only the specified training subset.
Training-target marginal controls ignore the held source's expression.  This is
a new controlled computational experiment in the existing three-donor cohort.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.human_wound_data import COND_ORDER, COND_TIME, parse_sample_map
from scripts.audit_population_math import weighted_energy
SPEC = importlib.util.spec_from_file_location('original_donor_curve', ROOT / 'scripts/evaluate_training_donor_count.py')
CURVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURVE)
DEFAULT_OUT = ROOT / 'outputs/scientific_revision_20260922/population'
LATENT = ROOT / 'outputs/analysis/human_temporal_flow/mu.npy'
OBS = ROOT / 'outputs/analysis/human_temporal_flow/obs_fibroblast.csv'
OLD_BENCH = ROOT / 'outputs/analysis/donor_conditioned_benchmark/report.json'
OLD_CURVE = ROOT / 'outputs/analysis/donor_curve_seed_stability/report.json'
META = ROOT / 'data/raw/GSE241132/GSE241132_cell_metadata.txt.gz'
ZIP_DIR = ROOT / 'data/raw/GSE241132/per_gsm'
METHODS = ['Shared CFM', 'Mean displacement', 'Training-target marginal (exact)', 'Training-target marginal (matched)', 'Unchanged source']

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def fit_scaler(mu, train_mask):
    train_mask = np.asarray(train_mask, bool)
    if train_mask.shape != (len(mu),) or train_mask.sum() < 2:
        raise ValueError('Scaler requires at least two permitted training cells')
    mean = mu[train_mask].mean(axis=0, keepdims=True)
    scale = mu[train_mask].std(axis=0, keepdims=True)
    scale[scale == 0] = 1
    return (mean, scale)

def transform(mu, mean, scale):
    return ((mu - mean) / scale).astype(np.float32)

def reference_prediction(prediction, train_mean, train_scale, reference_mean, reference_scale):
    raw = np.asarray(prediction, float) * train_scale + train_mean
    return (raw, (raw - reference_mean) / reference_scale)

def original_target_indices(n_target, cap=1200, seed=0):
    return np.random.default_rng(seed).choice(n_target, cap, replace=False) if n_target > cap else np.arange(n_target)

def marginal_draw(endpoint_rows, n_particles, draw):
    names = sorted(endpoint_rows)
    if n_particles < len(names):
        raise ValueError('Need at least one prediction particle per training donor')
    counts = np.full(len(names), n_particles // len(names), dtype=int)
    for j in range(n_particles % len(names)):
        counts[(draw + j) % len(names)] += 1
    rng = np.random.default_rng(draw)
    indices = [rng.choice(endpoint_rows[d], int(n), replace=False) for d, n in zip(names, counts)]
    weights = np.concatenate([np.full(n, 1 / (len(names) * n)) for n in counts])
    return (np.concatenate(indices), weights, dict(zip(names, counts.tolist())))

def exact_marginal(endpoint_rows):
    names = sorted(endpoint_rows)
    rows = np.concatenate([endpoint_rows[d] for d in names])
    weights = np.concatenate([np.full(len(endpoint_rows[d]), 1 / (len(names) * len(endpoint_rows[d]))) for d in names])
    return (rows, weights)

def cell_identity(saved):
    """Recover original barcodes without rereading the count matrices.

    The original loader concatenates sorted archives and unchanged barcode
    order. Its inner annotation join preserves that order, before fibroblast
    selection. Matching every saved metadata row verifies this reconstruction.
    """
    meta = pd.read_csv(META, sep='\t')
    meta = meta[[meta.columns[0], 'newMainCellTypes']].copy()
    meta.columns = ['author_key', 'celltype']
    mapping = parse_sample_map(str(ZIP_DIR))
    records = []
    for path in sorted(ZIP_DIR.glob('*.zip')):
        gsm = path.name.split('_')[0]
        spec = mapping[gsm]
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if 'barcodes' in Path(n).name and '__MACOSX' not in n and (not Path(n).name.startswith('._'))]
            if len(names) != 1:
                raise RuntimeError(f'Ambiguous barcode member: {path}')
            barcodes = gzip.decompress(archive.read(names[0])).decode().splitlines()
        for sample_row, barcode in enumerate(barcodes):
            records.append(dict(gsm=gsm, donor=spec['donor'], cond=spec['cond'], cell_id=f'{gsm}:{barcode}', author_key=f"{spec['label']}_{barcode}", archive_cell_row=sample_row))
    raw = pd.DataFrame(records)
    raw['raw_matrix_row'] = np.arange(len(raw))
    joined = raw.merge(meta, on='author_key', how='inner', sort=False, validate='one_to_one')
    fib = joined[joined.celltype.eq('Fibroblast')].reset_index(drop=True)
    if len(fib) != len(saved):
        raise RuntimeError('Recovered identity count differs from saved coordinates')
    for name in ['gsm', 'donor', 'cond', 'celltype']:
        if not np.array_equal(fib[name], saved[name]):
            raise RuntimeError(f'Saved coordinate order differs in {name}')
    fib.insert(0, 'latent_row', np.arange(len(fib)))
    if not fib.cell_id.is_unique or not fib.author_key.is_unique:
        raise RuntimeError('Cell identities are not unique')
    return fib

def quantile_summary(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()))

def descriptive_range(values, seed=23):
    values = np.asarray(values, float)
    samples = np.random.default_rng(seed).choice(values, (4000, len(values)), replace=True).mean(1)
    return np.quantile(samples, [0.025, 0.975]).tolist()

def historical_baseline(mu, obs, out):
    folder = out / 'historical_benchmark'
    folder.mkdir(exist_ok=True)
    original = json.loads(OLD_BENCH.read_text())
    donor, cond = (obs.donor.to_numpy(), obs.cond.to_numpy())
    summaries, draws, checks, exacts, indices = ([], [], [], [], [])
    for held in sorted(set(donor)):
        mean, scale = fit_scaler(mu, donor != held)
        z = transform(mu, mean, scale)
        sr = np.flatnonzero((donor == held) & (cond == 'Wound1'))
        tr = np.flatnonzero((donor == held) & (cond == 'Wound7'))
        tr = tr[original_target_indices(len(tr))]
        assert len(sr) < 1200
        endpoint_rows = {d: np.flatnonzero((donor == d) & (cond == 'Wound7')) for d in sorted(set(donor)) if d != held}
        target = z[tr]
        shared = original['per_donor']['M1_shared_cfm'][held]['ed']
        md = CURVE.mean_displacement_prediction(z, donor, cond, list(endpoint_rows), 'Wound1', 'Wound7', z[sr])
        for name, prediction in [('M0_standstill', z[sr]), ('M2_mean_displacement', md)]:
            value = weighted_energy(prediction, target)
            old = original['per_donor'][name][held]['ed']
            checks.append(dict(held_donor=held, method=name, reported_ed=old, recalculated_ed=value, difference=value - old))
        all_rows, weights = exact_marginal(endpoint_rows)
        exacts.append(dict(held_donor=held, energy_distance=weighted_energy(z[all_rows], target, weights), prediction_particles=len(all_rows), target_cells=len(tr)))
        held_draws = []
        for draw in range(50):
            rows, weights, counts = marginal_draw(endpoint_rows, len(sr), draw)
            uniform = weighted_energy(z[rows], target)
            equal = weighted_energy(z[rows], target, weights)
            held_draws.append(dict(held_donor=held, draw=draw, n_prediction=len(rows), n_target=len(tr), uniform_ed=uniform, equal_donor_ed=equal, shared_cfm_reported_ed=shared, shared_minus_equal_donor=shared - equal))
            indices.append(dict(held_donor=held, draw=draw, prediction_latent_rows=rows.tolist(), weights=weights.tolist(), counts=counts))
        draws.extend(held_draws)
        values = np.asarray([r['equal_donor_ed'] for r in held_draws])
        summaries.append(dict(held_donor=held, shared_cfm_ed=shared, matched_equal_donor_mean_ed=float(values.mean()), matched_equal_donor_min_ed=float(values.min()), matched_equal_donor_max_ed=float(values.max()), marginal_lower_draws=int(np.sum(values < shared)), draws=50))
        np.savez(folder / f'{held}_evaluation.npz', source_latent_rows=sr, target_latent_rows=tr, mean=mean, scale=scale)
    pd.DataFrame(draws).to_csv(folder / 'draw_scores.csv', index=False)
    pd.DataFrame(summaries).to_csv(folder / 'donor_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(folder / 'original_score_reproduction.csv', index=False)
    pd.DataFrame(exacts).to_csv(folder / 'exact_marginal_scores.csv', index=False)
    save_json(folder / 'draw_indices.json', indices)
    macro = pd.DataFrame(draws).groupby('draw')[['uniform_ed', 'equal_donor_ed']].mean().reset_index()
    macro.to_csv(folder / 'macro_draw_scores.csv', index=False)
    max_error = max((abs(r['difference']) for r in checks))
    if max_error > 1e-05:
        raise RuntimeError(f'Historical scorer failed reproduction: {max_error}')
    result = dict(original_shared_cfm_mean_ed=original['summary']['M1_shared_cfm']['mean_ed'], matched_equal_donor=quantile_summary(macro.equal_donor_ed), matched_uniform=quantile_summary(macro.uniform_ed), exact_marginal_mean_ed=float(np.mean([r['energy_distance'] for r in exacts])), n_macro_draws_lower_than_shared=int((macro.equal_donor_ed < original['summary']['M1_shared_cfm']['mean_ed']).sum()), maximum_original_score_reproduction_error=max_error, donor_summary=summaries, interpretation='50 draws characterize evaluation sampling in the same three donors, not biological replication; original shared-field values are historical, not newly trained.')
    save_json(folder / 'report.json', result)
    return result

def run_curve(mu, obs, out, args):
    donor, cond = (obs.donor.to_numpy(), obs.cond.to_numpy())
    donors = sorted(set(donor))
    segments = list(zip(COND_ORDER[:-1], COND_ORDER[1:]))
    old_runs = json.loads(OLD_CURVE.read_text())['runs']
    old_lookup = {(r['held_out_donor'], tuple(r['training_donors']), r['train_seed']): r for r in old_runs}
    scores, fits, control_draws, historical = ([], [], [], [])
    total = len(donors) * 3 * len(args.seeds)
    for held in donors:
        others = [d for d in donors if d != held]
        ref_mask = donor != held
        ref_mean, ref_scale = fit_scaler(mu, ref_mask)
        reference = transform(mu, ref_mean, ref_scale)
        sr = np.flatnonzero((donor == held) & (cond == 'Wound1'))
        all_tr = np.flatnonzero((donor == held) & (cond == 'Wound7'))
        tr = all_tr[original_target_indices(len(all_tr))]
        assert len(sr) < 1200
        reference_target = reference[tr]
        base = weighted_energy(reference[sr], reference_target)
        half_order = np.random.default_rng(0).permutation(len(all_tr))
        half = len(all_tr) // 2
        split_reference = weighted_energy(reference[all_tr[half_order[:half]]], reference[all_tr[half_order[half:]]])
        held_folder = out / 'fits' / held
        held_folder.mkdir(parents=True, exist_ok=True)
        np.savez(held_folder / 'reference.npz', mean=ref_mean, scale=ref_scale, fitting_latent_rows=np.flatnonzero(ref_mask), source_latent_rows=sr, target_all_latent_rows=all_tr, target_evaluation_latent_rows=tr)
        for k in (1, 2):
            for subset in itertools.combinations(others, k):
                config = f"k{k}_{'_'.join(subset)}"
                folder = held_folder / config
                folder.mkdir(exist_ok=True)
                train_mask = np.isin(donor, subset)
                train_mean, train_scale = fit_scaler(mu, train_mask)
                z = transform(mu, train_mean, train_scale)
                np.savez(folder / 'training_scaler.npz', mean=train_mean, scale=train_scale, fitting_latent_rows=np.flatnonzero(train_mask))
                md_z = CURVE.mean_displacement_prediction(z, donor, cond, list(subset), 'Wound1', 'Wound7', z[sr])
                md_mu, md_ref = reference_prediction(md_z, train_mean, train_scale, ref_mean, ref_scale)
                np.savez(folder / 'mean_displacement_prediction.npz', training_z=md_z, mu=md_mu, reference_z=md_ref, source_latent_rows=sr)
                endpoints = {d: np.flatnonzero((donor == d) & (cond == 'Wound7')) for d in subset}
                exact_rows, exact_weights = exact_marginal(endpoints)
                exact_ed = weighted_energy(reference[exact_rows], reference_target, exact_weights)
                draw_rows = []
                saved_indices, saved_weights = ([], [])
                for draw in range(50):
                    rows, weights, _ = marginal_draw(endpoints, len(sr), draw)
                    draw_rows.append(dict(held_donor=held, k=k, training_donors='|'.join(subset), draw=draw, equal_donor_ed=weighted_energy(reference[rows], reference_target, weights), uniform_ed=weighted_energy(reference[rows], reference_target)))
                    saved_indices.append(rows)
                    saved_weights.append(weights)
                control_draws.extend(draw_rows)
                np.savez(folder / 'target_marginal_predictions.npz', exact_latent_rows=exact_rows, exact_weights=exact_weights, matched_latent_rows=np.asarray(saved_indices), matched_weights=np.asarray(saved_weights), target_evaluation_latent_rows=tr)
                controls = {'Mean displacement': weighted_energy(md_ref, reference_target), 'Training-target marginal (exact)': exact_ed, 'Training-target marginal (matched)': float(np.mean([r['equal_donor_ed'] for r in draw_rows])), 'Unchanged source': base}
                for seed in args.seeds:
                    fit_dir = folder / f'seed{seed}'
                    fit_dir.mkdir(exist_ok=True)
                    started = time.perf_counter()
                    completed = fit_dir / 'fit.json'
                    if completed.exists():
                        if not args.resume:
                            raise FileExistsError(f'Already fitted: {completed}; use --resume for the identical protocol')
                        fit = json.loads(completed.read_text())
                        if fit['training_steps'] != args.steps or fit['batch_size'] != args.batch_size:
                            raise RuntimeError('Resume hyperparameters differ')
                        pred_z = np.load(fit_dir / 'prediction_training_z.npy')
                    else:
                        field, pairs = CURVE.train_shared_field(z, donor, cond, list(subset), segments, n_topics=mu.shape[1], dev=args.device, steps=args.steps, batch_size=args.batch_size, seed=seed)
                        pred_z = CURVE.integrate(field, torch.from_numpy(z[sr]).to(args.device), COND_TIME['Wound1'], COND_TIME['Wound7'], n_steps=50)
                        if not np.isfinite(pred_z).all():
                            raise RuntimeError(f'Non-finite prediction: {held}/{config}/{seed}')
                        state = {name: value.detach().cpu() for name, value in field.state_dict().items()}
                        torch.save(state, fit_dir / 'field.pt')
                        np.save(fit_dir / 'prediction_training_z.npy', pred_z)
                        fit = dict(held_donor=held, k=k, training_donors=list(subset), seed=seed, training_steps=args.steps, batch_size=args.batch_size, rk4_steps=50, donor_segments=[list(p) for p in pairs], n_training_cells=int(train_mask.sum()), source_count=len(sr), target_all_count=len(all_tr), target_evaluation_count=len(tr), training_seconds=time.perf_counter() - started, reference_scaler_used_by_training=False, source_latent_rows=sr.tolist(), target_evaluation_latent_rows=tr.tolist())
                        del field
                    raw_prediction, ref_prediction = reference_prediction(pred_z, train_mean, train_scale, ref_mean, ref_scale)
                    np.save(fit_dir / 'prediction_mu.npy', raw_prediction)
                    np.save(fit_dir / 'prediction_reference_z.npy', ref_prediction)
                    own_ed = weighted_energy(pred_z, z[tr])
                    ref_ed = weighted_energy(ref_prediction, reference_target)
                    fit.update(own_scale_energy_distance=own_ed, reference_energy_distance=ref_ed, reference_standstill_distance=base, reference_relative_improvement=1 - ref_ed / base, reference_split_half_distance=split_reference, path=str(fit_dir.relative_to(out)), artifact_sha256={p.name: sha(p) for p in sorted(fit_dir.glob('*')) if p.suffix in {'.pt', '.npy'}})
                    save_json(completed, fit)
                    fits.append(fit)
                    for method, ed in {'Shared CFM': ref_ed, **controls}.items():
                        scores.append(dict(held_donor=held, k=k, training_donors='|'.join(subset), seed=seed, method=method, energy_distance=ed, standstill_distance=base, gain=base - ed, relative_improvement=1 - ed / base, split_half_reference=split_reference, n_source=len(sr), n_target=len(tr)))
                    previous = old_lookup.get((held, tuple(subset), seed))
                    if previous is not None:
                        historical.append(dict(held_donor=held, k=k, training_donors='|'.join(subset), seed=seed, historical_own_ed=previous['ed_shared_cfm'], new_own_ed=own_ed, new_reference_ed=ref_ed, new_own_minus_historical=own_ed - previous['ed_shared_cfm']))
                    pd.DataFrame(scores).to_csv(out / 'curve_scores.csv', index=False)
                    print(f"[{len(fits)}/{total}] {held} {config} seed={seed}: own ED={own_ed:.6f}, reference ED={ref_ed:.6f}, gain={1 - ref_ed / base:.2%}, {fit['training_seconds']:.1f}s", flush=True)
    pd.DataFrame(control_draws).to_csv(out / 'curve_marginal_draw_scores.csv', index=False)
    pd.DataFrame(historical).to_csv(out / 'historical_curve_comparison.csv', index=False)
    save_json(out / 'fit_manifest.json', fits)
    return (pd.DataFrame(scores), fits)

def summarize_curve(frame, out):
    paired = []
    for (method, held, seed), block in frame.groupby(['method', 'held_donor', 'seed'], sort=False):
        k1, k2 = (block[block.k.eq(1)], block[block.k.eq(2)])
        assert len(k1) == 2 and len(k2) == 1
        a, b = (float(k1.energy_distance.mean()), float(k2.energy_distance.iloc[0]))
        paired.append(dict(method=method, held_donor=held, seed=int(seed), k1_ed=a, k2_ed=b, reduction=a - b, k1_relative_improvement=float(k1.relative_improvement.mean()), k2_relative_improvement=float(k2.relative_improvement.iloc[0])))
    paired = pd.DataFrame(paired)
    paired.to_csv(out / 'paired_donor_seed_scores.csv', index=False)
    numeric = ['k1_ed', 'k2_ed', 'reduction', 'k1_relative_improvement', 'k2_relative_improvement']
    seed = paired.groupby(['method', 'seed'], sort=False)[numeric].mean().reset_index()
    blocks = paired.groupby(['method', 'held_donor'], sort=False)[numeric].mean().reset_index()
    seed.to_csv(out / 'curve_by_seed.csv', index=False)
    blocks.to_csv(out / 'curve_donor_blocks.csv', index=False)
    rows = []
    for method in METHODS:
        values = blocks[blocks.method.eq(method)]
        sr = seed[seed.method.eq(method)]
        lower, upper = descriptive_range(values.reduction)
        rows.append(dict(method=method, n_donor_blocks=len(values), k1_mean_ed=float(values.k1_ed.mean()), k2_mean_ed=float(values.k2_ed.mean()), mean_reduction=float(values.reduction.mean()), descriptive_block_range_lower=lower, descriptive_block_range_upper=upper, donor_blocks_improved=int((values.reduction > 0).sum()), seed_reduction_min=float(sr.reduction.min()), seed_reduction_max=float(sr.reduction.max()), k1_relative_improvement=float(values.k1_relative_improvement.mean()), k2_relative_improvement=float(values.k2_relative_improvement.mean())))
    pd.DataFrame(rows).to_csv(out / 'curve_summary.csv', index=False)
    stability = frame.groupby(['method', 'held_donor', 'k', 'training_donors'], sort=False).agg(mean_ed=('energy_distance', 'mean'), training_sd=('energy_distance', 'std'), training_range=('energy_distance', lambda x: float(x.max() - x.min())), n_seeds=('seed', 'nunique'), split_half_reference=('split_half_reference', 'first')).reset_index()
    stability.to_csv(out / 'configuration_stability.csv', index=False)
    field_stability = stability[stability.method.eq('Shared CFM')]
    noise = float(field_stability.split_half_reference.mean())
    stochasticity = dict(mean_configuration_sd=float(field_stability.training_sd.mean()), maximum_configuration_range=float(field_stability.training_range.max()), mean_split_half_reference=noise, sd_to_split_half_ratio=float(field_stability.training_sd.mean()) / noise)
    return dict(summary=rows, by_seed=seed.to_dict('records'), donor_blocks=blocks.to_dict('records'), configuration_stability=stability.to_dict('records'), training_stochasticity=stochasticity, uncertainty='4000 percentile resamples of three held-donor blocks after averaging seeds. Descriptive only: training sets overlap. Seeds/subsets/draws are computational repetitions.')

def verify(out):
    manifest = json.loads((out / 'fit_manifest.json').read_text())
    identities = pd.read_csv(out / 'cell_identity.csv')
    mu = np.load(LATENT)
    max_inverse, max_score, max_checkpoint = (0.0, 0.0, 0.0)
    for fit in manifest:
        folder = out / fit['path']
        config = folder.parent
        held_folder = config.parent
        training = np.load(config / 'training_scaler.npz')
        reference = np.load(held_folder / 'reference.npz')
        rows = training['fitting_latent_rows']
        assert set(identities.iloc[rows].donor) == set(fit['training_donors'])
        assert fit['held_donor'] not in set(identities.iloc[reference['fitting_latent_rows']].donor)
        assert not fit['reference_scaler_used_by_training']
        sr, tr = (reference['source_latent_rows'], reference['target_evaluation_latent_rows'])
        assert np.array_equal(sr, fit['source_latent_rows'])
        assert np.array_equal(tr, fit['target_evaluation_latent_rows'])
        assert set(identities.iloc[sr].cond) == {'Wound1'} and set(identities.iloc[tr].cond) == {'Wound7'}
        pred = np.load(folder / 'prediction_training_z.npy')
        raw, ref = reference_prediction(pred, training['mean'], training['scale'], reference['mean'], reference['scale'])
        max_inverse = max(max_inverse, float(np.max(np.abs(raw - np.load(folder / 'prediction_mu.npy')))), float(np.max(np.abs(ref - np.load(folder / 'prediction_reference_z.npy')))))
        target = transform(mu, reference['mean'], reference['scale'])[tr]
        max_score = max(max_score, abs(weighted_energy(ref, target) - fit['reference_energy_distance']))
        for name, digest in fit['artifact_sha256'].items():
            assert sha(folder / name) == digest
        field = CURVE.LatentFlowField(mu.shape[1])
        field.load_state_dict(torch.load(folder / 'field.pt', map_location='cpu', weights_only=True))
        source = transform(mu, training['mean'], training['scale'])[sr]
        reloaded = CURVE.integrate(field, torch.from_numpy(source), COND_TIME['Wound1'], COND_TIME['Wound7'], 50)
        max_checkpoint = max(max_checkpoint, float(np.max(np.abs(reloaded - pred))))
    if max_inverse > 1e-12 or max_score > 1e-10 or max_checkpoint > 5e-05:
        raise RuntimeError(f'Saved fit verification failed: {max_inverse}, {max_score}, {max_checkpoint}')
    result = dict(status='passed', saved_fit_count=len(manifest), unique_cell_ids=bool(identities.cell_id.is_unique), training_donor_exclusion=True, reference_excludes_held_donor=True, source_target_ids_checked=True, maximum_inverse_transform_difference=max_inverse, maximum_score_reproduction_difference=max_score, maximum_cpu_checkpoint_prediction_difference=max_checkpoint, checkpoint_reload_tolerance=5e-05, explanation='Every saved checkpoint re-integrated on CPU; original fits use the reported device. Small float32 CPU/GPU differences are expected.')
    save_json(out / 'verification.json', result)
    return result

def write_integration(report, out):
    rows = report['curve']['summary']
    flow = next((r for r in rows if r['method'] == 'Shared CFM'))
    prior = report['historical_baseline']
    text = ['# 群体协议修复整合说明', '', '这是既有三供者队列上的新受控计算实验。没有新增生物重复，未修改旧报告或论文。', '', f"历史六方法 benchmark 的目标边际基线：严格同目标抽样/同预测粒子数/供者等质量，50 draws 宏平均 ED {prior['matched_equal_donor']['mean']:.6f}，范围 {prior['matched_equal_donor']['minimum']:.6f}–{prior['matched_equal_donor']['maximum']:.6f}；历史共享流 {prior['original_shared_cfm_mean_ed']:.6f}。这是对原方法集合的正式补充；shared 原数值沿用历史工件，不能与新模型点值混称同一次拟合。", '', f"统一评分空间的新 Shared CFM 曲线：k=1 ED {flow['k1_mean_ed']:.6f}，k=2 ED {flow['k2_mean_ed']:.6f}，平均下降 {flow['mean_reduction']:.6f}；三个 seed 的下降范围 {flow['seed_reduction_min']:.6f}–{flow['seed_reduction_max']:.6f}；{flow['donor_blocks_improved']}/3 个供者块下降。三块描述重采样范围 [{flow['descriptive_block_range_lower']:.6f}, {flow['descriptive_block_range_upper']:.6f}]。", '', '训练 scaler 只看各自 k 个供者。评分 reference scaler 看该 fold 的两位可用训练供者全部时点，绝不进入 k=1 模型拟合。模型先逆变换回 μ，再进入共同参考空间；这种评分定义是回顾性审计，不把第二位供者的信息伪称为 k=1 训练信息。', '', '每个方法统一尺度结果：', '']
    for row in rows:
        text.append(f"- {row['method']}: k1={row['k1_mean_ed']:.6f}, k2={row['k2_mean_ed']:.6f}, reduction={row['mean_reduction']:.6f}; relative improvement={row['k1_relative_improvement']:.2%}/{row['k2_relative_improvement']:.2%}.")
    text += ['', '应更新的主张与图表：', '', '- 摘要、Results 共享流最低、Discussion source 增量解释、图5及其逐供者表：加入合法目标边际基线；保留供者异质性，不声称新的总体排名。', '- 图6、Results 第二供者、种子表、donor-block 表：使用 curve_summary.csv、curve_by_seed.csv 和 curve_donor_blocks.csv 的共同尺度结果；旧0.165不再作统一误差降低。', '- Methods：明确训练/评分 scaler 的信息隔离、每配置 NumPy/Torch 重置、原 8000步/256批次/50 RK4不变，以及完整预测工件。', '- exact marginal 是训练供者目标经验分布的精确等质量混合，粒子数多于 source；matched marginal 是同source粒子数、50 draws误差的平均。两者必须分开标记。', '- 所有重采样范围均描述已有三供者；seed/subset/draw不是新增生物重复。此修复不解决时钟实验、conditioned/OT多seed或临床泛化。', '', '主要文件：', '', '- report.json：全部协议、结果、来源哈希及验证。', '- historical_benchmark/report.json、donor_summary.csv、draw_scores.csv、macro_draw_scores.csv、draw_indices.json：正式历史基线补充。', '- curve_scores.csv：27配置×5方法逐行统一尺度误差；curve_summary.csv、curve_by_seed.csv、curve_donor_blocks.csv：整合论文的主要表。', '- curve_marginal_draw_scores.csv：所有9训练配置的50抽样；paired_donor_seed_scores.csv：每donor/seed配对。', '- configuration_stability.csv：逐方法/held donor/训练子集的统一尺度seed均值、SD、范围与split-half参考；独立seed数仍为3。', '- historical_curve_comparison.csv：只供历史对照，不混合进新结果。', '- cell_identity.csv：原始barcode、GSM、矩阵行与latent行；fits/**：训练/参考scaler、检查点、training-z/μ/reference-z预测、源/目标身份。', '- verification.json：全部模型检查点重新载入积分、评分与信息排除验证。', '', '复现：`OPENBLAS_NUM_THREADS=1 python scripts/repair_population_protocol.py --output-dir outputs/scientific_revision_20260922/population`。已有完整运行用 `--verify-only` 验证；同一协议中断可用 `--resume`，不得无意覆盖原实验。', '']
    (out / 'INTEGRATION.md').write_text('\n'.join(text))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(',')]
    torch.set_num_threads(4)
    out = args.output_dir.resolve()
    if args.verify_only:
        print(json.dumps(verify(out), indent=2))
        return
    if out.exists() and any(out.iterdir()) and (not args.resume):
        raise FileExistsError('Use a fresh output directory, or --resume for the identical experiment')
    out.mkdir(parents=True, exist_ok=True)
    sources = [LATENT, OBS, META, OLD_BENCH, OLD_CURVE, Path(__file__), ROOT / 'scripts/evaluate_training_donor_count.py', ROOT / 'wound_models/human_wound_data.py', ROOT / 'wound_models/population_flow.py', ROOT / 'scripts/audit_population_math.py', *sorted(ZIP_DIR.glob('*.zip'))]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    run_contract = dict(schema_version=1, inputs=inputs, steps=args.steps, batch_size=args.batch_size, seeds=args.seeds, device=args.device, rk4_steps=50, evaluation_seed=0, marginal_draws=50, target_cap=1200)
    if (out / 'run_contract.json').exists():
        if json.loads((out / 'run_contract.json').read_text()) != run_contract:
            raise RuntimeError('Resume contract or input hashes differ')
    else:
        save_json(out / 'run_contract.json', run_contract)
    started = time.perf_counter()
    mu, saved = (np.load(LATENT), pd.read_csv(OBS))
    identities = cell_identity(saved)
    identities.to_csv(out / 'cell_identity.csv', index=False)
    baseline = historical_baseline(mu, identities, out)
    frame, fits = run_curve(mu, identities, out, args)
    curve_summary = summarize_curve(frame, out)
    validation = verify(out)
    report = dict(status='completed', experiment='fixed_reference_donor_curve_and_target_marginal', schema_version=1, biological_donors=3, independent_new_data=False, fits=len(fits), seeds=args.seeds, protocol=dict(training='Same scripts/evaluate_training_donor_count.py train_shared_field and integrate used by scripts/assess_training_seed_stability.py; 8000 steps, batch256, Adam1e-3, same paired donor-segment product coupling, rank clock, 50 RK4 by default.', randomness='Every fit resets torch.manual_seed(seed) and np.random.default_rng(seed) inside reused training function. Controls use separate draw RNGs; a new controlled computation, not a replacement of historical outputs.', scaling='Own scaler uses the specified training subset only. Reference scaler uses both non-held donors at all times, only to evaluate inverse-transformed predictions. Never supplied to k1 training.', evaluation='All held-source cells; original seed0 target sample capped1200; same reference geometry/target/source identities for all k and methods within held donor; float64 energy V-statistic.', marginal='Equal donor empirical target measure; exact score and source-count-matched50 draw average reported separately; no held-source expression used.', scope='Retrospective same-cohort protocol audit; computational seeds/subsets/draws are not independent biological units; no clinical/new-patient validation.'), historical_baseline=baseline, curve=curve_summary, verification=validation, provenance=dict(inputs=inputs, runtime=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda, device=args.device, gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None), elapsed_seconds=time.perf_counter() - started))
    save_json(out / 'report.json', report)
    write_integration(report, out)
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    save_json(out / 'output_manifest.json', outputs)
    print(json.dumps(dict(summary=curve_summary['summary'], verification=validation, elapsed_seconds=report['provenance']['elapsed_seconds']), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
