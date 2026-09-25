#!/usr/bin/env python3
"""Saved-input computational sensitivity."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.expand_biological_evidence import energy, interpolation_controls, sha
from wound_models.human_wound_data import COND_ORDER, COND_TIME, fold_standardize
OUT = ROOT / 'outputs/computational_extension'
PROTOCOL = ROOT / 'config/computational_extension_protocol.json'

def save(folder, report, inputs):
    report['provenance'] = {'script_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL), 'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs}, 'output_sha256': {p.name: sha(p) for p in sorted(folder.glob('*.csv'))}}
    (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'provenance'}, indent=2), flush=True)

def location_scale(source, anchor, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('Interpolation fraction must be between zero and one')
    a = np.asarray(source, float)
    b = np.asarray(anchor, float)
    sa = a.std(0)
    sb = b.std(0)
    scale = np.divide((1 - alpha) * sa + alpha * sb, sa, out=np.zeros_like(sa), where=sa > 1e-08)
    return (1 - alpha) * a.mean(0) + alpha * b.mean(0) + (a - a.mean(0)) * scale

def geometry_metrics(prediction, target):
    a = np.asarray(prediction, float)
    b = np.asarray(target, float)
    total = float(b.var(0).sum())
    if total <= 0:
        raise ValueError('Target variance must be positive')
    return dict(energy_distance=energy(a, b), centered_energy=energy(a - a.mean(0), b - b.mean(0)), centroid_error=float(np.linalg.norm(a.mean(0) - b.mean(0))), variance_ratio=float(a.var(0).sum() / total))

def population():
    folder = OUT / 'population'
    folder.mkdir(parents=True, exist_ok=True)
    bio = ROOT / 'outputs/biological_expansion/population'
    mu_path = ROOT / 'outputs/analysis/human_temporal_flow/mu.npy'
    obs_path = ROOT / 'outputs/analysis/human_temporal_flow/obs_fibroblast.csv'
    mu = np.load(mu_path)
    obs = pd.read_csv(obs_path)
    cond = obs.cond.to_numpy()
    donor = obs.donor.to_numpy()
    donors = sorted(set(donor))
    rows = []
    subsets = []
    inputs = [mu_path, obs_path, ROOT / 'scripts/expand_biological_evidence.py']
    for holdout in ['Wound7', 'Wound1']:
        z, _ = fold_standardize(mu, cond != holdout)
        index = COND_ORDER.index(holdout)
        src = COND_ORDER[index - 1]
        anchor = COND_ORDER[index + 1]
        alpha = (COND_TIME[holdout] - COND_TIME[src]) / (COND_TIME[anchor] - COND_TIME[src])
        source = cond == src
        for seed in [0, 1, 2]:
            path = bio / f'{holdout}_seed{seed}_prediction.npy'
            inputs.append(path)
            pred = np.load(path)
            rng = np.random.default_rng(seed)
            for d in donors:
                a = z[source & (donor == d)]
                b = z[(cond == anchor) & (donor == d)]
                t = z[(cond == holdout) & (donor == d)]
                candidates = interpolation_controls(a, b, alpha, rng)
                candidates['Shared flow'] = pred[donor[source] == d]
                candidates['Location-scale'] = location_scale(a, b, alpha)
                assert len(a) >= 200 and len(t) >= 200
                for method, c in candidates.items():
                    rows.append(dict(holdout=holdout, donor=d, seed=seed, method=method, **geometry_metrics(c, t)))
                sampling = np.random.default_rng(29)
                for draw in range(50):
                    ai = sampling.choice(len(a), 200, replace=False)
                    ti = sampling.choice(len(t), 200, replace=False)
                    for method, c in candidates.items():
                        subsets.append(dict(holdout=holdout, donor=d, seed=seed, draw=draw, method=method, energy_distance=energy(c[ai], t[ti])))
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'geometry_by_donor_seed.csv', index=False)
    metrics = ['energy_distance', 'centered_energy', 'centroid_error', 'variance_ratio']
    summary = frame.groupby(['holdout', 'method'], sort=False)[metrics].mean().reset_index()
    summary.to_csv(folder / 'geometry_summary.csv', index=False)
    draws = pd.DataFrame(subsets)
    draws.to_csv(folder / 'matched_cell_draws.csv', index=False)
    macro = draws.groupby(['holdout', 'draw', 'method'], sort=False).energy_distance.mean().unstack('method')
    comparisons = []
    for holdout in ['Wound7', 'Wound1']:
        current = macro.loc[holdout]
        for method in ['Centroid translation', 'Location-scale', 'Independent bridge']:
            difference = current['Shared flow'] - current[method]
            comparisons.append(dict(holdout=holdout, comparator=method, draws=len(difference), flow_minus_comparator_mean=float(difference.mean()), difference_min=float(difference.min()), difference_max=float(difference.max()), comparator_lower_count=int((difference > 0).sum())))
    pd.DataFrame(comparisons).to_csv(folder / 'matched_cell_summary.csv', index=False)
    old = pd.read_csv(bio / 'baseline_by_donor_seed.csv')
    check = old.merge(frame, on=['holdout', 'donor', 'seed', 'method'], suffixes=('_old', '_new'), validate='one_to_one')
    difference = float(np.max(np.abs(check.energy_distance_old - check.energy_distance_new)))
    if len(check) != len(old) or difference > 1e-10:
        raise RuntimeError(f'Historical baseline mismatch: {difference}')
    save(folder, dict(status='complete', biological_donors=3, seeds=[0, 1, 2], historical_max_error_difference=difference, geometry=summary.to_dict('records'), matched_sampling=comparisons, limits='Retrospective endpoint-informed reconstruction; centered ED is not an additive decomposition; draw ranges are not donor confidence intervals'), inputs + [bio / 'baseline_by_donor_seed.csv'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute saved-input computational sensitivity')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reruns/computational_extension')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if (OUT / 'population' / 'report.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    population()
