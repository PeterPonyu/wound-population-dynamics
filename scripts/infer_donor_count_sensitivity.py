#!/usr/bin/env python3
"""Uncertainty summary for the two-point donor-count learning curve.

The upstream seed-stability experiment already contains all fits.  This
script does not train another model.  It pairs k=1 and k=2 runs by held-out
donor and training seed, then reports donor-block and seed resampling intervals.
With only k=1 and k=2, no plateau or required donor count is estimable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs' / 'analysis' / 'donor_curve_seed_stability' / 'report.json'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'analysis' / 'donor_curve_inference'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def bootstrap_mean(values: np.ndarray, seed: int=0, n_boot: int=4000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-report', default=str(DEFAULT_SOURCE))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    t0 = time.time()
    source = Path(args.source_report)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(source.read_text(encoding='utf-8'))
    if source_report.get('status') != 'completed':
        raise RuntimeError('seed-stability source report is not completed')
    runs = pd.DataFrame(source_report.get('runs', []))
    required = {'held_out_donor', 'n_training_donors', 'train_seed', 'ed_shared_cfm'}
    if not required.issubset(runs.columns):
        raise RuntimeError(f'source report lacks columns: {sorted(required - set(runs.columns))}')
    if set(runs['n_training_donors'].unique()) != {1, 2}:
        raise RuntimeError('source report does not contain exactly k=1 and k=2')
    k1 = runs[runs['n_training_donors'] == 1].copy()
    k2 = runs[runs['n_training_donors'] == 2].copy()
    rows = []
    for donor in sorted(runs['held_out_donor'].unique()):
        for seed in sorted(runs['train_seed'].unique()):
            a = k1[(k1['held_out_donor'] == donor) & (k1['train_seed'] == seed)]
            b = k2[(k2['held_out_donor'] == donor) & (k2['train_seed'] == seed)]
            if len(a) != 2 or len(b) != 1:
                raise RuntimeError(f'expected two k=1 and one k=2 run for {donor}, seed {seed}')
            k1_mean = float(a['ed_shared_cfm'].mean())
            k2_ed = float(b['ed_shared_cfm'].iloc[0])
            rows.append({'held_out_donor': donor, 'train_seed': int(seed), 'k1_mean_ed_across_single_training_donors': k1_mean, 'k2_ed': k2_ed, 'reduction_k1_to_k2': k1_mean - k2_ed})
    paired = pd.DataFrame(rows)
    donor_summary = []
    for donor, block in paired.groupby('held_out_donor', sort=True):
        reductions = block['reduction_k1_to_k2'].to_numpy(float)
        donor_summary.append({'held_out_donor': donor, 'n_seeds': int(len(reductions)), 'k1_mean_ed': float(block['k1_mean_ed_across_single_training_donors'].mean()), 'k2_mean_ed': float(block['k2_ed'].mean()), 'mean_reduction': float(reductions.mean()), 'seed_resampled_95_ci': bootstrap_mean(reductions, seed=17), 'all_seed_reductions_positive': bool(np.all(reductions > 0))})
    donor_summary_df = pd.DataFrame(donor_summary)
    donor_reductions = donor_summary_df['mean_reduction'].to_numpy(float)
    paired.to_csv(output / 'paired_seed_scores.csv', index=False)
    out = {'experiment': 'donor_curve_inference', 'status': 'completed', 'scientific_scope': 'uncertainty audit of the existing GSE241132 two-point donor-count curve; no new model fits, no plateau, and no transfer to DFU', 'source_experiment': 'donor_curve_seed_stability', 'n_training_points': 2, 'plateau_claim_supported': False, 'paired_seed_results': paired.to_dict(orient='records'), 'held_out_donor_summary': donor_summary, 'aggregate': {'n_held_out_donors': int(len(donor_reductions)), 'mean_reduction_across_donor_blocks': float(donor_reductions.mean()), 'donor_block_bootstrap_95_ci': bootstrap_mean(donor_reductions, seed=23), 'all_donor_block_reductions_positive': bool(np.all(donor_reductions > 0)), 'interpretation': 'the second donor improves the observed two-point curve on these three held-out donors, but the interval is a three-block uncertainty summary and cannot identify a plateau or a required cohort size'}, 'training_stochasticity_context': {'mean_within_cell_sd': source_report['training_stochasticity']['mean_within_cell_sd'], 'max_within_cell_range': source_report['training_stochasticity']['max_within_cell_range'], 'seed_reduction_values': source_report['reduction_across_seeds']['values'], 'source_verdict': source_report['verdict']}, 'protocol': {'script_sha256': sha256(Path(__file__)), 'source_report_sha256': sha256(source), 'source_report': str(source), 'paired_by': 'held_out_donor and train_seed; k=1 averaged over its two single-donor fits', 'bootstrap_resamples': 4000, 'new_model_fits': False, 'plateau_claim_supported': False, 'elapsed_seconds': time.time() - t0}}
    (output / 'report.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'aggregate': out['aggregate'], 'donor_summary': donor_summary, 'output': str(output / 'report.json')}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
