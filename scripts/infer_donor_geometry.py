#!/usr/bin/env python3
"""Infer donor geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = ROOT / 'outputs' / 'analysis' / 'human_temporal_flow' / 'report.json'
DEFAULT_TRANSFER = ROOT / 'outputs' / 'analysis' / 'donor_transfer_flow' / 'report.json'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'analysis' / 'donor_geometry_inference'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def bootstrap_ratio(donor: np.ndarray, time_dist: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        d = rng.choice(donor, len(donor), replace=True)
        t = rng.choice(time_dist, len(time_dist), replace=True)
        out[i] = np.median(t) / np.median(d)
    return out

def bootstrap_median(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)

def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry-report', default=str(DEFAULT_GEOMETRY))
    ap.add_argument('--transfer-report', default=str(DEFAULT_TRANSFER))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    started = time.time()
    geometry_path = Path(args.geometry_report)
    transfer_path = Path(args.transfer_report)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
    transfer = json.loads(transfer_path.read_text(encoding='utf-8'))
    donor_block = geometry['donor_vs_time']['between_donor_pairs']
    time_block = geometry['donor_vs_time']['between_time_pairs']
    donor = np.asarray([row['d'] for row in donor_block], dtype=float)
    time_dist = np.asarray([row['d'] for row in time_block], dtype=float)
    if len(donor) != 12 or len(time_dist) != 18:
        raise RuntimeError(f'unexpected geometry sizes: donor={len(donor)}, time={len(time_dist)}')
    observed_d = float(np.median(donor))
    observed_t = float(np.median(time_dist))
    observed_ratio = observed_t / observed_d
    source = geometry['donor_vs_time']
    if not np.isclose(observed_d, source['median_between_donor_same_time']):
        raise RuntimeError('donor median does not reproduce source report')
    if not np.isclose(observed_t, source['median_between_time_same_donor']):
        raise RuntimeError('time median does not reproduce source report')
    donor_median_boot = bootstrap_median(donor, args.n_boot, args.seed + 7)
    time_median_boot = bootstrap_median(time_dist, args.n_boot, args.seed + 11)
    ratio_boot = bootstrap_ratio(donor, time_dist, args.n_boot, args.seed + 13)
    cosine_map = transfer['donor_trajectory_alignment']['pairwise_cosine']
    cosines = np.asarray(list(cosine_map.values()), dtype=float)
    cosine_boot = bootstrap_mean(cosines, args.n_boot, args.seed + 31)
    ratio_ci = np.percentile(ratio_boot, [2.5, 97.5])
    cosine_ci = np.percentile(cosine_boot, [2.5, 97.5])
    out = {'experiment': 'donor_geometry_inference', 'status': 'completed', 'scientific_unit': 'pairwise distances and donor displacement directions', 'patient_level_inference_available': False, 'scope': {'patient_level_inference_available': False, 'limit': 'three healthy acute-wound donors; descriptive geometry only; no DFU inference'}, 'scientific_scope': 'descriptive uncertainty audit of three healthy acute-wound donors in GSE241132; no new model fit, no DFU inference, and no universal donor/time claim', 'source_experiments': ['human_temporal_flow', 'donor_transfer_flow'], 'n_donors': 3, 'observed': {'median_between_donor_same_time': observed_d, 'median_between_time_same_donor': observed_t, 'donor_to_time_median_ratio': observed_d / observed_t, 'time_to_donor_median_ratio': observed_ratio, 'donor_dominates': bool(observed_d > observed_t), 'mean_pairwise_displacement_cosine': float(cosines.mean()), 'displacement_magnitude_spread': float(transfer['donor_trajectory_alignment']['magnitude_spread'])}, 'bootstrap': {'n_resamples': int(args.n_boot), 'median_donor_seed': int(args.seed + 7), 'median_time_seed': int(args.seed + 11), 'ratio_seed': int(args.seed + 13), 'cosine_seed': int(args.seed + 31), 'median_donor_ci95': [float(np.percentile(donor_median_boot, 2.5)), float(np.percentile(donor_median_boot, 97.5))], 'median_time_ci95': [float(np.percentile(time_median_boot, 2.5)), float(np.percentile(time_median_boot, 97.5))], 'median_ratio_ci95': [float(ratio_ci[0]), float(ratio_ci[1])], 'donor_to_time_ratio_ci95': [float(1.0 / ratio_ci[1]), float(1.0 / ratio_ci[0])], 'mean_cosine_ci95': [float(cosine_ci[0]), float(cosine_ci[1])], 'ratio_fraction_above_one': float(np.mean(ratio_boot > 1.0))}, 'pairwise_inputs': {'between_donor_same_time': donor_block, 'between_time_same_donor': time_block, 'pairwise_displacement_cosine': cosine_map}, 'interpretation': 'within this three-donor acute-wound cohort, same-donor temporal distances are larger than same-time donor distances and displacement directions are closely aligned; the bootstrap is descriptive and cannot establish a universal donor-versus-time ordering', 'protocol': {'script_sha256': sha256(Path(__file__)), 'geometry_report_sha256': sha256(geometry_path), 'transfer_report_sha256': sha256(transfer_path), 'geometry_report': str(geometry_path), 'transfer_report': str(transfer_path), 'bootstrap_resampling': 'pairwise distances resampled within comparison class; cosine pairs resampled with replacement', 'new_model_fits': False, 'descriptive_donor_count': 3, 'elapsed_seconds': time.time() - started}}
    (output / 'report.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'observed': out['observed'], 'bootstrap': out['bootstrap'], 'output': str(output / 'report.json')}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
