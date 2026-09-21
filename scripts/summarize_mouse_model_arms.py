#!/usr/bin/env python3
"""Summarize mouse model arms."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'outputs' / 'analysis'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _load(path: Path, expected_arm: str) -> dict:
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('status') != 'completed':
        raise RuntimeError(f'{path}: status is not completed')
    if report.get('model_arm') != expected_arm:
        raise RuntimeError(f"{path}: expected arm {expected_arm}, got {report.get('model_arm')}")
    protocol = report.get('protocol') or {}
    if protocol.get('holdout_timepoint_excluded_from_fit') != 'POD7':
        raise RuntimeError(f'{path}: POD7 was not held out')
    if protocol.get('time_axis') != 'rank':
        raise RuntimeError(f'{path}: time axis is not rank')
    if 'one animal per timepoint' not in str(protocol.get('scope', '')):
        raise RuntimeError(f'{path}: scope lost one-animal limitation')
    if protocol.get('scaler_fit_on_training_timepoints_only') is not True:
        raise RuntimeError(f'{path}: scaler is not training-timepoint-only')
    return report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--output-dir', default=str(DEFAULT_ROOT / 'mouse_all_arms_pod7_audit'))
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.root)
    paths = {'NDB': root / 'mouse_temporal_flow' / 'report.json', 'PDB': root / 'mouse_temporal_flow_PDB' / 'report.json', 'GDB': root / 'mouse_temporal_flow_GDB' / 'report.json'}
    reports = {arm: _load(path, arm) for arm, path in paths.items()}
    protocol_hashes = {key: (rep['protocol'].get('panel_sha256'), rep['protocol'].get('model_sha256')) for key, rep in reports.items()}
    if len(set(protocol_hashes.values())) != 1:
        raise RuntimeError(f'frozen encoder hashes differ across arms: {protocol_hashes}')
    arms = {}
    for arm, rep in reports.items():
        q1 = rep['Q1_cross_species']
        q2 = rep['Q2_holdout']
        q3 = rep['Q3_geometry']
        arms[arm] = {'panel_covered': int(q1['covered']), 'panel_size': int(q1['panel_size']), 'panel_fraction': float(q1['fraction']), 'n_fibroblasts': int(q1['n_fibroblasts']), 'mean_perplexity': float(q1['mean_perplexity']), 'n_cells_by_pod': q2['n_cells'], 'energy_distance_predicted': float(q2['energy_distance_predicted']), 'energy_distance_standstill': float(q2['energy_distance_standstill']), 'energy_distance_noise_floor': float(q2['energy_distance_noise_floor']), 'improvement_fraction': float(q2['improvement_fraction']), 'beats_baseline': bool(q2['beats_baseline']), 'passes_noise_gate': bool(q2['passes_noise_gate']), 'holdout_on_direct_path': bool(q3['holdout_on_direct_path'])}
    improvements = [v['improvement_fraction'] for v in arms.values()]
    out = {'experiment': 'mouse_all_arms_pod7_audit', 'status': 'completed', 'scientific_scope': 'three one-animal-per-timepoint GSE326622 mouse model-arm audits under one frozen cross-species projection protocol; computational only, not statistical inference and not DFU validation', 'not_statistical_inference': True, 'not_DFU_validation': True, 'arms': arms, 'aggregate': {'n_arms': 3, 'n_beats_standstill': int(sum((v['beats_baseline'] for v in arms.values()))), 'n_passes_noise_gate': int(sum((v['passes_noise_gate'] for v in arms.values()))), 'n_holdouts_on_direct_path': int(sum((v['holdout_on_direct_path'] for v in arms.values()))), 'mean_improvement_fraction': float(sum(improvements) / len(improvements)), 'improvement_range': [float(min(improvements)), float(max(improvements))], 'interpretation': 'arm-level computation is reported without treating cells or arms as independent clinical observations'}, 'protocol': {'script_sha256': sha256(Path(__file__)), 'source_report_sha256': {arm: sha256(path) for arm, path in paths.items()}, 'source_reports': {arm: str(path) for arm, path in paths.items()}, 'frozen_panel_sha256': protocol_hashes['NDB'][0], 'frozen_model_sha256': protocol_hashes['NDB'][1], 'same_protocol_across_arms': True, 'no_refit': True, 'elapsed_seconds': time.time() - t0}}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'arms': arms, 'aggregate': out['aggregate'], 'output': str(output / 'report.json')}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
