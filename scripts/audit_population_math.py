#!/usr/bin/env python3
"""Verify saved-input mathematical estimands."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.human_wound_data import fold_standardize

def donor_weights(donors):
    donors = np.asarray(donors)
    names, counts = np.unique(donors, return_counts=True)
    sizes = dict(zip(names, counts))
    return np.asarray([1.0 / (len(names) * sizes[name]) for name in donors])

def weighted_energy(x, y, a=None, b=None):
    """Exact weighted empirical energy distance, including self-distance zeros."""
    x, y = (np.asarray(x, float), np.asarray(y, float))
    a = np.full(len(x), 1 / len(x)) if a is None else np.asarray(a, float)
    b = np.full(len(y), 1 / len(y)) if b is None else np.asarray(b, float)
    for weights, n in ((a, len(x)), (b, len(y))):
        if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any() or (not np.isclose(weights.sum(), 1)):
            raise ValueError('Weights must be nonnegative finite unit-mass vectors')
    return float(2 * a @ cdist(x, y) @ b - a @ cdist(x, x) @ a - b @ cdist(y, y) @ b)

def comparison(label, source, prediction, target, a=None, b=None):
    baseline = weighted_energy(source, target, a, b)
    error = weighted_energy(prediction, target, a, b)
    if baseline <= 0:
        raise ValueError('Relative improvement requires a positive baseline')
    return dict(evaluation=label, n_source=len(source), n_target=len(target), stay_still_distance=baseline, predicted_distance=error, absolute_gain=baseline - error, relative_improvement=1 - error / baseline)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/mathematical_audit/population')
    args = ap.parse_args()
    if (args.output_dir / 'report.json').exists():
        raise FileExistsError('Choose a fresh output directory; reference reports are read only')
    folder = ROOT / 'outputs/analysis/human_temporal_flow'
    inputs = {'coordinates': folder / 'mu.npy', 'observations': folder / 'obs_fibroblast.csv', 'prediction': ROOT / 'outputs/robustness_extensions/predicted_cells.npy', 'field': ROOT / 'outputs/robustness_extensions/temporal_field.pt', 'fit_report': ROOT / 'outputs/robustness_extensions/report.json'}
    mu, obs = (np.load(inputs['coordinates']), pd.read_csv(inputs['observations']))
    assert len(obs) == len(mu) == 12259
    standardized, scaler = fold_standardize(mu, obs.cond.ne('Wound7').to_numpy())
    source_mask, target_mask = (obs.cond.eq('Wound1').to_numpy(), obs.cond.eq('Wound7').to_numpy())
    source, target = (standardized[source_mask], standardized[target_mask])
    prediction = np.load(inputs['prediction'])
    assert source.shape == prediction.shape == (1012, 15) and target.shape == (3821, 15)
    source_donors, target_donors = (obs.donor[source_mask].to_numpy(), obs.donor[target_mask].to_numpy())
    rows = [comparison('Cell-pooled', source, prediction, target), comparison('Equal donor mass', source, prediction, target, donor_weights(source_donors), donor_weights(target_donors))]
    for donor in sorted(np.unique(source_donors)):
        si, ti = (source_donors == donor, target_donors == donor)
        rows.append(comparison(str(donor), source[si], prediction[si], target[ti]))
    counts = [dict(donor=str(donor), source_cells=int((source_donors == donor).sum()), target_cells=int((target_donors == donor).sum()), source_cell_mass=float((source_donors == donor).mean()), target_cell_mass=float((target_donors == donor).mean())) for donor in sorted(np.unique(source_donors))]
    report = {'status': 'completed', 'scope': 'All-cell, fixed-field evaluation-weight sensitivity in three observed donors; donor rows use the same pooled time-holdout field, not leave-one-donor-out transfer.', 'evaluation': rows, 'donor_counts': counts, 'protocol': {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_sha256': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()}, 'retrained': False, 'subsampling': False, 'computation_dtype': 'float64', 'rk4_steps': 50, 'scaler': scaler, 'source_prediction_row_alignment': 'Saved in source-row order by the fixed-field sensitivity analysis'}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / 'evaluation_weights.csv', index=False)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))
if __name__ == '__main__':
    main()
