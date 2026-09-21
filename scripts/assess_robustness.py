"""Assess robustness."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.human_wound_data import COND_TIME, fold_standardize

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()

def energy(a, b):
    """Biased empirical energy statistic, with diagonal zero distances included."""
    return float(2 * cdist(a, b).mean() - 2 * pdist(a).sum() / len(a) ** 2 - 2 * pdist(b).sum() / len(b) ** 2)

def temporal_analysis(output, n_draws):
    folder = ROOT / 'outputs/analysis/human_temporal_flow'
    module_path = ROOT / 'scripts/reconstruct_held_out_timepoint.py'
    spec = importlib.util.spec_from_file_location('temporal_reconstruction', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mu = np.load(folder / 'mu.npy')
    obs = pd.read_csv(folder / 'obs_fibroblast.csv')
    assert len(obs) == len(mu) == 12259
    conditions = obs.cond.to_numpy()
    donors = obs.donor.to_numpy()
    training_mask = conditions != 'Wound7'
    standardized, scaler = fold_standardize(mu, training_mask)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Fitting one fixed-seed pooled field for numerical sensitivity', flush=True)
    field = module.train_field_paired(standardized, donors, conditions, np.ones(len(obs), dtype=bool), [('Skin', 'Wound1'), ('Wound1', 'Wound30')], mu.shape[1], dev, 8000, 256, np.random.default_rng(0), seed=0)
    field.eval()
    torch.save(field.state_dict(), output / 'temporal_field.pt')
    source = standardized[conditions == 'Wound1'].astype(np.float32)
    target = standardized[conditions == 'Wound7'].astype(np.float32)
    predictions = {}
    for steps in (25, 50, 100, 200):
        predictions[steps] = module.integrate(field, torch.from_numpy(source).to(dev), COND_TIME['Wound1'], COND_TIME['Wound7'], steps)
    np.save(output / 'predicted_cells.npy', predictions[50])
    chosen_target = np.random.default_rng(0).choice(len(target), 1500, replace=False)
    fixed_target = target[chosen_target]
    baseline = energy(source, fixed_target)
    ref = predictions[200]
    numerical = []
    for steps, pred in predictions.items():
        ed = energy(pred, fixed_target)
        numerical.append({'rk4_steps': steps, 'energy_distance': ed, 'relative_improvement': (baseline - ed) / baseline, 'rms_coordinate_difference_from_200': float(np.sqrt(np.mean((pred - ref) ** 2)))})
    draws = []
    for cap in (250, 500, 1000):
        for seed in range(n_draws):
            rng = np.random.default_rng(3000 + seed)
            si = rng.choice(len(source), min(cap, len(source)), replace=False)
            ti = rng.choice(len(target), min(cap, len(target)), replace=False)
            b = energy(source[si], target[ti])
            d = energy(predictions[50][si], target[ti])
            order = rng.permutation(len(target))[:2 * cap]
            noise = energy(target[order[:cap]], target[order[cap:]])
            draws.append({'cells_per_distribution': cap, 'seed': seed, 'stay_still_distance': b, 'predicted_distance': d, 'split_half_distance': noise, 'gain': b - d, 'relative_improvement': (b - d) / b, 'gain_exceeds_split_half': bool(b - d > noise)})
        print(f'Evaluation sensitivity: {cap} cells, {n_draws} draws completed', flush=True)
    grouped = []
    for cap in (250, 500, 1000):
        block = [r for r in draws if r['cells_per_distribution'] == cap]
        gains = [r['relative_improvement'] for r in block]
        grouped.append({'cells_per_distribution': cap, 'draws': n_draws, 'median_relative_improvement': float(np.median(gains)), 'central_95_draw_range': interval(gains), 'gain_exceeds_split_half_count': sum((r['gain_exceeds_split_half'] for r in block))})
    original = json.loads((folder / 'report.json').read_text())['all_donors']
    return {'fit': {'seed': 0, 'optimization_steps': 8000, 'batch_size': 256, 'device': str(dev), 'n_training_cells': int(training_mask.sum()), 'n_source_cells': len(source), 'n_target_cells': len(target), 'training_excludes_target': True, 'scaler': scaler}, 'numerical_sensitivity': {'reference_steps': 200, 'stay_still_distance': baseline, 'evaluation_target_cells': 1500, 'rows': numerical, 'historical_predicted_distance': original['predicted'], 'historical_baseline_distance': original['standstill']}, 'evaluation_sampling': {'summary': grouped, 'draws': draws, 'interval_type': 'central Monte Carlo range conditional on one fitted field and the observed cohort; not a confidence interval'}, 'source_fingerprints': {str(p.relative_to(ROOT)): fingerprint(p) for p in (folder / 'mu.npy', folder / 'obs_fibroblast.csv', folder / 'report.json', module_path)}}

def main():
    ap = argparse.ArgumentParser(description="Recompute the study robustness analysis")
    ap.add_argument('--output-dir', default=str(ROOT/'outputs/reruns/robustness'))
    ap.add_argument('--evaluation-draws', type=int, default=50)
    args=ap.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    started=time.time()
    result={'status':'completed','scope':'Retrospective sensitivity within the existing cohort; no additional biological replicates.', 'temporal':temporal_analysis(output, args.evaluation_draws)}
    result['protocol']={'script_sha256':fingerprint(Path(__file__)), 'elapsed_seconds':time.time()-started}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed: '+str(output/'report.json'), flush=True)
if __name__ == '__main__':
    main()
