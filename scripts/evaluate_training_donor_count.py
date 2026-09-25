#!/usr/bin/env python3
"""Is donor count the limiting resource? Measure the first two points.

The retired claim was that population flow fields fail on a new individual
because the cohort has too few donors. After the barcode-row repair the shared
field does transfer to every held-out donor, so the resource claim has to be
re-tested rather than assumed in either direction.

The only falsifiable version available with three donors is a learning curve
over the number of training donors. For each held-out donor the same protocol
is run twice: trained on one other donor, and trained on both other donors.
Everything else is held fixed - segments, steps, batch size, seed, integration,
energy distance, and the randomized split-half noise floor. The scaler for each
condition sees only that condition's training donors, so a k=1 run contains
information from exactly one individual.

A non-neural mean-displacement predictor is run on the identical folds.
It is a restricted predictor and can also be misspecified. Similar reductions
do not identify a unique mechanism or prove a neural capacity limitation.
This historical script scores each training subset in its own standardized
geometry; absolute cross-k differences confound training and evaluation scale.
Use the fixed-reference protocol for the current common-scale comparison.

Two points cannot establish a curve shape. The output is a direction and a
magnitude for the step from one donor to two, plus the explicit statement that
extrapolating to eight donors from this is not supported.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.population_flow import LatentFlowField, compute_cfm_loss
from wound_models.human_wound_data import COND_ORDER, COND_TIME, ANALYSIS_ROOT, preserve_reference_outputs, fold_standardize, load_lineage_latent, paired_endpoint_pairs, protocol_block

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def energy_distance(a: np.ndarray, b: np.ndarray, n_max: int=1200, seed: int=0) -> float:
    rng = np.random.default_rng(seed)
    if a.shape[0] > n_max:
        a = a[rng.choice(a.shape[0], n_max, replace=False)]
    if b.shape[0] > n_max:
        b = b[rng.choice(b.shape[0], n_max, replace=False)]

    def md(x, y):
        return float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean())
    return float(2 * md(a, b) - md(a, a) - md(b, b))

def integrate(field, z0, t_a, t_b, n_steps: int=50):
    dt = (t_b - t_a) / n_steps
    with torch.no_grad():
        z = z0.clone()
        for s in range(n_steps):
            t = t_a + s * dt
            k1 = field(z, t)
            k2 = field(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = field(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = field(z + dt * k3, t + dt)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z.cpu().numpy()

def train_shared_field(mu_z, donors, conds, train_donors, segments, *, n_topics, dev, steps, batch_size, seed):
    pairs = paired_endpoint_pairs(donors, conds, train_donors, segments)
    cache = {}
    for donor, src, tgt in pairs:
        for cond in (src, tgt):
            key = (donor, cond)
            if key not in cache:
                cache[key] = torch.from_numpy(mu_z[(donors == donor) & (conds == cond)]).to(dev)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    field = LatentFlowField(latent_dim=n_topics).to(dev)
    opt = torch.optim.Adam(field.parameters(), lr=0.001)
    for step in range(steps):
        donor, src, tgt = pairs[step % len(pairs)]
        za, zb = (cache[donor, src], cache[donor, tgt])
        ia = torch.from_numpy(rng.integers(0, za.shape[0], batch_size)).to(dev)
        ib = torch.from_numpy(rng.integers(0, zb.shape[0], batch_size)).to(dev)
        opt.zero_grad()
        compute_cfm_loss(field, zb[ib], za[ia], t_0=COND_TIME[src], t_1=COND_TIME[tgt]).backward()
        opt.step()
    return (field, pairs)

def mean_displacement_prediction(mu_z, donors, conds, train_donors, source, target, src_cells):
    deltas = []
    for donor in train_donors:
        a = mu_z[(donors == donor) & (conds == source)]
        b = mu_z[(donors == donor) & (conds == target)]
        if len(a) >= 20 and len(b) >= 20:
            deltas.append(b.mean(axis=0) - a.mean(axis=0))
    if not deltas:
        return None
    return src_cells + np.mean(deltas, axis=0)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-gsm-dir', default='data/raw/GSE241132/per_gsm')
    ap.add_argument('--cell-metadata', default='data/raw/GSE241132/GSE241132_cell_metadata.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'donor_count_curve'))
    ap.add_argument('--lineage', default='Fibroblast')
    ap.add_argument('--source', default='Wound1')
    ap.add_argument('--target', default='Wound7')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    log('loading GSE241132 with the raw-row contract')
    packed = load_lineage_latent(args.per_gsm_dir, args.cell_metadata, args.discovery_dir, lineage=args.lineage)
    mu, obs_f = (packed['latent'], packed['obs'])
    n_topics, dev = (packed['n_topics'], packed['device'])
    conds = obs_f['cond'].to_numpy()
    donors = obs_f['donor'].to_numpy()
    donor_list = sorted(set(donors))
    segments = [(COND_ORDER[i], COND_ORDER[i + 1]) for i in range(3)]
    log(f'{args.lineage}s: {len(obs_f)}; donors {donor_list}')
    runs = []
    for held in donor_list:
        others = [d for d in donor_list if d != held]
        subsets = [list(c) for k in (1, 2) for c in itertools.combinations(others, k)]
        for train_donors in subsets:
            train_mask = np.isin(donors, train_donors)
            mu_z, scaler = fold_standardize(mu, train_mask)
            src = mu_z[(donors == held) & (conds == args.source)]
            tgt = mu_z[(donors == held) & (conds == args.target)]
            if len(src) < 20 or len(tgt) < 20:
                log(f'  skip {held} (k={len(train_donors)}): held-out endpoint too small')
                continue
            ed_base = energy_distance(src, tgt)
            perm = np.random.default_rng(args.seed).permutation(len(tgt))
            half = len(tgt) // 2
            noise = energy_distance(tgt[perm[:half]], tgt[perm[half:]])
            field, pairs = train_shared_field(mu_z, donors, conds, train_donors, segments, n_topics=n_topics, dev=dev, steps=args.steps, batch_size=args.batch_size, seed=args.seed)
            pred = integrate(field, torch.from_numpy(src).to(dev), COND_TIME[args.source], COND_TIME[args.target])
            ed_cfm = energy_distance(pred, tgt)
            md_pred = mean_displacement_prediction(mu_z, donors, conds, train_donors, args.source, args.target, src)
            ed_md = energy_distance(md_pred, tgt) if md_pred is not None else None
            runs.append({'held_out_donor': held, 'n_training_donors': len(train_donors), 'training_donors': train_donors, 'n_training_pairs': len(pairs), 'n_source_cells': int(len(src)), 'n_target_cells': int(len(tgt)), 'ed_standstill': ed_base, 'noise_floor': noise, 'shared_cfm': {'ed': ed_cfm, 'gain': ed_base - ed_cfm, 'improvement': (ed_base - ed_cfm) / ed_base if ed_base > 0 else None, 'passes_noise_gate': bool(ed_base - ed_cfm > noise)}, 'mean_displacement': None if ed_md is None else {'ed': ed_md, 'gain': ed_base - ed_md, 'improvement': (ed_base - ed_md) / ed_base if ed_base > 0 else None, 'passes_noise_gate': bool(ed_base - ed_md > noise)}, 'scaler': scaler})
            log(f"  [{held}] k={len(train_donors)} ({'+'.join(train_donors)}): cfm={ed_cfm:.4f} md={ed_md:.4f} standstill={ed_base:.4f} noise={noise:.4f} -> {('PASS' if runs[-1]['shared_cfm']['passes_noise_gate'] else 'fail')}")
    curve = {}
    for k in (1, 2):
        block = [r for r in runs if r['n_training_donors'] == k]
        if not block:
            continue
        curve[f'k_{k}'] = {'n_runs': len(block), 'shared_cfm_mean_ed': float(np.mean([r['shared_cfm']['ed'] for r in block])), 'shared_cfm_mean_improvement': float(np.mean([r['shared_cfm']['improvement'] for r in block])), 'shared_cfm_pass_count': int(sum((r['shared_cfm']['passes_noise_gate'] for r in block))), 'mean_displacement_mean_ed': float(np.mean([r['mean_displacement']['ed'] for r in block])), 'mean_displacement_pass_count': int(sum((r['mean_displacement']['passes_noise_gate'] for r in block))), 'mean_noise_floor': float(np.mean([r['noise_floor'] for r in block]))}
    step = None
    if 'k_1' in curve and 'k_2' in curve:
        d_cfm = curve['k_1']['shared_cfm_mean_ed'] - curve['k_2']['shared_cfm_mean_ed']
        d_md = curve['k_1']['mean_displacement_mean_ed'] - curve['k_2']['mean_displacement_mean_ed']
        floor = float(np.mean([r['noise_floor'] for r in runs]))
        paired = []
        for held in donor_list:
            at1 = [r['shared_cfm']['ed'] for r in runs if r['held_out_donor'] == held and r['n_training_donors'] == 1]
            at2 = [r['shared_cfm']['ed'] for r in runs if r['held_out_donor'] == held and r['n_training_donors'] == 2]
            if at1 and at2:
                paired.append({'held_out_donor': held, 'mean_ed_k1': float(np.mean(at1)), 'ed_k2': float(at2[0]), 'improved_with_second_donor': bool(at2[0] < np.mean(at1))})
        step = {'shared_cfm_ed_reduction_k1_to_k2': float(d_cfm), 'mean_displacement_ed_reduction_k1_to_k2': float(d_md), 'mean_noise_floor': floor, 'cfm_reduction_exceeds_noise_floor': bool(abs(d_cfm) > floor), 'per_held_out_donor': paired, 'donors_improved_by_second_training_donor': int(sum((p['improved_with_second_donor'] for p in paired)))}
    if step is None:
        verdict = 'insufficient_folds'
    elif not step['cfm_reduction_exceeds_noise_floor']:
        verdict = 'no_measurable_gain_from_the_second_donor'
    elif step['shared_cfm_ed_reduction_k1_to_k2'] > 0:
        verdict = 'second_donor_reduced_held_out_error'
    else:
        verdict = 'second_donor_increased_held_out_error'
    report = {'experiment': 'donor_count_learning_curve', 'status': 'completed', 'question': 'Does held-out-donor prediction improve when the training set grows from one donor to two?', 'source': args.source, 'target': args.target, 'donors': donor_list, 'runs': runs, 'curve': curve, 'step_one_to_two_donors': step, 'verdict': verdict, 'scope': {'max_training_donors': 2, 'points_on_curve': 2, 'cohort': 'GSE241132; three healthy volunteers; acute experimental wounds', 'not_supported': ['a common-scale absolute error reduction from subset-specific distances', 'an extrapolated donor requirement such as >=8', 'a claim that donor count is or is not the binding constraint', 'any transfer to diabetic foot ulcer patients']}, 'protocol': protocol_block(__file__, args.discovery_dir, args.seed, folds=[f'holdout_{d}' for d in donor_list], extra={'paired_same_donor_endpoints': True, 'scaler_uses_only_that_conditions_training_donors': True, 'identical_steps_across_k': True, 'non_neural_control': 'mean displacement over training donors', 'source': args.source, 'target': args.target, 'steps': args.steps, 'batch_size': args.batch_size, 'elapsed_seconds': time.time() - t0})}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    print(json.dumps({'curve': curve, 'step': step, 'verdict': verdict}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
