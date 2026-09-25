#!/usr/bin/env python3
"""Historical single-seed sensitivity to four specified time coordinates.

Training excludes the held-out time and pairs endpoints within donor. Axes
also change optimization targets. Endpoint draws advance between axes, so this
experiment does not isolate a causal clock effect. The scan reads a nearest
grid point and uses held-target data to describe an oracle path minimum; that
minimum is not an independently validated prediction or proof of path shape.
Use the paired exact-target audit for a newly controlled comparison. Existing
reports retain their historical values and producing-source fingerprints.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.population_flow import LatentFlowField, compute_cfm_loss
from wound_models.human_wound_data import COND_ORDER, ANALYSIS_ROOT, preserve_reference_outputs, fold_standardize, load_lineage_latent, paired_endpoint_pairs, protocol_block
COND_DAYS = {'Skin': 0.0, 'Wound1': 1.0, 'Wound7': 7.0, 'Wound30': 30.0}

def time_axes() -> dict:
    d = COND_DAYS
    raw = {'rank': {c: i / 3 for i, c in enumerate(COND_ORDER)}, 'days': {c: d[c] / 30.0 for c in COND_ORDER}, 'sqrt_days': {c: np.sqrt(d[c]) / np.sqrt(30.0) for c in COND_ORDER}, 'log_days': {c: np.log1p(d[c]) / np.log1p(30.0) for c in COND_ORDER}}
    return {k: {c: float(v) for c, v in m.items()} for k, m in raw.items()}

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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-gsm-dir', default='data/raw/GSE241132/per_gsm')
    ap.add_argument('--cell-metadata', default='data/raw/GSE241132/GSE241132_cell_metadata.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'time_parameterisation'))
    ap.add_argument('--lineage', default='Fibroblast')
    ap.add_argument('--holdout', default='Wound7')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--scan-steps', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'holdout': args.holdout, 'axes': time_axes()}
    log('loading GSE241132 with author metadata (raw-row contract)')
    packed = load_lineage_latent(args.per_gsm_dir, args.cell_metadata, args.discovery_dir, lineage=args.lineage)
    mu = packed['latent']
    obs_f = packed['obs']
    n_topics = packed['n_topics']
    dev = packed['device']
    log(f"{args.lineage}s: {len(obs_f)}; panel {packed['covered']}/{packed['n_panel']}")
    conds = obs_f['cond'].to_numpy()
    donors = obs_f['donor'].to_numpy()
    train_conds = [c for c in COND_ORDER if c != args.holdout]
    mu_z, scaler = fold_standardize(mu, np.isin(conds, train_conds))
    Zt = {c: torch.from_numpy(mu_z[conds == c]).to(dev) for c in COND_ORDER}
    Znp = {c: mu_z[conds == c] for c in COND_ORDER}
    segments = [(train_conds[i], train_conds[i + 1]) for i in range(len(train_conds) - 1)]
    prior, last = (train_conds[-2], train_conds[-1])
    actual = Znp[args.holdout]
    ed_standstill = energy_distance(Znp[prior], actual)
    perm = np.random.default_rng(args.seed).permutation(len(actual))
    half = len(actual) // 2
    noise_floor = energy_distance(actual[perm[:half]], actual[perm[half:]])
    log(f"segments {', '.join((f'{a}->{b}' for a, b in segments))}")
    log(f'stand-still baseline d({prior},{args.holdout}) = {ed_standstill:.4f}; noise floor (random split of {args.holdout}) = {noise_floor:.4f}')
    log(f'descriptive reference: gain > {noise_floor:.4f}; the split-half distance is not a hypothesis-test threshold')
    pairs = paired_endpoint_pairs(donors, conds, sorted(set(donors)), segments)
    cache = {}
    for donor, src, tgt in pairs:
        for cond in (src, tgt):
            key = (donor, cond)
            if key not in cache:
                cache[key] = torch.from_numpy(mu_z[(donors == donor) & (conds == cond)]).to(dev)
    rng = np.random.default_rng(args.seed)
    results = {}
    for axis_name, T in time_axes().items():
        torch.manual_seed(args.seed)
        field = LatentFlowField(latent_dim=n_topics).to(dev)
        opt = torch.optim.Adam(field.parameters(), lr=0.001)
        for step in range(args.steps):
            donor, src, tgt = pairs[step % len(pairs)]
            za = cache[donor, src]
            zb = cache[donor, tgt]
            ia = torch.from_numpy(rng.integers(0, za.shape[0], args.batch_size)).to(dev)
            ib = torch.from_numpy(rng.integers(0, zb.shape[0], args.batch_size)).to(dev)
            opt.zero_grad()
            compute_cfm_loss(field, zb[ib], za[ia], t_0=T[src], t_1=T[tgt]).backward()
            opt.step()
        t_a, t_b = (T[prior], T[last])
        dt = (t_b - t_a) / args.scan_steps
        z = Zt[prior].clone()
        scan = []
        with torch.no_grad():
            for s in range(args.scan_steps + 1):
                frac = s / args.scan_steps
                scan.append({'frac': frac, 't': t_a + s * dt, 'ed': energy_distance(z.cpu().numpy(), actual)})
                if s == args.scan_steps:
                    break
                t = t_a + s * dt
                k1 = field(z, t)
                k2 = field(z + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = field(z + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = field(z + dt * k3, t + dt)
                z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_hold = T[args.holdout]
        frac_hold = (t_hold - t_a) / (t_b - t_a)
        at_hold = min(scan, key=lambda r: abs(r['frac'] - frac_hold))
        best = min(scan, key=lambda r: r['ed'])
        results[axis_name] = {'t_holdout': t_hold, 'claimed_fraction': frac_hold, 'ed_at_claimed': at_hold['ed'], 'ed_best_on_path': best['ed'], 'best_fraction': best['frac'], 'beats_standstill_at_claimed': bool(ed_standstill - at_hold['ed'] > noise_floor), 'beats_standstill_anywhere': bool(ed_standstill - best['ed'] > noise_floor), 'gain_at_claimed': ed_standstill - at_hold['ed'], 'gain_best_on_path': ed_standstill - best['ed'], 'improvement_at_claimed': (ed_standstill - at_hold['ed']) / ed_standstill, 'diverged': bool(at_hold['ed'] > 10 * ed_standstill)}
        r = results[axis_name]
        flag = ' [DIVERGED]' if r['diverged'] else ''
        log(f"  [{axis_name:<9}] claimed {frac_hold:5.1%}: ED={at_hold['ed']:8.4f}{flag}  gain={r['gain_at_claimed']:+.4f} ({('PASS' if r['beats_standstill_at_claimed'] else 'fail')})  | best on path {best['frac']:5.1%}: ED={best['ed']:.4f} gain={r['gain_best_on_path']:+.4f} ({('PASS' if r['beats_standstill_anywhere'] else 'fail')})")
        report.setdefault('scans', {})[axis_name] = scan
    report['standstill'] = ed_standstill
    report['noise_floor'] = noise_floor
    report['axes_results'] = results
    diverged = [k for k, v in results.items() if v['diverged']]
    if diverged:
        log()
        log(f'Large observed target error for axes {diverged}; the 10x-baseline flag does not identify numerical instability or its cause. Changing a segment span also changes the velocity target.')
    report['diverged_axes'] = diverged
    any_path_works = any((v['beats_standstill_anywhere'] for v in results.values()))
    winners = [k for k, v in results.items() if v['beats_standstill_at_claimed']]
    log()
    log('=' * 74)
    if winners:
        best_axis = min(winners, key=lambda k: results[k]['ed_at_claimed'])
        w = results[best_axis]
        log(f"Observed single-fit result: '{best_axis}' places {args.holdout} at {w['claimed_fraction']:.1%} and beats stand-still ({w['ed_at_claimed']:.4f} vs {ed_standstill:.4f}, {w['improvement_at_claimed']:+.1%})")
        log('This selected observed minimum does not identify a biological clock or path mechanism.')
        verdict = f'observed_reference_criterion_met:{best_axis}'
    elif any_path_works:
        b = min(results.items(), key=lambda kv: kv[1]['ed_best_on_path'])
        log(f"PARTIAL: no axis places {args.holdout} correctly, but the learned path does pass near it ({b[1]['ed_best_on_path']:.4f} at {b[1]['best_fraction']:.1%} of the segment, axis '{b[0]}', gain {b[1]['gain_best_on_path']:+.4f} vs noise floor {noise_floor:.4f}).")
        log('A target-guided path minimum is optimistic; it does not validate a new schedule.')
        verdict = 'oracle_scan_only_meets_reference'
    else:
        best_gain = max((v['gain_best_on_path'] for v in results.values()))
        log(f'No tested single fit meets the descriptive reference at its target, and the best point anywhere on any learned path improves on standing still by only {best_gain:+.4f}, which is inside the {noise_floor:.4f} noise floor.')
        log('This result does not separate optimization, representation, clock or path effects.')
        verdict = 'tested_single_fits_below_descriptive_reference'
    log('=' * 74)
    report['verdict'] = verdict
    report['protocol'] = protocol_block(__file__, args.discovery_dir, args.seed, folds=[f'axis:{name}' for name in time_axes()], extra={'paired_same_donor_endpoints': True, 'holdout_timepoint': args.holdout, 'scaler': scaler, 'steps': args.steps, 'batch_size': args.batch_size})
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
