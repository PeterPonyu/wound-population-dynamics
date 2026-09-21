#!/usr/bin/env python3
"""Reconstruct held out timepoint."""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.population_flow import LatentFlowField, compute_cfm_loss
from wound_models.human_wound_data import COND_ORDER, COND_TIME, ANALYSIS_ROOT, preserve_reference_outputs, fold_standardize, load_lineage_latent, paired_endpoint_pairs, protocol_block

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def energy_distance(a: np.ndarray, b: np.ndarray, n_max: int=1500, seed: int=0) -> float:
    rng = np.random.default_rng(seed)
    if a.shape[0] > n_max:
        a = a[rng.choice(a.shape[0], n_max, replace=False)]
    if b.shape[0] > n_max:
        b = b[rng.choice(b.shape[0], n_max, replace=False)]

    def md(x, y):
        return float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean())
    return float(2 * md(a, b) - md(a, a) - md(b, b))

def train_field_paired(mu_z, donors, conds, eval_mask, segments, n_topics, dev, steps, batch, rng, seed=0):
    torch.manual_seed(seed)
    field = LatentFlowField(latent_dim=n_topics).to(dev)
    opt = torch.optim.Adam(field.parameters(), lr=0.001)
    donor_list = sorted(set(donors[eval_mask]))
    pairs = paired_endpoint_pairs(donors, conds, donor_list, segments)
    cache = {}
    for donor, src, tgt in pairs:
        for cond in (src, tgt):
            key = (donor, cond)
            if key not in cache:
                cells = mu_z[eval_mask & (donors == donor) & (conds == cond)]
                cache[key] = torch.from_numpy(np.asarray(cells, dtype=np.float32)).to(dev)
    for step in range(steps):
        donor, src, tgt = pairs[step % len(pairs)]
        za = cache[donor, src]
        zb = cache[donor, tgt]
        ia = torch.from_numpy(rng.integers(0, za.shape[0], batch)).to(dev)
        ib = torch.from_numpy(rng.integers(0, zb.shape[0], batch)).to(dev)
        opt.zero_grad()
        compute_cfm_loss(field, zb[ib], za[ia], t_0=COND_TIME[src], t_1=COND_TIME[tgt]).backward()
        opt.step()
    return field

def integrate(field, z0, t_a, t_b, n_steps=50):
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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-gsm-dir', default='data/raw/GSE241132/per_gsm')
    ap.add_argument('--cell-metadata', default='data/raw/GSE241132/GSE241132_cell_metadata.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'human_temporal_flow'))
    ap.add_argument('--lineage', default='Fibroblast')
    ap.add_argument('--holdout', default='Wound7')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {'holdout': args.holdout, 'time_axis': COND_TIME, 'lineage_source': "authors' newMainCellTypes, not our marker panel"}
    log('loading GSE241132 with author metadata (raw-row contract)')
    packed = load_lineage_latent(args.per_gsm_dir, args.cell_metadata, args.discovery_dir, lineage=args.lineage)
    mu = packed['latent']
    obs_f = packed['obs']
    n_topics = packed['n_topics']
    n_panel = packed['n_panel']
    dev = packed['device']
    log(f"joined {packed['n_cells_joined']}/{packed['n_cells_loaded']} cells; {args.lineage}s={len(obs_f)}; panel {packed['covered']}/{n_panel}")
    report['qc'] = {'cells_loaded': packed['n_cells_loaded'], 'cells_after_join': packed['n_cells_joined']}
    report['panel_coverage'] = {'covered': packed['covered'], 'total': int(n_panel), 'fraction': packed['covered'] / n_panel}
    log('cells per donor x condition:\n' + obs_f.groupby(['donor', 'cond'], observed=True).size().unstack(fill_value=0).to_string())
    conds = obs_f['cond'].to_numpy()
    donors = obs_f['donor'].to_numpy()
    train_conds = [c for c in COND_ORDER if c != args.holdout]
    segments = [(train_conds[i], train_conds[i + 1]) for i in range(len(train_conds) - 1)]
    prior = max((c for c in train_conds if COND_TIME[c] < COND_TIME[args.holdout]), key=lambda c: COND_TIME[c])
    log(f"training segments: {', '.join((f'{a}->{b}' for a, b in segments))}; integrating {prior} -> {args.holdout}")

    def run(eval_mask: np.ndarray, scaler_mask: np.ndarray, tag: str) -> dict:
        mu_z, scaler = fold_standardize(mu, scaler_mask)
        Z = {c: torch.from_numpy(mu_z[eval_mask & (conds == c)].astype(np.float32)).to(dev) for c in COND_ORDER}
        if any((Z[c].shape[0] < 50 for c in COND_ORDER)):
            return {}
        field = train_field_paired(mu_z, donors, conds, eval_mask, segments, n_topics, dev, args.steps, args.batch_size, rng, seed=args.seed)
        pred = integrate(field, Z[prior], COND_TIME[prior], COND_TIME[args.holdout])
        actual = Z[args.holdout].cpu().numpy()
        start = Z[prior].cpu().numpy()
        ed_p, ed_b = (energy_distance(pred, actual), energy_distance(start, actual))
        perm = np.random.default_rng(args.seed).permutation(len(actual))
        half = len(actual) // 2
        ed_n = energy_distance(actual[perm[:half]], actual[perm[half:]])
        d = {'predicted': ed_p, 'standstill': ed_b, 'noise_floor': ed_n, 'improvement': (ed_b - ed_p) / ed_b if ed_b > 0 else float('nan'), 'beats_baseline': bool(ed_p < ed_b), 'n_cells': {c: int(Z[c].shape[0]) for c in COND_ORDER}, 'scaler': scaler}
        log(f"  [{tag}] predicted={ed_p:.4f} standstill={ed_b:.4f} noise={ed_n:.4f} improvement={d['improvement']:+.1%} -> {('BEATS' if d['beats_baseline'] else 'loses to')} baseline")
        return d
    log()
    log('all donors pooled:')
    pooled_scaler = np.isin(conds, train_conds)
    overall = run(np.ones(len(obs_f), dtype=bool), pooled_scaler, 'all')
    report['all_donors'] = overall
    log()
    log('leave-one-donor-out:')
    loo = {}
    for d in sorted(set(donors)):
        loo[d] = run(donors != d, (donors != d) & np.isin(conds, train_conds), f'drop {d}')
    report['leave_one_donor_out'] = loo
    report['protocol'] = protocol_block(__file__, args.discovery_dir, args.seed, folds=['pooled'] + [f'drop_{d}' for d in sorted(set(donors))], extra={'paired_same_donor_endpoints': True, 'holdout_timepoint': args.holdout, 'steps': args.steps, 'batch_size': args.batch_size})
    donor_list = sorted(set(donors))
    mu_diag, _ = fold_standardize(mu, np.isin(conds, train_conds))
    same_time_diff_donor, same_donor_diff_time = ([], [])
    for c in COND_ORDER:
        for i in range(len(donor_list)):
            for j in range(i + 1, len(donor_list)):
                a = mu_diag[(conds == c) & (donors == donor_list[i])]
                b = mu_diag[(conds == c) & (donors == donor_list[j])]
                if len(a) >= 50 and len(b) >= 50:
                    same_time_diff_donor.append({'cond': c, 'pair': f'{donor_list[i]}|{donor_list[j]}', 'd': energy_distance(a, b)})
    for d in donor_list:
        for i in range(len(COND_ORDER)):
            for j in range(i + 1, len(COND_ORDER)):
                a = mu_diag[(donors == d) & (conds == COND_ORDER[i])]
                b = mu_diag[(donors == d) & (conds == COND_ORDER[j])]
                if len(a) >= 50 and len(b) >= 50:
                    same_donor_diff_time.append({'donor': d, 'pair': f'{COND_ORDER[i]}|{COND_ORDER[j]}', 'd': energy_distance(a, b)})
    md_donor = float(np.median([x['d'] for x in same_time_diff_donor]))
    md_time = float(np.median([x['d'] for x in same_donor_diff_time]))
    donor_dominates = bool(md_donor > md_time)
    log()
    log(f'donor vs time: median energy distance between DONORS at the same timepoint = {md_donor:.4f} (n={len(same_time_diff_donor)} pairs)')
    log(f'               median energy distance between TIMEPOINTS within a donor = {md_time:.4f} (n={len(same_donor_diff_time)} pairs)')
    log(f"  -> donor identity {('DOMINATES' if donor_dominates else 'does not dominate')} healing time in this latent ({md_donor / md_time:.2f}x)")
    report['donor_vs_time'] = {'median_between_donor_same_time': md_donor, 'median_between_time_same_donor': md_time, 'ratio': md_donor / md_time if md_time > 0 else float('nan'), 'donor_dominates': donor_dominates, 'between_donor_pairs': same_time_diff_donor, 'between_time_pairs': same_donor_diff_time}
    zp = {c: mu_diag[conds == c] for c in COND_ORDER}
    d_ph = energy_distance(zp[prior], zp[args.holdout])
    d_hl = energy_distance(zp[args.holdout], zp[train_conds[-1]])
    d_pl = energy_distance(zp[prior], zp[train_conds[-1]])
    on_path = bool(d_pl >= max(d_ph, d_hl))
    log()
    log(f'geometry: d({prior},{args.holdout})={d_ph:.3f}  d({args.holdout},{train_conds[-1]})={d_hl:.3f}  d({prior},{train_conds[-1]})={d_pl:.3f}')
    log(f'  {args.holdout} endpoint-distance ordering satisfied: {on_path}; this is not a geodesic or fairness test')
    report['geometry'] = {'d_prior_holdout': d_ph, 'd_holdout_last': d_hl, 'd_prior_last': d_pl, 'holdout_on_direct_path': on_path}
    log()
    log('=' * 70)
    if overall:
        log(f"HUMAN HELD-OUT {args.holdout}: {('BEATS' if overall['beats_baseline'] else 'LOSES TO')} stand-still ({overall['improvement']:+.1%})")
    agree = [v.get('beats_baseline') for v in loo.values() if v]
    log(f'leave-one-donor-out agreement: {sum((1 for a in agree if a))}/{len(agree)} folds beat the baseline')

    log('=' * 70)
    obs_f[['gsm', 'donor', 'cond', 'celltype']].to_csv(os.path.join(args.output_dir, 'obs_fibroblast.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'mu.npy'), mu)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
