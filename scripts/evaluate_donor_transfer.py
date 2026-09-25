#!/usr/bin/env python3
"""Leave-one-donor-out distribution reconstruction in acute human wounds.

Training uses all four observed times in the non-held donors. Held-source
Wound1 cells initialize the field; held Wound7 cells are used for evaluation.
This information set differs from a globally missing time point. Improvement
over unchanged source does not establish clinical prediction, identifiable
cell trajectories, or incremental value beyond a training-target marginal.
The three donors and split-half reference support descriptive cohort auditing.
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
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'donor_transfer_flow'))
    ap.add_argument('--lineage', default='Fibroblast')
    ap.add_argument('--source', default='Wound1')
    ap.add_argument('--target', default='Wound7')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    report = {'source': args.source, 'target': args.target}
    log('loading GSE241132 with author metadata (raw-row contract)')
    packed = load_lineage_latent(args.per_gsm_dir, args.cell_metadata, args.discovery_dir, lineage=args.lineage)
    mu = packed['latent']
    obs_f = packed['obs']
    n_topics = packed['n_topics']
    dev = packed['device']
    log(f'{args.lineage}s: {len(obs_f)}')
    conds = obs_f['cond'].to_numpy()
    donors = obs_f['donor'].to_numpy()
    donor_list = sorted(set(donors))
    segments = [(COND_ORDER[i], COND_ORDER[i + 1]) for i in range(3)]
    log(f'donors: {donor_list}; training segments now include {args.source}->{args.target} (from the other donors)')
    rng = np.random.default_rng(args.seed)
    per_donor = {}
    for held in donor_list:
        train_mask = donors != held
        mu_z, scaler = fold_standardize(mu, train_mask)
        train_donors = [d for d in donor_list if d != held]
        pairs = paired_endpoint_pairs(donors, conds, train_donors, segments)
        cache = {}
        for donor, src, tgt in pairs:
            for cond in (src, tgt):
                key = (donor, cond)
                if key not in cache:
                    cache[key] = torch.from_numpy(mu_z[(donors == donor) & (conds == cond)]).to(dev)
        if any((cache[k].shape[0] < 50 for k in cache)):
            log(f'  skip {held}: a training timepoint has <50 cells')
            continue
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
            compute_cfm_loss(field, zb[ib], za[ia], t_0=COND_TIME[src], t_1=COND_TIME[tgt]).backward()
            opt.step()
        src = mu_z[(donors == held) & (conds == args.source)]
        tgt = mu_z[(donors == held) & (conds == args.target)]
        pred = integrate(field, torch.from_numpy(src).to(dev), COND_TIME[args.source], COND_TIME[args.target])
        ed_pred = energy_distance(pred, tgt)
        ed_base = energy_distance(src, tgt)
        perm = np.random.default_rng(args.seed).permutation(len(tgt))
        half = len(tgt) // 2
        noise = energy_distance(tgt[perm[:half]], tgt[perm[half:]])
        gain = ed_base - ed_pred
        passes = bool(gain > noise)
        per_donor[held] = {'n_source': int(len(src)), 'n_target': int(len(tgt)), 'ed_predicted': ed_pred, 'ed_standstill': ed_base, 'noise_floor': noise, 'gain': gain, 'improvement': gain / ed_base if ed_base > 0 else float('nan'), 'passes_noise_gate': passes, 'scaler': scaler}
        log(f"  [{held}] n={len(src)}->{len(tgt)}  predicted={ed_pred:.4f}  standstill={ed_base:.4f}  noise={noise:.4f}  gain={gain:+.4f} ({gain / ed_base:+.1%})  -> {('PASS' if passes else 'fail')}")
    mu_align, _ = fold_standardize(mu, conds == args.source)
    deltas = {}
    for d in donor_list:
        a = mu_align[(donors == d) & (conds == args.source)]
        b = mu_align[(donors == d) & (conds == args.target)]
        if len(a) >= 50 and len(b) >= 50:
            deltas[d] = b.mean(axis=0) - a.mean(axis=0)
    cos = {}
    for i, di in enumerate(sorted(deltas)):
        for dj in sorted(deltas)[i + 1:]:
            u, v = (deltas[di], deltas[dj])
            denom = float(np.linalg.norm(u) * np.linalg.norm(v))
            cos[f'{di}|{dj}'] = float(u @ v / denom) if denom > 0 else float('nan')
    norms = {d: float(np.linalg.norm(v)) for d, v in deltas.items()}
    log()
    log(f'displacement vectors {args.source} -> {args.target} per donor:')
    for d, n in norms.items():
        log(f'  |delta[{d}]| = {n:.3f}')
    log('pairwise cosine similarity between donor displacements:')
    for k, v in cos.items():
        log(f'  cos({k}) = {v:+.3f}')
    mean_cos = float(np.mean(list(cos.values()))) if cos else float('nan')
    spread = max(norms.values()) / min(norms.values()) if norms else float('nan')
    aligned = bool(mean_cos > 0.5)
    log(f'  mean cosine = {mean_cos:+.3f}; magnitude spread = {spread:.1f}x')
    log(f'  Mean displacement alignment exceeds the descriptive 0.5 cut: {aligned}; this does not establish field specification')
    report['donor_trajectory_alignment'] = {'displacement_norms': norms, 'pairwise_cosine': cos, 'mean_cosine': mean_cos, 'magnitude_spread': spread, 'aligned': aligned}
    report['per_donor'] = per_donor
    n_pass = sum((1 for v in per_donor.values() if v['passes_noise_gate']))
    mean_imp = float(np.mean([v['improvement'] for v in per_donor.values()])) if per_donor else float('nan')
    report['n_donors'] = len(per_donor)
    report['n_pass'] = n_pass
    report['mean_improvement'] = mean_imp
    log()
    log('=' * 74)
    log(f'DONOR TRANSFER {args.source} -> {args.target}: {n_pass}/{len(per_donor)} held-out donors beat stand-still by more than their own noise floor; mean improvement {mean_imp:+.1%}')
    if n_pass == len(per_donor) and per_donor:
        log('All observed donor folds meet the descriptive reference criterion; population generalization requires additional independent donors.')
        verdict = 'transfer_works'
    elif n_pass > 0:
        log('Mixed. Transfer works for some donors and not others, so any claim must be per-donor, not aggregate.')
        verdict = 'transfer_partial'
    elif not aligned:
        log('Transfer fails with heterogeneous mean displacements. This association does not identify the reason for failure or prove that conditioning would repair it.')
        verdict = 'transfer_fails_donors_unaligned'
    else:
        log('Transfer fails despite aligned average displacements; this diagnostic does not locate the source of error.')
        verdict = 'transfer_fails_despite_alignment'
    log('=' * 74)
    report['verdict'] = verdict
    report['protocol'] = protocol_block(__file__, args.discovery_dir, args.seed, folds=[f'holdout_{d}' for d in donor_list], extra={'paired_same_donor_endpoints': True, 'source': args.source, 'target': args.target, 'steps': args.steps, 'batch_size': args.batch_size})
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
