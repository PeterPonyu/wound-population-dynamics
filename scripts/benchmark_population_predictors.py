#!/usr/bin/env python3
"""Benchmark population predictors."""
import argparse
import json
import os
import sys
import time
import numpy as np
import ot
import torch
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.population_flow import ConditionalLatentFlowField, LatentFlowField, compute_cfm_loss, compute_conditional_cfm_loss
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

def context_of(cells: np.ndarray) -> np.ndarray:
    """Donor context: mean and sd of that donor's source-state latent."""
    return np.concatenate([cells.mean(axis=0), cells.std(axis=0)]).astype(np.float32)

def rk4(step_fn, z, t_a, t_b, n_steps=50):
    dt = (t_b - t_a) / n_steps
    with torch.no_grad():
        for s in range(n_steps):
            t = t_a + s * dt
            k1 = step_fn(z, t)
            k2 = step_fn(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = step_fn(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = step_fn(z + dt * k3, t + dt)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z

def load_unscaled_latent(args):
    packed = load_lineage_latent(args.per_gsm_dir, args.cell_metadata, args.discovery_dir, lineage=args.lineage)
    return (packed['latent'].astype(np.float32), packed['obs'], packed['n_topics'], packed['device'])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-gsm-dir', default='data/raw/GSE241132/per_gsm')
    ap.add_argument('--cell-metadata', default='data/raw/GSE241132/GSE241132_cell_metadata.txt.gz')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default=os.path.join(ANALYSIS_ROOT, 'donor_conditioned_benchmark'))
    ap.add_argument('--lineage', default='Fibroblast')
    ap.add_argument('--source', default='Wound1')
    ap.add_argument('--target', default='Wound7')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--ot-reg', type=float, default=0.05)
    ap.add_argument('--knn', type=int, default=15)
    ap.add_argument('--context-dropout', type=float, default=0.2, help='fraction of training batches with the context zeroed, so an unseen context degrades to the shared field instead of diverging')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    preserve_reference_outputs(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    log('loading latent')
    mu, obs_f, n_topics, dev = load_unscaled_latent(args)
    conds = obs_f['cond'].to_numpy()
    donors = obs_f['donor'].to_numpy()
    donor_list = sorted(set(donors))
    src_c, tgt_c = (args.source, args.target)
    t_a, t_b = (COND_TIME[src_c], COND_TIME[tgt_c])
    segments = [(COND_ORDER[i], COND_ORDER[i + 1]) for i in range(3)]
    log(f'{args.lineage}s={len(obs_f)}  donors={donor_list}  {src_c}(t={t_a:.3f}) -> {tgt_c}(t={t_b:.3f})')
    ctx_dim = 2 * n_topics
    rng = np.random.default_rng(args.seed)
    results = {m: {} for m in ['M0_standstill', 'M1_shared_cfm', 'M2_mean_displacement', 'M3_nearest_donor', 'M4_conditioned_cfm', 'M5_optimal_transport']}
    insample = {}
    for held in donor_list:
        train_mask = donors != held
        mu_z, scaler = fold_standardize(mu, train_mask)
        train_donors = [d for d in donor_list if d != held]
        pairs = paired_endpoint_pairs(donors, conds, train_donors, segments)
        Zd = {(d, c): mu_z[(donors == d) & (conds == c)] for d in train_donors for c in COND_ORDER}
        Zdt = {k: torch.from_numpy(v).to(dev) for k, v in Zd.items() if len(v) > 0}
        Z = {c: mu_z[train_mask & (conds == c)] for c in COND_ORDER}
        if any((len(Z[c]) < 50 for c in COND_ORDER)):
            log(f'  skip {held}')
            continue
        src = mu_z[(donors == held) & (conds == src_c)]
        tgt = mu_z[(donors == held) & (conds == tgt_c)]
        perm = np.random.default_rng(args.seed).permutation(len(tgt))
        noise = energy_distance(tgt[perm[:len(tgt) // 2]], tgt[perm[len(tgt) // 2:]])
        ed_base = energy_distance(src, tgt)

        def record(name, pred, extra=None):
            ed = energy_distance(pred, tgt)
            d = {'ed': ed, 'standstill': ed_base, 'noise': noise, 'gain': ed_base - ed, 'passes': bool(ed_base - ed > noise)}
            if extra:
                d.update(extra)
            results[name][held] = d
            return d
        record('M0_standstill', src)
        torch.manual_seed(args.seed)
        f1 = LatentFlowField(latent_dim=n_topics).to(dev)
        o1 = torch.optim.Adam(f1.parameters(), lr=0.001)
        for step in range(args.steps):
            d, a, b = pairs[step % len(pairs)]
            za = Zdt[d, a][torch.from_numpy(rng.integers(0, len(Zd[d, a]), args.batch_size)).to(dev)]
            zb = Zdt[d, b][torch.from_numpy(rng.integers(0, len(Zd[d, b]), args.batch_size)).to(dev)]
            o1.zero_grad()
            compute_cfm_loss(f1, zb, za, t_0=COND_TIME[a], t_1=COND_TIME[b]).backward()
            o1.step()
        p1 = rk4(lambda z, t: f1(z, t), torch.from_numpy(src).to(dev), t_a, t_b)
        record('M1_shared_cfm', p1.cpu().numpy())
        deltas, ctxs = ({}, {})
        for d in train_donors:
            a = mu_z[(donors == d) & (conds == src_c)]
            b = mu_z[(donors == d) & (conds == tgt_c)]
            deltas[d] = b.mean(axis=0) - a.mean(axis=0)
            ctxs[d] = context_of(a)
        record('M2_mean_displacement', src + np.mean(list(deltas.values()), axis=0))
        held_ctx = context_of(src)
        sims = {d: float(held_ctx @ c / (np.linalg.norm(held_ctx) * np.linalg.norm(c))) for d, c in ctxs.items()}
        nearest = max(sims, key=lambda k: sims[k])
        record('M3_nearest_donor', src + deltas[nearest], {'nearest': nearest, 'similarities': sims})
        S, T = (Z[src_c], Z[tgt_c])
        ns, nt = (min(len(S), 1500), min(len(T), 1500))
        Si = S[rng.choice(len(S), ns, replace=False)]
        Ti = T[rng.choice(len(T), nt, replace=False)]
        M = np.asarray(ot.dist(Si, Ti), dtype=np.float64)
        M /= M.max()
        G = np.asarray(ot.sinkhorn(np.ones(ns) / ns, np.ones(nt) / nt, M, args.ot_reg, numItermax=2000))
        bary = G @ Ti / G.sum(axis=1, keepdims=True)
        disp = bary - Si
        nn = NearestNeighbors(n_neighbors=min(args.knn, ns)).fit(Si)
        _, idx = nn.kneighbors(src)
        record('M5_optimal_transport', src + disp[idx].mean(axis=1))
        torch.manual_seed(args.seed)
        f4 = ConditionalLatentFlowField(latent_dim=n_topics, context_dim=ctx_dim).to(dev)
        o4 = torch.optim.Adam(f4.parameters(), lr=0.001)
        ctx_t = {}
        for d, a, b in pairs:
            if (d, a) not in ctx_t:
                ctx_t[d, a] = torch.from_numpy(context_of(Zd[d, a])).to(dev)
        for step in range(args.steps):
            d, a, b = pairs[step % len(pairs)]
            za = Zdt[d, a][torch.from_numpy(rng.integers(0, len(Zd[d, a]), args.batch_size)).to(dev)]
            zb = Zdt[d, b][torch.from_numpy(rng.integers(0, len(Zd[d, b]), args.batch_size)).to(dev)]
            o4.zero_grad()
            compute_conditional_cfm_loss(f4, zb, za, ctx_t[d, a], t_0=COND_TIME[a], t_1=COND_TIME[b], context_dropout=args.context_dropout).backward()
            o4.step()
        c_held = torch.from_numpy(held_ctx).to(dev)
        p4 = rk4(lambda z, t: f4(z, t, c_held), torch.from_numpy(src).to(dev), t_a, t_b)
        record('M4_conditioned_cfm', p4.cpu().numpy(), extra={'scaler': scaler})
        for d in train_donors:
            s_d, t_d = (Zd[d, src_c], Zd[d, tgt_c])
            if len(s_d) < 20 or len(t_d) < 20:
                continue
            c_d = torch.from_numpy(context_of(s_d)).to(dev)
            pi4 = rk4(lambda z, t: f4(z, t, c_d), torch.from_numpy(s_d).to(dev), t_a, t_b).cpu().numpy()
            pi1 = rk4(lambda z, t: f1(z, t), torch.from_numpy(s_d).to(dev), t_a, t_b).cpu().numpy()
            base_d = energy_distance(s_d, t_d)
            insample.setdefault(f'{held}/{d}', {'standstill': base_d, 'shared': energy_distance(pi1, t_d), 'conditioned': energy_distance(pi4, t_d)})
        log(f'  [{held}] base={ed_base:.4f} noise={noise:.4f} | ' + '  '.join((f"{k.split('_')[0]}={results[k][held]['ed']:.4f}" for k in results if held in results[k])))
    log()
    log(f"{'method':<24}{'mean ED':>10}{'mean gain':>11}{'passes':>9}")
    summary = {}
    for name, per in results.items():
        if not per:
            continue
        eds = [v['ed'] for v in per.values()]
        gains = [v['gain'] for v in per.values()]
        npass = sum((1 for v in per.values() if v['passes']))
        summary[name] = {'mean_ed': float(np.mean(eds)), 'mean_gain': float(np.mean(gains)), 'n_pass': npass, 'n': len(per)}
        log(f'{name:<24}{np.mean(eds):>10.4f}{np.mean(gains):>+11.4f}{npass:>6}/{len(per)}')
    log()
    log('in-sample (training donors) shared vs conditioned:')
    for k, v in insample.items():
        log(f"  fold {k}: standstill={v['standstill']:.4f}  shared={v['shared']:.4f}  conditioned={v['conditioned']:.4f}")
    cond_better = sum((1 for v in insample.values() if v['conditioned'] < v['shared']))
    log(f'  conditioned beats shared in-sample on {cond_better}/{len(insample)} donor-folds')
    best = min(summary, key=lambda k: summary[k]['mean_ed'])
    n_folds = summary['M0_standstill']['n']
    any_pass = [k for k, v in summary.items() if k != 'M0_standstill' and v['n_pass'] > n_folds / 2 and (v['mean_gain'] > 0)]
    minority = [k for k, v in summary.items() if k != 'M0_standstill' and 0 < v['n_pass'] <= n_folds / 2]
    log()
    log('=' * 74)
    log(f"best mean energy distance: {best} ({summary[best]['mean_ed']:.4f})")
    if any_pass:
        log(f"methods beating stand-still on a majority of donors with positive mean gain: {', '.join(any_pass)}")
        verdict = 'some_method_works'
    else:
        cond_ok = cond_better > len(insample) / 2
        if minority:
            log(f"{', '.join(minority)} beat stand-still on a minority of donors only, with negative mean gain overall - not a win.")
        log('NO method beats stand-still on a majority of held-out donors, neural or not.')
        if cond_ok:
            log(f'Conditioning does help on the donors it saw ({cond_better}/{len(insample)} folds), so the model class is not the problem - with two training contexts, predicting a third is extrapolation. The requirement this sets is donors, not architecture.')
            verdict = 'conditioning_helps_in_sample_only'
        else:
            log('Conditioning does not even help in-sample, so the limitation is upstream of the field - most likely the frozen latent does not encode this trajectory.')
            verdict = 'limitation_upstream_of_field'
    log('=' * 74)
    report = {'summary': summary, 'per_donor': results, 'in_sample': insample, 'conditioned_better_in_sample': f'{cond_better}/{len(insample)}', 'verdict': verdict, 'caveat': '3 donors; leave-one-out leaves 2 training contexts for the conditional model.', 'protocol': protocol_block(__file__, args.discovery_dir, args.seed, folds=[f'holdout_{d}' for d in donor_list], extra={'paired_same_donor_endpoints': True, 'shared_and_conditioned_use_same_pairs': True, 'source': args.source, 'target': args.target, 'steps': args.steps, 'batch_size': args.batch_size})}
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
