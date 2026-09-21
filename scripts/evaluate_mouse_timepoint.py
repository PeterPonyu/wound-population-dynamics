#!/usr/bin/env python3
"""Evaluate mouse timepoint."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wound_models.celltype_markers import MARKER_PANEL_MOUSE, score_cell_types
from wound_models.data_adapter import library_normalize, load_10x_mtx_dir, mito_fraction
from wound_models.population_flow import LatentFlowField, compute_cfm_loss
from wound_models.topic_model import TopicModel
POD_ORDER = ['POD0', 'POD2', 'POD7', 'POD30']
POD_TIME = {'POD0': 0.0, 'POD2': 1 / 3, 'POD7': 2 / 3, 'POD30': 1.0}
_COND_RE = re.compile('(GSM\\d+)_([A-Z]+)_(POD\\d+)_')

def log(msg: str='') -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else '', flush=True)

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _rows(m) -> int:
    r, _ = m.shape
    return int(r)

def energy_distance(a: np.ndarray, b: np.ndarray, n_max: int=1500, seed: int=0) -> float:
    """
    Two-sample energy distance: 2*E|a-b| - E|a-a'| - E|b-b'|.
    Zero iff the distributions match; no distributional assumptions.
    """
    rng = np.random.default_rng(seed)
    if a.shape[0] > n_max:
        a = a[rng.choice(a.shape[0], n_max, replace=False)]
    if b.shape[0] > n_max:
        b = b[rng.choice(b.shape[0], n_max, replace=False)]

    def md(x, y):
        d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        return float(d.mean())
    return float(2 * md(a, b) - md(a, a) - md(b, b))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-gsm-dir', default='data/raw/GSE326622/per_gsm')
    ap.add_argument('--discovery-dir', default='outputs/expression_representation')
    ap.add_argument('--output-dir', default='outputs/analysis/mouse_temporal_flow')
    ap.add_argument('--model-arm', default='NDB', help='NDB (non-diabetic), PDB (diet), GDB (genetic)')
    ap.add_argument('--lineage', default='fibroblast')
    ap.add_argument('--holdout', default='POD7')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--min-genes-per-cell', type=int, default=200)
    ap.add_argument('--max-mito', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.model_arm not in {'NDB', 'PDB', 'GDB'}:
        raise ValueError(f'model-arm must be one of NDB/PDB/GDB, got {args.model_arm!r}')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    report = {'experiment': 'repaired_mouse_temporal_flow', 'status': 'running', 'model_arm': args.model_arm, 'holdout': args.holdout, 'pod_time_axis': POD_TIME}
    cond = {}
    for base in sorted(os.listdir(args.per_gsm_dir)):
        m = _COND_RE.match(base)
        if m:
            cond[m.group(1)] = {'arm': m.group(2), 'pod': m.group(3)}
    log(f'conditions parsed from filenames: {len(cond)} GSMs')
    if not cond:
        log('FATAL: no GSM_ARM_POD filenames found')
        return 2
    panel = np.load(os.path.join(args.discovery_dir, 'panel.npy'), allow_pickle=True)
    beta = np.load(os.path.join(args.discovery_dir, 'beta.npy'))
    state = torch.load(os.path.join(args.discovery_dir, 'topic_model.pt'), map_location='cpu')
    n_topics, n_panel = beta.shape
    log(f'loading {args.per_gsm_dir}')
    t0 = time.time()
    coh = load_10x_mtx_dir(args.per_gsm_dir)
    log(f'loaded {coh!r} in {time.time() - t0:.1f}s')
    obs = coh.obs.copy()
    obs['arm'] = obs['gsm'].map(lambda g: cond.get(g, {}).get('arm'))
    obs['pod'] = obs['gsm'].map(lambda g: cond.get(g, {}).get('pod'))
    log('cells per condition: ' + json.dumps(obs.groupby(['arm', 'pod'], observed=True).size().unstack(fill_value=0).to_dict()))
    mito = mito_fraction(coh.X, coh.genes)
    gpc = np.asarray((coh.X > 0).sum(axis=1)).ravel()
    keep = (obs['arm'] == args.model_arm).to_numpy() & (gpc >= args.min_genes_per_cell) & (mito <= args.max_mito)
    idx = np.flatnonzero(keep)
    X = coh.X[idx]
    obs = obs.iloc[idx].reset_index(drop=True)
    log(f'{args.model_arm} cells after QC: {idx.size} (median mouse mito {np.median(mito):.1%})')
    labels, _, found = score_cell_types(X, coh.genes, panel=MARKER_PANEL_MOUSE)
    obs['celltype'] = labels
    log(f'mouse markers found: { {k: v for k, v in found.items() if v}}')
    log(f'composition: {pd.Series(labels).value_counts().to_dict()}')
    mouse_panel = np.array([str(g).capitalize() for g in panel], dtype=object)
    pos = pd.Index(coh.genes).get_indexer(mouse_panel)
    covered = int((pos >= 0).sum())
    log(f'Q1 orthologue coverage: {covered}/{n_panel} ({covered / n_panel:.1%}) of the human panel has a capitalised mouse symbol present')
    report['Q1_cross_species'] = {'panel_size': int(n_panel), 'covered': covered, 'fraction': covered / n_panel}
    if covered / n_panel < 0.5:
        log('FATAL: under half the panel maps; cross-species transfer is not defensible at this coverage')
        return 2
    fib_mask = (obs['celltype'] == args.lineage).to_numpy()
    Xf = X[np.flatnonzero(fib_mask)]
    obs_f = obs.iloc[np.flatnonzero(fib_mask)].reset_index(drop=True)
    present = np.flatnonzero(pos >= 0)
    sub = Xf[:, pos[present]].tocoo()
    Xp = sp.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(_rows(Xf), n_panel), dtype=np.int32)
    Xn = library_normalize(Xp)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    topic = TopicModel(input_dim=n_panel, num_topics=n_topics).to(dev)
    topic.load_state_dict(state)
    topic.eval()
    mus, thetas = ([], [])
    with torch.no_grad():
        for s in range(0, _rows(Xn), 4096):
            xb = torch.from_numpy(np.asarray(Xn[s:s + 4096].todense(), dtype=np.float32)).to(dev)
            o = topic(xb)
            mus.append(o['mu'].cpu().numpy())
            thetas.append(o['theta'].cpu().numpy())
    mu = np.vstack(mus)
    theta = np.vstack(thetas)
    occupancy = float((theta.max(axis=1) > 2.0 / n_topics).mean())
    log(f'Q1 latent sanity: {len(mu)} {args.lineage}s; {occupancy:.1%} of cells have a dominant topic (theta_max > 2/K); mean theta_max = {theta.max(axis=1).mean():.3f}')
    report['Q1_cross_species']['n_fibroblasts'] = int(len(mu))
    report['Q1_cross_species']['dominant_topic_fraction'] = occupancy
    pods = obs_f['pod'].to_numpy()
    by_pod = {p: mu[pods == p] for p in POD_ORDER if (pods == p).sum() > 0}
    log('fibroblasts per POD: ' + json.dumps({p: int(v.shape[0]) for p, v in by_pod.items()}))
    train_pods = [p for p in POD_ORDER if p in by_pod and p != args.holdout]
    if args.holdout not in by_pod or len(train_pods) < 2:
        log('FATAL: not enough timepoints to train and hold one out')
        return 2
    train_cell_mask = np.isin(pods, np.asarray(train_pods, dtype=object))
    scaler_mu = mu[train_cell_mask].mean(axis=0, keepdims=True)
    scaler_sd = mu[train_cell_mask].std(axis=0, keepdims=True)
    scaler_sd[scaler_sd == 0] = 1.0
    Z = {p: torch.from_numpy(((v - scaler_mu) / scaler_sd).astype(np.float32)).to(dev) for p, v in by_pod.items()}
    field = LatentFlowField(latent_dim=n_topics).to(dev)
    opt = torch.optim.Adam(field.parameters(), lr=0.001)
    segments = [(train_pods[i], train_pods[i + 1]) for i in range(len(train_pods) - 1)]
    log(f'training segments (holdout {args.holdout} excluded): ' + ', '.join((f'{a}->{b}' for a, b in segments)))
    t_start = time.time()
    for step in range(1, args.steps + 1):
        a, b = segments[step % len(segments)]
        za = Z[a][torch.from_numpy(rng.integers(0, Z[a].shape[0], args.batch_size)).to(dev)]
        zb = Z[b][torch.from_numpy(rng.integers(0, Z[b].shape[0], args.batch_size)).to(dev)]
        opt.zero_grad()
        loss = compute_cfm_loss(field, zb, za, t_0=POD_TIME[a], t_1=POD_TIME[b])
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0:
            log(f'  step {step}: cfm loss={loss.item():.4f}')
    log(f'flow field trained in {time.time() - t_start:.1f}s')
    prior = max((p for p in train_pods if POD_TIME[p] < POD_TIME[args.holdout]), key=lambda p: POD_TIME[p])
    t_a, t_b = (POD_TIME[prior], POD_TIME[args.holdout])
    log(f'integrating {prior} (t={t_a:.3f}) -> {args.holdout} (t={t_b:.3f})')
    z0 = Z[prior]
    n_steps = 50
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
    pred = z.cpu().numpy()
    actual = Z[args.holdout].cpu().numpy()
    start = z0.cpu().numpy()
    ed_pred = energy_distance(pred, actual)
    ed_base = energy_distance(start, actual)
    split = rng.permutation(len(actual))
    half = len(actual) // 2
    ed_self = energy_distance(actual[split[:half]], actual[split[half:]])
    improvement = (ed_base - ed_pred) / ed_base if ed_base > 0 else float('nan')
    beats_baseline = bool(ed_pred < ed_base)
    gain = float(ed_base - ed_pred)
    passes_noise_gate = bool(gain > ed_self)
    zp = {p: Z[p].cpu().numpy() for p in by_pod}
    d_2_7 = energy_distance(zp[prior], zp[args.holdout])
    d_7_30 = energy_distance(zp[args.holdout], zp[train_pods[-1]])
    d_2_30 = energy_distance(zp[prior], zp[train_pods[-1]])
    on_path = bool(d_2_30 >= max(d_2_7, d_7_30))
    log()
    log(f'Q3 geometry check: d({prior},{args.holdout})={d_2_7:.3f}  d({args.holdout},{train_pods[-1]})={d_7_30:.3f}  d({prior},{train_pods[-1]})={d_2_30:.3f}')
    log(f"   {args.holdout} lies {('BETWEEN the training endpoints' if on_path else 'OFF the direct path')} -> the monotone holdout test is {('fair' if on_path else 'UNFAIR by construction')}")
    report['Q3_geometry'] = {'d_prior_holdout': d_2_7, 'd_holdout_last': d_7_30, 'd_prior_last': d_2_30, 'holdout_on_direct_path': on_path}
    ent = -(theta * np.log(theta + 1e-12)).sum(axis=1)
    perp = float(np.exp(ent).mean())
    log(f'Q1 latent perplexity on mouse: {perp:.2f} of K={n_topics} (human discovery cohort was 3.69)')
    report['Q1_cross_species']['mean_perplexity'] = perp
    log()
    log('=' * 70)
    log(f'Q1 CROSS-SPECIES  {covered}/{n_panel} panel genes map ({covered / n_panel:.1%}); {occupancy:.1%} of mouse fibroblasts get a dominant topic')
    log(f'Q2 HELD-OUT {args.holdout}')
    log(f'   energy distance  predicted vs actual : {ed_pred:.4f}')
    log(f'                    stand-still baseline: {ed_base:.4f}')
    log(f'                    actual split-half   : {ed_self:.4f}  (noise floor)')
    log(f'   improvement over standing still: {improvement:+.1%}')
    log(f"   -> flow field {('BEATS' if beats_baseline else 'DOES NOT BEAT')} the stand-still baseline")
    log('=' * 70)
    report['Q2_holdout'] = {'prior_timepoint': prior, 'holdout': args.holdout, 'n_cells': {p: int(v.shape[0]) for p, v in by_pod.items()}, 'energy_distance_predicted': ed_pred, 'energy_distance_standstill': ed_base, 'energy_distance_noise_floor': ed_self, 'improvement_fraction': improvement, 'beats_baseline': beats_baseline, 'gain_over_standstill': gain, 'passes_noise_gate': passes_noise_gate, 'caveat': 'n=1 animal per model per timepoint; this is a computational trajectory experiment, not a statistical result about wound healing. Cells within a timepoint share one animal.'}
    report['protocol'] = {'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size, 'holdout_timepoint_excluded_from_fit': args.holdout, 'scaler_fit_on_training_timepoints_only': True, 'noise_floor_split': 'randomized_split_half', 'time_axis': 'rank', 'script_sha256': sha256(__file__), 'panel_sha256': sha256(os.path.join(args.discovery_dir, 'panel.npy')), 'model_sha256': sha256(os.path.join(args.discovery_dir, 'topic_model.pt')), 'scope': f'one {args.model_arm} mouse arm; one animal per timepoint; computational audit only'}
    report['status'] = 'completed'
    np.save(os.path.join(args.output_dir, 'mu.npy'), mu)
    np.save(os.path.join(args.output_dir, 'scaler_mu.npy'), scaler_mu)
    np.save(os.path.join(args.output_dir, 'scaler_sd.npy'), scaler_sd)
    obs_f[['gsm', 'arm', 'pod', 'celltype']].to_csv(os.path.join(args.output_dir, 'obs_fibroblast.csv'), index=False)
    out = os.path.join(args.output_dir, 'report.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
