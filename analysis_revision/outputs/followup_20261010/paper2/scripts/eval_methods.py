"""Evaluate every predictor of one representation under the harness paired-v2 draws.

Usage: python eval_methods.py REP
Writes results/eval_{REP}_draws.csv.gz   (task, pair, draw, method, seed, energy)
       results/eval_{REP}_fullcell.csv     (full-cell energy, centroid error, variance ratio per donor/method/seed)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from p2common import (OUT, PROTOCOL, ROOT, alpha_of, anchor_of, cached_coords, create, draw_plan, energy_distance,
                      exact_ot_plan, group_rows, harness_tasks, independent_bridge, load_adata, mccann_barycentric,
                      mccann_sample, method_inputs, task_data, total_variance)
from expansion.population.worker import _method_params  # noqa: E402

REP = sys.argv[1] if len(sys.argv) > 1 else "mu_std"
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
adata = load_adata()
obs = adata.obs
FULLCELL_CAP = None  # all cells (Table S25 style)


def ot_exact_knn_transfer(data, source, pair, k=15):
    """Held-donor exact-OT analogue: per training donor exact balanced OT between its observed
    source- and target-time cells; barycentric displacement; extended to held cells by the mean
    displacement of k nearest training-source cells; training donors weighted equally."""
    shifts = []
    for (ind, t), cells in sorted(data.train.items()):
        if not np.isclose(t, pair.source_time) or (ind, float(pair.target_time)) not in data.train:
            continue
        xs = np.asarray(cells, float); yt = np.asarray(data.train[(ind, float(pair.target_time))], float)
        plan = exact_ot_plan(xs, yt)
        disp = plan @ yt / plan.sum(1, keepdims=True) - xs
        _, idx = NearestNeighbors(n_neighbors=min(k, len(xs))).fit(xs).kneighbors(source)
        shifts.append(disp[idx].mean(1))
    return source + np.mean(shifts, axis=0)


def flow_fits(task_name):
    folder = OUT / "fits" / REP / task_name
    out = []
    for path in sorted(folder.glob("*_s[0-9].npz")):
        f = np.load(path)
        meta = json.loads(str(f["meta"]))
        out.append((meta["tag"], meta["seed"], f))
    return out


def published_predictions(task_name, pair, src_rows):
    """Manuscript (Table S25) 2x128 fields, saved predictions; only for the frozen-mu standardized coordinates."""
    if REP != "mu_std" or not task_name.startswith("MT_"):
        return {}
    hold = "Wound7" if task_name == "MT_day7" else "Wound1"
    folder = ROOT / "outputs/repaired_20260918/human_temporal_flow"
    o = pd.read_csv(folder / "obs_fibroblast.csv")
    src = {"Wound7": "Wound1", "Wound1": "Skin"}[hold]
    is_src = (o.cond == src).to_numpy()
    out = {}
    for seed in (0, 1, 2):
        pred = np.load(ROOT / f"outputs/biological_expansion/population/{hold}_seed{seed}_prediction.npy")
        sel = (o.donor.to_numpy()[is_src] == pair.individual)
        assert np.array_equal(np.flatnonzero(is_src)[sel], src_rows)
        out[seed] = pred[sel]
    return out


draw_rows, full_rows = [], []
for task_name, task in harness_tasks().items():
    if only and task_name not in only:
        continue
    coords = cached_coords(REP, task_name)
    data = task_data(adata, task, coords)
    inputs = method_inputs(data)
    names = ["identity", "mean_shift", "location_scale"]
    names += ["ot_displacement"] if task.kind == "T-MT" else ["target_marginal", "population_ot"]
    harness = {}
    for m in names:
        meth = create(m, **_method_params(PROTOCOL, m)); meth.fit(inputs, seed=0, device="cpu"); harness[m] = meth
    fits = flow_fits(task_name)
    for pi, pair in enumerate(task.pairs):
        src_rows = group_rows(obs, pair.individual, pair.source_time)
        tgt_rows = group_rows(obs, pair.individual, pair.target_time)
        source, target_full = coords[src_rows].astype(np.float64), coords[tgt_rows].astype(np.float64)
        call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time,
                    anchor_time=pair.anchor_time)
        det = {}  # (method, seed) -> full-source prediction (deterministic)
        for m in ["identity", "mean_shift", "location_scale"]:
            det[(m, 0)] = harness[m].predict(source, **call)
        for tag, seed, f in fits:
            det[(tag, seed)] = f[f"pred{pi}"]
        for seed, pred in published_predictions(task_name, pair, src_rows).items():
            det[("cfm_independent_published2x128", seed)] = pred
        stochastic = {}  # method -> callable(draw_seed) -> full-source prediction
        if task.kind == "T-MT":
            alpha, anchor = alpha_of(data.clock, pair), anchor_of(data, pair)
            plan = exact_ot_plan(source, anchor)
            det[("ot_mccann_barycentric_exact", 0)] = mccann_barycentric(source, anchor, plan, alpha)
            stochastic["independent_bridge"] = lambda s, a=anchor, al=alpha: independent_bridge(
                source, a, al, np.random.default_rng([s, 41]))
            stochastic["ot_mccann_exact"] = lambda s, a=anchor, al=alpha, pl=plan: mccann_sample(
                source, a, pl, al, np.random.default_rng([s, 43]))
            stochastic["ot_displacement_harness"] = lambda s: harness["ot_displacement"].predict(source, **call, seed=s)
        else:
            det[("ot_exact_knn_transfer", 0)] = ot_exact_knn_transfer(data, source, pair)
            stochastic["population_ot_harness"] = lambda s: harness["population_ot"].predict(source, **call, seed=s)
        # full-cell geometry (Table S25 style), one realization (seed 0 stream) for stochastic predictors
        tv = total_variance(target_full)
        full_preds = dict(det)
        for m, fn in stochastic.items():
            full_preds[(m, 0)] = fn(0)
        if task.kind == "T-LIO":
            full_preds[("target_marginal", 0)] = harness["target_marginal"].predict(source, **call, seed=0)
        for (m, seed), pred in full_preds.items():
            pred = np.asarray(pred, np.float64)
            full_rows.append(dict(task=task_name, individual=pair.individual, source_time=pair.source_time,
                                  target_time=pair.target_time, method=m, seed=seed,
                                  energy_full=energy_distance(pred, target_full),
                                  centroid_error=float(np.linalg.norm(pred.mean(0) - target_full.mean(0))),
                                  variance_ratio=total_variance(pred) / tv,
                                  centered_energy=energy_distance(pred - pred.mean(0), target_full - target_full.mean(0))))
        for draw in range(50):
            dseed, sel, tsel = draw_plan(task, pair, len(src_rows), len(tgt_rows), draw)
            target = target_full[tsel]
            base = dict(task=task_name, individual=pair.individual, source_time=pair.source_time,
                        target_time=pair.target_time, draw=draw)
            for (m, seed), pred in det.items():
                draw_rows.append({**base, "method": m, "seed": seed, "energy": energy_distance(pred[sel], target)})
            for m, fn in stochastic.items():
                draw_rows.append({**base, "method": m, "seed": 0, "energy": energy_distance(fn(dseed)[sel], target)})
            if task.kind == "T-LIO":
                pred = harness["target_marginal"].predict(source[sel], **call, seed=dseed)
                draw_rows.append({**base, "method": "target_marginal", "seed": 0, "energy": energy_distance(pred, target)})
        print(REP, task_name, pair.individual, pair.source_time, "done", flush=True)

suffix = "" if only is None else "_" + "_".join(sorted(only))
pd.DataFrame(draw_rows).to_csv(OUT / f"eval_{REP}{suffix}_draws.csv.gz", index=False)
pd.DataFrame(full_rows).to_csv(OUT / f"eval_{REP}{suffix}_fullcell.csv", index=False)
