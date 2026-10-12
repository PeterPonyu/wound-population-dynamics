"""E3a: error versus interpolation fraction alpha in [0, 1] (step 0.05), per donor, both global missing-time tasks.

Predictors at alpha: centroid translation, location-scale, independent bridge, exact McCann interpolant
(own source -> own anchor, frozen mu standardized coordinates) and every fixed rank-time field (harness
3x256 independent / OT / SB couplings, 40k-step independent, and the manuscript's saved 2x128 fields)
integrated from the source to s(alpha) = s_source + alpha (s_anchor - s_source).
Evaluation: harness paired-v2 matched draws (identical source/target indices for all alphas and methods).
This is a retrospective oracle scan: the target is used only to *display* the curve, never to fit.
"""
import json

import numpy as np
import pandas as pd
import torch

from p2common import (ALPHAS, OUT, ROOT, anchor_of, cached_coords, draw_plan, energy_distance, exact_ot_plan,
                      group_rows, harness_tasks, independent_bridge, load_adata, mccann_sample, task_data)
from core_engines.flow_matching_engine import LatentFlowField  # noqa: E402

adata = load_adata()
CLOCK_ALPHA = {"MT_day7": {"rank": 0.5, "log_days": float(np.log(4) / np.log(15.5)),
                           "sqrt_days": float((np.sqrt(7 / 30) - np.sqrt(1 / 30)) / (1 - np.sqrt(1 / 30))),
                           "days": 6 / 29},
               "MT_day1": {"rank": 0.5, "log_days": float(np.log(2) / np.log(8)),
                           "sqrt_days": float(np.sqrt(1 / 7)), "days": 1 / 7}}


def published_alpha_grid(task_name, source):
    hold = "Wound7" if task_name == "MT_day7" else "Wound1"
    s0, s1 = (1 / 3, 1.0) if hold == "Wound7" else (0.0, 2 / 3)
    out = {}
    for seed in (0, 1, 2):
        field = LatentFlowField(latent_dim=15)
        field.load_state_dict(torch.load(ROOT / f"outputs/biological_expansion/population/{hold}_seed{seed}_field.pt",
                                         map_location="cpu"))
        field.eval()
        grid = []
        for a in ALPHAS:
            n = int(round(100 * a))
            z = torch.tensor(source, dtype=torch.float32)
            if n:
                dt = a * (s1 - s0) / n
                with torch.no_grad():
                    for k in range(n):
                        t = s0 + k * dt
                        k1 = field(z, t); k2 = field(z + .5 * dt * k1, t + .5 * dt)
                        k3 = field(z + .5 * dt * k2, t + .5 * dt); k4 = field(z + dt * k3, t + dt)
                        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            grid.append(z.numpy())
        out[seed] = np.stack(grid)
    return out


rows = []
for task_name in ["MT_day7", "MT_day1"]:
    task = harness_tasks()[task_name]
    coords = cached_coords("mu_std", task_name)
    data = task_data(adata, task, coords)
    fits = {}
    for path in sorted((OUT / "fits/mu_std" / task_name).glob("*_s[0-9].npz")):
        f = np.load(path); meta = json.loads(str(f["meta"]))
        fits[(meta["tag"], meta["seed"])] = f
    for pi, pair in enumerate(task.pairs):
        src_rows = group_rows(adata.obs, pair.individual, pair.source_time)
        tgt_rows = group_rows(adata.obs, pair.individual, pair.target_time)
        source, target = coords[src_rows].astype(float), coords[tgt_rows].astype(float)
        anchor = anchor_of(data, pair)
        plan = exact_ot_plan(source, anchor)
        grids = {(tag, seed): f[f"alpha{pi}"] for (tag, seed), f in fits.items()}
        for seed, g in published_alpha_grid(task_name, source).items():
            grids[("cfm_independent_published2x128", seed)] = g
        sa, sb = source.std(0), anchor.std(0)
        draws = [draw_plan(task, pair, len(src_rows), len(tgt_rows), d) for d in range(50)]
        for ai, a in enumerate(ALPHAS):
            preds = {("mean_shift", 0): source + a * (anchor.mean(0) - source.mean(0)),
                     ("location_scale", 0): (1 - a) * source.mean(0) + a * anchor.mean(0)
                     + (source - source.mean(0)) * np.where(sa > 1e-8, ((1 - a) * sa + a * sb) / np.where(sa > 1e-8, sa, 1), 0)}
            for key, g in grids.items():
                preds[key] = g[ai]
            for draw, (dseed, sel, tsel) in enumerate(draws):
                t = target[tsel]
                for (m, seed), p in preds.items():
                    rows.append(dict(task=task_name, individual=pair.individual, alpha=a, method=m, seed=seed, draw=draw,
                                     energy=energy_distance(p[sel], t)))
                for m, p in (("independent_bridge", independent_bridge(source, anchor, a, np.random.default_rng([dseed, 41]))),
                             ("ot_mccann_exact", mccann_sample(source, anchor, plan, a, np.random.default_rng([dseed, 43])))):
                    rows.append(dict(task=task_name, individual=pair.individual, alpha=a, method=m, seed=0, draw=draw,
                                     energy=energy_distance(p[sel], t)))
        print(task_name, pair.individual, flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "e3_alpha_draws.csv.gz", index=False)
curve = (frame.groupby(["task", "individual", "alpha", "method", "seed"]).energy.mean()
         .groupby(["task", "individual", "alpha", "method"]).mean().reset_index())
curve.to_csv(OUT / "e3_alpha_curve_by_donor.csv", index=False)
macro = curve.groupby(["task", "alpha", "method"]).energy.mean().reset_index()
macro.to_csv(OUT / "e3_alpha_curve_macro.csv", index=False)
best = []
for (task, ind, m), g in pd.concat([curve, macro.assign(individual="macro")]).groupby(["task", "individual", "method"]):
    i = g.energy.idxmin()
    best.append(dict(task=task, individual=ind, method=m, best_alpha=float(g.loc[i, "alpha"]),
                     best_energy=float(g.loc[i, "energy"]),
                     energy_at_rank_alpha=float(g[np.isclose(g.alpha, 0.5)].energy.iloc[0]),
                     **{f"energy_at_{c}": float(np.interp(v, g.alpha, g.energy)) for c, v in CLOCK_ALPHA[task].items()}))
pd.DataFrame(best).to_csv(OUT / "e3_alpha_best.csv", index=False)
json.dump(CLOCK_ALPHA, open(OUT / "e3_clock_implied_alpha.json", "w"), indent=1)
print(pd.DataFrame(best).to_string())
