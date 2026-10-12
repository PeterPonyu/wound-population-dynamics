"""E3b: nested, training-only hyperparameter selection for the held-donor flow, then refit.

Inner folds (H = outer held donor, never used): train on one remaining donor, evaluate the other donor's
day-1 -> day-7 prediction with matched paired-v2 draws (200 x 50). Grid (seed 0):
coupling {independent, OT, SB} x width {64, 128, 256} x steps {2k, 8k, 20k (independent only)} x lr {1e-3, 3e-4};
depth 3, batch 512, grad-clip 1 (harness). Selection criterion: mean inner energy over the two inner folds.
The selected configuration is refitted on both training donors with seeds 0-2 (tag 'tuned_nested').
Usage: python e3b_nested.py select   |   python e3b_nested.py refit
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from p2common import DONORS, OUT, cached_coords, draw_plan, energy_distance, group_rows, inner_tasks, load_adata

stage = sys.argv[1]
if stage == "select":
    adata = load_adata()
    rows = []
    for name, task in inner_tasks().items():
        coords = cached_coords("mu_std", name)
        pi = [i for i, p in enumerate(task.pairs) if p.source_time == 1.0][0]
        pair = task.pairs[pi]
        src = group_rows(adata.obs, pair.individual, 1.0); tgt = group_rows(adata.obs, pair.individual, 7.0)
        plans = [draw_plan(task, pair, len(src), len(tgt), d) for d in range(50)]
        for path in sorted((OUT / "fits/mu_std" / name).glob("*_s0.npz")):
            f = np.load(path); meta = json.loads(str(f["meta"]))
            if meta["tag"] in ("cfm_independent", "tuned_nested"):
                continue
            pred = f[f"pred{pi}"]
            e = np.mean([energy_distance(pred[s], coords[tgt[t]]) for _, s, t in plans])
            rows.append(dict(inner_task=name, outer_held=name.split("_")[1], inner_eval=pair.individual,
                             tag=meta["tag"], method=meta["method"], hidden=meta["params"]["hidden"],
                             steps=meta["params"]["steps"], lr=meta["params"]["lr"], inner_energy=e,
                             fit_seconds=meta["fit_seconds"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "e3b_inner_scores.csv", index=False)
    agg = frame.groupby(["outer_held", "tag", "method", "hidden", "steps", "lr"]).agg(
        inner_energy=("inner_energy", "mean"), folds=("inner_energy", "size")).reset_index()
    agg = agg[agg.folds == 2]
    selection = {}
    for h, g in agg.groupby("outer_held"):
        best = g.sort_values("inner_energy").iloc[0]
        default = g[(g.method == "cfm_independent") & (g.hidden == 256) & (g.steps == 8000) & (g.lr == 1e-3)]
        selection[h] = dict(tag=best.tag, method=best.method, hidden=int(best.hidden), steps=int(best.steps),
                            lr=float(best.lr), inner_energy=float(best.inner_energy), configs=len(g),
                            default_config_inner_energy=float(default.inner_energy.iloc[0]) if len(default) else None,
                            top5=g.sort_values("inner_energy").head(5)[["tag", "inner_energy"]].to_dict("records"))
    json.dump(selection, open(OUT / "e3b_selection.json", "w"), indent=1)
    print(json.dumps(selection, indent=1))
elif stage == "refit":
    selection = json.load(open(OUT / "e3b_selection.json"))
    procs = []
    for h, s in selection.items():
        for seed in (0, 1, 2):
            cmd = [sys.executable, "fit_worker.py", "--task", f"LIO_{h}", "--method", s["method"], "--seed", str(seed),
                   "--tag", "tuned_nested", "--set", f"hidden={s['hidden']}", f"steps={s['steps']}", f"lr={s['lr']}"]
            procs.append(subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__))))
    for p in procs:
        p.wait()
