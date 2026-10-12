"""Fit every representation on each task's permitted cells only and cache the coordinates."""
import sys
from p2common import COORD_DIR, all_tasks, harness_tasks, load_adata, task_coords, write_json

reps = sys.argv[1:] or ["mu_std", "mu_raw", "mu_white", "mu_clr_std", "pca30"]
adata = load_adata()
info = {}
for rep in reps:
    (COORD_DIR / rep).mkdir(parents=True, exist_ok=True)
    tasks = all_tasks() if rep == "mu_std" else harness_tasks()
    for name, task in tasks.items():
        import numpy as np
        coords, meta = task_coords(adata, task, rep)
        np.save(COORD_DIR / rep / f"{name}.npy", coords)
        info[f"{rep}/{name}"] = meta
        print(rep, name, meta, flush=True)
write_json(COORD_DIR / f"coords_info_{'_'.join(reps)}.json", info)
