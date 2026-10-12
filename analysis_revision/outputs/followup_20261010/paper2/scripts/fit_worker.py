"""Fit one harness flow (one representation, task, coupling/config, seed); save full-source predictions.

Predictions are pushed from the *full* observed source of every task pair (as in the harness);
for global missing-time tasks the field is additionally integrated to s(alpha) on the
source->anchor rank-time segment for the alpha grid (E3).
"""
import argparse
import json
import time

import numpy as np
import torch

from p2common import (ALPHAS, FLOW_DEFAULTS, OUT, all_tasks, cached_coords, create, group_rows, load_adata,
                      method_inputs, rk4_to, task_data)

ap = argparse.ArgumentParser()
ap.add_argument("--rep", default="mu_std"); ap.add_argument("--task", required=True)
ap.add_argument("--method", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--no-alpha", action="store_true"); ap.add_argument("--force", action="store_true")
args = ap.parse_args()
folder = OUT / "fits" / args.rep / args.task
folder.mkdir(parents=True, exist_ok=True)
path = folder / f"{args.tag}_s{args.seed}.npz"
if path.exists() and not args.force:
    raise SystemExit(0)
torch.set_num_threads(2)
overrides = {}
for item in args.set:
    k, v = item.split("=")
    overrides[k] = float(v) if k == "lr" else int(v)
params = {**FLOW_DEFAULTS, **overrides}
adata = load_adata()
task = all_tasks()[args.task]
coords = cached_coords(args.rep, args.task)
data = task_data(adata, task, coords)
method = create(args.method, **params)
torch.cuda.synchronize(); t0 = time.perf_counter()
info = method.fit(method_inputs(data), seed=args.seed, device="cuda")
torch.cuda.synchronize(); fit_seconds = time.perf_counter() - t0
info.pop("sb_batch_diagnostics", None)
arrays = {}
clock = data.clock
for pi, pair in enumerate(task.pairs):
    rows = group_rows(adata.obs, pair.individual, pair.source_time)
    src = coords[rows]
    arrays[f"pred{pi}"] = method.predict(src, individual=pair.individual, source_time=pair.source_time,
                                         target_time=pair.target_time, anchor_time=pair.anchor_time)
    if task.kind == "T-MT" and not args.no_alpha:
        s0, s1 = clock[pair.source_time], clock[pair.anchor_time]
        arrays[f"alpha{pi}"] = np.stack([rk4_to(method, src, s0, s0 + a * (s1 - s0)) for a in ALPHAS])
meta = dict(rep=args.rep, task=args.task, method=args.method, tag=args.tag, seed=args.seed, params=params,
            fit_seconds=fit_seconds, fit_info=info)
np.savez_compressed(path, meta=np.array(json.dumps(meta, default=float)), **arrays)
torch.save(method.field.state_dict(), folder / f"{args.tag}_s{args.seed}_field.pt")
print(args.rep, args.task, args.tag, args.seed, f"{fit_seconds:.1f}s", "loss_last", info.get("loss_last"), flush=True)
