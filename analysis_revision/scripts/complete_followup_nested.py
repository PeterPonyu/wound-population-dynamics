#!/usr/bin/env python3
"""Complete selected outer refits in a fresh tree, preserving the original search.

No outer-target observations enter selection or fitting. The input manifest is
sealed before the first fit; stored predictions and weights are both retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/followup_20261010/paper2"
sys.path.insert(0, str(SOURCE / "scripts"))
from p2common import (DONORS, FLOW_DEFAULTS, H5AD, H5AD_SHA256, OUT,
                      cached_coords, create, group_rows, harness_tasks,
                      load_adata, method_inputs, sha256, task_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    selection_path = OUT / "e3b_selection.json"
    selection = json.loads(selection_path.read_text())
    if set(selection) != set(DONORS) or not all(s["configs"] == 28 for s in selection.values()):
        raise ValueError("Three complete two-fold, 28-configuration selections are required")
    scores = json.loads((SOURCE / "results/e0_reproduction.json").read_text())
    if scores["harness_reproduction"]["missing"] or scores["harness_reproduction"]["rows"] != 2700:
        raise ValueError("Incomplete original harness reproduction")
    if not torch.cuda.is_available():
        raise RuntimeError("Original CUDA backend is required; no CPU substitution")
    torch.set_num_threads(2)
    adata = load_adata()
    tasks = harness_tasks()
    inputs = [H5AD, selection_path, OUT / "e3b_inner_scores.csv",
              ROOT / "output/local-p2-20261009/protocol.json",
              ROOT / "output/local-p2-20261009/jobs.jsonl", Path(__file__).resolve()]
    inputs += sorted((SOURCE / "scripts").glob("*.py"))
    inputs += sorted((ROOT / "expansion/population").rglob("*.py"))
    inputs += [ROOT / "expansion/common/rng.py", ROOT / "expansion/common/io.py"]
    inputs += [OUT / f"coords/mu_std/LIO_{d}.npy" for d in DONORS]
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    if manifest[str(H5AD.relative_to(ROOT))] != H5AD_SHA256:
        raise ValueError("H5AD differs from the frozen harness input")
    protocol = dict(selection=selection, inputs=manifest, seeds=[0, 1, 2],
                    selection_scope="inner leave-one-training-donor-out, seed 0",
                    outer_target_used_for_selection=False,
                    device="cuda", torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), biological_donors=3)
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for held in DONORS:
        task = tasks[f"LIO_{held}"]
        coords = cached_coords("mu_std", f"LIO_{held}")
        data = task_data(adata, task, coords)
        if any(k[0] == held for k in data.train):
            raise ValueError("Outer held donor found in fitting data")
        chosen = selection[held]
        params = {**FLOW_DEFAULTS, **{k: chosen[k] for k in ("hidden", "steps", "lr")}}
        for seed in (0, 1, 2):
            method = create(chosen["method"], **params)
            torch.cuda.synchronize()
            start = time.perf_counter()
            info = method.fit(method_inputs(data), seed=seed, device="cuda")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            info.pop("sb_batch_diagnostics", None)
            arrays = {}
            for i, pair in enumerate(task.pairs):
                source = coords[group_rows(adata.obs, pair.individual, pair.source_time)]
                arrays[f"pred{i}"] = method.predict(
                    source, individual=pair.individual, source_time=pair.source_time,
                    target_time=pair.target_time, anchor_time=pair.anchor_time)
                if not np.isfinite(arrays[f"pred{i}"]).all():
                    raise ValueError("Nonfinite outer prediction")
            folder = dest / held
            folder.mkdir(exist_ok=True)
            meta = dict(rep="mu_std", task=f"LIO_{held}", tag="tuned_nested",
                        method=chosen["method"], params=params, seed=seed,
                        fit_seconds=elapsed, fit_info=info,
                        protocol_sha256=sha256(dest / "protocol.json"))
            path = folder / f"tuned_nested_s{seed}.npz"
            np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
            weights = path.with_name(path.stem + "_field.pt")
            torch.save(method.field.state_dict(), weights)
            records.append(dict(held=held, seed=seed, method=chosen["method"],
                                fit_seconds=elapsed, predictions_sha256=sha256(path),
                                weights_sha256=sha256(weights)))
            print(f"{held} seed={seed} {chosen['method']} {elapsed:.1f}s saved", flush=True)
    if any(sha256(ROOT / p) != digest for p, digest in manifest.items()):
        raise ValueError("Input or source changed during refitting")
    (dest / "completion.json").write_text(json.dumps(dict(completed=9, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
