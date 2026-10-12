#!/usr/bin/env python3
"""Audit saved follow-up artifacts and export compact, matched-draw summaries.

This audit does not refit predictors or overwrite original experiment outputs.
The retrospective manifest binds current files; it does not assert historical
training provenance that was not recorded when those fields were fitted.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/followup_20261010/paper2"
sys.path.insert(0, str(SOURCE / "scripts"))
from p2common import (DONORS, H5AD, H5AD_SHA256, OUT, REPRESENTATIONS,
                      all_tasks, cached_coords, create, draw_plan, energy_distance,
                      group_rows, load_adata, method_inputs, sha256, task_data,
                      task_coords)

KEYS = ["task", "individual", "source_time", "target_time"]


def paired_summary(frame, comparator):
    """Complete 50-draw pairs; average draws, then seeds, within each donor."""
    identity = KEYS + ["method", "seed", "draw"]
    if frame.duplicated(identity).any() or not np.isfinite(frame.energy).all() or (frame.energy <= 0).any():
        raise ValueError("Duplicate, missing or nonpositive evaluation energy")
    groups = frame.groupby(KEYS + ["method", "seed"]).draw.agg(lambda x: set(x))
    if not all(g == set(range(50)) for g in groups):
        raise ValueError("Every evaluation fit requires exactly the same 50 draw identities")
    baseline = frame[frame.method == comparator][KEYS + ["draw", "energy"]]
    if baseline.duplicated(KEYS + ["draw"]).any():
        raise ValueError("Comparator must be unique within donor and draw")
    paired = frame.merge(baseline.rename(columns={"energy": "baseline"}),
                         on=KEYS + ["draw"], how="left", validate="many_to_one")
    if paired.baseline.isna().any():
        raise ValueError("Missing matched comparator")
    paired["log2_ratio"] = np.log2(paired.energy / paired.baseline)
    fitted = paired.groupby(KEYS + ["method", "seed"]).agg(
        log2_ratio=("log2_ratio", "mean"), energy=("energy", "mean")).reset_index()
    seed_counts = fitted.groupby(KEYS + ["method"]).seed.agg(lambda x: set(x))
    for key, seeds in seed_counts.items():
        required = {0, 1, 2} if key[-1].startswith(("cfm_", "tuned_")) else {0}
        if seeds != required:
            raise ValueError(f"Incomplete seeds for {key}: {seeds}")
    summary = fitted.groupby(KEYS + ["method"]).agg(
        log2_ratio=("log2_ratio", "mean"), seed_min=("log2_ratio", "min"),
        seed_max=("log2_ratio", "max"), energy=("energy", "mean"),
        seeds=("seed", "nunique")).reset_index()
    return summary.assign(comparator=comparator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nested-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    nested = args.nested_dir.resolve()
    completion = json.loads((nested / "completion.json").read_text())
    if completion["completed"] != 9 or len(completion["records"]) != 9:
        raise ValueError("All nine selected outer refits must finish before export")
    adata = load_adata()
    tasks = all_tasks()
    # Coordinates and input identities are rechecked, independently of scalar tables.
    coordinate_checks = []
    for rep in REPRESENTATIONS:
        for name, task in tasks.items():
            path = OUT / "coords" / rep / f"{name}.npy"
            if not path.exists():
                if name.startswith("INNER_") and rep != "mu_std":
                    continue
                raise ValueError(f"Missing coordinate file: {path}")
            stored = np.load(path)
            recomputed, info = task_coords(adata, task, rep)
            delta = float(np.max(np.abs(stored.astype(float) - recomputed.astype(float))))
            if not np.allclose(stored, recomputed, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Coordinate reproduction failed: {rep}/{name}: {delta}")
            data = task_data(adata, task, stored)
            if name.startswith(("LIO_", "INNER_")) and any(k[0] == task.held_individual for k in data.train):
                raise ValueError("Held donor entered training data")
            if task.family == "missing_time" and any(k[1] == task.held_time for k in data.train):
                raise ValueError("Held time entered training data")
            coordinate_checks.append(dict(rep=rep, task=name, max_abs_error=delta,
                                          fit_cells=info["n_fit"], sha256=sha256(path)))
            print(f"coordinates {rep}/{name}: max_abs={delta:.3g}", flush=True)
    inputs = [H5AD, ROOT / "output/local-p2-20261009/protocol.json",
              ROOT / "output/local-p2-20261009/jobs.jsonl", OUT / "e3b_inner_scores.csv",
              OUT / "e3b_selection.json", nested / "protocol.json", Path(__file__).resolve()]
    inputs += sorted((SOURCE / "scripts").glob("*.py"))
    inputs += sorted((ROOT / "expansion/population").rglob("*.py"))
    inputs += [ROOT / "expansion/common/rng.py", ROOT / "expansion/common/io.py"]
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    if fingerprints[str(H5AD.relative_to(ROOT))] != H5AD_SHA256:
        raise ValueError("Frozen H5AD hash changed")
    # Every declared original fit must exist; never trust queue exit text alone.
    from run_fits import jobs
    declared = [j for stage in ("e1", "e2", "e3b", "e6") for j in jobs(stage)]
    expected = {OUT / f"fits/{j['rep']}/{j['task']}/{j['tag']}_s{j['seed']}.npz" for j in declared}
    paths = sorted((OUT / "fits").rglob("*.npz"))
    if set(paths) != expected:
        raise ValueError(f"Original fit grid differs: missing={len(expected-set(paths))}, extra={len(set(paths)-expected)}")
    paths += sorted(nested.glob("*/tuned_nested_s*.npz"))
    field_checks = []
    if args.replay:
        import torch
        from expansion.population.methods.flows import VelocityField, DEFAULTS
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA replay requested but original backend is unavailable")
        torch.set_num_threads(2)
    for idx, path in enumerate(paths):
        with np.load(path) as saved:
            meta = json.loads(str(saved["meta"]))
            task = tasks[meta["task"]]
            coords = cached_coords(meta["rep"], meta["task"])
            data = task_data(adata, task, coords)
            weights = path.with_name(path.stem + "_field.pt")
            if not weights.is_file():
                raise ValueError(f"Missing saved weights: {weights}")
            check = dict(path=str(path.relative_to(ROOT)), predictions_sha256=sha256(path),
                         weights_sha256=sha256(weights), predictions_finite=True,
                         cuda_replayed=False, maximum_prediction_error=None)
            if args.replay:
                params = {**DEFAULTS, **meta["params"]}
                method = create(meta["method"], **meta["params"])
                method.inputs = method_inputs(data)
                method.device = torch.device("cuda")
                method.interval = float(np.diff(sorted(data.clock.values())).min())
                method.field = VelocityField(coords.shape[1], hidden=params["hidden"],
                                             depth=params["depth"], time_dim=params["time_dim"]).to(method.device)
                method.field.load_state_dict(torch.load(weights, map_location=method.device, weights_only=True))
                method.field.eval()
                check["cuda_replayed"] = True
                check["maximum_prediction_error"] = 0.0
            for pi, pair in enumerate(task.pairs):
                source = coords[group_rows(adata.obs, pair.individual, pair.source_time)]
                pred = saved[f"pred{pi}"]
                if pred.shape != source.shape or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid saved prediction: {path}/pred{pi}")
                if args.replay:
                    fresh = method.predict(source, individual=pair.individual,
                                           source_time=pair.source_time, target_time=pair.target_time,
                                           anchor_time=pair.anchor_time)
                    delta = float(np.max(np.abs(fresh.astype(float) - pred.astype(float))))
                    check["maximum_prediction_error"] = max(check["maximum_prediction_error"], delta)
                    if not np.allclose(fresh, pred, atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Unchanged CUDA replay tolerance failed: {path}: {delta}")
            field_checks.append(check)
        if (idx + 1) % 30 == 0:
            print(f"saved fields checked: {idx+1}/{len(paths)}", flush=True)
    summaries = []
    counts = {}
    for rep in REPRESENTATIONS:
        path = OUT / f"eval_{rep}_draws.csv.gz"
        frame = pd.read_csv(path)
        counts[rep] = len(frame)
        fingerprints[str(path.relative_to(ROOT))] = sha256(path)
        mt = frame[frame.task.str.startswith("MT")]
        lio = frame[frame.task.str.startswith("LIO") & (frame.source_time == 1.0)]
        summaries += [paired_summary(mt, "mean_shift").assign(rep=rep),
                      paired_summary(lio, "target_marginal").assign(rep=rep)]
    original = pd.concat(summaries, ignore_index=True)
    original.to_csv(dest / "representation_donor.csv", index=False)
    original.assign(task_group=np.where(original.task.str.startswith("LIO"), "LIO_1to7", original.task)).groupby(
        ["rep", "task_group", "comparator", "method"]).agg(
            log2_ratio=("log2_ratio", "mean"), donors=("individual", "nunique"),
            donors_below=("log2_ratio", lambda v: int((v < 0).sum()))).reset_index().to_csv(
                dest / "representation_macro.csv", index=False)
    # Fresh outer predictions use the exact existing matched target/source draws.
    frame = pd.read_csv(OUT / "eval_mu_std_draws.csv.gz")
    rows = []
    for path in sorted(nested.glob("*/tuned_nested_s*.npz")):
        with np.load(path) as saved:
            meta = json.loads(str(saved["meta"]))
            task = tasks[meta["task"]]
            coords = cached_coords("mu_std", meta["task"])
            for pi, pair in enumerate(task.pairs):
                sr = group_rows(adata.obs, pair.individual, pair.source_time)
                tr = group_rows(adata.obs, pair.individual, pair.target_time)
                for draw in range(50):
                    _, source_idx, target_idx = draw_plan(task, pair, len(sr), len(tr), draw)
                    rows.append(dict(task=meta["task"], individual=pair.individual,
                                     source_time=pair.source_time, target_time=pair.target_time,
                                     method="tuned_nested", seed=meta["seed"], draw=draw,
                                     energy=energy_distance(saved[f"pred{pi}"][source_idx], coords[tr[target_idx]])))
    fresh = pd.DataFrame(rows)
    fresh.to_csv(dest / "nested_draws.csv.gz", index=False)
    combined = pd.concat([frame[frame.task.str.startswith("LIO")], fresh], ignore_index=True)
    nested_summary = paired_summary(combined, "target_marginal")
    nested_summary.to_csv(dest / "nested_donor.csv", index=False)
    # Preserve each alpha optimum's status as a target-informed diagnostic.
    alpha = pd.read_csv(OUT / "e3_alpha_best.csv")
    alpha.to_csv(dest / "alpha_diagnostic.csv", index=False)
    # Retrospective bindings cover original result files, all coordinates,
    # weights, raw predictions, search scores and new results, without copies.
    originals = [p for p in SOURCE.rglob("*") if p.is_file() and
                 not any(x in p.parts for x in ("__pycache__", ".omc"))]
    originals += [p for p in nested.rglob("*") if p.is_file()]
    bindings = {str(p.relative_to(ROOT)): sha256(p) for p in originals}
    for p, digest in fingerprints.items():
        if sha256(ROOT / p) != digest:
            raise ValueError(f"Input changed during audit: {p}")
    report = dict(status="passed", original_fits=len(expected), new_nested_fits=9,
                  biological_donors=3, coordinates=coordinate_checks,
                  field_checks=field_checks, draw_rows=counts,
                  cuda_replay=args.replay, atol=1e-5, rtol=1e-5,
                  alpha_grid_replayed=False,
                  historical_training_input_hashes_available=False,
                  retrospective_manifest_scope="current inputs and saved artifacts, not proof of original fitting provenance",
                  inputs=fingerprints, original_artifacts=bindings)
    (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS {len(expected)} original fields + 9 nested refits; {len(coordinate_checks)} coordinate reproductions", flush=True)


if __name__ == "__main__":
    main()
