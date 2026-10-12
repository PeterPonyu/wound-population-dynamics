#!/usr/bin/env python3
"""External cross-sectional mouse benchmark on GSE186527 (frozen protocol, diagnostic only).

``admit``  verifies the downloaded payload, audits every library's hashtag measurements, applies the
           frozen demultiplexing rule and admission floors, and writes standardized per-gate data.
``run``    executes the frozen missing-day task with the repository's own population methods and metrics.

The protocol (config/external_gse186527_protocol_20261009.json) was frozen before any RNA-level analysis.
Nothing here changes a threshold after inspection; a group that fails a floor is reported as not admitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
import platform
import re
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expansion.common.io import (atomic_save_npz, atomic_write_json, content_hash,
                                 sanitize, sha256_file)

PROTOCOL = ROOT / "config/external_gse186527_protocol_20261009.json"
BASE = ROOT / "output/external-gse186527-20261009"
INPUTS = BASE / "inputs"
MEMBERS = INPUTS / "members"
ADMISSION = BASE / "admission"
RESULTS = BASE / "results"
ADMITTED_LIBRARIES = {"T-7P": ("PBS", 7), "T-7V": ("Verteporfin", 7), "T-14P": ("PBS", 14),
                      "T-14V": ("Verteporfin", 14), "T-30P": ("PBS", 30), "T-30V": ("Verteporfin", 30)}
EXCLUDED_LIBRARIES = ("T1", "T2", "T3")


def sha256(path) -> str:
    return sha256_file(path)


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


# ----------------------------------------------------------------------------- dictionary and demultiplexing
ENTRY = re.compile(r"((?:Unwounded|POD\d+)[^:]*?(?:Lin\+|Lin-)\s*Mouse\s*\d+):\s*(\S+)\s+(TotalSeq-B-\d+),(\S+)")


def parse_dictionary(soft_gz: Path) -> pd.DataFrame:
    """Author-provided (library, HTO) -> state, gate and mouse table; labels are never corrected."""
    text = gzip.open(soft_gz, "rt").read()
    sections = {line for line in text.splitlines()
                if line.startswith("!Sample_data_processing") and "TotalSeq-B-0301" in line}
    if len(sections) != 1:
        raise ValueError("the nine records must carry one identical dictionary")
    body = sections.pop().split("=", 1)[1]
    rows = []
    for label, library, hto, tag in ENTRY.findall(body[body.index("Unwounded"):]):
        state = re.sub(r"\s+", " ", label.split("Lin")[0]).strip()
        gate = "Lin+" if "Lin+" in label else "Lin-"
        mouse = int(re.search(r"Mouse\s*(\d+)", label).group(1))
        arm = "PBS" if "PBS" in state else "Verteporfin" if "Verteporfin" in state else "none"
        day = 0 if state.startswith("Unwounded") else int(re.search(r"POD(\d+)", state).group(1))
        rows.append(dict(label=label, state=state, library=library, hto=hto, tag=tag, gate=gate, mouse=mouse, arm=arm, day=day))
    table = pd.DataFrame(rows)
    if len(table) != 90 or table.duplicated(["library", "hto"]).any():
        raise ValueError("dictionary must contain 90 unique (library, HTO) entries")
    return table


def classify_hto(counts: np.ndarray, *, min_top: float, min_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (status, top index): 'singlet', 'doublet' or 'negative' for each cell."""
    counts = np.asarray(counts, dtype=np.float64)
    order = np.argsort(-counts, axis=1, kind="stable")
    top = np.take_along_axis(counts, order[:, :1], 1)[:, 0]
    second = np.take_along_axis(counts, order[:, 1:2], 1)[:, 0]
    negative = top < min_top
    doublet = ~negative & (second >= np.maximum(min_top, top / min_ratio))
    singlet = ~negative & ~doublet & (top >= min_ratio * np.maximum(second, 1.0))
    status = np.where(negative, "negative", np.where(singlet, "singlet", "doublet"))
    return status, order[:, 0]


def read_library(gsm: str, name: str):
    features = pd.read_csv(MEMBERS / f"{gsm}_{name}_features.tsv.gz", sep="\t", header=None, names=["id", "name", "type"])
    barcodes = pd.read_csv(MEMBERS / f"{gsm}_{name}_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).to_numpy()
    with gzip.open(MEMBERS / f"{gsm}_{name}_matrix.mtx.gz", "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"{gsm}: matrix, feature and barcode dimensions disagree")
    return features, barcodes, matrix


def library_files() -> dict:
    out = {}
    for path in sorted(MEMBERS.glob("*_matrix.mtx.gz")):
        gsm, name = path.name.split("_matrix")[0].split("_", 1)
        out[name] = gsm
    return out


# ----------------------------------------------------------------------------- admission
def admit() -> dict:
    if ADMISSION.exists():
        raise FileExistsError("existing admission evidence must not be overwritten")
    protocol = load_protocol()
    spec = protocol["inputs"]
    archive, soft = INPUTS / spec["archive"], INPUTS / "GSE186527_family.soft.gz"
    assert archive.stat().st_size == spec["archive_bytes"] and sha256(archive) == spec["archive_sha256"], "archive differs from the reviewed payload"
    assert sha256(soft) == spec["soft_sha256"], "SOFT metadata differs"
    if not MEMBERS.exists():
        MEMBERS.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(MEMBERS, filter="data")
    dictionary = parse_dictionary(soft)
    ADMISSION.mkdir(parents=True)
    dictionary.to_csv(ADMISSION / "hto_dictionary.csv", index=False)
    rules = {"primary": protocol["hto_demultiplexing"]["singlet_rule_primary"],
             "sensitivity_a": protocol["hto_demultiplexing"]["sensitivity_rules"][0],
             "sensitivity_b": protocol["hto_demultiplexing"]["sensitivity_rules"][1]}
    libraries, cells, matrices = [], [], {}
    for name, gsm in library_files().items():
        features, barcodes, matrix = read_library(gsm, name)
        hto_rows = np.flatnonzero(features["type"].to_numpy() == "Antibody Capture")
        hto_names = features["id"].to_numpy()[hto_rows]
        counts = matrix[hto_rows].toarray().T
        record = dict(library=name, gsm=gsm, cells=int(matrix.shape[1]), genes=int((features["type"] == "Gene Expression").sum()),
                      hto_features=int(len(hto_rows)), cells_with_any_hto=int((counts.sum(1) > 0).sum()),
                      total_hto_counts=int(counts.sum()))
        record["usable_hto_measurements"] = bool(record["cells_with_any_hto"] > 0)
        if name in EXCLUDED_LIBRARIES:
            record["admitted"] = False
            record["reason"] = "protocol exclusion; zero HTO counts" if not record["usable_hto_measurements"] else "protocol exclusion"
            libraries.append(record)
            continue
        arm, day = ADMITTED_LIBRARIES[name]
        status = {}
        for rule_name, rule in rules.items():
            state, top = classify_hto(counts, min_top=rule["min_top_count"], min_ratio=rule["min_top_over_second"])
            status[rule_name] = (state, top)
            record[f"singlet_{rule_name}"] = int((state == "singlet").sum())
            record[f"doublet_{rule_name}"] = int((state == "doublet").sum())
            record[f"negative_{rule_name}"] = int((state == "negative").sum())
        primary_state, primary_top = status["primary"]
        for rule_name in ("sensitivity_a", "sensitivity_b"):
            state, top = status[rule_name]
            both = (primary_state == "singlet") & (state == "singlet")
            record[f"primary_singlets_kept_{rule_name}"] = float(both.sum() / max((primary_state == "singlet").sum(), 1))
            record[f"primary_singlets_relabelled_{rule_name}"] = float(((top != primary_top) & both).sum() / max(both.sum(), 1))
        record.update(admitted=True, arm=arm, day=day)
        libraries.append(record)
        frame = pd.DataFrame({"barcode": barcodes, "library": name, "arm": arm, "day": day,
                              "hto_primary": [hto_names[i] if s == "singlet" else "" for i, s in zip(primary_top, primary_state)],
                              "status_primary": primary_state,
                              "status_sensitivity_a": status["sensitivity_a"][0], "status_sensitivity_b": status["sensitivity_b"][0],
                              "hto_sensitivity_a": [hto_names[i] for i in status["sensitivity_a"][1]],
                              "hto_sensitivity_b": [hto_names[i] for i in status["sensitivity_b"][1]]})
        frame["column"] = np.arange(len(frame))
        cells.append(frame)
        matrices[name] = (features, matrix)
    cells = pd.concat(cells, ignore_index=True)
    lookup = dictionary.set_index(["library", "hto"])
    for suffix, column in (("primary", "hto_primary"), ("sensitivity_a", "hto_sensitivity_a"), ("sensitivity_b", "hto_sensitivity_b")):
        keep = cells[f"status_{suffix}"] == "singlet"
        gate, mouse = [], []
        for library, tag, ok in zip(cells["library"], cells[column], keep):
            if not ok or (library, tag) not in lookup.index:
                gate.append(""); mouse.append(0)
            else:
                entry = lookup.loc[(library, tag)]
                gate.append(entry["gate"]); mouse.append(int(entry["mouse"]))
        cells[f"gate_{suffix}"] = gate
        cells[f"mouse_{suffix}"] = mouse
    cells["mouse5_conflict"] = cells["hto_primary"].isin(["TotalSeq-B-0309", "TotalSeq-B-0310"])
    # Floors are applied to the frozen primary rule; sensitivity rules are reported but do not rescue a group.
    floors = protocol["admission_floors"]
    groups = []
    for analysis, gate in (("Lin-", "Lin-"), ("Lin+", "Lin+")):
        for arm in ("PBS", "Verteporfin"):
            for day in (7, 14, 30):
                row = {}
                for suffix in ("primary", "sensitivity_a", "sensitivity_b"):
                    mask = (cells["arm"] == arm) & (cells["day"] == day) & (cells[f"gate_{suffix}"] == gate)
                    if gate == "Lin+":
                        mask &= cells[f"mouse_{suffix}"] != 5  # mouse 5 (HTO9/HTO10 label conflict) is excluded, never relabelled
                    row[f"cells_{suffix}"] = int(mask.sum())
                    row[f"mice_{suffix}"] = int(cells.loc[mask, f"mouse_{suffix}"].nunique())
                row.update(analysis=analysis, arm=arm, day=day)
                row["admitted"] = bool(row["cells_primary"] >= floors["min_cells_per_group"] and row["mice_primary"] >= floors["min_mice_per_group"])
                groups.append(row)
    groups = pd.DataFrame(groups)
    groups.to_csv(ADMISSION / "groups.csv", index=False)
    pd.DataFrame(libraries).to_csv(ADMISSION / "libraries.csv", index=False)
    cells.drop(columns=["column"]).to_csv(ADMISSION / "cell_assignments.csv.gz", index=False)
    # Standardized count matrices per analysis: admitted groups only, primary singlets only.
    result = {"protocol_sha256": sha256(PROTOCOL), "libraries": libraries, "groups": groups.to_dict("records"), "analyses": {}}
    import anndata as ad
    for analysis, gate in (("Lin-", "Lin-"), ("Lin+", "Lin+")):
        admitted = groups[(groups["analysis"] == analysis) & groups["admitted"]]
        arms = sorted({a for a in admitted["arm"] if (admitted[admitted["arm"] == a]["day"].isin([7, 14, 30]).sum() == 3)})
        result["analyses"][analysis] = {"arms_with_all_three_days": arms, "task_admitted": len(arms) >= 1,
                                       "arms_not_admitted": sorted(set(["PBS", "Verteporfin"]) - set(arms))}
        if not arms:
            continue
        blocks, obs_rows = [], []
        for arm in arms:
            for day in (7, 14, 30):
                library = [k for k, v in ADMITTED_LIBRARIES.items() if v == (arm, day)][0]
                mask = (cells["library"] == library) & (cells["gate_primary"] == gate)
                if gate == "Lin+":
                    mask &= ~cells["mouse5_conflict"] & (cells["mouse_primary"] != 5)
                sub = cells[mask]
                features, matrix = matrices[library]
                gene_rows = np.flatnonzero(features["type"].to_numpy() == "Gene Expression")
                columns = sub["column"].to_numpy()
                blocks.append(matrix[gene_rows][:, columns].T.tocsr())
                obs_rows.append(pd.DataFrame({"individual": arm, "time_value": float(day), "sample": library, "mouse": sub["mouse_primary"].astype(int).astype(str).to_numpy(),
                                              "barcode": sub["barcode"].to_numpy(), "gate": gate}))
        features = matrices["T-7P"][0]
        genes = features[features["type"] == "Gene Expression"].reset_index(drop=True)
        X = sp.vstack(blocks).tocsr().astype(np.float32)
        detected = np.asarray((X > 0).sum(0)).ravel()
        keep = detected >= 3
        obs = pd.concat(obs_rows, ignore_index=True)
        obs.index = obs["sample"] + ":" + obs["barcode"]
        adata = ad.AnnData(X[:, keep], obs=obs, var=pd.DataFrame({"gene_id": genes["id"].to_numpy()[keep]}, index=genes["name"].astype(str).to_numpy()[keep]))
        adata.var_names_make_unique()
        path = ADMISSION / f"standard_{analysis.replace('+', 'pos').replace('-', 'neg')}.h5ad"
        adata.write_h5ad(path)
        result["analyses"][analysis].update(path=str(path.relative_to(ROOT)), sha256=sha256(path), cells=int(adata.n_obs), genes=int(adata.n_vars))
    (ADMISSION / "admission.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


# ----------------------------------------------------------------------------- benchmark
def producing_sources() -> dict:
    paths = [Path(__file__).resolve(), *sorted((ROOT / "expansion").rglob("*.py"))]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def fit_plan(protocol, admission) -> list:
    """Every admitted fit, including the frozen clock sensitivity, exactly once."""
    from expansion.population.methods import baselines, flows  # noqa: F401
    from expansion.population.methods.base import REGISTRY
    plan = []
    for analysis, info in admission["analyses"].items():
        if not info.get("task_admitted"):
            continue
        for clock in [protocol["clock"]["primary"], *protocol["clock"]["sensitivity"]]:
            for name in protocol["methods"]:
                if clock != protocol["clock"]["primary"] and name not in protocol["clock"]["sensitivity_methods"]:
                    continue
                stochastic = REGISTRY[name].spec.stochastic
                for seed in protocol["seeds"]["stochastic" if stochastic else "deterministic"]:
                    job_id = f"{analysis.replace('+', 'pos').replace('-', 'neg')}-{clock}-{name}-seed{seed}"
                    plan.append(dict(job_id=job_id, analysis=analysis, clock=clock, method=name, seed=int(seed)))
    if not plan or len({item["job_id"] for item in plan}) != len(plan):
        raise ValueError("no admitted fits or duplicate frozen jobs")
    return plan


def fresh_run_directory(path):
    if path is None:
        raise ValueError("run requires an explicit fresh --output-dir")
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("run destination cannot be symlinked")
    if (path.parent != ROOT / "output" or not re.fullmatch(r"external-gse186527-\d{8}-[A-Za-z0-9_-]+", path.name)):
        raise ValueError("choose output/external-gse186527-YYYYMMDD-<new-run-name>")
    if path.exists():
        raise FileExistsError("existing experiment outputs must not be overwritten")
    return path


def checked_admission(protocol):
    admission = json.loads((ADMISSION / "admission.json").read_text())
    if admission["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("protocol changed after admission")
    for info in admission["analyses"].values():
        if info.get("task_admitted"):
            path = (ROOT / info["path"]).resolve()
            if not path.is_relative_to(ADMISSION.resolve()) or sha256(path) != info["sha256"]:
                raise ValueError("admitted input changed or escapes the admission directory")
    return admission


def run(output_dir=None, *, device=None) -> dict:
    import anndata as ad
    from expansion.common.rng import derive_seed
    from expansion.population.methods import baselines, flows  # noqa: F401  (register methods)
    from expansion.population.methods.base import REGISTRY, create
    from expansion.population.metrics import energy_distance, evaluate
    from expansion.population.represent import task_coordinates
    from expansion.population.tasks import (Pair, TaskSpec, build_task_data, make_clock, method_inputs,
                                            representation_mask)
    from expansion.population.paired import group_rows, save_model
    protocol = load_protocol()
    admission = checked_admission(protocol)
    output_dir = fresh_run_directory(output_dir)
    caps = protocol["evaluation"]["caps"]
    projections = int(protocol["evaluation"]["sliced_projections"])
    metric_names = [protocol["evaluation"]["primary_metric"], *protocol["evaluation"]["secondary_metrics"]]
    flow_defaults = protocol["flow_defaults"]
    seeds = protocol["seeds"]
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested execution backend unavailable")
    sources = producing_sources()
    input_hashes = {str(path.relative_to(ROOT)): sha256(path)
                    for path in sorted(ADMISSION.iterdir()) if path.is_file()}
    plan = fit_plan(protocol, admission)
    manifest = {"schema": "external-population-reproduction-v1", "protocol_sha256": sha256(PROTOCOL),
                "protocol": protocol, "code_files": sources, "code_digest": content_hash(sources, 64),
                "input_files": input_hashes, "planned_fits": plan,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": device, "torch_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "torch_threads": torch.get_num_threads(), "pid": os.getpid(),
                    "versions": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "anndata", "POT")},
                    "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                "scope": "Retrospective cross-sectional diagnostic; no new biological units. Prediction replay, not optimizer resume."}
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_json(output_dir / "STATUS.json", {"status": "running", "planned": len(plan), "done": 0, "pid": os.getpid()})
    rows, mouse_rows, noise_rows, fits = [], [], [], []
    report = {"protocol_sha256": sha256(PROTOCOL), "device": device, "analyses": {}}
    for analysis, info in admission["analyses"].items():
        if not info.get("task_admitted"):
            report["analyses"][analysis] = {"status": "not_admitted", "reason": "floors failed", "arms_not_admitted": info["arms_not_admitted"]}
            continue
        path = ROOT / info["path"]
        adata = ad.read_h5ad(path)
        arms = info["arms_with_all_three_days"]
        task_cfg = protocol["tasks"]["T-MT"]
        pairs = tuple(Pair(arm, float(task_cfg["source_time"]), float(task_cfg["held_time"]), float(task_cfg["anchor_time"])) for arm in arms)
        dataset_id = f"GSE186527-{analysis.replace('+', 'pos').replace('-', 'neg')}"
        task = TaskSpec(dataset_id, "T-MT", "missing_time", float(task_cfg["held_time"]), None, pairs)
        rep_cfg = {"kind": "pca_train", **{k: protocol["representation"][k] for k in ("n_hvg", "n_components", "max_fit_cells")}}
        fit_mask = representation_mask(adata.obs, task)
        coords, rep_info, rep = task_coordinates(adata, fit_mask, rep_cfg,
            seed=int(protocol["representation"]["seed"]), return_state=True)
        obs = adata.obs.reset_index(drop=True)
        report["analyses"][analysis] = {"status": "run", "arms": arms, "cells": int(adata.n_obs), "representation": rep_info}
        names = protocol["methods"]
        for clock_name in ["rank", *protocol["clock"]["sensitivity"]]:
            clock = make_clock(obs["time_value"].unique(), clock_name)
            data = build_task_data(coords, obs, task, clock, min_cells=int(protocol["admission_floors"]["min_cells_per_group"]))
            train_rows = {key: group_rows(obs, *key) for key in data.train}
            for name in names:
                if clock_name != "rank" and name not in protocol["clock"]["sensitivity_methods"]:
                    continue
                spec = REGISTRY[name].spec
                params = dict(protocol["method_params"].get(name, {}))
                if spec.family == "flow":
                    params = {**flow_defaults, **params}
                for seed in (seeds["stochastic"] if spec.stochastic else seeds["deterministic"]):
                    planned = next(item for item in plan if (item["analysis"], item["clock"], item["method"], item["seed"])
                                   == (analysis, clock_name, name, seed))
                    directory = output_dir / "fits" / planned["job_id"]
                    directory.mkdir(parents=True)
                    method = create(name, **params)
                    started = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught_fit:
                        warnings.simplefilter("always")
                        info_fit = method.fit(method_inputs(data), seed=int(seed), device=device)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    fit_seconds = time.perf_counter() - started
                    fits.append({"analysis": analysis, "clock": clock_name, "method": name, "seed": seed, "fit_seconds": fit_seconds,
                                 "variant": method.variant})
                    binding = {**planned, "protocol_sha256": manifest["protocol_sha256"],
                               "code_digest": manifest["code_digest"], "dataset_sha256": info["sha256"]}
                    save_model(directory / "model_replay.npz", method, rep, rep_config=rep_cfg,
                        train_rows=train_rows, binding=binding,
                        min_cells=int(protocol["admission_floors"]["min_cells_per_group"]), clock_name=clock_name)
                    evaluations = []
                    for pair in pairs:
                        source = data.sources[(pair.individual, pair.source_time)]
                        target = data.targets[(pair.individual, pair.target_time)]
                        call = dict(individual=pair.individual, source_time=pair.source_time,
                                    target_time=pair.target_time, anchor_time=pair.anchor_time, seed=int(seed))
                        with warnings.catch_warnings(record=True) as caught_prediction:
                            warnings.simplefilter("always")
                            prediction = method.predict(source, **call)
                        eval_seed = derive_seed(dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
                        scores = evaluate(prediction, target, seed=eval_seed, caps=caps, n_projections=projections, metric_names=metric_names)
                        rows.append({"analysis": analysis, "clock": clock_name, "arm": pair.individual, "method": name, "seed": seed,
                                     "variant": method.variant, "n_source": len(source), "n_target": len(target), **scores})
                        target_rows = np.flatnonzero((obs["individual"] == pair.individual) & (obs["time_value"] == pair.target_time))
                        mice = obs.loc[target_rows, "mouse"].to_numpy()
                        source_rows = group_rows(obs, pair.individual, pair.source_time)
                        artifact = f"evaluation{len(evaluations)}.npz"
                        atomic_save_npz(directory / artifact, prediction=prediction, source=source, target=target,
                            source_rows=source_rows, target_rows=target_rows,
                            source_ids=adata.obs_names.to_numpy(dtype=str)[source_rows],
                            target_ids=adata.obs_names.to_numpy(dtype=str)[target_rows],
                            mice=np.asarray(mice, dtype=str), representation_fit_rows=np.flatnonzero(fit_mask))
                        mouse_scores = {}
                        for mouse in sorted(set(mice)):
                            cells = target[mice == mouse]
                            mouse_score = evaluate(prediction, cells, seed=eval_seed, caps=caps, n_projections=projections,
                                                   metric_names=["energy"])["energy"]
                            mouse_rows.append({"analysis": analysis, "clock": clock_name, "arm": pair.individual, "method": name,
                                               "seed": seed, "variant": method.variant, "mouse": mouse, "n_mouse_cells": len(cells),
                                               "energy": mouse_score})
                            mouse_scores[str(mouse)] = mouse_score
                        evaluations.append({"artifact": artifact, "call": call, "score_seed": int(eval_seed),
                            "scores": scores, "mouse_energy": mouse_scores,
                            "prediction_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught_prediction]})
                    fit_record = {"binding": binding, "task": task.to_dict(), "fit_seconds": fit_seconds,
                        "fit_diagnostics": sanitize(info_fit), "fit_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught_fit],
                        "evaluations": evaluations}
                    atomic_write_json(directory / "fit.json", fit_record)
                    atomic_write_json(directory / "done.json", {"job": planned,
                        "files": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}})
                    atomic_write_json(output_dir / "STATUS.json", {"status": "running", "planned": len(plan), "done": len(fits), "pid": os.getpid()})
                    print(planned["job_id"], "saved", flush=True)
                print(analysis, clock_name, name, "done", flush=True)
        # Descriptive scale references: distance between held-time mice and between neighbouring days.
        for pair in pairs:
            target_idx = np.flatnonzero((obs["individual"] == pair.individual) & (obs["time_value"] == pair.target_time))
            mice = obs.loc[target_idx, "mouse"].to_numpy()
            held = coords[target_idx]
            for mouse in sorted(set(mice)):
                rest = held[mice != mouse]
                noise_rows.append({"analysis": analysis, "arm": pair.individual, "reference": "held_mouse_vs_other_held_mice",
                                   "mouse": mouse, "energy": energy_distance(held[mice == mouse][:2000], rest[:2000])})
            for label, day in (("source_day_vs_held_day", pair.source_time), ("anchor_day_vs_held_day", pair.anchor_time)):
                other = coords[np.flatnonzero((obs["individual"] == pair.individual) & (obs["time_value"] == day))]
                noise_rows.append({"analysis": analysis, "arm": pair.individual, "reference": label, "mouse": "pooled",
                                   "energy": energy_distance(other[:2000], held[:2000])})
    frame, mouse_frame, noise = pd.DataFrame(rows), pd.DataFrame(mouse_rows), pd.DataFrame(noise_rows)
    frame.to_csv(output_dir / "population_scores.csv", index=False)
    mouse_frame.to_csv(output_dir / "mouse_scores.csv", index=False)
    noise.to_csv(output_dir / "scale_references.csv", index=False)
    pd.DataFrame(fits).to_csv(output_dir / "fits.csv", index=False)
    summary = {}
    for (analysis, clock, arm), part in frame.groupby(["analysis", "clock", "arm"]):
        energy = part.groupby("method")["energy"].agg(["mean", "std", "min", "max", "count"])
        reference = energy.loc[protocol["analysis"]["primary_comparator"], "mean"]
        energy["log_ratio_to_comparator"] = np.log(energy["mean"] / reference)
        mice = mouse_frame[(mouse_frame["analysis"] == analysis) & (mouse_frame["clock"] == clock) & (mouse_frame["arm"] == arm)]
        per_mouse = mice.groupby(["method", "mouse"])["energy"].mean().groupby("method").agg(["mean", "min", "max"])
        summary[f"{analysis}|{clock}|{arm}"] = {"population_energy": energy.round(6).to_dict("index"),
                                                  "per_mouse_energy": per_mouse.round(6).to_dict("index")}
    report["summary"] = summary
    report["scale_references"] = noise.round(6).to_dict("records")
    report["not_admitted"] = {k: v for k, v in report["analyses"].items() if v["status"] != "run"}
    if len(fits) != len(plan):
        raise ValueError("not every frozen fit completed")
    if producing_sources() != sources or any(sha256(ROOT / name) != digest for name, digest in input_hashes.items()):
        raise ValueError("producing code or admitted input changed during execution")
    atomic_write_json(output_dir / "report.json", sanitize(report))
    files = {str(p.relative_to(output_dir)): sha256(p) for p in sorted(output_dir.rglob("*"))
             if p.is_file() and p.name != "STATUS.json"}
    if sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) > 128 * 1024 ** 2:
        raise ValueError("new experiment exceeds the 128 MiB local-output budget")
    atomic_write_json(output_dir / "STATUS.json", {"status": "complete", "planned": len(plan), "done": len(fits),
        "files": files, "replay_completed": False, "submission_ready": False})
    return report


def checked_file(base, name):
    """Artifact metadata supplies locators, never authority to leave its root."""
    relative = Path(name)
    path = base / relative
    if relative.is_absolute() or ".." in relative.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe artifact locator")
    if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("missing or escaping artifact")
    return path


def replay(output_dir, *, device="cpu") -> dict:
    """Rebuild permitted representations and replay saved fits, never refit a predictor."""
    import anndata as ad
    import torch
    from expansion.common.rng import derive_seed
    from expansion.population.methods import baselines, flows  # noqa: F401
    from expansion.population.metrics import evaluate
    from expansion.population.paired import (_check_representation_state, _expected_flow_interval,
                                             group_rows, load_model)
    from expansion.population.tasks import Pair, TaskSpec, build_task_data, make_clock, representation_mask

    output_dir = Path(output_dir).absolute()
    status = json.loads(checked_file(output_dir, "STATUS.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("all frozen fits must complete before replay")
    for name, digest in status["files"].items():
        if sha256(checked_file(output_dir, name)) != digest:
            raise ValueError("completed artifact hash differs: " + name)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    protocol = load_protocol()
    admission = checked_admission(protocol)
    if (manifest.get("schema") != "external-population-reproduction-v1"
            or manifest["protocol_sha256"] != sha256(PROTOCOL) or manifest["protocol"] != protocol
            or manifest["code_files"] != producing_sources()
            or manifest["code_digest"] != content_hash(manifest["code_files"], 64)):
        raise ValueError("producing protocol/code binding changed")
    current_inputs = {str(path.relative_to(ROOT)): sha256(path)
                      for path in sorted(ADMISSION.iterdir()) if path.is_file()}
    if manifest["input_files"] != current_inputs:
        raise ValueError("admission inputs changed since execution")
    expected_plan = fit_plan(protocol, admission)
    if (manifest["planned_fits"] != expected_plan or status["planned"] != len(expected_plan)
            or status["done"] != len(expected_plan)):
        raise ValueError("completed jobs differ from the frozen plan")
    actual_jobs = sorted(p.name for p in (output_dir / "fits").iterdir())
    if actual_jobs != sorted(item["job_id"] for item in expected_plan):
        raise ValueError("missing or extra fit directories")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested replay backend unavailable")
    if device == "cuda" and (torch.backends.cuda.matmul.allow_tf32 != manifest["runtime"]["cuda_matmul_tf32"]
            or torch.backends.cudnn.allow_tf32 != manifest["runtime"]["cudnn_tf32"]):
        raise ValueError("CUDA numerical settings differ from the producing run")

    population_frame = pd.read_csv(output_dir / "population_scores.csv")
    mouse_frame = pd.read_csv(output_dir / "mouse_scores.csv", dtype={"mouse": str})
    fit_frame = pd.read_csv(output_dir / "fits.csv")
    fit_keys = ["analysis", "clock", "method", "seed"]
    population_keys = [*fit_keys, "arm"]
    expected_pairs = sum(len(admission["analyses"][job["analysis"]]["arms_with_all_three_days"]) for job in expected_plan)
    if (len(fit_frame) != len(expected_plan) or fit_frame.duplicated(fit_keys).any()
            or len(population_frame) != expected_pairs or population_frame.duplicated(population_keys).any()
            or mouse_frame.duplicated([*population_keys, "mouse"]).any()):
        raise ValueError("aggregate tables contain incomplete or duplicate fit/evaluation rows")
    cache, results, verified_mouse_rows = {}, [], 0
    begun = time.perf_counter()
    for job in expected_plan:
        analysis, clock_name = job["analysis"], job["clock"]
        info = admission["analyses"][analysis]
        if analysis not in cache:
            cache[analysis] = ad.read_h5ad(ROOT / info["path"])
        adata = cache[analysis]
        cfg = protocol["tasks"]["T-MT"]
        pairs = tuple(Pair(arm, float(cfg["source_time"]), float(cfg["held_time"]), float(cfg["anchor_time"]))
                      for arm in info["arms_with_all_three_days"])
        dataset_id = f"GSE186527-{analysis.replace('+', 'pos').replace('-', 'neg')}"
        task = TaskSpec(dataset_id, "T-MT", "missing_time", float(cfg["held_time"]), None, pairs)
        directory = output_dir / "fits" / job["job_id"]
        done = json.loads(checked_file(directory, "done.json").read_text())
        if done["job"] != job or set(done["files"]) != {"fit.json", "model_replay.npz", *[f"evaluation{i}.npz" for i in range(len(pairs))]}:
            raise ValueError("completion receipt is incomplete or belongs to another fit")
        for name, digest in done["files"].items():
            if sha256(checked_file(directory, name)) != digest:
                raise ValueError("fit artifact changed")
        record = json.loads((directory / "fit.json").read_text())
        binding = {**job, "protocol_sha256": manifest["protocol_sha256"],
                   "code_digest": manifest["code_digest"], "dataset_sha256": info["sha256"]}
        if record["binding"] != binding or record["task"] != task.to_dict() or len(record["evaluations"]) != len(pairs):
            raise ValueError("saved fit task/binding differs from the frozen task")
        fit_row = fit_frame
        for key in fit_keys:
            fit_row = fit_row[fit_row[key] == job[key]]
        if len(fit_row) != 1:
            raise ValueError("fit table does not match the saved job")
        np.testing.assert_allclose(fit_row.iloc[0].fit_seconds, record["fit_seconds"], rtol=1e-12, atol=1e-12)
        method, rep, train_rows, meta = load_model(directory / "model_replay.npz",
            expected_sha256=done["files"]["model_replay.npz"], expected_binding=binding)
        rep_cfg = {"kind": "pca_train", **{k: protocol["representation"][k] for k in ("n_hvg", "n_components", "max_fit_cells")}}
        min_cells = int(protocol["admission_floors"]["min_cells_per_group"])
        params = dict(protocol["method_params"].get(job["method"], {}))
        if method.spec.family == "flow":
            params = {**protocol["flow_defaults"], **params}
        if (meta["method"] != job["method"] or meta["params"] != params or meta["clock_name"] != clock_name
                or meta["representation"]["config"] != rep_cfg or meta["min_cells_per_group"] != min_cells
                or meta["kind"] != task.kind or meta["held_time"] != task.held_time or meta["held_individual"] is not None):
            raise ValueError("saved method settings differ from the frozen protocol")
        mask = representation_mask(adata.obs, task)
        _check_representation_state(rep, adata, np.flatnonzero(mask), rep_cfg,
                                    seed=int(protocol["representation"]["seed"]))
        coords = rep.transform_counts(adata.X)
        clock = make_clock(adata.obs.time_value.unique(), clock_name)
        data = build_task_data(coords, adata.obs, task, clock, min_cells=min_cells)
        if method.inputs.clock != clock or set(method.inputs.train) != set(data.train):
            raise ValueError("saved training information set differs from the admitted inputs")
        for key in data.train:
            np.testing.assert_array_equal(train_rows[key], group_rows(adata.obs, *key))
            np.testing.assert_array_equal(method.inputs.train[key], data.train[key])
        if method.spec.family == "flow":
            if method.interval != _expected_flow_interval(clock):
                raise ValueError("saved integration interval differs from the frozen clock")
            method.device = torch.device(device)
            method.field.to(method.device).eval()
        maximum = 0.0
        for i, (pair, evaluation) in enumerate(zip(pairs, record["evaluations"])):
            call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time,
                        anchor_time=pair.anchor_time, seed=job["seed"])
            score_seed = derive_seed(dataset_id, task.kind, task.family, pair.individual, pair.source_time, pair.target_time)
            if evaluation["call"] != call or evaluation["score_seed"] != score_seed or evaluation["artifact"] != f"evaluation{i}.npz":
                raise ValueError("saved evaluation identity/seed differs from the frozen task")
            with np.load(directory / evaluation["artifact"], allow_pickle=False) as saved:
                source_rows = group_rows(adata.obs, pair.individual, pair.source_time)
                target_rows = group_rows(adata.obs, pair.individual, pair.target_time)
                np.testing.assert_array_equal(saved["representation_fit_rows"], np.flatnonzero(mask))
                for prefix, rows in (("source", source_rows), ("target", target_rows)):
                    np.testing.assert_array_equal(saved[prefix + "_rows"], rows)
                    np.testing.assert_array_equal(saved[prefix + "_ids"], adata.obs_names.to_numpy(dtype=str)[rows])
                    np.testing.assert_array_equal(saved[prefix], coords[rows])
                mice = adata.obs.iloc[target_rows]["mouse"].astype(str).to_numpy()
                np.testing.assert_array_equal(saved["mice"], mice)
                again = method.predict(saved["source"], **call)
                np.testing.assert_allclose(again, saved["prediction"], rtol=1e-5, atol=1e-5,
                                           err_msg=job["job_id"] + ": prediction replay failed")
                maximum = max(maximum, float(np.max(np.abs(again - saved["prediction"]))))
                options = dict(seed=score_seed, caps=protocol["evaluation"]["caps"],
                               n_projections=int(protocol["evaluation"]["sliced_projections"]))
                scores = evaluate(saved["prediction"], saved["target"], **options,
                    metric_names=[protocol["evaluation"]["primary_metric"], *protocol["evaluation"]["secondary_metrics"]])
                if set(scores) != set(evaluation["scores"]) or set(evaluation["mouse_energy"]) != set(mice):
                    raise ValueError("saved population/mouse metric set differs")
                for name, value in scores.items():
                    np.testing.assert_allclose(value, evaluation["scores"][name], rtol=1e-12, atol=1e-12)
                aggregate_row = population_frame
                mouse_rows = mouse_frame
                for key in fit_keys:
                    aggregate_row = aggregate_row[aggregate_row[key] == job[key]]
                    mouse_rows = mouse_rows[mouse_rows[key] == job[key]]
                aggregate_row = aggregate_row[aggregate_row["arm"] == pair.individual]
                mouse_rows = mouse_rows[mouse_rows["arm"] == pair.individual]
                if (len(aggregate_row) != 1 or len(mouse_rows) != len(set(mice))
                        or aggregate_row.iloc[0]["variant"] != method.variant
                        or fit_row.iloc[0]["variant"] != method.variant
                        or aggregate_row.iloc[0]["n_source"] != len(source_rows)
                        or aggregate_row.iloc[0]["n_target"] != len(target_rows)):
                    raise ValueError("aggregate evaluation identity/count/variant differs")
                for name, value in scores.items():
                    np.testing.assert_allclose(aggregate_row.iloc[0][name], value, rtol=1e-12, atol=1e-12)
                for mouse in sorted(set(mice)):
                    score = evaluate(saved["prediction"], saved["target"][mice == mouse], **options, metric_names=["energy"])["energy"]
                    np.testing.assert_allclose(score, evaluation["mouse_energy"][mouse], rtol=1e-12, atol=1e-12)
                    row = mouse_rows[mouse_rows["mouse"] == mouse]
                    if len(row) != 1 or row.iloc[0]["n_mouse_cells"] != int((mice == mouse).sum()) or row.iloc[0]["variant"] != method.variant:
                        raise ValueError("aggregate mouse identity/count/variant differs")
                    np.testing.assert_allclose(row.iloc[0]["energy"], score, rtol=1e-12, atol=1e-12)
                    verified_mouse_rows += 1
        results.append({**job, "max_abs_prediction_difference": maximum})
        print(job["job_id"], "replayed", flush=True)
    if verified_mouse_rows != len(mouse_frame):
        raise ValueError("aggregate mouse table contains extra rows")
    return {"schema": "external-population-replay-receipt-v1", "backend": device, "all_planned_fits": len(results),
            "manifest_sha256": sha256(output_dir / "run_manifest.json"), "status_sha256": sha256(output_dir / "STATUS.json"),
            "input_revalidated": True, "representation_fit_rows_revalidated": True, "score_recomputed": True,
            "population_score_rows_revalidated": len(population_frame), "mouse_score_rows_revalidated": verified_mouse_rows,
            "predictor_refitted": False, "atol": 1e-5, "rtol": 1e-5, "jobs": results,
            "max_abs_prediction_difference": max(row["max_abs_prediction_difference"] for row in results),
            "wall_seconds": time.perf_counter() - begun, "submission_ready": False,
            "scope": "Input-backed saved-state/prediction/metric consistency, not optimizer provenance, CPU portability on another backend, or biological validation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("admit", "run", "replay"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("one worker supports 1--4 numeric threads")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    from threadpoolctl import threadpool_limits
    numeric_limit = threadpool_limits(limits=args.threads)
    import torch
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.25)
    if args.phase == "replay":
        if args.output_dir is None or args.receipt is None:
            parser.error("replay requires --output-dir and a fresh --receipt")
        if args.receipt.exists() or any(p.is_symlink() for p in (args.receipt.absolute(), *args.receipt.absolute().parents)):
            raise FileExistsError("replay receipt must be a fresh nonsymlinked file")
        if not args.receipt.resolve().is_relative_to(args.output_dir.resolve()) or args.receipt.suffix != ".json":
            parser.error("replay receipt must be a new JSON file inside the selected run")
        outcome = replay(args.output_dir, device=args.device or "cpu")
        atomic_write_json(args.receipt, outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != "jobs"}, indent=2))
    else:
        outcome = admit() if args.phase == "admit" else run(args.output_dir, device=args.device)
        print(json.dumps(outcome.get("analyses"), indent=2, default=str))
