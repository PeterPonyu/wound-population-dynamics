"""Shared utilities for the paper-2 reviewer follow-up analyses (2026-10-10).

Everything here re-uses the frozen 107-job harness (``expansion.population``) by
import: task specifications are read from ``output/local-p2-20261009/jobs.jsonl``,
the representation is the harness training-only standardizer of the frozen 15-dim
encoder means, flows are the harness ``cfm_*`` implementations with the protocol
``flow_defaults`` and the matched evaluation re-derives the harness
``train-side-paired-v2`` draws (200 particles x 50 draws, draw seed 20261005).
No project file outside ``outputs/followup_20261010/paper2`` is written.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from expansion.common.rng import derive_seed  # noqa: E402
from expansion.population.methods import create  # noqa: E402
from expansion.population.metrics import energy_distance  # noqa: E402
from expansion.population.represent import fit_pca, fit_standardizer  # noqa: E402
from expansion.population.tasks import (TaskSpec, build_task_data, make_clock,  # noqa: E402
                                        method_inputs, representation_mask)

H5AD = ROOT / "output/data-smoke-20260929/data/standardized/GSE241132/standard.h5ad"
H5AD_SHA256 = "620aa6980e8ee5c6846532b65b22e860bca1faa1b64f54c17bac553315624848"
MU = ROOT / "outputs/repaired_20260918/human_temporal_flow/mu.npy"
OBS = ROOT / "outputs/repaired_20260918/human_temporal_flow/obs_fibroblast.csv"
HARNESS = ROOT / "output/local-p2-20261009"
PROTOCOL = json.loads((HARNESS / "protocol.json").read_text())
CONTRACT = PROTOCOL["evaluation"]["contract"]
FLOW_DEFAULTS = dict(PROTOCOL["flow_defaults"])
FLOW_DEFAULTS.pop("profile_stages", None)
DONORS = ("PWH26", "PWH27", "PWH28")
SEEDS = (0, 1, 2)
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


_ADATA = {}


def load_adata(counts: bool = False):
    import anndata as ad
    key = "counts" if counts else "latent"
    if key not in _ADATA:
        adata = ad.read_h5ad(H5AD) if counts else ad.read_h5ad(H5AD, backed="r")
        if not counts:
            obs = adata.obs.copy()
            latent = np.asarray(adata.obsm["X_frozen_mu"], dtype=np.float64)
            adata.file.close()

            class Lite:  # minimal stand-in exposing what the harness reads
                pass
            lite = Lite()
            lite.obs, lite.obsm, lite.X = obs, {"X_frozen_mu": latent}, None
            adata = lite
        _ADATA[key] = adata
    return _ADATA[key]


def harness_tasks() -> dict:
    """Return {name: TaskSpec} for the five primary harness tasks (exact job specs)."""
    seen = {}
    for line in open(HARNESS / "jobs.jsonl"):
        job = json.loads(line)
        task = job["task"]
        if task["family"] == "missing_time":
            name = f"MT_day{int(task['held_time'])}"
        elif task["family"] == "leave_individual":
            name = f"LIO_{task['held_individual']}"
        else:
            continue
        seen.setdefault(name, TaskSpec.from_dict(task))
    return dict(sorted(seen.items()))


def group_rows(obs, individual, time_value):
    return np.flatnonzero((obs.individual.astype(str).to_numpy() == str(individual))
                          & (obs.time_value.to_numpy(dtype=float) == float(time_value)))


# ----------------------------------------------------------------------------------------------
# Representations (all fitted on the task's permitted cells only: held time / held donor excluded)
# ----------------------------------------------------------------------------------------------
REPRESENTATIONS = ("mu_std", "mu_raw", "mu_white", "mu_clr_std", "pca30")


def task_coords(adata, task: TaskSpec, rep: str = "mu_std") -> tuple[np.ndarray, dict]:
    mask = representation_mask(adata.obs, task)
    latent = np.asarray(adata.obsm["X_frozen_mu"], dtype=np.float64)
    if rep == "mu_std":  # identical to the harness obsm_standardized representation
        state = fit_standardizer(latent[mask])
        return state.transform_latent(latent), {"rep": rep, "n_fit": int(mask.sum())}
    if rep == "mu_raw":  # unscaled encoder means; centring only (distances unchanged)
        mean = latent[mask].mean(0)
        return (latent - mean).astype(np.float32), {"rep": rep, "n_fit": int(mask.sum())}
    if rep == "mu_white":  # ZCA whitening with the training covariance (tiny relative ridge)
        mean = latent[mask].mean(0)
        cov = np.cov(latent[mask] - mean, rowvar=False, bias=True)
        values, vectors = np.linalg.eigh(cov)
        floor = 1e-6 * float(values.mean())
        w = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, floor))) @ vectors.T
        return ((latent - mean) @ w).astype(np.float32), {
            "rep": rep, "n_fit": int(mask.sum()), "eigen_min": float(values.min()), "eigen_max": float(values.max())}
    if rep == "mu_clr_std":  # project onto the sum-zero hyperplane (softmax offset removed), then standardize
        clr = latent - latent.mean(1, keepdims=True)
        state = fit_standardizer(clr[mask])
        return state.transform_latent(clr), {"rep": rep, "n_fit": int(mask.sum()), "rank": 14}
    if rep == "pca30":  # harness pca_train: 2000 training HVGs, 30 PCs, training-standardized
        counts = load_adata(counts=True)
        import scipy.sparse as sp
        X = sp.csr_matrix(counts.X)
        state = fit_pca(X[np.flatnonzero(mask)], n_hvg=2000, n_components=30, seed=0)
        return state.transform_counts(X), {"rep": rep, "n_fit": int(mask.sum()), **state.params}
    raise ValueError(rep)


def task_data(adata, task, coords):
    clock = make_clock(adata.obs.time_value.unique(), "rank")
    return build_task_data(coords, adata.obs, task, clock, min_cells=int(PROTOCOL.get("min_cells_per_group", 1)))


# ----------------------------------------------------------------------------------------------
# Matched paired-v2 evaluation draws (identical random streams to the harness)
# ----------------------------------------------------------------------------------------------
def draw_plan(task: TaskSpec, pair, n_source: int, n_target: int, draw_index: int):
    draw_seed = derive_seed("paired-draw-v2", CONTRACT["draw_seed"], task.dataset_id, task.kind, task.family,
                            pair.individual, pair.source_time, pair.target_time, draw_index)
    count = min(CONTRACT["sample_size"], n_source, n_target)
    selected = np.sort(np.random.default_rng([draw_seed, 1]).choice(n_source, count, replace=False))
    target = np.sort(np.random.default_rng([draw_seed, 2]).choice(n_target, count, replace=False))
    return draw_seed, selected, target


# ----------------------------------------------------------------------------------------------
# Additional predictors (not in the harness)
# ----------------------------------------------------------------------------------------------
def alpha_of(clock, pair):
    return (clock[pair.target_time] - clock[pair.source_time]) / (clock[pair.anchor_time] - clock[pair.source_time])


def anchor_of(data, pair):
    return np.asarray(data.train[(pair.individual, float(pair.anchor_time))], dtype=np.float64)


def independent_bridge(source, anchor, alpha, rng):
    return (1 - alpha) * source + alpha * anchor[rng.integers(0, len(anchor), len(source))]


def exact_ot_plan(x, y):
    import ot
    return ot.emd(np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)), ot.dist(x, y), numItermax=10_000_000)


def mccann_sample(source, anchor, plan, alpha, rng):
    """Exact McCann interpolant: pair each source atom with an anchor atom drawn from its plan row."""
    rows = plan / plan.sum(1, keepdims=True)
    cdf = np.cumsum(rows, 1)
    u = rng.random(len(source))[:, None]
    j = np.minimum((cdf < u).sum(1), len(anchor) - 1)
    return (1 - alpha) * source + alpha * anchor[j]


def mccann_barycentric(source, anchor, plan, alpha):
    mapped = plan @ anchor / plan.sum(1, keepdims=True)
    return source + alpha * (mapped - source)


def rk4_to(method, source, s0, s1):
    """Integrate a fitted harness flow from rank time s0 to an arbitrary s1 (harness RK4 step rule)."""
    import torch
    p = {**FLOW_DEFAULTS, **method.params}
    steps = max(int(math.ceil(p["rk4_steps_per_interval"] * abs(s1 - s0) / method.interval - 1e-9)), 1)
    if s1 == s0:
        return np.asarray(source, dtype=np.float32).copy()
    dt = (s1 - s0) / steps
    z = torch.as_tensor(np.asarray(source, dtype=np.float32), device=method.device)
    with torch.no_grad():
        def v(state, t):
            return method.field(state, torch.full((len(state),), float(t), device=method.device), None)
        for k in range(steps):
            s = s0 + k * dt
            k1 = v(z, s)
            k2 = v(z + 0.5 * dt * k1, s + 0.5 * dt)
            k3 = v(z + 0.5 * dt * k2, s + 0.5 * dt)
            k4 = v(z + dt * k3, s + dt)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z.cpu().numpy()


def total_variance(x):
    return float(np.asarray(x, dtype=np.float64).var(0).sum())


def weighted_energy(x, wx, y, wy):
    from scipy.spatial.distance import cdist
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(2 * wx @ cdist(x, y) @ wy - wx @ cdist(x, x) @ wx - wy @ cdist(y, y) @ wy)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=float, allow_nan=True) + "\n")


def summarize_log2(frame, comparator, keys=("task", "individual", "source_time", "target_time")):
    """Harness estimand: log2(method/comparator) per matched draw -> mean over draws within fit ->
    mean over seeds within donor. Deterministic comparators are broadcast over fit seeds."""
    keys = list(keys)
    comp = frame[frame.method == comparator].groupby(keys + ["draw"]).energy.mean().rename("comp")
    m = frame.merge(comp.reset_index(), on=keys + ["draw"], how="left")
    m["log2"] = np.log2(m.energy / m.comp)
    per_fit = m.groupby(keys + ["method", "seed"]).agg(log2=("log2", "mean"), energy=("energy", "mean"),
                                                        frac_lower=("log2", lambda v: float((v < 0).mean()))).reset_index()
    out = per_fit.groupby(keys + ["method"]).agg(
        log2_ratio=("log2", "mean"), seed_min=("log2", "min"), seed_max=("log2", "max"),
        energy_mean=("energy", "mean"), frac_draws_lower=("frac_lower", "mean"), seeds=("seed", "nunique")).reset_index()
    out["comparator"] = comparator
    return out


# ----------------------------------------------------------------------------------------------
# Inner leave-one-training-donor-out folds (nested selection; E3b and E6)
# ----------------------------------------------------------------------------------------------
def inner_tasks() -> dict:
    """For outer held donor H and training donors {A, B}: train on B only, evaluate A (H excluded everywhere)."""
    outer = harness_tasks()
    out = {}
    for h in DONORS:
        base = outer[f"LIO_{h}"]
        for a in DONORS:
            if a == h:
                continue
            pairs = tuple(type(p)(a, p.source_time, p.target_time, None) for p in base.pairs)
            out[f"INNER_{h}_eval{a}"] = TaskSpec(base.dataset_id, "T-LIO", "inner_lodo", None, a, pairs, substitute=h)
    return out


def all_tasks() -> dict:
    return {**harness_tasks(), **inner_tasks()}


COORD_DIR = OUT / "coords"


def cached_coords(rep: str, task_name: str) -> np.ndarray:
    return np.load(COORD_DIR / rep / f"{task_name}.npy")


ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
