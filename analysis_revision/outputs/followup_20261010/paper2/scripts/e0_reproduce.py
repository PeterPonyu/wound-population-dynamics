"""E0: reproduce published numbers before any new analysis.

(a) Table S25 full-cell donor-averaged energy distance for global day-7 holdout:
    unchanged source 1.4725, centroid translation 0.6674 (also day-1: 1.3460, 0.9293).
(b) Harness paired-v2 matched draws: re-derive identity / mean_shift / location_scale /
    target_marginal energies for every task, pair and draw and compare with the archived
    scalar_records.csv.gz of output/local-p2-20261009 (via publication-quality-20261009/p2).
"""
import numpy as np
import pandas as pd

from p2common import (OUT, ROOT, create, draw_plan, energy_distance, group_rows, harness_tasks, load_adata,
                      method_inputs, task_coords, task_data, write_json, PROTOCOL)
from expansion.population.worker import _method_params  # noqa: E402

sys_out = {}
# (a) Table S25 estimand with the manuscript code path (core_engines.gse241132.fold_standardize)
from core_engines.gse241132 import COND_ORDER, COND_TIME, fold_standardize  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('ebe', ROOT / 'scripts/expand_biological_evidence.py')
_ebe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ebe)
energy = _ebe.energy

mu = np.load(ROOT / "outputs/repaired_20260918/human_temporal_flow/mu.npy")
obs = pd.read_csv(ROOT / "outputs/repaired_20260918/human_temporal_flow/obs_fibroblast.csv")
cond, donor = obs.cond.to_numpy(), obs.donor.to_numpy()
table = {}
for hold in ["Wound7", "Wound1"]:
    z, _ = fold_standardize(mu, cond != hold)
    i = COND_ORDER.index(hold); src, anc = COND_ORDER[i - 1], COND_ORDER[i + 1]
    alpha = (COND_TIME[hold] - COND_TIME[src]) / (COND_TIME[anc] - COND_TIME[src])
    un, tr = [], []
    for d in sorted(set(donor)):
        a = z[(cond == src) & (donor == d)]; b = z[(cond == anc) & (donor == d)]; t = z[(cond == hold) & (donor == d)]
        un.append(energy(a, t)); tr.append(energy(a + alpha * (b.mean(0) - a.mean(0)), t))
    table[hold] = {"unchanged_source": float(np.mean(un)), "translation": float(np.mean(tr)),
                   "per_donor_unchanged": un, "per_donor_translation": tr}
print(table)
assert abs(table["Wound7"]["unchanged_source"] - 1.4725) < 5e-5 and abs(table["Wound7"]["translation"] - 0.6674) < 5e-5
sys_out["table_s25_full_cell"] = table

# (b) harness matched-draw reproduction
archived = pd.read_csv(ROOT / "output/publication-quality-20261009/p2/scalar_records.csv.gz")
archived = archived[archived.family.isin(["missing_time", "leave_individual"]) & (archived.source_variant == "own")
                    & archived.method.isin(["identity", "mean_shift", "location_scale", "target_marginal"])]
adata = load_adata()
rows = []
for name, task in harness_tasks().items():
    coords, _ = task_coords(adata, task, "mu_std")
    data = task_data(adata, task, coords)
    methods = ["identity", "mean_shift", "location_scale"] + (["target_marginal"] if task.kind == "T-LIO" else [])
    fitted = {}
    for m in methods:
        meth = create(m, **_method_params(PROTOCOL, m)); meth.fit(method_inputs(data), seed=0, device="cpu"); fitted[m] = meth
    for pi, pair in enumerate(task.pairs):
        src_rows = group_rows(adata.obs, pair.individual, pair.source_time)
        tgt_rows = group_rows(adata.obs, pair.individual, pair.target_time)
        call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time,
                    anchor_time=pair.anchor_time)
        full = {m: fitted[m].predict(coords[src_rows], **call) for m in methods if m != "target_marginal"}
        for draw in range(50):
            seed, sel, tsel = draw_plan(task, pair, len(src_rows), len(tgt_rows), draw)
            target = coords[tgt_rows[tsel]]
            for m in methods:
                pred = (fitted[m].predict(coords[src_rows][sel], **call | {}, seed=seed) if m == "target_marginal"
                        else full[m][sel])
                rows.append(dict(kind=task.kind, method=m, individual=pair.individual, source_time=pair.source_time,
                                 target_time=pair.target_time, draw_index=draw, energy_new=energy_distance(pred, target)))
new = pd.DataFrame(rows)
merged = new.merge(archived, on=["kind", "method", "individual", "source_time", "target_time", "draw_index"],
                   how="left", validate="one_to_one")
diff = (merged.energy_new - merged.energy).abs()
print("matched rows", len(merged), "missing", int(merged.energy.isna().sum()), "max abs diff", float(diff.max()))
sys_out["harness_reproduction"] = {"rows": len(merged), "missing": int(merged.energy.isna().sum()),
                                   "max_abs_energy_difference": float(diff.max()),
                                   "methods": sorted(new.method.unique().tolist())}
write_json(OUT / "e0_reproduction.json", sys_out)
