"""Summaries for E1 (coupling estimand) from eval_mu_std_* outputs, plus the ideal-marginal check."""
import json

import numpy as np
import pandas as pd

from p2common import (OUT, alpha_of, anchor_of, cached_coords, energy_distance, group_rows, harness_tasks,
                      independent_bridge, load_adata, summarize_log2, task_data, total_variance, weighted_energy)

REP = "mu_std"
draws = pd.read_csv(OUT / f"eval_{REP}_draws.csv.gz")
full = pd.read_csv(OUT / f"eval_{REP}_fullcell.csv")
keys = ["task", "individual", "source_time", "target_time"]
out = []
for comp in ["mean_shift", "target_marginal"]:
    sub = draws[draws.task.str.startswith("LIO")] if comp == "target_marginal" else draws
    out.append(summarize_log2(sub, comp, keys))
table = pd.concat(out)
geo = full.groupby(keys + ["method"])[["energy_full", "variance_ratio", "centroid_error", "centered_energy"]].mean().reset_index()
table = table.merge(geo, on=keys + ["method"], how="left")
table.to_csv(OUT / "e1_donor_log2_ratios.csv", index=False)

primary = table[((table.task.str.startswith("MT")) & (table.comparator == "mean_shift"))
                | ((table.task.str.startswith("LIO")) & (table.source_time == 1.0))]
primary = primary.assign(task_group=np.where(primary.task.str.startswith("LIO"), "LIO_1to7", primary.task))
macro = primary.groupby(["task_group", "comparator", "method"]).agg(
    macro_log2=("log2_ratio", "mean"), donors_log2_le0=("log2_ratio", lambda v: int((v <= 0).sum())),
    min_donor=("log2_ratio", "min"), max_donor=("log2_ratio", "max"), macro_energy_draw=("energy_mean", "mean"),
    macro_energy_full=("energy_full", "mean"), macro_variance_ratio=("variance_ratio", "mean")).reset_index()
macro.to_csv(OUT / "e1_macro_summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 500):
    print(macro.round(4).to_string())

# success criterion stated by the reviewer: OT-CFM log2 <= 0 in >= 2/3 donors for both held days
crit = {}
for t in ["MT_day7", "MT_day1"]:
    for m in ["cfm_ot", "cfm_sb", "cfm_independent", "ot_mccann_exact"]:
        g = primary[(primary.task == t) & (primary.method == m) & (primary.comparator == "mean_shift")]
        crit[f"{t}|{m}"] = {"donor_log2": dict(zip(g.individual, g.log2_ratio.round(4))),
                            "donors_le0": int((g.log2_ratio <= 0).sum())}

# ---- ideal independent-CFM marginal check (global missing-time tasks) ----
adata = load_adata()
ideal = []
for task_name in ["MT_day7", "MT_day1"]:
    task = harness_tasks()[task_name]
    coords = cached_coords(REP, task_name)
    data = task_data(adata, task, coords)
    fits = {}
    for path in sorted((OUT / "fits" / REP / task_name).glob("*_s[0-9].npz")):
        f = np.load(path); meta = json.loads(str(f["meta"]))
        fits[(meta["tag"], meta["seed"])] = f
    pooled = {}
    rng = np.random.default_rng(7)
    for pi, pair in enumerate(task.pairs):
        src = coords[group_rows(adata.obs, pair.individual, pair.source_time)].astype(float)
        tgt = coords[group_rows(adata.obs, pair.individual, pair.target_time)].astype(float)
        alpha, anchor = alpha_of(data.clock, pair), anchor_of(data, pair)
        bridge = independent_bridge(src, anchor, alpha, np.random.default_rng([0, 41]))
        trans = src + alpha * (anchor.mean(0) - src.mean(0))
        sub = rng.choice(len(src), min(250, len(src)), replace=False)
        pooled.setdefault("bridge", []).append(bridge[sub]); pooled.setdefault("translation", []).append(trans[sub])
        pooled.setdefault("target", []).append(tgt[rng.choice(len(tgt), min(250, len(tgt)), replace=False)])
        for (tag, seed), f in fits.items():
            p = f[f"pred{pi}"].astype(float)
            pooled.setdefault((tag, seed), []).append(p[sub])
            ideal.append(dict(task=task_name, individual=pair.individual, method=tag, seed=seed, scope="donor",
                              ed_to_bridge=energy_distance(p, bridge), ed_to_translation=energy_distance(p, trans),
                              ed_to_target=energy_distance(p, tgt),
                              var_ratio_to_bridge=total_variance(p) / total_variance(bridge),
                              var_ratio_to_target=total_variance(p) / total_variance(tgt)))
        ideal.append(dict(task=task_name, individual=pair.individual, method="translation", seed=0, scope="donor",
                          ed_to_bridge=energy_distance(trans, bridge), ed_to_translation=0.0,
                          ed_to_target=energy_distance(trans, tgt),
                          var_ratio_to_bridge=total_variance(trans) / total_variance(bridge),
                          var_ratio_to_target=total_variance(trans) / total_variance(tgt)))
        ideal.append(dict(task=task_name, individual=pair.individual, method="independent_bridge", seed=0, scope="donor",
                          ed_to_bridge=0.0, ed_to_translation=energy_distance(bridge, trans),
                          ed_to_target=energy_distance(bridge, tgt), var_ratio_to_bridge=1.0,
                          var_ratio_to_target=total_variance(bridge) / total_variance(tgt)))
    # donor-balanced pooled measures: the population-level target of an ideal independent-coupling CFM is the
    # donor-balanced mixture of bridges, when started from the donor-balanced source mixture.
    def pool(key):
        parts = pooled[key]
        return np.vstack(parts), np.concatenate([np.full(len(q), 1 / (len(parts) * len(q))) for q in parts])
    B, wb = pool("bridge")
    for key in [k for k in pooled if k != "bridge"]:
        X, wx = pool(key)
        T, wt = pool("target")
        ideal.append(dict(task=task_name, individual="pooled_donor_balanced", method=key if isinstance(key, str) else key[0],
                          seed=0 if isinstance(key, str) else key[1], scope="pooled",
                          ed_to_bridge=weighted_energy(X, wx, B, wb), ed_to_target=weighted_energy(X, wx, T, wt)))
ideal = pd.DataFrame(ideal)
ideal.to_csv(OUT / "e1_ideal_marginal_check.csv", index=False)
ideal_s = ideal.groupby(["task", "scope", "method"])[["ed_to_bridge", "ed_to_translation", "ed_to_target",
                                                       "var_ratio_to_bridge", "var_ratio_to_target"]].mean().reset_index()
ideal_s.to_csv(OUT / "e1_ideal_marginal_summary.csv", index=False)
with pd.option_context("display.width", 250):
    print(ideal_s.round(4).to_string())
json.dump({"reviewer_success_criterion": crit}, open(OUT / "e1_criterion.json", "w"), indent=1)
print(json.dumps(crit, indent=1))
