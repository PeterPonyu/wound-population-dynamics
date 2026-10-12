"""(i) Held-donor shared CFM vs target marginal for ALL intervals (Table S2 source: harness paired_ratios.csv).
(ii) GSE186527 admitted population identity (protocol + admission groups + marker check on the admitted cells)."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from p2common import OUT, ROOT, write_json

ratios = pd.read_csv(ROOT / "output/publication-quality-20261009/p2/paired_ratios.csv")
x = ratios[(ratios.kind == "T-LIO") & (ratios.family == "leave_individual") & (ratios.metric == "energy")
           & (ratios.comparator == "target_marginal")]
keep = ["method", "individual", "source_time", "target_time", "log2_ratio", "seed_min", "seed_max",
        "fraction_paired_draws_lower"]
x = x[keep].sort_values(["method", "individual", "source_time"])
x.to_csv(OUT / "heldDonor_all_intervals_vs_target_marginal.csv", index=False)
summary = {}
for m, g in x.groupby("method"):
    if m == "target_marginal":
        continue
    summary[m] = {"cells": len(g), "wins_log2_lt_0": int((g.log2_ratio < 0).sum()),
                  "wins_all_seeds_lt_0": int((g.seed_max < 0).sum()),
                  "ties_abs_lt_0.05": int((g.log2_ratio.abs() < 0.05).sum()),
                  "mean_log2": float(g.log2_ratio.mean()),
                  "interval_1_7_mean_log2": float(g[g.source_time == 1].log2_ratio.mean()),
                  "win_cells": g[g.log2_ratio < 0][["individual", "source_time", "target_time", "log2_ratio"]].to_dict("records")}
print(json.dumps(summary, indent=1))

base = ROOT / "output/external-gse186527-20261009"
protocol = json.loads((ROOT / "config/external_gse186527_protocol_20261009.json").read_text())
groups = pd.read_csv(base / "admission/groups.csv")
report = json.loads((ROOT / "output/external-gse186527-20261010-replay-r1/report.json").read_text())
a = ad.read_h5ad(base / "admission/standard_Linpos.h5ad")
X = sp.csr_matrix(a.X); names = list(map(str, a.var_names))
markers = {}
for g in ["Col1a1", "Col1a2", "Pdgfra", "Dcn", "Ptprc", "Lyz2", "Cd74", "Cd3e", "Adgre1", "Pecam1", "Cdh5", "Krt14"]:
    if g in names:
        markers[g] = float((np.asarray(X[:, names.index(g)].todense()).ravel() > 0).mean())
gse = {"protocol_gate_definitions": protocol["unit_definitions"]["gate"],
       "protocol_analyses": protocol["tasks"]["analyses"],
       "admission_groups": groups.to_dict("records"),
       "analysed": {k: v.get("status") for k, v in report["analyses"].items()},
       "lin_minus_reason": report["analyses"]["Lin-"],
       "lin_plus_admitted_arms": report["analyses"]["Lin+"].get("arms"),
       "admitted_cells": int(a.n_obs), "admitted_gate_values": sorted(a.obs.gate.astype(str).unique().tolist()),
       "fraction_cells_detecting_marker_in_admitted_population": markers}
print(json.dumps({k: v for k, v in gse.items() if k != "admission_groups"}, indent=1))
write_json(OUT / "existing_tables_summary.json", {"held_donor_all_intervals": summary, "gse186527": gse})
