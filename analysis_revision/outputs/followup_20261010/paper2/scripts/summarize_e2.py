"""E2: representation invariance -- sign agreement of donor-level log2 ratios across representations."""
import numpy as np
import pandas as pd

from p2common import OUT, summarize_log2

REPS = ["mu_std", "mu_raw", "mu_white", "mu_clr_std", "pca30"]
METHODS = ["identity", "location_scale", "mean_shift", "cfm_independent", "cfm_ot", "ot_mccann_exact",
           "independent_bridge", "target_marginal", "ot_exact_knn_transfer"]
keys = ["task", "individual", "source_time", "target_time"]
tables = []
for rep in REPS:
    d = pd.read_csv(OUT / f"eval_{rep}_draws.csv.gz")
    d = d[d.method.isin(METHODS)]
    mt = summarize_log2(d[d.task.str.startswith("MT")], "mean_shift", keys)
    lio = d[d.task.str.startswith("LIO") & (d.source_time == 1.0)]
    tables += [mt.assign(rep=rep), summarize_log2(lio, "target_marginal", keys).assign(rep=rep),
               summarize_log2(lio, "mean_shift", keys).assign(rep=rep)]
t = pd.concat(tables)
t["task_group"] = np.where(t.task.str.startswith("LIO"), "LIO_1to7", t.task)
t.to_csv(OUT / "e2_log2_by_rep.csv", index=False)
wide = t.pivot_table(index=["task_group", "comparator", "method", "individual"], columns="rep", values="log2_ratio")
wide = wide[REPS]
sgn = np.sign(wide.round(6))
wide["n_reps"] = wide[REPS].notna().sum(1)
wide["n_negative"] = (wide[REPS] < 0).sum(1)
wide["n_same_sign_as_mu_std"] = (sgn[REPS].eq(sgn["mu_std"], axis=0)).sum(1)
wide["majority_agreement"] = np.maximum(wide.n_negative, wide.n_reps - wide.n_negative)
wide = wide[~((wide[REPS].abs() < 1e-12).all(1))]  # drop comparator vs itself
wide.reset_index().to_csv(OUT / "e2_sign_agreement.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 300):
    print(wide.round(3).to_string())
macro = t.groupby(["task_group", "comparator", "method", "rep"]).log2_ratio.mean().unstack("rep")[REPS]
macro.reset_index().to_csv(OUT / "e2_macro_log2_by_rep.csv", index=False)
with pd.option_context("display.width", 250):
    print(macro.round(3).to_string())
