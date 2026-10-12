"""E6: held-source covariate shift and a nested shrinkage predictor (held-donor task, frozen mu standardized).

(a) OOD score of each held donor's day-1 source relative to the permitted training day-1 cells:
    energy distance (donor-balanced weights) and mean 15-NN distance, both in the fold's training-only
    standardization; compared with the real-minus-substituted gaps of the two existing designs:
      A. harness T-SRC (fields trained on the single remaining donor; own and substitute both held out):
         log2(ED_substituted / ED_own), output/publication-quality-20261009/p2/source_substitution.csv
      B. mixture audit (two-donor fields), outputs/scientific_revision_20260923/population_mixture/source_per_donor.csv
(b) Shrinkage predictor: mixture w * transported held source + (1 - w) * training-donor target marginal.
    w is chosen on a 0.05 grid by inner leave-one-training-donor-out (train on one training donor,
    evaluate the other; the outer held donor never enters), then applied once to the outer fold.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from p2common import (DONORS, OUT, PROTOCOL, ROOT, cached_coords, create, draw_plan, group_rows, harness_tasks,
                      inner_tasks, load_adata, method_inputs, task_data, weighted_energy, write_json)
from expansion.population.represent import fit_standardizer  # noqa: E402
from expansion.population.worker import _method_params  # noqa: E402

adata = load_adata()
obs = adata.obs
latent = adata.obsm["X_frozen_mu"]
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def knn_mean(x, ref, k=15):
    d, _ = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(x)
    return float(d.mean())


def balanced_energy(x, refs):
    y = np.vstack(refs); wy = np.concatenate([np.full(len(r), 1 / (len(refs) * len(r))) for r in refs])
    return weighted_energy(x, np.full(len(x), 1 / len(x)), y, wy)


# ---------------- (a) OOD scores ----------------
ood = []
day1 = {d: group_rows(obs, d, 1.0) for d in DONORS}
for h in DONORS:
    train = [d for d in DONORS if d != h]
    # design B geometry: training-only scaler of the two remaining donors (harness LIO fold)
    z = cached_coords("mu_std", f"LIO_{h}").astype(float)
    src = z[day1[h]]; refs = [z[day1[d]] for d in train]
    ref = np.vstack(refs)
    cross = np.mean([knn_mean(z[day1[a]], z[day1[b]]) for a in train for b in train if a != b])
    ood.append(dict(held=h, design="B_two_donor_field", training=",".join(train),
                    energy_to_training_day1=balanced_energy(src, refs), knn15_to_training_day1=knn_mean(src, ref),
                    knn15_reference_between_training_donors=cross))
    # design A geometry: each single remaining donor r trains the field; substitute s is the other held donor
    for s in train:
        r = [d for d in DONORS if d not in (h, s)][0]
        mask = (obs.individual.astype(str) == r).to_numpy()
        za = fit_standardizer(latent[mask]).transform_latent(latent).astype(float)
        ood.append(dict(held=h, design="A_single_donor_field", training=r, substitute=s,
                        energy_own_to_training_day1=balanced_energy(za[day1[h]], [za[day1[r]]]),
                        energy_sub_to_training_day1=balanced_energy(za[day1[s]], [za[day1[r]]]),
                        knn15_own=knn_mean(za[day1[h]], za[day1[r]]), knn15_sub=knn_mean(za[day1[s]], za[day1[r]])))
ood = pd.DataFrame(ood)

swap = pd.read_csv(ROOT / "output/publication-quality-20261009/p2/source_substitution.csv")
swap = swap[swap.source_time == 1.0]
swap["substitute"] = swap.source_variant.str.replace("substitute:", "")
gapA = swap.groupby(["method", "individual", "substitute"]).log2_ratio.mean().reset_index()
mixture = pd.read_csv(ROOT / "outputs/scientific_revision_20260923/population_mixture/source_per_donor.csv")
rowsA = []
for _, r in ood[ood.design == "A_single_donor_field"].iterrows():
    g = gapA[(gapA.individual == r.held) & (gapA.substitute == r.substitute)]
    rowsA.append({**r.dropna().to_dict(), **{f"log2_sub_over_own_{m}": float(v) for m, v in zip(g.method, g.log2_ratio)},
                  "ood_difference_energy_own_minus_sub": r.energy_own_to_training_day1 - r.energy_sub_to_training_day1})
designA = pd.DataFrame(rowsA)
designB = ood[ood.design == "B_two_donor_field"].dropna(axis=1).merge(
    mixture[["held_donor", "real_ed", "individual_wrong_mean_ed", "individual_wrong_minus_real", "mixed_wrong_minus_real"]],
    left_on="held", right_on="held_donor")
designA.to_csv(OUT / "e6_ood_designA_single_donor_fields.csv", index=False)
designB.to_csv(OUT / "e6_ood_designB_two_donor_fields.csv", index=False)
print(designB.round(4).to_string()); print(designA.round(4).to_string())
pairwise = {f"{a}|{b}": balanced_energy(cached_coords("mu_std", "MT_day7").astype(float)[day1[a]],
                                         [cached_coords("mu_std", "MT_day7").astype(float)[day1[b]]])
            for i, a in enumerate(DONORS) for b in DONORS[i + 1:]}
print("pairwise day-1 energy (MT_day7 scaler):", pairwise)


# ---------------- (b) nested shrinkage ----------------
def transported_and_marginal(task_name, task, pi, pair, transport, seed=None):
    coords = cached_coords("mu_std", task_name)
    data = task_data(adata, task, coords)
    src_rows = group_rows(obs, pair.individual, pair.source_time)
    tgt_rows = group_rows(obs, pair.individual, pair.target_time)
    call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time, anchor_time=None)
    if transport == "mean_shift":
        m = create("mean_shift"); m.fit(method_inputs(data), seed=0)
        pred = m.predict(coords[src_rows], **call)
    else:
        f = np.load(OUT / "fits/mu_std" / task_name / f"cfm_independent_s{seed}.npz")
        pred = f[f"pred{pi}"]
    tm = create("target_marginal"); tm.fit(method_inputs(data), seed=0)
    out = []
    for d in range(50):
        dseed, sel, tsel = draw_plan(task, pair, len(src_rows), len(tgt_rows), d)
        marg = tm.predict(coords[src_rows][sel], **call, seed=dseed)
        out.append((pred[sel].astype(float), marg.astype(float), coords[tgt_rows][tsel].astype(float)))
    return out


def curve(samples):
    """Mean over draws of the weighted-mixture energy for every w in W (exact weighted V-statistic)."""
    vals = np.zeros(len(W))
    for p, m, t in samples:
        x = np.vstack([p, m]); n = len(p)
        dxx, dxt, dtt = cdist(x, x), cdist(x, t), cdist(t, t)
        wt = np.full(len(t), 1 / len(t))
        for i, w in enumerate(W):
            wx = np.r_[np.full(n, w / n), np.full(len(m), (1 - w) / len(m))]
            vals[i] += 2 * wx @ dxt @ wt - wx @ dxx @ wx - wt @ dtt @ wt
    return vals / len(samples)


outer_tasks, inner = harness_tasks(), inner_tasks()
rows = []
for transport, seeds in [("mean_shift", [None]), ("cfm_independent", [0, 1, 2])]:
    for h in DONORS:
        for pi in range(3):
            # inner selection (cfm inner fields: seed 0 only)
            inner_curves = []
            for a in [d for d in DONORS if d != h]:
                name = f"INNER_{h}_eval{a}"
                task = inner[name]
                inner_curves.append(curve(transported_and_marginal(name, task, pi, task.pairs[pi], transport,
                                                                   seed=None if transport == "mean_shift" else 0)))
            w_star = float(W[int(np.argmin(np.mean(inner_curves, 0)))])
            task = outer_tasks[f"LIO_{h}"]
            for seed in seeds:
                c = curve(transported_and_marginal(f"LIO_{h}", task, pi, task.pairs[pi], transport, seed))
                rows.append(dict(transport=transport, held=h, source_time=task.pairs[pi].source_time,
                                 target_time=task.pairs[pi].target_time, seed=seed, w_selected=w_star,
                                 energy_selected=float(np.interp(w_star, W, c)), energy_marginal=float(c[0]),
                                 energy_transport=float(c[-1]), w_oracle=float(W[int(np.argmin(c))]),
                                 energy_oracle=float(c.min()), inner_curve=json.dumps(np.mean(inner_curves, 0).round(5).tolist())))
            print(transport, h, pi, w_star, flush=True)
shrink = pd.DataFrame(rows)
# note: these are log2 of draw-averaged energies (ratio of means), not the harness mean of per-draw log ratios
shrink["log2_selected_vs_marginal"] = np.log2(shrink.energy_selected / shrink.energy_marginal)
shrink["log2_transport_vs_marginal"] = np.log2(shrink.energy_transport / shrink.energy_marginal)
shrink.to_csv(OUT / "e6_shrinkage.csv", index=False)
summ = shrink.groupby(["transport", "held", "source_time", "target_time"]).agg(
    w_selected=("w_selected", "first"), w_oracle=("w_oracle", "mean"),
    log2_selected_vs_marginal=("log2_selected_vs_marginal", "mean"),
    log2_transport_vs_marginal=("log2_transport_vs_marginal", "mean")).reset_index()
summ.to_csv(OUT / "e6_shrinkage_summary.csv", index=False)
print(summ.round(4).to_string())
write_json(OUT / "e6_summary.json", {"pairwise_day1_energy_MTday7_scaler": pairwise,
                                     "designB": designB.to_dict("records"), "shrinkage": summ.to_dict("records")})
