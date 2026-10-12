"""E7: expression-space evaluation by kNN decoding (k = 15 nearest *permitted training* cells).

Each predicted particle (full-source prediction, frozen mu standardized coordinates of the task) is mapped
to the mean common-panel expression profile (5,383 genes present in all assay catalogues; per-cell
proportions, as in the manuscript's observable audit) of its 15 nearest permitted training cells.
Scores: TV and JS (natural log) between predicted and observed target mean profiles; Spearman correlation
of predicted vs observed source->target log2 fold changes on (i) 2,000 training-HVGs within the panel
and (ii) a 50-gene signature (25 up / 25 down) from permitted training cells only
(held-donor: training donors' source->target change; missing time: source->anchor change).
Reference: kNN-decoding of the observed target cells themselves (decoder floor).
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from p2common import (OUT, PROTOCOL, ROOT, cached_coords, create, group_rows, harness_tasks, load_adata,
                      method_inputs, task_data)
from expansion.population.worker import _method_params  # noqa: E402

counts = load_adata(counts=True)
obs = counts.obs
genes = pd.read_csv(ROOT / "outputs/scientific_revision_20260925/population_observable/features.csv").gene.to_numpy()
col = pd.Index(counts.var_names).get_indexer(genes)
assert (col >= 0).all() and len(col) == 5383
X = sp.csr_matrix(counts.X)[:, col].astype(np.float64)
tot = np.asarray(X.sum(1)).ravel()
assert (tot > 0).all()
P = sp.csr_matrix(sp.diags(1 / tot) @ X)  # per-cell proportions on the common panel
logn = P.copy(); logn.data = np.log1p(1e4 * logn.data)
latent_adata = load_adata()


def mean_profile(rows):
    return np.asarray(P[rows].mean(0)).ravel()


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def lfc(a, b, eps=1e-6):
    return np.log2((a + eps) / (b + eps))


rows_out = []
for task_name in ["LIO_PWH26", "LIO_PWH27", "LIO_PWH28", "MT_day7"]:
    task = harness_tasks()[task_name]
    coords = cached_coords("mu_std", task_name)
    data = task_data(latent_adata, task, coords)
    train_rows = np.concatenate([group_rows(obs, *k) for k in sorted(data.train)])
    nn = NearestNeighbors(n_neighbors=15).fit(coords[train_rows])
    # training-only HVGs within the panel
    tr = logn[train_rows]
    var = np.asarray(tr.multiply(tr).mean(0)).ravel() - np.asarray(tr.mean(0)).ravel() ** 2
    hvg = np.argsort(-var)[:2000]
    pair = [p for p in task.pairs if p.source_time == 1.0][0]
    pi = list(task.pairs).index(pair)
    src_rows = group_rows(obs, pair.individual, pair.source_time)
    tgt_rows = group_rows(obs, pair.individual, pair.target_time)
    # signature from permitted training cells only
    if task.kind == "T-LIO":
        trainers = sorted({k[0] for k in data.train})
        s_prof = np.mean([mean_profile(group_rows(obs, d, 1.0)) for d in trainers], 0)
        t_prof = np.mean([mean_profile(group_rows(obs, d, 7.0)) for d in trainers], 0)
    else:
        s_prof = np.mean([mean_profile(group_rows(obs, p.individual, 1.0)) for p in task.pairs], 0)
        t_prof = np.mean([mean_profile(group_rows(obs, p.individual, 30.0)) for p in task.pairs], 0)
    expressed = np.flatnonzero((s_prof + t_prof) > 1e-5)
    ch = lfc(t_prof, s_prof)[expressed]
    signature = np.r_[expressed[np.argsort(-ch)[:25]], expressed[np.argsort(ch)[:25]]]
    src_obs, tgt_obs = mean_profile(src_rows), mean_profile(tgt_rows)
    obs_lfc = lfc(tgt_obs, src_obs)

    def decode(points):
        _, idx = nn.kneighbors(np.asarray(points, dtype=np.float32))
        flat = train_rows[idx.ravel()]
        return np.asarray(P[flat].mean(0)).ravel()  # mean over particles of mean over neighbours

    preds = {}
    call = dict(individual=pair.individual, source_time=pair.source_time, target_time=pair.target_time,
                anchor_time=pair.anchor_time)
    names = ["identity", "mean_shift", "location_scale"] + (["target_marginal"] if task.kind == "T-LIO" else [])
    for m in names:
        meth = create(m, **_method_params(PROTOCOL, m)); meth.fit(method_inputs(data), seed=0)
        preds[(m, 0)] = meth.predict(coords[src_rows], **call, **({"seed": 0} if m == "target_marginal" else {}))
    for path in sorted((OUT / "fits/mu_std" / task_name).glob("*_s[0-9].npz")):
        f = np.load(path); meta = json.loads(str(f["meta"]))
        preds[(meta["tag"], meta["seed"])] = f[f"pred{pi}"]
    preds[("observed_target_knn_decoded_floor", 0)] = coords[tgt_rows]
    for (m, seed), p in preds.items():
        prof = decode(p)
        pl = lfc(prof, src_obs)
        rows_out.append(dict(task=task_name, individual=pair.individual, method=m, seed=seed,
                             tv=tv(prof, tgt_obs), js=js(prof, tgt_obs),
                             spearman_lfc_hvg=float(spearmanr(pl[hvg], obs_lfc[hvg])[0]),
                             spearman_lfc_signature=float(spearmanr(pl[signature], obs_lfc[signature])[0])))
    rows_out.append(dict(task=task_name, individual=pair.individual, method="observed_source_persistence_raw", seed=0,
                         tv=tv(src_obs, tgt_obs), js=js(src_obs, tgt_obs), spearman_lfc_hvg=np.nan,
                         spearman_lfc_signature=np.nan))
    print(task_name, flush=True)
frame = pd.DataFrame(rows_out)
frame.to_csv(OUT / "e7_knn_decode_by_fit.csv", index=False)
summ = frame.groupby(["task", "method"])[["tv", "js", "spearman_lfc_hvg", "spearman_lfc_signature"]].mean().reset_index()
summ.to_csv(OUT / "e7_knn_decode_summary.csv", index=False)
lio = summ[summ.task.str.startswith("LIO")].groupby("method")[["tv", "js", "spearman_lfc_hvg", "spearman_lfc_signature"]].mean()
lio.to_csv(OUT / "e7_knn_decode_heldDonor_macro.csv")
with pd.option_context("display.width", 200):
    print(summ.round(4).to_string()); print(lio.round(4).to_string())
