#!/usr/bin/env python3
"""POP05: frozen population predictions versus actual held-day7 expression.

Prepare first, inspect the locked protocol, then run without changing it.
All computations use CPU; only the existing checkpoint verification uses torch.
No manuscript, original research artifact, network or model-training writes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys
import zipfile
for _thread_variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[_thread_variable] = '4'
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wound_models.data_adapter import _read_10x_triplet, library_normalize
DEFAULT_PROTOCOL = ROOT / 'config/population_observable_protocol.json'
OUTPUT = ROOT / 'outputs/scientific_revision_20260925/population_observable'
METRICS = ['total_variation', 'js_divergence_nats', 'gene_rmse']
METHODS = ['shared_cfm_decoded', 'persistence_decoded', 'mean_displacement_decoded', 'training_target_marginal_decoded_exact', 'persistence_observed', 'training_target_marginal_observed_exact']
PREPARATION_FILES = frozenset({'cell_identity.csv', 'features.csv', 'input_sha256.json', 'model_provenance.json', 'observed_common_counts.npz', 'panel_coverage.csv', 'preparation_started.json', 'prepared.json', 'projection_verification.json', 'protocol.json', 'raw_sample_audit.csv', 'split_identity_rows.npz', 'split_verification.json', 'training_signatures.json'})

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

def read_json(path):
    return json.loads(Path(path).read_text())

def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim in (1, 2) and values.size > 0 and np.isfinite(values).all() and (values >= 0).all(), 'Invalid probability array')
    require(np.allclose(values.sum(axis=-1), 1, atol=2e-06, rtol=0), 'Probability mass must sum to one')
    return values

def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), 'Invalid latent coordinates')
    exponential = np.exp(values - values.max(axis=1, keepdims=True))
    return exponential / exponential.sum(axis=1, keepdims=True)

def inverse_scaler(z, mean, scale):
    z, mean, scale = (np.asarray(a, dtype=np.float64) for a in (z, mean, scale))
    require(z.ndim == 2 and mean.shape == scale.shape == (1, z.shape[1]) and np.isfinite(z).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid training scaler')
    return z * scale + mean

def weights_for(n, weights=None):
    require(n > 0, 'Empty population')
    w = np.full(n, 1 / n) if weights is None else np.asarray(weights, dtype=float)
    require(w.shape == (n,) and np.isfinite(w).all() and (w >= 0).all() and np.isclose(w.sum(), 1, rtol=0, atol=1e-12), 'Invalid particle weights')
    return w

def decode_cells(mu, beta, common):
    """Condition each decoded particle AFTER mixing full-panel topic rows."""
    beta = check_probabilities(beta)
    theta = softmax(mu)
    common = np.asarray(common)
    require(beta.ndim == 2 and theta.shape[1] == beta.shape[0] and (common.ndim == 1) and (common.dtype.kind in 'iu') and (len(common) > 0) and (len(np.unique(common)) == len(common)) and (common >= 0).all() and (common < beta.shape[1]).all(), 'Invalid decoder/feature mapping')
    decoded = theta @ beta[:, common]
    retained = decoded.sum(axis=1)
    full_mass = theta @ beta.sum(axis=1)
    require((retained > 0).all(), 'Zero decoded common-panel mass')
    conditional = decoded / retained[:, None]
    return (check_probabilities(conditional), 1 - retained / full_mass)

def normalized_counts(counts):
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    require(counts.ndim == 2 and counts.shape[0] > 0 and (counts.shape[1] > 0) and np.isfinite(counts.data).all() and (counts.data >= 0).all(), 'Invalid counts')
    totals = np.asarray(counts.sum(axis=1)).ravel()
    require((totals > 0).all(), 'Zero common-panel count total; refusing silent exclusion')
    return sparse.csr_matrix(sparse.diags(1 / totals) @ counts)

def mean_profile(probabilities, rows=None, weights=None):
    selected = probabilities if rows is None else probabilities[np.asarray(rows)]
    w = weights_for(selected.shape[0], weights)
    result = np.asarray(w @ selected).ravel()
    return check_probabilities(result)

def profile_metrics(prediction, observed):
    p, q = (check_probabilities(prediction), check_probabilities(observed))
    require(p.ndim == q.ndim == 1 and p.shape == q.shape, 'Profile dimensions disagree')
    middle = (p + q) / 2
    nonzero = middle > 0
    js = 0.5 * (xlogy(p[nonzero], p[nonzero] / middle[nonzero]).sum() + xlogy(q[nonzero], q[nonzero] / middle[nonzero]).sum())
    return dict(total_variation=float(0.5 * np.abs(p - q).sum()), js_divergence_nats=float(js), gene_rmse=float(np.sqrt(np.mean((p - q) ** 2))))

def checked_rows(identity, rows, donors, condition=None):
    rows = np.asarray(rows)
    require(rows.ndim == 1 and rows.dtype.kind in 'iu' and (len(rows) > 0) and (rows >= 0).all() and (rows < len(identity)).all() and (len(np.unique(rows)) == len(rows)), 'Invalid or duplicated latent rows')
    selected = identity.iloc[rows]
    require(selected.donor.isin(donors).all(), 'Disallowed donor in rows')
    if condition is not None:
        require(selected.cond.eq(condition).all(), 'Disallowed time in rows')
    return rows.astype(np.int64, copy=False)

def rows_for(identity, donor, condition):
    rows = np.flatnonzero(identity.donor.eq(donor) & identity.cond.eq(condition))
    return checked_rows(identity, rows, [donor], condition)

def validate_scaler(scaler, mu, identity, permitted, held):
    require(held not in permitted and len(permitted) == len(set(permitted)), 'Leaking training subset')
    rows = checked_rows(identity, scaler['fitting_latent_rows'], permitted)
    require(np.array_equal(rows, np.flatnonzero(identity.donor.isin(permitted))), 'Scaler fitting rows must be exactly the training subset')
    expected_mean = mu[rows].mean(axis=0, keepdims=True)
    expected_scale = mu[rows].std(axis=0, keepdims=True)
    expected_scale[expected_scale == 0] = 1
    require(np.array_equal(scaler['mean'], expected_mean) and np.array_equal(scaler['scale'], expected_scale), 'Scaler does not reproduce from training data')

def exact_marginal(identity, permitted, held):
    require(permitted and held not in permitted and (len(permitted) == len(set(permitted))), 'Target-marginal includes held donor or duplicate donor')
    parts = [rows_for(identity, donor, 'Wound7') for donor in permitted]
    rows = np.concatenate(parts)
    weights = np.concatenate([np.full(len(part), 1 / (len(parts) * len(part))) for part in parts])
    return (rows, weights)

def training_signature(proportions, identity, genes, permitted, held, n=25):
    """Only permitted training rows are accessed; held expression is not used."""
    require(permitted and held not in permitted and (len(permitted) == len(set(permitted))), 'Signature training subset leaks')
    genes = np.asarray(genes, dtype=str)
    require(len(set(genes)) == len(genes) and proportions.shape == (len(identity), len(genes)), 'Signature feature/identity dimensions differ')
    delta = np.mean([mean_profile(proportions, rows_for(identity, donor, 'Wound7')) - mean_profile(proportions, rows_for(identity, donor, 'Wound1')) for donor in permitted], axis=0)
    positive, negative = (np.flatnonzero(delta > 0), np.flatnonzero(delta < 0))
    require(len(positive) >= n and len(negative) >= n, 'Insufficient signed training features')
    up = sorted(positive, key=lambda i: (-delta[i], genes[i]))[:n]
    down = sorted(negative, key=lambda i: (delta[i], genes[i]))[:n]
    return dict(held_donor=held, training_donors=list(permitted), up_indices=[int(i) for i in up], down_indices=[int(i) for i in down], up_genes=genes[up].tolist(), down_genes=genes[down].tolist(), up_training_deltas=delta[up].tolist(), down_training_deltas=delta[down].tolist())

def signature_scores(profile, definition):
    up = float(profile[definition['up_indices']].sum())
    down = float(profile[definition['down_indices']].sum())
    return dict(up_mass=up, down_mass=down, signed_mass=up - down)

def hierarchical_summary(frame, expected_donors=None):
    keys = ['held_donor', 'k', 'training_donors', 'seed', 'target_set', 'method']
    require(not frame.empty and (not frame.duplicated(keys).any()) and np.isfinite(frame[METRICS].to_numpy()).all(), 'Invalid score rows')
    config = frame.groupby([k for k in keys if k != 'seed'], sort=True)[METRICS].mean().reset_index()
    blocks = config.groupby(['held_donor', 'k', 'target_set', 'method'], sort=True)[METRICS].mean().reset_index()
    groups = blocks.groupby(['k', 'target_set', 'method'], sort=True)
    if expected_donors is not None:
        expected = set(expected_donors)
        require(expected and all((set(group.held_donor) == expected for _, group in groups)), 'Incomplete biological donor grid')
    macro = groups[METRICS].mean().reset_index()
    counts = groups.held_donor.nunique().reset_index(name='biological_donors')
    macro = macro.merge(counts, on=['k', 'target_set', 'method'], validate='one_to_one')
    macro['biological_ci_available'] = False
    return (config, blocks, macro)

def runtime_evidence(path, requested):
    result = dict(requested_model=requested, runtime_reported_model=None, runtime_metadata_available=False, verification='unavailable; request alone is not verification')
    if path is not None:
        latest = None
        with Path(path).open() as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get('type') == 'turn_context' and item.get('payload', {}).get('cwd') == str(ROOT):
                    latest = item
        if latest:
            model = latest['payload'].get('model')
            result.update(runtime_reported_model=model, runtime_metadata_available=True, runtime_timestamp_utc=latest['timestamp'], metadata_source=str(Path(path).resolve()), verification='local session turn_context, not just requested parameter', matches_request=model == requested)
            require(model == requested, 'Runtime model differs from requested model')
    return result

def safe_path(base, relative):
    name = Path(relative)
    require(not name.is_absolute() and '..' not in name.parts and (name.as_posix() == relative), 'Unsafe manifest path')
    path = base / name
    require(path.is_file() and (not any((p.is_symlink() for p in [path, *path.parents]))), 'Missing/symlink input')
    return path

def bind_file(path, inputs, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    require(expected is None or value == expected, f'Input hash mismatch: {path}')
    inputs[str(path)] = value
    return value

def bind_population(protocol, inputs):
    pop = ROOT / protocol['population_directory']
    manifest_path = pop / 'output_manifest.json'
    bind_file(manifest_path, inputs)
    manifest = read_json(manifest_path)

    def bind(relative):
        require(relative in manifest, f'Unmanifested population input: {relative}')
        path = safe_path(pop, relative)
        bind_file(path, inputs, manifest[relative])
        return path
    identity = pd.read_csv(bind('cell_identity.csv'))
    contract = read_json(bind('run_contract.json'))
    fits = read_json(bind('fit_manifest.json'))
    expected = protocol['expected']
    require(sorted(identity.donor.unique()) == expected['donors'] and len(identity) == expected['latent_cells'] and identity.cell_id.is_unique and identity.author_key.is_unique and identity.celltype.eq('Fibroblast').all() and (not identity.isna().any().any()) and np.array_equal(identity.latent_row, np.arange(len(identity))), 'Identity table mismatch')
    require(contract['seeds'] == expected['seeds'] and contract['target_cap'] == 1200 and (contract['evaluation_seed'] == 0) and (contract['rk4_steps'] == 50), 'Original fit contract differs')
    latent = ROOT / protocol['latent_path']
    bind_file(latent, inputs, contract['inputs'][protocol['latent_path']])
    mu = np.load(latent, allow_pickle=False)
    require(mu.shape == (len(identity), expected['topics']) and np.isfinite(mu).all(), 'Latent shape/value mismatch')
    obs_path = ROOT / protocol['observation_path']
    bind_file(obs_path, inputs, contract['inputs'][protocol['observation_path']])
    obs = pd.read_csv(obs_path)
    for key in ['gsm', 'donor', 'cond', 'celltype']:
        require(np.array_equal(identity[key], obs[key]), f'Saved observation order differs: {key}')
    grid = {(held, tuple(subset), seed) for held in expected['donors'] for k in (1, 2) for subset in itertools.combinations([d for d in expected['donors'] if d != held], k) for seed in expected['seeds']}
    require(len(fits) == expected['fits'] == len(grid), 'Wrong saved fit count')
    seen = set()
    for fit in fits:
        held, subset, seed = (fit['held_donor'], fit['training_donors'], fit['seed'])
        key = (held, tuple(subset), seed)
        relative = f"fits/{held}/k{len(subset)}_{'_'.join(subset)}/seed{seed}"
        require(key in grid and key not in seen and (fit['path'] == relative) and (fit['k'] == len(subset)) and (fit['reference_scaler_used_by_training'] is False) and (fit['training_steps'] == contract['steps']) and (fit['batch_size'] == contract['batch_size']) and (fit['rk4_steps'] == 50), 'Invalid/duplicate saved fit')
        seen.add(key)
        require(read_json(bind(relative + '/fit.json')) == fit, 'Fit metadata differs from manifest')
        for artifact, checksum in fit['artifact_sha256'].items():
            bound = bind(relative + '/' + artifact)
            require(inputs[str(bound.resolve())] == checksum, 'Fit artifact digest differs')
        parent = Path(relative).parent.as_posix()
        for name in ['training_scaler.npz', 'mean_displacement_prediction.npz', 'target_marginal_predictions.npz']:
            bind(parent + '/' + name)
        bind(f'fits/{held}/reference.npz')
    require(seen == grid, 'Missing fit configuration')
    for relative in [protocol['metadata_path'], *sorted(contract['inputs'])]:
        if relative == protocol['metadata_path'] or relative.startswith(protocol['raw_directory'] + '/'):
            bind_file(ROOT / relative, inputs, contract['inputs'][relative])
    return (identity, mu, fits, contract)

def catalogue(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if '__MACOSX' not in n and (not Path(n).name.startswith('._'))]
        members = {}
        for kind in ['features', 'barcodes', 'matrix']:
            matches = [n for n in names if kind in Path(n).name and (not n.endswith('/'))]
            require(len(matches) == 1, f'Ambiguous {kind} member: {path}')
            members[kind] = matches[0]
        features = pd.read_csv(io.BytesIO(gzip.decompress(archive.read(members['features']))), sep='\t', header=None, dtype=str)
        barcodes = gzip.decompress(archive.read(members['barcodes'])).decode().splitlines()
    require(len(set(barcodes)) == len(barcodes), 'Duplicate original barcodes')
    return (features.iloc[:, 1].to_numpy(dtype=str), np.asarray(barcodes), members)

def align_archive(identity, gsm, label, donor, condition, barcodes, offset, meta):
    records = identity.loc[identity.gsm.eq(gsm)]
    rows = records.archive_cell_row.to_numpy()
    require(rows.dtype.kind in 'iu' and len(rows) > 0 and (rows >= 0).all() and (rows < len(barcodes)).all() and (len(np.unique(rows)) == len(rows)), 'Invalid archive row identities')
    keys = np.asarray([f'{label}_{b}' for b in barcodes])
    author = meta.reindex(keys).newMainCellTypes.to_numpy()
    expected = np.flatnonzero(author == 'Fibroblast')
    require(np.array_equal(rows, expected), 'Author fibroblast inclusion/order differs')
    require(np.array_equal(records.cell_id, [f'{gsm}:{b}' for b in barcodes[rows]]) and np.array_equal(records.author_key, keys[rows]) and np.array_equal(records.raw_matrix_row, offset + rows) and records.donor.eq(donor).all() and records.cond.eq(condition).all(), 'Barcode/donor/matrix alignment failure')
    return (records.latent_row.to_numpy(), rows)

def raw_counts(protocol, identity, panel, out):
    paths = sorted((ROOT / protocol['raw_directory']).glob('*.zip'))
    require(len(paths) == 12, 'Expected twelve original sample archives')
    inventories = {path: catalogue(path) for path in paths}
    common_genes = set(panel)
    for genes, _, _ in inventories.values():
        common_genes.intersection_update(genes)
    common = np.flatnonzero(np.isin(panel, list(common_genes)))
    require(len(common) == protocol['expected']['common_genes'], 'Assay coverage differs from frozen protocol')
    genes = np.asarray(panel[common], dtype=str)
    pd.DataFrame(dict(common_index=np.arange(len(common)), panel_index=common, gene=genes)).to_csv(out / 'features.csv', index=False)
    pd.DataFrame(dict(panel_index=np.arange(len(panel)), gene=panel, observed_in_all_archives=np.isin(np.arange(len(panel)), common))).to_csv(out / 'panel_coverage.csv', index=False)
    meta = pd.read_csv(ROOT / protocol['metadata_path'], sep='\t')
    require(meta.iloc[:, 0].is_unique, 'Duplicate author metadata key')
    meta = meta.set_index(meta.columns[0])
    blocks, global_rows, audits = ([], [], [])
    offset = 0
    for path, (symbols, barcodes, members) in inventories.items():
        match = re.fullmatch('(GSM\\d+)_(PWH\\d+)(D\\d+)\\.zip', path.name)
        require(match is not None, 'Unknown sample archive naming')
        gsm, donor, day = match.groups()
        condition = {'D0': 'Skin', 'D1': 'Wound1', 'D7': 'Wound7', 'D30': 'Wound30'}[day]
        latent_rows, sample_rows = align_archive(identity, gsm, donor + day, donor, condition, barcodes, offset, meta)
        with zipfile.ZipFile(path) as archive:
            streams = [io.BytesIO(gzip.decompress(archive.read(members[k]))) for k in ['features', 'barcodes', 'matrix']]
            matrix, collapsed_genes, decoded_barcodes = _read_10x_triplet(*streams)
        require(np.array_equal(decoded_barcodes, barcodes) and matrix.shape[0] == len(barcodes) and (matrix.data >= 0).all(), 'Raw matrix/barcode dimensions or counts invalid')
        positions = pd.Index(collapsed_genes).get_indexer(genes)
        require((positions >= 0).all(), 'Common gene absent after duplicate-symbol collapse')
        selected = matrix[sample_rows]
        common_counts = selected[:, positions].tocsr()
        totals = np.asarray(common_counts.sum(axis=1)).ravel()
        require((totals > 0).all(), 'Empty observed common-panel cell')
        assay_totals = np.asarray(selected.sum(axis=1)).ravel()
        audits.append(dict(gsm=gsm, donor=donor, condition=condition, archive_cells=len(barcodes), fibroblasts=len(sample_rows), assay_features=len(symbols), unique_assay_symbols=len(collapsed_genes), duplicate_symbol_rows=len(symbols) - len(collapsed_genes), panel_genes=len(panel), common_genes=len(common), zero_common_panel_cells=0, common_count_total=int(totals.sum()), full_assay_count_total=int(assay_totals.sum()), cell_mean_common_fraction_of_full_assay=float(np.mean(totals / assay_totals))))
        blocks.append(common_counts)
        global_rows.extend(latent_rows.tolist())
        offset += len(barcodes)
        print(f'raw {gsm} {donor}{day}: {len(sample_rows)} identities verified', flush=True)
    require(np.array_equal(global_rows, np.arange(len(identity))), 'Raw concatenation does not match saved latent order')
    counts = sparse.vstack(blocks, format='csr')
    sparse.save_npz(out / 'observed_common_counts.npz', counts)
    pd.DataFrame(audits).to_csv(out / 'raw_sample_audit.csv', index=False)
    return (counts, common, genes, audits)

def check_projection(protocol, counts, common, beta, mu, checkpoint):
    import torch
    from wound_models.topic_model import TopicModel
    torch.set_num_threads(4)
    beta = check_probabilities(beta)
    mu, common = (np.asarray(mu), np.asarray(common))
    counts = sparse.csr_matrix(counts)
    require(beta.ndim == 2 and mu.ndim == 2 and (mu.shape[0] > 0) and (mu.shape[1] == beta.shape[0]) and np.isfinite(mu).all(), 'Invalid saved mu/beta shape or nonfinite values')
    require(common.ndim == 1 and common.dtype.kind in 'iu' and (len(common) > 0) and (len(np.unique(common)) == len(common)) and (common >= 0).all() and (common < beta.shape[1]).all() and (counts.shape == (len(mu), len(common))) and np.isfinite(counts.data).all() and (counts.data >= 0).all() and (counts.data <= np.iinfo(np.int32).max).all() and (counts.data == np.floor(counts.data)).all(), 'Invalid projection counts/feature mapping')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state and all((isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() for value in state.values())), 'Nonfinite or invalid checkpoint state')
    model = TopicModel(input_dim=beta.shape[1], num_topics=beta.shape[0]).cpu()
    model.load_state_dict(state)
    model.eval()
    actual_beta = model.decoder.beta.detach().numpy()
    require(actual_beta.shape == beta.shape and np.isfinite(actual_beta).all(), 'Nonfinite or wrong-shaped checkpoint beta')
    beta_error = float(np.max(np.abs(actual_beta - beta)))
    require(beta_error <= protocol['checks']['checkpoint_beta_max_abs'], 'Saved beta/checkpoint disagree')
    maximum = 0.0
    coo = counts.tocoo()
    mapped = sparse.csr_matrix((coo.data, (coo.row, common[coo.col])), shape=(len(mu), beta.shape[1]), dtype=np.int32)
    proportions = library_normalize(mapped)
    with torch.no_grad():
        for start in range(0, len(mu), 4096):
            values = np.asarray(proportions[start:start + 4096].todense(), dtype=np.float32)
            require(values.shape == (min(4096, len(mu) - start), beta.shape[1]) and np.isfinite(values).all(), 'Nonfinite or wrong-shaped encoder input')
            actual, logvar = model.encoder(torch.from_numpy(values))
            actual, logvar = (actual.numpy(), logvar.numpy())
            expected = mu[start:start + len(values)]
            require(actual.shape == logvar.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(logvar).all(), 'Nonfinite or wrong-shaped encoder batch output')
            difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            require(np.isfinite(difference).all(), 'Nonfinite encoder difference')
            maximum = max(maximum, float(np.max(difference)))
    require(maximum <= protocol['checks']['cpu_encoder_mu_max_abs'], 'Raw expression does not reproduce saved mu')
    return dict(device='cpu', cpu_threads=4, checkpoint_beta_max_abs_error=beta_error, beta_tolerance=protocol['checks']['checkpoint_beta_max_abs'], all_cells_encoder_mu_max_abs_error=maximum, mu_tolerance=protocol['checks']['cpu_encoder_mu_max_abs'], checked_cells=len(mu), saved_beta_used_for_scores=True, no_model_training=True, checkpoint_inputs_and_batch_outputs_explicitly_finite=True)

def validate_split(protocol, fit, identity, mu):
    pop = ROOT / protocol['population_directory']
    folder = pop / fit['path']
    training = load_npz(folder.parent / 'training_scaler.npz')
    reference = load_npz(folder.parent.parent / 'reference.npz')
    held, permitted = (fit['held_donor'], fit['training_donors'])
    validate_scaler(training, mu, identity, permitted, held)
    source, target_all = (rows_for(identity, held, 'Wound1'), rows_for(identity, held, 'Wound7'))
    selected = target_all[np.random.default_rng(0).choice(len(target_all), 1200, replace=False)] if len(target_all) > 1200 else target_all
    for key, expected in [('source_latent_rows', source), ('target_all_latent_rows', target_all), ('target_evaluation_latent_rows', selected)]:
        require(np.array_equal(reference[key], expected), f'Reference identity differs: {key}')
    require(np.array_equal(fit['source_latent_rows'], source) and np.array_equal(fit['target_evaluation_latent_rows'], selected), 'Fit identity mismatch')
    pred_z = np.load(folder / 'prediction_training_z.npy', allow_pickle=False)
    pred_mu = inverse_scaler(pred_z, training['mean'], training['scale'])
    require(pred_mu.shape == mu[source].shape, 'Prediction row count differs from source')
    inverse_error = float(np.max(np.abs(pred_mu - np.load(folder / 'prediction_mu.npy', allow_pickle=False))))
    require(inverse_error <= protocol['checks']['inverse_max_abs'], 'Saved inverse transform differs')
    displacement = load_npz(folder.parent / 'mean_displacement_prediction.npz')
    z = ((mu - training['mean']) / training['scale']).astype(np.float32)
    delta = np.mean([z[rows_for(identity, d, 'Wound7')].mean(0) - z[rows_for(identity, d, 'Wound1')].mean(0) for d in permitted], axis=0)
    md_error = float(np.max(np.abs(z[source] + delta - displacement['training_z'])))
    require(np.array_equal(displacement['source_latent_rows'], source) and md_error <= protocol['checks']['mean_displacement_z_max_abs'], 'Mean displacement not same split')
    md_mu = inverse_scaler(displacement['training_z'], training['mean'], training['scale'])
    require(np.max(np.abs(md_mu - displacement['mu'])) <= protocol['checks']['inverse_max_abs'], 'Mean displacement inverse differs')
    marginal = load_npz(folder.parent / 'target_marginal_predictions.npz')
    marginal_rows, weights = exact_marginal(identity, permitted, held)
    require(np.array_equal(marginal['exact_latent_rows'], marginal_rows) and np.array_equal(marginal['exact_weights'], weights) and np.array_equal(marginal['target_evaluation_latent_rows'], selected), 'Marginal identity/weight mismatch')
    return dict(source=source, original_evaluation=selected, all_day7=target_all, marginal_rows=marginal_rows, marginal_weights=weights, prediction_mu=pred_mu, displacement_mu=md_mu, inverse_error=inverse_error, displacement_error=md_error)

def config_key(fit):
    return str(Path(fit['path']).parent)

def prepare(protocol_path, out, runtime_log):
    require(not out.exists() or (out.is_dir() and (not any(out.iterdir()))), 'Prepare requires a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    require(protocol['schema_version'] == 1 and protocol['metrics']['all'] == METRICS and (protocol['methods'] == METHODS), 'Unsupported protocol definition')
    inputs = {}
    for path in [protocol_path, Path(__file__), ROOT / 'wound_models/data_adapter.py', ROOT / 'wound_models/topic_model.py']:
        bind_file(path, inputs)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'preparation_started.json', dict(timestamp_utc=utc_now(), no_predictive_scores_computed=True, script_sha256=inputs[str(Path(__file__).resolve())], protocol_sha256=inputs[str(protocol_path.resolve())]))
    write_json(out / 'model_provenance.json', runtime_evidence(runtime_log, protocol['requested_agent_model']))
    identity, mu, fits, contract = bind_population(protocol, inputs)
    discovery = ROOT / protocol['discovery_directory']
    for name in ['panel.npy', 'beta.npy', 'topic_model.pt']:
        bind_file(discovery / name, inputs)
    projection_path = ROOT / protocol['projection_report']
    bind_file(projection_path, inputs)
    projection_provenance = read_json(projection_path)['protocol']
    require(inputs[str(discovery / 'panel.npy')] == projection_provenance['panel_sha256'] and inputs[str(discovery / 'topic_model.pt')] == projection_provenance['model_sha256'], 'Original representation differs')
    panel = np.load(discovery / 'panel.npy', allow_pickle=True).astype(str)
    beta = np.load(discovery / 'beta.npy', allow_pickle=False)
    require(len(panel) == protocol['expected']['panel_genes'] and len(set(panel)) == len(panel) and (beta.shape == (protocol['expected']['topics'], len(panel))), 'Decoder panel shape/identity differs')
    identity.to_csv(out / 'cell_identity.csv', index=False)
    counts, common, genes, sample_audits = raw_counts(protocol, identity, panel, out)
    projection = check_projection(protocol, counts, common, beta, mu, discovery / 'topic_model.pt')
    proportions = normalized_counts(counts)
    signatures, splits, split_indices = ({}, [], {})
    for fit in fits:
        checked = validate_split(protocol, fit, identity, mu)
        key = config_key(fit)
        if key not in signatures:
            signatures[key] = training_signature(proportions, identity, genes, fit['training_donors'], fit['held_donor'], protocol['signatures']['genes_per_direction'])
            row_key = key.replace('/', '__')
            for name in ['source', 'original_evaluation', 'all_day7', 'marginal_rows', 'marginal_weights']:
                split_indices[f'{row_key}__{name}'] = checked[name]
            split_indices[f'{row_key}__training_scaler_rows'] = np.flatnonzero(identity.donor.isin(fit['training_donors']))
        splits.append(dict(path=fit['path'], held_donor=fit['held_donor'], training_donors=fit['training_donors'], seed=fit['seed'], source_count=len(checked['source']), target_count=len(checked['original_evaluation']), target_all_count=len(checked['all_day7']), inverse_max_abs_error=checked['inverse_error'], mean_displacement_z_max_abs_error=checked['displacement_error']))
    write_json(out / 'training_signatures.json', signatures)
    write_json(out / 'projection_verification.json', projection)
    write_json(out / 'split_verification.json', splits)
    np.savez_compressed(out / 'split_identity_rows.npz', **split_indices)
    write_json(out / 'input_sha256.json', inputs)
    require(all((sha(Path(path)) == value for path, value in inputs.items())), 'Input snapshot changed during preparation')
    write_json(out / 'prepared.json', dict(status='prepared', timestamp_utc=utc_now(), no_predictive_scores_computed=True, common_genes=len(common), panel_genes=len(panel), fit_count=len(fits), training_only_signatures=len(signatures), raw_count_cells=counts.shape[0], biological_donors=3, input_files=len(inputs), no_new_training=True))
    write_json(out / 'preparation_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(prepared=str(out), projection=projection, common_genes=len(common), fits=len(fits)), indent=2))

def validate_preparation(protocol_path, out):
    require((out / 'prepared.json').is_file() and (out / 'preparation_manifest.json').is_file(), 'Run requires successful preparation')
    manifest = read_json(out / 'preparation_manifest.json')
    require(set(manifest) == PREPARATION_FILES, 'Incomplete or unexpected preparation closure')
    require(set((p.name for p in out.iterdir())) == set(manifest) | {'preparation_manifest.json'}, 'Unexpected/existing output; use a fresh preparation, never overwrite results')
    for relative, value in manifest.items():
        require(sha(safe_path(out, relative)) == value, f'Prepared artifact changed: {relative}')
    inputs = read_json(out / 'input_sha256.json')
    require(str(protocol_path.resolve()) in inputs and str(Path(__file__).resolve()) in inputs, 'Preparation not bound to this code/protocol')
    for path, value in inputs.items():
        require(sha(Path(path)) == value, f'Frozen input changed: {path}')
    require(read_json(protocol_path) == read_json(out / 'protocol.json'), 'Protocol snapshot changed')
    protocol, prepared = (read_json(out / 'protocol.json'), read_json(out / 'prepared.json'))
    expected = protocol['expected']
    require(prepared.get('status') == 'prepared' and prepared.get('no_predictive_scores_computed') is True and (prepared.get('no_new_training') is True), 'Preparation not successfully frozen')
    for key, value in {'common_genes': expected['common_genes'], 'panel_genes': expected['panel_genes'], 'fit_count': expected['fits'], 'raw_count_cells': expected['latent_cells'], 'biological_donors': len(expected['donors']), 'input_files': len(inputs)}.items():
        require(prepared.get(key) == value, f'Prepared count differs from frozen protocol: {key}')
    identity, features = (pd.read_csv(out / 'cell_identity.csv'), pd.read_csv(out / 'features.csv'))
    counts = sparse.load_npz(out / 'observed_common_counts.npz')
    require(len(identity) == expected['latent_cells'] and identity.cell_id.is_unique and (set(identity.donor) == set(expected['donors'])) and (len(features) == expected['common_genes']) and features.gene.is_unique and (counts.shape == (len(identity), len(features))), 'Prepared identity/feature/count dimensions differ')
    splits, signatures = (read_json(out / 'split_verification.json'), read_json(out / 'training_signatures.json'))
    require(len(splits) == expected['fits'] and len({r['path'] for r in splits}) == len(splits) and (set(signatures) == {config_key(r) for r in splits}) and (prepared.get('training_only_signatures') == len(signatures)), 'Incomplete prepared split/signature closure')
    projection = read_json(out / 'projection_verification.json')
    require(projection.get('checkpoint_inputs_and_batch_outputs_explicitly_finite') is True and projection.get('checked_cells') == len(identity), 'Missing finite projection verification')
    for field, tolerance in [('checkpoint_beta_max_abs_error', 'checkpoint_beta_max_abs'), ('all_cells_encoder_mu_max_abs_error', 'cpu_encoder_mu_max_abs')]:
        value = projection.get(field, float('nan'))
        require(np.isfinite(value) and 0 <= value <= protocol['checks'][tolerance], 'Invalid projection verification error')
    return inputs

def run(protocol_path, out, runtime_log):
    inputs = validate_preparation(protocol_path, out)
    protocol = read_json(out / 'protocol.json')
    started = utc_now()
    runtime = runtime_evidence(runtime_log, protocol['requested_agent_model'])
    pop = ROOT / protocol['population_directory']
    fits = read_json(pop / 'fit_manifest.json')
    identity = pd.read_csv(out / 'cell_identity.csv')
    feature_map = pd.read_csv(out / 'features.csv')
    common, genes = (feature_map.panel_index.to_numpy(), feature_map.gene.to_numpy(dtype=str))
    proportions = normalized_counts(sparse.load_npz(out / 'observed_common_counts.npz'))
    mu = np.load(ROOT / protocol['latent_path'], allow_pickle=False)
    beta = np.load(ROOT / protocol['discovery_directory'] / 'beta.npy', allow_pickle=False)
    decoded, missing_mass = decode_cells(mu, beta, common)
    definitions = read_json(out / 'training_signatures.json')
    metrics, decoded_metrics, reference_scores, programme_rows, mass_rows = ([], [], [], [], [])
    profiles, target_profiles, rows_by_target = ({}, {}, {})
    for held in protocol['expected']['donors']:
        reference = load_npz(pop / 'fits' / held / 'reference.npz')
        for target_set, key in [('original_evaluation', 'target_evaluation_latent_rows'), ('all_day7', 'target_all_latent_rows')]:
            rows = reference[key]
            target = mean_profile(proportions, rows)
            decoded_target = mean_profile(decoded, rows)
            target_profiles[held, target_set] = (target, decoded_target)
            profiles[f'observed__{held}__{target_set}'] = target
            profiles[f'decoded_target_reference__{held}__{target_set}'] = decoded_target
            rows_by_target[f'{held}__{target_set}'] = rows
            reference_scores.append(dict(held_donor=held, target_set=target_set, n_target=len(rows), target_kind='actual_observed_expression', method='decoder_reconstruction_reference_not_forecast', missing_panel_mass_mean=float(missing_mass[rows].mean()), **profile_metrics(decoded_target, target)))
    for number, fit in enumerate(fits, 1):
        held, subset = (fit['held_donor'], fit['training_donors'])
        checked = validate_split(protocol, fit, identity, mu)
        key = config_key(fit)
        signature = training_signature(proportions, identity, genes, subset, held, protocol['signatures']['genes_per_direction'])
        require(signature == definitions[key], 'Training signature differs from pre-execution freeze')
        source, marginal_rows, marginal_weights = (checked[k] for k in ['source', 'marginal_rows', 'marginal_weights'])
        flow, flow_missing = decode_cells(checked['prediction_mu'], beta, common)
        displacement, md_missing = decode_cells(checked['displacement_mu'], beta, common)
        predictions = dict(shared_cfm_decoded=mean_profile(flow), persistence_decoded=mean_profile(decoded, source), mean_displacement_decoded=mean_profile(displacement), training_target_marginal_decoded_exact=mean_profile(decoded, marginal_rows, marginal_weights), persistence_observed=mean_profile(proportions, source), training_target_marginal_observed_exact=mean_profile(proportions, marginal_rows, marginal_weights))
        common_meta = dict(held_donor=held, k=fit['k'], training_donors='|'.join(subset), seed=fit['seed'], fit_path=fit['path'])
        dropped = dict(shared_cfm_decoded=(flow_missing, None), persistence_decoded=(missing_mass[source], None), mean_displacement_decoded=(md_missing, None), training_target_marginal_decoded_exact=(missing_mass[marginal_rows], marginal_weights))
        for method, (values, weight) in dropped.items():
            mass_rows.append(dict(**common_meta, method=method, missing_panel_genes=beta.shape[1] - len(common), excluded_probability_mass_mean=float(weights_for(len(values), weight) @ values), excluded_probability_mass_min=float(values.min()), excluded_probability_mass_max=float(values.max())))
        source_score = signature_scores(predictions['persistence_observed'], signature)
        for method, prediction in predictions.items():
            profile_key = f"{held}__k{fit['k']}_{'_'.join(subset)}__seed{fit['seed']}__{method}"
            profiles[profile_key] = prediction
            pred_scores = signature_scores(prediction, signature)
            for target_set in ['original_evaluation', 'all_day7']:
                target, decoded_target = target_profiles[held, target_set]
                meta = dict(**common_meta, method=method, target_set=target_set, n_source=len(source), n_target=len(checked[target_set]), n_common_genes=len(common))
                metrics.append(dict(**meta, target_kind='actual_observed_expression', **profile_metrics(prediction, target)))
                decoded_metrics.append(dict(**meta, target_kind='decoded_observed_mu_reference_not_truth', **profile_metrics(prediction, decoded_target)))
                obs_scores = signature_scores(target, signature)
                for programme in pred_scores:
                    programme_rows.append(dict(**meta, programme=programme, predicted_score=pred_scores[programme], observed_score=obs_scores[programme], source_observed_score=source_score[programme], predicted_change_from_observed_source=pred_scores[programme] - source_score[programme], observed_change_from_source=obs_scores[programme] - source_score[programme], absolute_error=abs(pred_scores[programme] - obs_scores[programme])))
        print(f"[{number}/27] {held} k={fit['k']} seed={fit['seed']} observed profiles scored", flush=True)
    frame = pd.DataFrame(metrics)
    require(len(frame) == 27 * len(METHODS) * 2, 'Incomplete score matrix')
    config, blocks, macro = hierarchical_summary(frame, protocol['expected']['donors'])
    contrasts = []
    for (held, k, target_set), group in blocks.groupby(['held_donor', 'k', 'target_set']):
        indexed = group.set_index('method')
        for baseline in METHODS[1:]:
            contrasts.append(dict(held_donor=held, k=int(k), target_set=target_set, baseline=baseline, **{m + '_flow_minus_baseline': float(indexed.loc[METHODS[0], m] - indexed.loc[baseline, m]) for m in METRICS}))
    contrast = pd.DataFrame(contrasts)
    contrast_columns = [m + '_flow_minus_baseline' for m in METRICS]
    macro_contrast = contrast.groupby(['k', 'target_set', 'baseline'])[contrast_columns].mean().reset_index()
    for name in METRICS:
        wins = contrast.groupby(['k', 'target_set', 'baseline'])[name + '_flow_minus_baseline'].apply(lambda s: int((s < 0).sum())).reset_index(name=name + '_donors_lower')
        macro_contrast = macro_contrast.merge(wins, validate='one_to_one')
    paired_k = []
    for (held, target_set, method), group in blocks.groupby(['held_donor', 'target_set', 'method']):
        pair = group.set_index('k')
        paired_k.append(dict(held_donor=held, target_set=target_set, method=method, **{m + '_k2_minus_k1': float(pair.loc[2, m] - pair.loc[1, m]) for m in METRICS}))
    programme = pd.DataFrame(programme_rows)
    program_config = programme.groupby(['held_donor', 'k', 'training_donors', 'target_set', 'method', 'programme'])['absolute_error'].mean().reset_index()
    program_blocks = program_config.groupby(['held_donor', 'k', 'target_set', 'method', 'programme'])['absolute_error'].mean().reset_index()
    program_macro = program_blocks.groupby(['k', 'target_set', 'method', 'programme'])['absolute_error'].mean().reset_index()
    require(all((sha(Path(path)) == value for path, value in inputs.items())), 'Input snapshot changed during scoring')
    validate_preparation(protocol_path, out)
    tables = {'observed_expression_per_fit.csv': frame, 'decoded_target_reference_per_fit.csv': pd.DataFrame(decoded_metrics), 'decoder_reconstruction_reference.csv': pd.DataFrame(reference_scores), 'configuration_means.csv': config, 'donor_blocks.csv': blocks, 'macro_scores.csv': macro, 'paired_method_contrasts.csv': contrast, 'macro_method_contrasts.csv': macro_contrast, 'paired_k_contrasts.csv': pd.DataFrame(paired_k), 'training_programme_per_fit.csv': programme, 'training_programme_donor_blocks.csv': program_blocks, 'training_programme_macro.csv': program_macro, 'excluded_decoder_mass.csv': pd.DataFrame(mass_rows)}
    for name, table in tables.items():
        with (out / name).open('x') as stream:
            table.to_csv(stream, index=False)
    np.savez_compressed(out / 'mean_profiles.npz', common_genes=genes, **profiles)
    np.savez_compressed(out / 'target_identities.npz', **rows_by_target)
    write_json(out / 'execution_model_provenance.json', runtime)
    report = dict(schema_version=1, status='completed', experiment='POP05_observed_expression_saved_prediction_audit', protocol_id=protocol['protocol_id'], prepared_utc=read_json(out / 'prepared.json')['timestamp_utc'], execution_started_utc=started, execution_completed_utc=utc_now(), predictive_scores_started_after_protocol_feature_freeze=True, protocol_sha256=sha(out / 'protocol.json'), preparation_manifest_sha256=sha(out / 'preparation_manifest.json'), biological_donors=3, new_biological_donors=0, saved_fits=27, trained_models=0, observed_count_cells=len(identity), common_genes=len(common), panel_genes=beta.shape[1], primary_target_kind='actual raw observed count-derived proportions, not decoded latent truth', primary_metric='total_variation', primary_target_set='original_evaluation', scores=len(frame), macro=macro.to_dict('records'), macro_method_contrasts=macro_contrast.to_dict('records'), decoder_reference=pd.DataFrame(reference_scores).to_dict('records'), verification=dict(status='passed', all_input_hashes_unchanged=True, complete_fit_grid=True, raw_barcode_author_label_and_latent_alignment=True, training_scalers_recomputed=True, saved_inverse_transform_verified=True, same_split_baselines=True, training_only_signatures=True, no_cell_as_biological_replicate=True, projection=read_json(out / 'projection_verification.json')), remaining_limits=['Only three healthy wound donors, overlapping training subsets and the existing frozen discovery representation.', 'No external/clinical validation and no new biological donors.', 'Mean normalized expression/programme summaries do not establish single-cell transitions or complete distributional fidelity.', '1619 frozen-panel genes are absent from the assay; common-panel conditioning changes the expression estimand.', 'No absolute counts or library-size prediction; no biological confidence interval or significance claim.', 'Training-derived signed gene sets are secondary descriptive signatures, not validated pathways.', 'The decoded observed-target reference is not truth, a forecast or a mathematical lower error bound.', 'POP05 is partially addressed for observable mean expression; broader biological validation remains open.'], runtime=dict(device='cpu', cpu_thread_limit=4, python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    write_json(out / 'report.json', report)
    write_json(out / 'output_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(macro.loc[macro.target_set.eq('original_evaluation')].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--runtime-log', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(out == OUTPUT or OUTPUT in out.parents, 'Output must remain within the owned observable audit directory')
    with threadpool_limits(limits=4):
        (prepare if args.prepare else run)(args.protocol.resolve(), out, args.runtime_log)
if __name__ == '__main__':
    main()
