"""Saved scalar reporting for this study; no model fitting or historical cross-study loader."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
FLOW_METHODS = ('cfm_independent', 'cfm_ot', 'cfm_sb', 'cfm_conditioned')
BASELINES = ('identity', 'mean_shift', 'location_scale', 'target_marginal')
DONORS = ('PWH26', 'PWH27', 'PWH28')

def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite saved scalar')
    return result

def validate_population(report: dict) -> None:
    records = report['records']
    grid = {(d, method, steps, seed, source, target) for d in DONORS for method in FLOW_METHODS + BASELINES for steps in ((120, 600) if method in FLOW_METHODS else (None,)) for seed in ((0, 1) if method in FLOW_METHODS else (0,)) for source, target in ((0.0, 1.0), (1.0, 7.0), (7.0, 30.0))}
    keys = [(r['held_individual'], r['method'], r['steps'], r['seed'], r['source_time'], r['target_time']) for r in records]
    if len(keys) != len(grid) or set(keys) != grid:
        raise ValueError('population reporting grid incomplete or duplicated')
    summaries = report['within_donor_summaries']
    summary_keys = [(r['held_donor'], r['method'], r['steps']) for r in summaries]
    expected = {(d, m, budget) for d in DONORS for m in FLOW_METHODS + BASELINES for budget in ((120, 600) if m in FLOW_METHODS else (None,))}
    if len(summary_keys) != len(expected) or set(summary_keys) != expected:
        raise ValueError('population summary grid incomplete or duplicated')
    for row in summaries:
        selected = [r for r in records if (r['held_individual'], r['method'], r['steps']) == (row['held_donor'], row['method'], row['steps'])]
        mean = math.fsum((finite(r['energy']) for r in selected)) / len(selected)
        marginal = [r for r in records if r['held_individual'] == row['held_donor'] and r['method'] == 'target_marginal']
        target_mean = math.fsum((finite(r['energy']) for r in marginal)) / len(marginal)
        if row['records_averaged'] != len(selected) or not math.isclose(finite(row['mean_energy']), mean, rel_tol=0, abs_tol=1e-12) or (not math.isclose(finite(row['mean_energy_minus_seed0_target_draw']), mean - target_mean, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('population within-donor arithmetic differs')
    if (report['new_fits'], report['new_biological_units'], report['biological_donors'], report['historical_flow_fits'], report['historical_baseline_jobs'], report['historical_prediction_pairs'], report['sb_batches'], report['sb_nonconverged']) != (0, 0, 3, 48, 12, 180, 4320, 20):
        raise ValueError('population diagnostic units or numerical failures changed')
    if any((report[key] for key in ('cross_fold_raw_distance_pooling_authorized', 'adaptive_MMD_method_ranking_authorized', 'population_ranking_authorized', 'replayable_fields_saved'))):
        raise ValueError('population reporting exceeds saved-diagnostic scope')

def fmt(value, digits=5):
    return f'{float(value):.{digits}f}'

def population_table_rows(report: dict) -> list[list[str]]:
    rows = report['within_donor_summaries']
    validate_population(report)
    result = []
    for donor in DONORS:
        for budget in (120, 600):
            selected = [next((r for r in rows if (r['held_donor'], r['method'], r['steps']) == (donor, method, budget))) for method in FLOW_METHODS]
            result.append([donor, str(budget), *(fmt(r['mean_energy_minus_seed0_target_draw'], 4) for r in selected)])
    return result

def population_detail_rows(report: dict) -> list[list[str]]:
    names = dict(zip(FLOW_METHODS + BASELINES, ('Independent', 'Minibatch OT', 'SB', 'Conditioned', 'Persistence', 'Mean shift', 'Location scale', 'Target draw')))
    rows = []
    for donor in DONORS:
        for method in FLOW_METHODS + BASELINES:
            for steps in (120, 600) if method in FLOW_METHODS else (None,):
                r = next((r for r in report['within_donor_summaries'] if (r['held_donor'], r['method'], r['steps']) == (donor, method, steps)))
                rows.append([donor, names[method], 'None' if steps is None else str(steps), str(r['records_averaged']), fmt(r['mean_energy'], 4)])
    return rows
