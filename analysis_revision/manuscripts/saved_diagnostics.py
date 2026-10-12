"""Small saved-result reporting bridge; no numerical libraries or model imports.

The historical verification is authenticated, then split into independent
study reports. This module never reads a count matrix, fits a model, or replays
predictions. Its summaries are descriptive arithmetic on saved scalar records.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HISTORICAL_VERIFICATION = "output/sdd-local-validation-20260929/verification-v2/verification.json"
HISTORICAL_SHA256 = "a19780ffa89ffa6aa844f860477cf669345033ab15c72cd80bc0610fdf9ed692"
REPORT_BASE = "output/submission-preparation-20261002-r1"
FLOW_METHODS = ("cfm_independent", "cfm_ot", "cfm_sb", "cfm_conditioned")
BASELINES = ("identity", "mean_shift", "location_scale", "target_marginal")
DONORS = ("PWH26", "PWH27", "PWH28")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_history(root: Path) -> dict:
    path = root / HISTORICAL_VERIFICATION
    if path.stat().st_size > 2 * 1024 * 1024 or sha(path) != HISTORICAL_SHA256:
        raise ValueError("historical saved verification changed or exceeds small-report cap")
    evidence = json.loads((root / "output/sdd-local-validation-20260929/evidence/evidence_hashes.json").read_text())
    if evidence.get("verification-v2/verification.json") != HISTORICAL_SHA256:
        raise ValueError("historical seal does not bind the selected verification")
    data = json.loads(path.read_text())
    if data["status"] != "passed_bounded_diagnostic_only":
        raise ValueError("historical diagnostic verification did not pass")
    return data


def finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite saved scalar")
    return result


def measurement_report(data: dict) -> dict:
    inherited, training = data["topic_inherited"], data["topic_training"]
    partition = inherited["partition"]
    if (len(partition["training_patients"]), len(partition["held_patients"]),
            partition["optimization_cells"], partition["held_cells"]) != (16, 4, 60861, 15600):
        raise ValueError("topic partition changed")
    if partition["end_to_end_independent"] or partition["selection_uses_outcomes"]:
        raise ValueError("topic independence or outcome-selection scope changed")
    if inherited["cross_panel_raw_crossentropy_ranking_authorized"] or training["cross_panel_raw_crossentropy_ranking_authorized"]:
        raise ValueError("cross-panel reconstruction ranking is not permitted")
    rows = []
    for group in (inherited, training):
        for fit in group["fits"]:
            held = fit["partitions"]["held"]
            ce = held["equal_patient_crossentropy"]
            rows.append({
                "panel_kind": fit["panel_kind"], "panel_size": fit["panel_size"],
                "seed": fit["seed"], "steps": fit["steps"],
                "held_objective": finite(fit["final_held_objective"]),
                "held_reconstruction": finite(ce["model"]),
                "equal_patient_training_null": finite(ce["training_equal_patient_mean"]),
                "within_panel_model_minus_null": finite(held["model_minus_null"]["training_equal_patient_mean"]),
                "decoder_cosine_median": finite(fit["decoder_cosine"]["median"]),
                "decoder_cosine_max": finite(fit["decoder_cosine"]["max"]),
                "held_effective_topic_usage": finite(held["effective_mean_usage_topics"]),
                "held_argmax_nonempty_topics": held["argmax_nonempty_topics"],
                "training_plateau_screen_passed": fit["partitions"]["train"]["plateau"]["screen_passed"],
                "held_plateau_screen_passed": held["plateau"]["screen_passed"],
                "historical_reference0_correlation": finite(fit["reference0_assigned_correlation"]),
            })
    result = {
        "schema_version": 1, "kind": "measurement_saved_budget_panel_diagnostic",
        "status": "integrated_saved_results_not_new_experiment",
        "upstream_verification_sha256": HISTORICAL_SHA256,
        "new_fits": 0, "new_biological_units": 0,
        "all_group_patients": 20, "outcome_patients": 11,
        "training_patients": 16, "held_patients": 4,
        "training_cells": 60861, "held_cells": 15600,
        "evaluation_cells_per_patient": 128, "batch_size": 128,
        "topic_count": 15, "kl_weight": 0.01, "lr": 0.001,
        "aggregation": "equal_patient_reconstruction_on_fixed_128_cell_draws",
        "historical_clinical_readouts": "equal_specimen_patient_readouts_not_primary_cell_pooled_estimator",
        "end_to_end_independent": False,
        "panel_fit_uses_held_expression": False,
        "reference_matching_uses_discovery_model_including_held_expression": True,
        "fit_selection_authorized": False, "cross_panel_raw_ranking_authorized": False,
        "plateau_threshold": 0.001, "fits": rows,
    }
    validate_measurement(result)
    return result


def validate_measurement(report: dict) -> None:
    rows = report["fits"]
    grid = {(kind, steps, seed) for kind, budgets in (("inherited", (20000, 40000)),
                                                    ("training", (20000,)))
            for steps in budgets for seed in (0, 1)}
    keys = [(r["panel_kind"], r["steps"], r["seed"]) for r in rows]
    if len(keys) != 6 or set(keys) != grid:
        raise ValueError("topic grid incomplete or duplicated")
    for r in rows:
        if r["panel_size"] != (7002 if r["panel_kind"] == "inherited" else 7003):
            raise ValueError("topic panel identity differs")
        for key in ("held_objective", "held_reconstruction", "equal_patient_training_null",
                    "within_panel_model_minus_null", "decoder_cosine_median", "decoder_cosine_max",
                    "held_effective_topic_usage", "historical_reference0_correlation"):
            finite(r[key])
        if not math.isclose(r["held_reconstruction"] - r["equal_patient_training_null"],
                            r["within_panel_model_minus_null"], abs_tol=1e-12):
            raise ValueError("within-panel null contrast differs")
        if not 0 <= r["decoder_cosine_median"] <= r["decoder_cosine_max"] <= 1:
            raise ValueError("invalid decoder cosine")
    if (report["new_fits"] != 0 or report["new_biological_units"] != 0 or
            report["end_to_end_independent"] or report["fit_selection_authorized"] or
            report["cross_panel_raw_ranking_authorized"]):
        raise ValueError("measurement reporting exceeds diagnostic scope")


def population_report(data: dict) -> dict:
    population = data["population"]
    raw = population["paired_records"]
    grid = {(d, method, steps, seed, source, target)
            for d in DONORS for method in FLOW_METHODS + BASELINES
            for steps in ((120, 600) if method in FLOW_METHODS else (None,))
            for seed in ((0, 1) if method in FLOW_METHODS else (0,))
            for source, target in ((0., 1.), (1., 7.), (7., 30.))}
    keys = [(r["held_individual"], r["method"], r["steps"], r["seed"], r["source_time"], r["target_time"]) for r in raw]
    if len(keys) != 180 or set(keys) != grid or population["jobs"] != 60:
        raise ValueError("population saved grid incomplete or duplicated")
    records = [{k: r[k] for k in ("held_individual", "method", "steps", "seed", "source_time", "target_time")}
               | {"energy": finite(r["energy"])} for r in raw]
    baseline = {(r["held_individual"], r["method"], r["source_time"], r["target_time"]): r["energy"]
                for r in records if r["method"] in BASELINES}
    for r in raw:
        if r["method"] in FLOW_METHODS:
            for name in BASELINES:
                difference = finite(r["energy"]) - baseline[(r["held_individual"], name, r["source_time"], r["target_time"])]
                if not math.isclose(difference, finite(r["energy_minus_baseline"][name]), abs_tol=1e-12):
                    raise ValueError("saved flow-minus-baseline contrast differs")
    summaries = []
    for donor in DONORS:
        for method in FLOW_METHODS + BASELINES:
            for steps in ((120, 600) if method in FLOW_METHODS else (None,)):
                selected = [r for r in records if (r["held_individual"], r["method"], r["steps"]) == (donor, method, steps)]
                score = math.fsum(r["energy"] for r in selected) / len(selected)
                marginal = math.fsum(v for (d, m, _, _), v in baseline.items() if d == donor and m == "target_marginal") / 3
                summaries.append({"held_donor": donor, "method": method, "steps": steps,
                                  "records_averaged": len(selected), "mean_energy": score,
                                  "mean_energy_minus_seed0_target_draw": score - marginal})
    result = {
        "schema_version": 1, "kind": "population_saved_finite_budget_diagnostic",
        "status": "integrated_saved_results_not_new_experiment",
        "upstream_verification_sha256": HISTORICAL_SHA256,
        "new_fits": 0, "new_biological_units": 0, "biological_donors": 3,
        "historical_flow_fits": 48, "historical_baseline_jobs": 12,
        "historical_prediction_pairs": 180,
        "representation": "training_side_HVG200_PCA15_fold_specific_scales",
        "source_rule": "observed_source_for_each_adjacent_interval_not_autonomous_rollout",
        "hidden": 64, "depth": 2, "time_dim": 16, "batch_size": 64,
        "clock": "rank", "energy_cap": 200, "baseline_draw_seed": 0,
        "cross_fold_raw_distance_pooling_authorized": False,
        "adaptive_MMD_method_ranking_authorized": False,
        "population_ranking_authorized": False, "replayable_fields_saved": False,
        "sb_batches": population["sb_batches"], "sb_nonconverged": population["sb_nonconverged"],
        "sb_max_row_l1": finite(population["sb_max_row_l1"]),
        "sb_max_column_l1": finite(population["sb_max_column_l1"]),
        "records": records, "within_donor_summaries": summaries,
    }
    validate_population(result)
    return result


def validate_population(report: dict) -> None:
    records = report["records"]
    grid = {(d, method, steps, seed, source, target)
            for d in DONORS for method in FLOW_METHODS + BASELINES
            for steps in ((120, 600) if method in FLOW_METHODS else (None,))
            for seed in ((0, 1) if method in FLOW_METHODS else (0,))
            for source, target in ((0., 1.), (1., 7.), (7., 30.))}
    keys = [(r["held_individual"], r["method"], r["steps"], r["seed"], r["source_time"], r["target_time"]) for r in records]
    if len(keys) != len(grid) or set(keys) != grid:
        raise ValueError("population reporting grid incomplete or duplicated")
    summaries = report["within_donor_summaries"]
    summary_keys = [(r["held_donor"], r["method"], r["steps"]) for r in summaries]
    expected = {(d, m, budget) for d in DONORS for m in FLOW_METHODS + BASELINES
                for budget in ((120, 600) if m in FLOW_METHODS else (None,))}
    if len(summary_keys) != len(expected) or set(summary_keys) != expected:
        raise ValueError("population summary grid incomplete or duplicated")
    for row in summaries:
        selected = [r for r in records if (r["held_individual"], r["method"], r["steps"])
                    == (row["held_donor"], row["method"], row["steps"])]
        mean = math.fsum(finite(r["energy"]) for r in selected) / len(selected)
        marginal = [r for r in records if r["held_individual"] == row["held_donor"] and r["method"] == "target_marginal"]
        target_mean = math.fsum(finite(r["energy"]) for r in marginal) / len(marginal)
        if (row["records_averaged"] != len(selected) or
                not math.isclose(finite(row["mean_energy"]), mean, rel_tol=0, abs_tol=1e-12) or
                not math.isclose(finite(row["mean_energy_minus_seed0_target_draw"]), mean - target_mean,
                                 rel_tol=0, abs_tol=1e-12)):
            raise ValueError("population within-donor arithmetic differs")
    if (report["new_fits"], report["new_biological_units"], report["biological_donors"],
            report["historical_flow_fits"], report["historical_baseline_jobs"],
            report["historical_prediction_pairs"], report["sb_batches"], report["sb_nonconverged"]) != (0, 0, 3, 48, 12, 180, 4320, 20):
        raise ValueError("population diagnostic units or numerical failures changed")
    if any(report[key] for key in ("cross_fold_raw_distance_pooling_authorized",
                                   "adaptive_MMD_method_ranking_authorized", "population_ranking_authorized",
                                   "replayable_fields_saved")):
        raise ValueError("population reporting exceeds saved-diagnostic scope")


def fmt(value, digits=5):
    return f"{float(value):.{digits}f}"


def measurement_table_rows(report: dict) -> list[list[str]]:
    validate_measurement(report)
    return [[("Inherited" if r["panel_kind"] == "inherited" else "Train-selected") + f" {r['panel_size']}",
             f"{r['steps']//1000}k / {r['seed']}", fmt(r["held_reconstruction"]),
             fmt(r["within_panel_model_minus_null"]), fmt(r["decoder_cosine_median"], 3),
             fmt(r["held_effective_topic_usage"], 3),
             "Pass / fail" if r["training_plateau_screen_passed"] else "Fail / fail"] for r in report["fits"]]


def population_table_rows(report: dict) -> list[list[str]]:
    rows = report["within_donor_summaries"]
    validate_population(report)
    result = []
    for donor in DONORS:
        for budget in (120, 600):
            selected = [next(r for r in rows if (r["held_donor"], r["method"], r["steps"]) == (donor, method, budget))
                        for method in FLOW_METHODS]
            result.append([donor, str(budget), *(fmt(r["mean_energy_minus_seed0_target_draw"], 4) for r in selected)])
    return result


def population_detail_rows(report: dict) -> list[list[str]]:
    names = dict(zip(FLOW_METHODS + BASELINES,
                     ("Independent", "Minibatch OT", "SB", "Conditioned", "Persistence", "Mean shift", "Location scale", "Target draw")))
    rows = []
    for donor in DONORS:
        for method in FLOW_METHODS + BASELINES:
            for steps in ((120, 600) if method in FLOW_METHODS else (None,)):
                r = next(r for r in report["within_donor_summaries"] if (r["held_donor"], r["method"], r["steps"]) == (donor, method, steps))
                rows.append([donor, names[method], "None" if steps is None else str(steps),
                             str(r["records_averaged"]), fmt(r["mean_energy"], 4)])
    return rows
