#!/usr/bin/env python3
"""Prepare saved-evidence reporting only. No training or experiment entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manuscripts.saved_diagnostics import (REPORT_BASE, load_verified_history,
                                           measurement_report, population_report, sha)


def prepare(out: Path, *, start=False):
    policy = json.loads((ROOT / "config/submission_prepare_policy_20261002.json").read_text())
    if start or any(policy[key] is not False for key in
                    ("training_enabled", "research_download_enabled", "batch_reanalysis_enabled", "remote_execution_enabled", "public_release_enabled")):
        raise ValueError("this is preparation only; experiment dispatch is forbidden")
    if any(int(os.environ.get(name, "1")) != 1 for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("preparation requires one numeric thread")
    if out.resolve() != ROOT / REPORT_BASE:
        raise ValueError("use the fixed fresh revision root; old sealed results must not be changed")
    targets = [out / study / "report.json" for study in ("measurement", "population")]
    if any(target.exists() for target in targets):
        raise ValueError("reporting snapshots already exist; verify instead of overwriting")
    saved = load_verified_history(ROOT)
    reports = (measurement_report(saved), population_report(saved))
    # Create each independent snapshot once, after all validation completes.
    for target, report in zip(targets, reports):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return {"status": "prepared_saved_reporting_only", "new_fits": 0,
            "raw_matrix_reads": 0, "field_integrations": 0, "research_downloads": 0,
            "report_sha256": {str(p.relative_to(ROOT)): sha(p) for p in targets}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / REPORT_BASE))
    parser.add_argument("--start", action="store_true", help="always refused; no experiment dispatch")
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.out), start=args.start), indent=2))


if __name__ == "__main__":
    main()
