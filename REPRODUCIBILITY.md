# Reproduction guide

## Data and biological units

The acute human time course contains three donors. The mouse comparison has one animal per arm and time point. Numerical resolutions, seeds and evaluation draws do not increase the donor count.

Source counts and annotations are supplied by the following GEO records, under their source-study terms:

- [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816)
- [GSE241132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241132)
- [GSE326622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE326622)

Download counts into `data/raw/<accession>/`. Acquisition helpers are in scripts/fetch_*.sh. Human wound cell annotations must be joined by barcode before selecting fibroblasts. GSE165816 supplies the frozen representation. Packaging a copy does not make its source data or fitted representation statistically independent; it does not supply a temporal healing target.

## Analysis order

Run from the archive root. Each program documents options with `--help`. Some programs consume completed outputs from earlier steps; preserve reference reports and use fresh output directories for alternative runs. A complete raw-data rerun requires count downloads and substantially more computation than rendering saved results.

1. Fit the 7,002-gene, 15-coordinate discovery representation with `python3 scripts/fit_expression_representation.py --steps 40000 --seed 0`. The generic CLI default is shorter than the manuscript's 40,000 steps. Hardware-dependent training may differ from the frozen reference weights.
2. Run `reconstruct_held_out_timepoint.py`, then `compare_time_coordinates.py`. These evaluate a time point omitted from fitting; training-only scaling excludes target cells.
3. Run `evaluate_donor_transfer.py`, then `benchmark_population_predictors.py`. These evaluate a donor omitted from fitting and must not be conflated with a time-point holdout. Training endpoints are paired within donor.
4. Run `evaluate_training_donor_count.py`, `assess_training_seed_stability.py`, `infer_donor_count_sensitivity.py` and `infer_donor_geometry.py` for the donor and seed analyses.
5. Run `evaluate_mouse_timepoint.py --model-arm NDB`, and analogously PDB and GDB with separate output directories, before `summarize_mouse_model_arms.py`. Arm names come from the source study.
6. Run `assess_robustness.py` after the human temporal analysis to fit one seed-0 field (8,000 steps), compare 25/50/100/200 RK4 steps and evaluate 50 subsampling draws at each of 250/500/1,000 cells per distribution. The energy distance is the empirical V-statistic. Central evaluation-draw ranges measure Monte Carlo sensitivity conditional on one field and the observed cohort; they are not generalization confidence intervals for new donors.

## Verification scope

The archive verifier checks the checksum manifest, Python syntax, matching citation metadata and CPU CLI imports. The full reproduction package additionally checks the frozen decoder, deterministic projection, training-only standardization, current narrative anchors, report hashes and recursive checkpoint manifests. Manuscript builds check protected narrative sections, adjacent equation references, table citations, embedded fonts, complete citations and figure placement. Figure builds enforce black lettering with bold panel labels, vector content and unchanged numerical source files.

The software-only archive contains programs, not pretrained weights or completed observations. Its checksum/import checks do not claim that a full raw-count pipeline has been rerun on every platform. Provenance fingerprints inside reports describe the original computation; MANIFEST.sha256.json hashes the files in the exact distributed artifact.

## Mathematical audit

Run `python3 -m pytest -p no:cacheprovider tests -q` for all analytic implementation and isolation checks, including unittest and pytest test styles. Run `python3 scripts/audit_population_math.py --output-dir outputs/reruns/mathematical_audit` after producing its saved inputs. The full reproduction package includes those inputs; the software-only archive requires the preceding analysis steps. Historical report hashes describe the original computation, while the release manifest describes the current exported files. Documentation corrections are not a claim that every original model has been refitted.

## Biological extension

After generating the saved inputs above, run `python3 scripts/expand_biological_evidence.py population` and `python3 scripts/expand_biological_evidence.py expression`. Outputs go to `outputs/reruns/biological_expansion/`; choose a fresh `--output-dir` for another run. The expression command requires the public raw counts. The full package already contains the reference display tables under `outputs/biological_expansion/population/`. The protocol is retrospective, not a preregistered clinical analysis. `strengthen_computational_evidence.py` recomputes the saved-input computational sensitivity.

## Current scientific revision

The current revision entry points are repair_population_protocol.py, assess_source_information.py and audit_population_mixtures.py; inspect their `--help` before rerunning. The full package includes the corresponding scientific_revision_20260922 and scientific_revision_20260923 reports, numeric inputs and recursive artifacts. Historical analyses above remain provenance and do not replace these revised estimands. The protocol repair trains fields if executed; it is not a smoke test. The source-information and mixture audits use existing checkpoints. Target-marginal and individual substituted-donor controls, mixture energy identity, common scoring coordinates and three-donor limits are retained.

## Paired clocks and observed expression

The full reproduction package includes the completed 12-fit paired-clock grid and its checkpoints, scaler, predictions, donor/seed scores and output manifest. Rerun with `python3 scripts/assess_paired_clock_sensitivity.py --latent outputs/analysis/human_temporal_flow/mu.npy --obs outputs/analysis/human_temporal_flow/obs_fibroblast.csv --holdout Wound7 --seeds 0,1,2 --steps 8000 --batch-size 256 --rk4-steps 50 --evaluation-cap 1200 --device cpu --output-dir outputs/reruns/paired_clock`. CUDA is an optional device choice. The software-only scope requires those inputs from the preceding analysis. Do not resume relocated historical checkpoints: their run-contract input paths/hashes describe the original execution environment, not the exported program. A fresh run creates its own contract. No GPU bundle or bundled runtime is distributed here.

`config/population_observable_protocol.json` fixes observed-expression evaluation of the saved 27-fit donor grid. After acquiring the listed public raw counts and metadata, run `python3 scripts/audit_population_observable.py --prepare --output-dir outputs/scientific_revision_20260925/population_observable/reproduction-01`, then repeat with `--run` and the same fresh directory. The output must be a fresh child of the audit's guarded directory; choose a different unused child for later reruns. The included completed preparation and results are historical evidence; they are not a reusable preparation after relocation. The analysis uses common-panel cellular proportions, training-only signatures and separately labelled observed/decoded baselines. Its decoded-target reconstruction is a reference, not a proven error floor. This within-cohort analysis adds no biological donors or clinical validation.

EXPORT_PROVENANCE.json links native source hashes to exported files. provenance/originals contains byte-exact originals for transformed reports and tables. Historical fingerprints inside reports and run contracts remain historical; execution manifests (output_manifest.json, preparation_manifest.json and artifact_manifest.json) are regenerated only for the exported artifact tree, with their originals preserved. Historical report-to-manifest hashes refer to those original producing manifests, not a relocated resume/preparation. No new raw-count, training, or clinical validation is claimed by packaging.

The mixture audit validates the current execution manifest and resolves its historical source-ablation producer hash through EXPORT_PROVENANCE.json to the byte-exact original manifest. This packaging adapter does not change the scientific report or claim it was computed with the exported source.
