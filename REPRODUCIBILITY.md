# Reproduction guide

## Data and biological units

The acute human time course contains three donors. The mouse comparison has one animal per arm and time point. Numerical resolutions, seeds and evaluation draws do not increase the donor count.

Source counts and annotations are supplied by the following GEO records, under their source-study terms:

- [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816)
- [GSE241132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241132)
- [GSE326622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE326622)

Download counts into `data/raw/<accession>/`. Acquisition helpers are in scripts/fetch_*.sh. Human wound cell annotations must be joined by barcode before selecting fibroblasts. GSE165816 supplies the independently frozen representation; it does not supply a temporal healing target.

## Analysis order

Run from the archive root. Each program documents options with `--help`. Some programs consume completed outputs from earlier steps; preserve reference reports and use fresh output directories for alternative runs. A complete raw-data rerun requires count downloads and substantially more computation than rendering saved results.

1. Fit the 7,002-gene, 15-coordinate discovery representation with `python3 scripts/fit_expression_representation.py --steps 40000 --seed 0`. The generic CLI default is shorter than the manuscript's 40,000 steps. Hardware-dependent training may differ from the frozen reference weights.
2. Run `reconstruct_held_out_timepoint.py`, then `compare_time_coordinates.py`. These evaluate a time point omitted from fitting; training-only scaling excludes target cells.
3. Run `evaluate_donor_transfer.py`, then `benchmark_population_predictors.py`. These evaluate a donor omitted from fitting and must not be conflated with a time-point holdout. Training endpoints are paired within donor.
4. Run `evaluate_training_donor_count.py`, `assess_training_seed_stability.py`, `infer_donor_count_sensitivity.py` and `infer_donor_geometry.py` for the donor and seed analyses.
5. Run `evaluate_mouse_timepoint.py --model-arm NDB`, and analogously PDB and GDB with separate output directories, before `summarize_mouse_model_arms.py`. Arm names come from the source study.
6. Run `assess_robustness.py` after the human temporal analysis to fit one seed-0 field (8,000 steps), compare 25/50/100/200 RK4 steps and evaluate 50 subsampling draws at each of 250/500/1,000 cells per distribution. The energy distance is the empirical V-statistic. Central evaluation-draw ranges measure Monte Carlo sensitivity conditional on one field and the observed cohort; they are not generalization confidence intervals for new donors.

## Verification scope

The archive verifier checks the checksum manifest, Python syntax, matching citation metadata and CLI imports. The full reproduction package additionally checks the frozen decoder, deterministic projection and training-only standardization. Manuscript builds check protected narrative sections, adjacent equation references, table citations, embedded fonts, complete citations and figure placement. Figure builds enforce black bold lettering, vector content and unchanged numerical source files.

The software-only archive contains programs, not pretrained weights or completed observations. Its checksum/import checks do not claim that a full raw-count pipeline has been rerun on every platform. Provenance fingerprints inside reports describe the original computation; MANIFEST.sha256.json hashes the files in the exact distributed artifact.

## Mathematical audit

Run `python3 -m unittest discover -s tests -v` for analytic implementation checks. Run `python3 scripts/audit_population_math.py --output-dir outputs/reruns/mathematical_audit` after producing its saved inputs. The full reproduction package includes those inputs; the software-only archive requires the preceding analysis steps. Historical report hashes describe the original computation, while the release manifest describes the current exported files. Documentation corrections are not a claim that every original model has been refitted.
