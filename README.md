# Conditional flow matching in acute human wounds: analysis software

Author: Zeyu Fu. Software version: 0.4.2.

Analysis software for a three-donor audit of population reconstruction by conditional flow matching in acute human wounds. Product-coupled fields operate in a frozen 15-dimensional expression representation. The study distinguishes global time holdout from donor transfer and compares endpoint controls, training-target marginals, paired clocks, distribution shape and observed expression. Simple permitted-data predictors outperform flow in important comparisons. The study does not identify individual cell trajectories or validate clinical wound prognosis.

The acute human time course contains three donors. The mouse comparison has one animal per arm and time point. Numerical resolutions, seeds and evaluation draws do not increase the donor count. Bootstrap draws and cells are not additional independent patients.

[Source code](https://github.com/PeterPonyu/wound-population-dynamics) · [Versioned downloads](https://github.com/PeterPonyu/wound-population-dynamics/releases/tag/v0.4.2) · [Software DOI](https://doi.org/10.5281/zenodo.22962339)

Read the [manuscript](output/pdf/wound_population_dynamics.pdf) and the [figure collection](manuscripts/figures/figures.pdf). The package contains 7 editable R/TikZ vector figures, numerical tables, completed reports, and its own copy of the frozen expression model. RESULTS_INDEX.json maps numerical reports to manuscript use. A copy of the model is included here so the project runs independently.

![Study design](manuscripts/figures/figure1_workflow.png)

## Rebuild from saved numerical inputs

First download and extract the complete reproduction ZIP from the versioned Releases page. The commands below run from that extracted root, not the lightweight Git checkout.

```sh
python3 verify_archive.py --smoke
python3 manuscripts/latex/export_tables.py
python3 manuscripts/build_main_figures.py
python3 manuscripts/latex/build.py --render
python3 -m pytest -p no:cacheprovider tests -q
```

The figure and PDF commands require R with ggplot2, patchwork, tikzDevice, jsonlite and digest; XeLaTeX/BibTeX/latexmk; Poppler; and installed Arial and TeX Gyre fonts. Fonts are not redistributed. Python package versions are recorded in requirements.txt and environment.json. CPU CLI smoke checks and saved-input rendering do not train models. Some optional analysis commands below fit models; they are not part of the release verification workflow.

## Rights

The MIT License applies to software, including analysis and rendering programs. Manuscripts, figures, tables, saved models, numerical results and source-study data retain their respective rights as set out in NOTICE. The full reproduction ZIP and the software-only Zenodo ZIP are different artifacts with separate manifests and checksums.

## Citation and software archiving

Use CITATION.cff for software attribution: [10.5281/zenodo.22962339](https://doi.org/10.5281/zenodo.22962339). This software DOI does not identify the manuscript. ARCHIVING.md describes artifact scopes and checksum verification.

## Data and research-material availability

Public processed count matrices and author annotations are available from these source records:

- [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816)
- [GSE241132](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241132)
- [GSE326622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE326622)

REPRODUCIBILITY.md describes downloads, input locations and analysis order. Public study materials require no request to the author. Source-study permissions remain in force: processed public matrices are not the same as unrestricted human sequencing reads. The complete reproduction package includes derived numerical reports, frozen weights, editable figures and the manuscript; these materials are openly downloadable from this repository's versioned Releases page without author approval or an access request. They retain the rights stated in NOTICE; public availability does not relicense source-study data.

## Mathematical verification

Version 0.4.2 corrects method descriptions and adds explicit estimands, analytic unit tests and a frozen-input sensitivity report. Read METHODS_CONTRACT.md and CHANGELOG.md for the interpretation and provenance boundaries. Recompute the added analysis with `python3 scripts/audit_population_math.py --output-dir outputs/reruns/mathematical_audit`. This uses saved inputs and does not refit the original representation or temporal fields.

## Observed-cell and experimental extension

Version 0.4.2 adds observed-cell maps, raw-count expression context and explicitly bounded experiments. Two globally missing times are compared against unchanged source, within-donor centroid translation and an independent-endpoint bridge across three seeds. The centroid predictor has lower mean donor error in both tasks; the earlier held-donor ranking is task-specific. The frozen original reports remain unchanged.

## Complete reproduction download

Download [wound-population-dynamics-0.4.2.zip](https://github.com/PeterPonyu/wound-population-dynamics/releases/tag/v0.4.2) for the exact manuscript, figures, frozen models, completed numerical reports and checksum manifest. The Git source checkout excludes large research-output directories; extract the complete ZIP before running saved-input figure/PDF rebuilds. The software-only ZIP is sufficient for code inspection and CPU import/unit tests, but is not a pretrained-result bundle.
