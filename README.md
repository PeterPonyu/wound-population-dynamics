# Conditional flow matching in acute human wounds: analysis software and reproduction archive

**Read the [corrected inline main manuscript](journal/scientific-reports-manuscript-inline.pdf).**
The [submission-layout main manuscript](journal/scientific-reports-manuscript.pdf)
and [Supplementary Information](journal/scientific-reports-supplement.pdf) are also included.

Software/reproduction version: **0.5.0**. DOI **10.5281/zenodo.23308074** was reserved at archive
assembly; remote publication is a separate operation. That identifier identifies
the software/reproduction release, not a manuscript or preprint. The immutable
prior software-only snapshot has DOI 10.5281/zenodo.22884874 and does not archive this revision.

A three-donor audit of population reconstruction by conditional flow matching in acute human wounds. The release includes the producing programs, public processed counts, frozen coordinates, saved fields and predictions, numerical evidence, protocols and corrected journal documents. It distinguishes global time holdout from donor transfer, retains adverse control comparisons and records strict replay limits. The analysis does not identify individual cell trajectories or validate clinical wound prognosis.

The full archive includes exact producing analysis code, recursive local imports,
public processed inputs, saved states, numerical tables, gene/mapping lists and
protocols. Current native paths are preserved beneath `analysis_revision/`.
Start with [the reproduction guide](REPRODUCIBILITY.md) and
[the current analysis guide](analysis_revision/REPRODUCIBILITY.md).

After downloading and extracting the complete version archive, run:

```sh
python3 verify_archive.py --smoke
```

The checksum/import/CLI smoke performs no fitting or downloads. Run the saved-input
checks documented in REPRODUCIBILITY.md using fresh output directories. The exact
journal source, figures, font contracts and build configuration are in
`journal/source/population/`; journal/README.md describes a fresh-copy rebuild.

The frozen representation and source data retain their original shared provenance.
Independent packaging does not add statistical independence or biological donors.
Original-CUDA replay evidence, prior CPU/PCA failures and historical state-retention limits remain in the analysis verification records. Imports and checksum checks do not certify strict saved-field CPU replay or full retraining.

MIT applies to authored software. Scholarly documents, figures, saved research
outputs, source-study materials and third-party fonts retain the rights stated in
[NOTICE](NOTICE). Public processed count files are distinct from controlled human
sequencing reads; the latter were not used or redistributed. Font binaries and
third-party assets are documented with their own license/availability contracts.

Historical manuscript/analysis snapshots remain as provenance. The corrected
documents linked above are the default reading products for this release.


## Complete versioned reproduction package

This Git checkout and the source-and-documents asset provide current producing
source, corrected manuscripts and compact saved results. Large numerical arrays,
model states and inputs are in the complete version archive at
[software DOI 10.5281/zenodo.23308074](https://doi.org/10.5281/zenodo.23308074).
The public record determines publication status; the sealed package was assembled
while its DOI was reserved. Download every numbered ZIP part and follow
REASSEMBLE.md and SHA256SUMS. The reconstructed ZIP has SHA256
`29d6045135a55e3f4857603773d24dc0d2098f8bc274643c67c50709a37ee7d3`.

Extract the full archive before running full-manifest checks or scripts that use
the large inputs. The full archive's release-manifest.json describes that archive;
SOURCE_DOCUMENTS_MANIFEST.json describes this compact source/docs selection.
The DOI identifies software and research sources, not a manuscript publication.
