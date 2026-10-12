# Corrected journal documents

Read [the inline main manuscript](scientific-reports-manuscript-inline.pdf),
[the submission-layout main manuscript](scientific-reports-manuscript.pdf), and
[Supplementary Information](scientific-reports-supplement.pdf).

Title: Conditional flow matching in acute human wounds: a three-donor audit of population reconstruction. The software/reproduction DOI 10.5281/zenodo.23308074 was reserved
at archive assembly; it is not a manuscript or preprint DOI.

The corrected source tree is `source/population/`. It contains the full canonical and
journal TeX, bibliography, layout, journal display configuration, native and
typography-adjusted figures, build/styling programs and producing checks. All
copied scientific bytes are linked by JOURNAL_SOURCE_PROVENANCE.json.

Run from this `journal/` directory and rebuild in a fresh copy so the reference files remain unchanged:

```sh
mkdir -p tmp/journal-rebuild-01
cp -a source/population tmp/journal-rebuild-01/population
python3 tmp/journal-rebuild-01/population/scripts/build_scientific_reports.py
```

The output appears in `tmp/journal-rebuild-01/population/journal/`. The command uses
XeLaTeX, latexmk, BibTeX, TeXcount and Poppler; it does not fit models. Matching
TeX Gyre Termes/Termes Math fonts are bundled with their GUST license. Arial is a
separate licensed font requirement and its original name/hash contract is in
`source/population/fonts/FONT_CONTRACT.json`; proprietary raw Arial binaries are not
redistributed. The delivered PDFs retain their embedded fonts and can be read
without installing fonts. Optional typography regeneration uses the included
styling program with `--root` set to the fresh `population` directory and `--font-dir`
set to a licensed Arial installation.
