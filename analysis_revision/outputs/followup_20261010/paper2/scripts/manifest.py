"""SHA256 manifest of every input read by the follow-up scripts, plus scripts and result files."""
from pathlib import Path

from p2common import H5AD, H5AD_SHA256, OUT, ROOT, sha256, write_json

inputs = [
    H5AD, ROOT / "outputs/repaired_20260918/human_temporal_flow/mu.npy",
    ROOT / "outputs/repaired_20260918/human_temporal_flow/obs_fibroblast.csv",
    ROOT / "output/local-p2-20261009/protocol.json", ROOT / "output/local-p2-20261009/jobs.jsonl",
    ROOT / "output/publication-quality-20261009/p2/scalar_records.csv.gz",
    ROOT / "output/publication-quality-20261009/p2/paired_ratios.csv",
    ROOT / "output/publication-quality-20261009/p2/source_substitution.csv",
    ROOT / "outputs/scientific_revision_20260923/population_mixture/source_per_donor.csv",
    ROOT / "outputs/scientific_revision_20260925/population_observable/features.csv",
    ROOT / "config/external_gse186527_protocol_20261009.json",
    ROOT / "output/external-gse186527-20261009/admission/groups.csv",
    ROOT / "output/external-gse186527-20261009/admission/standard_Linpos.h5ad",
    ROOT / "output/external-gse186527-20261010-replay-r1/report.json",
    ROOT / "scripts/expand_biological_evidence.py", ROOT / "scripts/18_human_temporal_flow.py",
    ROOT / "core_engines/gse241132.py", ROOT / "core_engines/flow_matching_engine.py",
] + sorted((ROOT / "outputs/biological_expansion/population").glob("Wound*_seed*_*.*")) \
  + sorted((ROOT / "expansion/population").rglob("*.py")) + [ROOT / "expansion/common/rng.py", ROOT / "expansion/common/io.py"]
manifest = {"inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
            "h5ad_matches_harness_datasets_json": sha256(H5AD) == H5AD_SHA256,
            "scripts": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "results": {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.glob("*.*"))}}
write_json(OUT / "input_sha256_manifest.json", manifest)
print(manifest["h5ad_matches_harness_datasets_json"], len(manifest["inputs"]))
