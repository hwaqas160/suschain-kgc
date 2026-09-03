"""Builds the Zenodo release archive.

Includes the code, the pipeline, every result file, the two source datasets
and the enriched sustainability labels. Excludes the 1.2 GB Open Food Facts
dump and the sentence-transformer embedding cache: both are third-party or
regenerable, and scripts/08_offmatch_lodhalal.py rebuilds them.

Run: python scripts/make_release.py
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "suschain-kgc-replication.zip"

INCLUDE_DIRS = ["src", "scripts", "configs", "results", "tests"]
INCLUDE_FILES = ["README.md", "requirements.txt", "pyproject.toml",
                 ".zenodo.json", "CITATION.cff"]
INCLUDE_DATA = [
    "data/raw/lodhalal",
    "data/raw/anonnet",
    "data/processed/lodhalal_food_off_matched.parquet",
    "data/external/owid_co2_data.csv",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints",
              "text_embeddings", "checkpoints"}


def keep(p: Path) -> bool:
    if any(part in SKIP_PARTS for part in p.parts):
        return False
    return p.suffix not in {".pyc", ".pt", ".gz"}


def add(zf, path: Path, count: list):
    rel = path.relative_to(ROOT)
    zf.write(path, arcname=str(Path("suschain-kgc") / rel))
    count[0] += 1
    count[1] += path.stat().st_size


def main():
    count = [0, 0]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and keep(p):
                    add(zf, p, count)
        for f in INCLUDE_FILES:
            p = ROOT / f
            if p.is_file():
                add(zf, p, count)
        for d in INCLUDE_DATA:
            p = ROOT / d
            if p.is_file():
                add(zf, p, count)
            elif p.is_dir():
                for q in sorted(p.rglob("*")):
                    if q.is_file() and keep(q):
                        add(zf, q, count)

    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}")
    print(f"  files      : {count[0]}")
    print(f"  uncompressed: {count[1] / 1e6:.1f} MB")
    print(f"  archive    : {OUT.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256     : {sha}")
    (OUT.parent / "suschain-kgc-replication.zip.sha256").write_text(
        f"{sha}  {OUT.name}\n")


if __name__ == "__main__":
    main()
