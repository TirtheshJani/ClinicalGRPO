"""Download CMS 2018 ICD-9 CM -> ICD-10 CM GEM crosswalk.

Usage: python scripts/download_gem.py [--out data/raw/gem/2018_I9gem.txt]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

# CMS archive URL (returns 404 since the GEM-archive page reorg). The mirror
# below is the same file content (2018 forward GEM, ICD-9 -> ICD-10) checked
# into a public NCBI-Hackathons repo.
GEM_URL_CMS = "https://www.cms.gov/files/zip/2018-icd-10-cm-and-gems.zip"
GEM_URL_MIRROR = (
    "https://raw.githubusercontent.com/NCBI-Hackathons/"
    "Design-of-ICD-9-to-10-conversion-function-for-the-R-package-icd/"
    "master/R/2018_I9gem.txt"
)
GEM_FILENAME = "2018_I9gem.txt"


def _try_cms_zip(out_path: Path) -> bool:
    try:
        with urllib.request.urlopen(GEM_URL_CMS, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False
    zf = zipfile.ZipFile(io.BytesIO(data))
    gem_entry = next(
        (n for n in zf.namelist() if n.upper().endswith(GEM_FILENAME.upper())), None
    )
    if gem_entry is None:
        return False
    out_path.write_bytes(zf.read(gem_entry))
    return True


def _try_mirror(out_path: Path) -> bool:
    with urllib.request.urlopen(GEM_URL_MIRROR, timeout=30) as resp:
        data = resp.read()
    if len(data) < 100_000:
        return False
    out_path.write_bytes(data)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/gem/2018_I9gem.txt"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Trying CMS source: {GEM_URL_CMS}")
    if _try_cms_zip(args.out):
        print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes) from CMS")
        return

    print(f"CMS unavailable, falling back to mirror: {GEM_URL_MIRROR}")
    if _try_mirror(args.out):
        print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes) from mirror")
        return

    raise RuntimeError(
        "Could not fetch GEM file from CMS or mirror. "
        f"Manually place {GEM_FILENAME} at {args.out}."
    )


if __name__ == "__main__":
    main()
