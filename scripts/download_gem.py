"""Download CMS 2018 ICD-9 CM → ICD-10 CM GEM crosswalk.

Usage: python scripts/download_gem.py [--out data/raw/gem/2018_I9gem.txt]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

GEM_URL = "https://www.cms.gov/files/zip/2018-icd-10-cm-and-gems.zip"
GEM_FILENAME = "2018_I9gem.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/gem/2018_I9gem.txt"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GEM_URL} ...")
    with urllib.request.urlopen(GEM_URL) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    # Find the GEM file (case-insensitive, may be in a subdirectory)
    gem_entry = next(
        (n for n in zf.namelist() if n.upper().endswith(GEM_FILENAME.upper())), None
    )
    if gem_entry is None:
        raise FileNotFoundError(f"{GEM_FILENAME} not found in ZIP. Names: {zf.namelist()}")
    args.out.write_bytes(zf.read(gem_entry))
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
