"""
download_visdrone.py
=====================
Downloads and extracts the VisDrone2019 DET + VID splits used by
process_visdrone.py.

File IDs below are the official links published at
https://github.com/VisDrone/VisDrone-Dataset (verified against the shared
folder https://drive.google.com/drive/folders/1TX4TAFiOOkOcId2s8dJYD-hB2mPD9kvS
— same 7 files, matching sizes). VID-train is intentionally not included:
it is not present in that shared folder.

Idempotent: re-running skips any split whose zip already exists (validated
as a real zip, not just present) and whose extraction folder already has
content. Safe to re-run after an interrupted or rate-limited download.

Usage
-----
    python download_visdrone.py                              # all 7 splits
    python download_visdrone.py --splits VisDrone2019-VID-val
    python download_visdrone.py --data_root /custom/path
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("[ERROR] gdown is not installed. Run: pip install -r requirements.txt")

# split name -> (google_drive_file_id, zip_filename)
_SPLITS = {
    "VisDrone2019-DET-train":          ("1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn", "VisDrone2019-DET-train.zip"),
    "VisDrone2019-DET-val":            ("1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59", "VisDrone2019-DET-val.zip"),
    "VisDrone2019-DET-test-dev":       ("1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V", "VisDrone2019-DET-test-dev.zip"),
    "VisDrone2019-DET-test-challenge": ("1KN8R3oioOvSXH492GEVk-Hx74nWHAcXT", "VisDrone2019-DET-test-challenge.zip"),
    "VisDrone2019-VID-val":            ("1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B", "VisDrone2019-VID-val.zip"),
    "VisDrone2019-VID-test-dev":       ("1-BEq--FcjshTF1UwUabby_LHhYj41os5", "VisDrone2019-VID-test-dev.zip"),
    "VisDrone2019-VID-test-challenge": ("1Qwyp_cEpGyXGqJ8IbusEzuNHgbM403NP", "VisDrone2019-VID-test-challenge.zip"),
}


def _download_one(name: str, file_id: str, zip_name: str, archive_dir: Path, retries: int = 3) -> Path:
    dest = archive_dir / zip_name
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[skip] {name}: already downloaded ({dest})")
        return dest

    url = f"https://drive.google.com/uc?id={file_id}"
    for attempt in range(1, retries + 1):
        try:
            gdown.download(url, str(dest), quiet=False)
            if dest.exists() and zipfile.is_zipfile(dest):
                return dest
            print(f"[warn] {name}: downloaded file is not a valid zip (attempt {attempt}/{retries})")
        except Exception as exc:
            print(f"[warn] {name}: download failed: {exc} (attempt {attempt}/{retries})")
        if dest.exists():
            dest.unlink()
        if attempt < retries:
            time.sleep(5 * attempt)

    raise RuntimeError(
        f"failed after {retries} attempts. Google Drive sometimes rate-limits "
        f"anonymous downloads of popular files ('too many users have viewed or "
        f"downloaded this file'). Wait a few minutes and re-run this script — "
        f"already-completed splits are skipped automatically."
    )


def _extract_one(name: str, zip_path: Path, data_root: Path) -> None:
    # VisDrone zips contain one top-level folder matching the archive name;
    # used only as a skip-check heuristic, extraction itself is unconditional
    # and safe to repeat if the guess is wrong.
    marker = data_root / zip_path.stem
    if marker.is_dir() and any(marker.iterdir()):
        print(f"[skip] {name}: already extracted ({marker})")
        return
    print(f"[extract] {name} -> {data_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/VisDrone", type=Path,
                     help="Where to extract the dataset (default: data/VisDrone)")
    ap.add_argument("--splits", nargs="+", choices=list(_SPLITS) + ["all"], default=["all"],
                     help="Which split(s) to download (default: all)")
    args = ap.parse_args()

    splits = list(_SPLITS) if "all" in args.splits else args.splits

    archive_dir = args.data_root / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[visdrone] {len(splits)} split(s) -> {args.data_root.resolve()}")
    failed = []
    for name in splits:
        file_id, zip_name = _SPLITS[name]
        try:
            zip_path = _download_one(name, file_id, zip_name, archive_dir)
            _extract_one(name, zip_path, args.data_root)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            failed.append(name)

    print("\n" + "=" * 60)
    ok = len(splits) - len(failed)
    print(f"[visdrone] {ok}/{len(splits)} split(s) ready in {args.data_root.resolve()}")
    if failed:
        print(f"[visdrone] FAILED: {', '.join(failed)} — re-run this script to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
