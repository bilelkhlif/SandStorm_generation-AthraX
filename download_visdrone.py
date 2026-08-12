"""
download_visdrone.py
=====================
Downloads and extracts the VisDrone2019 DET + VID splits used by
process_visdrone.py.

File IDs below point into
https://drive.google.com/drive/folders/1TX4TAFiOOkOcId2s8dJYD-hB2mPD9kvS —
the same 7 files (by name and size) as the official links published at
https://github.com/VisDrone/VisDrone-Dataset, but distinct Drive copies
with their own file IDs, not the same file objects. That matters in
practice: the official IDs are so heavily downloaded that Google routinely
rate-limits them ("too many users have viewed or downloaded this file"),
while this folder's copies are not — confirmed by testing an actual
download against both. VID-train is intentionally not included: it is not
present in that shared folder.

Idempotent: re-running skips any split whose zip already exists (validated
as a real zip, not just present) and whose extraction folder already has
content. Safe to re-run after an interrupted or rate-limited download.

Google Drive rate-limiting: these are popular official files, and Google
sometimes blocks anonymous downloads globally ("too many users have viewed
or downloaded this file recently") for anywhere from minutes to ~24h — this
is not specific to your machine or this script. Rather than give up after a
few seconds, each file is retried patiently (default: every 5 minutes, for
up to 6 hours) so you can start this once and leave it running; it grabs
the file as soon as Google unblocks it. Tune with --retry_interval_min /
--max_wait_hours, or just re-run the script later (already-downloaded
files are skipped).

Usage
-----
    python download_visdrone.py                              # all 7 splits
    python download_visdrone.py --splits VisDrone2019-VID-val
    python download_visdrone.py --data_root /custom/path
    python download_visdrone.py --max_wait_hours 12           # more patient
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
# IDs are this project's own Drive folder copies (see module docstring) --
# verified working (DET-val downloaded clean) when the official IDs were
# rate-limited.
_SPLITS = {
    "VisDrone2019-DET-train":          ("1RqAJfK5k-hvMYDaWJBUSHBDTA75ON60R", "VisDrone2019-DET-train.zip"),
    "VisDrone2019-DET-val":            ("1G35pedLyEwWWOpfY91QJ_fVilMvitOe1", "VisDrone2019-DET-val.zip"),
    "VisDrone2019-DET-test-dev":       ("1_V-g3y5KjqRi1vO4MRGq9R4l6qhkTRYO", "VisDrone2019-DET-test-dev.zip"),
    "VisDrone2019-DET-test-challenge": ("1yRo7eEP4ZSC5uyRNS5nyGTy_4hTBtmxP", "VisDrone2019-DET-test-challenge.zip"),
    "VisDrone2019-VID-val":            ("1LX22RAKBVFotS0fQFPomRZ38lXd0stGM", "VisDrone2019-VID-val.zip"),
    "VisDrone2019-VID-test-dev":       ("1fn1hTFd3ygyrKTvV0AkYi9Dk9tIbxJkb", "VisDrone2019-VID-test-dev.zip"),
    "VisDrone2019-VID-test-challenge": ("1-KIo5O4_zhz3TcWeiAlwM6-cr2Sa0cdX", "VisDrone2019-VID-test-challenge.zip"),
}


def _download_one(name: str, file_id: str, zip_name: str, archive_dir: Path,
                   max_wait_hours: float = 6.0, retry_interval_min: float = 5.0) -> Path:
    """Download one file, patiently retrying on Google Drive's "too many
    users" rate limit instead of giving up after a few seconds. Keeps
    trying every retry_interval_min minutes for up to max_wait_hours before
    raising -- meant to be started once and left running.
    """
    dest = archive_dir / zip_name
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[skip] {name}: already downloaded ({dest})")
        return dest

    url = f"https://drive.google.com/uc?id={file_id}"
    deadline = time.time() + max_wait_hours * 3600
    attempt = 0
    while True:
        attempt += 1
        try:
            gdown.download(url, str(dest), quiet=False)
            if dest.exists() and zipfile.is_zipfile(dest):
                return dest
            print(f"[warn] {name}: downloaded file is not a valid zip (attempt {attempt})")
        except Exception as exc:
            print(f"[warn] {name}: {exc} (attempt {attempt})")
        if dest.exists():
            dest.unlink()

        remaining_sec = deadline - time.time()
        if remaining_sec <= 0:
            raise RuntimeError(
                f"still rate-limited after {max_wait_hours:.1f}h and {attempt} attempts. "
                f"Google says this can take up to 24h to clear. Re-run this script later "
                f"(add --max_wait_hours to wait longer next time) — already-completed "
                f"splits are skipped automatically."
            )
        wait_sec = min(retry_interval_min * 60, remaining_sec)
        print(f"[wait] {name}: Google Drive is rate-limiting this file right now — "
              f"retrying in {wait_sec / 60:.0f} min (will keep trying for up to "
              f"{remaining_sec / 3600:.1f}h more) ...")
        time.sleep(wait_sec)


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
    ap.add_argument("--max_wait_hours", type=float, default=6.0,
                     help="Keep retrying a rate-limited file for up to this many hours before "
                          "giving up on it and moving to the next split (default: 6)")
    ap.add_argument("--retry_interval_min", type=float, default=5.0,
                     help="Minutes to wait between retries while rate-limited (default: 5)")
    args = ap.parse_args()

    splits = list(_SPLITS) if "all" in args.splits else args.splits

    archive_dir = args.data_root / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[visdrone] {len(splits)} split(s) -> {args.data_root.resolve()}")
    failed = []
    for name in splits:
        file_id, zip_name = _SPLITS[name]
        try:
            zip_path = _download_one(name, file_id, zip_name, archive_dir,
                                      max_wait_hours=args.max_wait_hours,
                                      retry_interval_min=args.retry_interval_min)
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
