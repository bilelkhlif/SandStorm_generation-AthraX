#!/usr/bin/env bash
# One-command VisDrone setup + run — for Lightning AI Studio terminals (or any
# Linux box with a GPU): clone the repo, then
#
#   bash run_visdrone.sh
#
# does everything: installs deps, downloads + extracts the VisDrone2019
# DET/VID splits, and runs the sand/dust degradation pipeline over all of
# them. Safe to re-run: already-downloaded/extracted/processed work is
# skipped, so an interrupted run just picks up where it left off.
#
# Optional env var overrides:
#   VISDRONE_SPLITS   space-separated split name(s), or "all"    (default: all)
#   MAX_UNITS         cap units (sequences/images) per split, 0=unlimited (default: 0)
#   MAX_FRAMES        cap frames per sequence, 0=unlimited        (default: 0)
#   DATA_ROOT         where to download/extract VisDrone          (default: data/VisDrone)
#   OUTPUT_DIR        where degraded output goes                  (default: output_visdrone)
#
# Recommended first call — smoke-test the full path (download -> depth ->
# degrade -> encode) on one short sequence in a couple of minutes, before
# committing GPU time and disk to the full dataset:
#
#   VISDRONE_SPLITS="VisDrone2019-VID-val" MAX_UNITS=1 MAX_FRAMES=20 bash run_visdrone.sh
#
# To see the full scope (unit/frame counts and a disk-size floor) without
# processing anything:
#
#   python process_visdrone.py --dry_run

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VISDRONE_SPLITS="${VISDRONE_SPLITS:-all}"
MAX_UNITS="${MAX_UNITS:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DATA_ROOT="${DATA_ROOT:-data/VisDrone}"
OUTPUT_DIR="${OUTPUT_DIR:-output_visdrone}"

SPLIT_ARGS=()
if [ "$VISDRONE_SPLITS" != "all" ]; then
  read -ra SPLIT_ARR <<< "$VISDRONE_SPLITS"
  SPLIT_ARGS=(--splits "${SPLIT_ARR[@]}")
fi

echo "=== [1/3] Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== [2/3] Downloading + extracting VisDrone ==="
python download_visdrone.py --data_root "$DATA_ROOT" "${SPLIT_ARGS[@]}"

echo "=== [3/3] Running degradation pipeline ==="
python process_visdrone.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --max_units_per_split "$MAX_UNITS" \
  --max_frames_per_unit "$MAX_FRAMES" \
  "${SPLIT_ARGS[@]}"

echo ""
echo "Done. Degraded VisDrone output: $OUTPUT_DIR"
