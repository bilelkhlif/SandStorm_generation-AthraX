#!/usr/bin/env bash
# One-command VisDrone setup + run — for Lightning AI Studio terminals (or any
# Linux box with a GPU): clone the repo, then
#
#   bash run_visdrone.sh
#
# does everything: installs deps (+ rclone if missing), downloads + extracts
# the VisDrone2019 DET/VID splits (numbered image sequences, VisDrone ships
# no video files), degrades every sequence/image found with several
# randomised variants each, packages a lightweight original+sand video (+
# metadata) bundle per unit, and auto-syncs that bundle to Google Drive as
# soon as each split finishes. Safe to re-run: already-downloaded/extracted/
# processed/exported/uploaded work is skipped, so an interrupted run just
# picks up where it left off.
#
# Optional env var overrides:
#   VISDRONE_SPLITS   space-separated split name(s), or "all"    (default: all)
#   VARIANT_MODE       "jitter" or "random"                       (default: jitter)
#   N_VARIANTS         variants per unit in jitter mode            (default: 3)
#   JITTER_FRAC        jitter window, fraction of each param's Table 4 range (default: 0.35)
#   MAX_UNITS          cap units (sequences/images) per split, 0=unlimited (default: 0)
#   MAX_FRAMES         cap frames per sequence, 0=unlimited        (default: 0)
#   DATA_ROOT          where to download/extract VisDrone          (default: data/VisDrone)
#   OUTPUT_DIR         raw per-frame ground truth, local only      (default: output_visdrone)
#   EXPORT_DIR         video+metadata bundle, this is what's uploaded (default: output_visdrone_drive)
#   RCLONE_REMOTE      rclone remote name; empty disables upload   (default: gdrive)
#   DRIVE_FOLDER_ID    target Google Drive folder ID                (default: project's folder)
#   PRUNE_LOCAL        1 = delete each split's EXPORT copy locally after a verified upload (default: 0)
#
# Every input is degraded N_VARIANTS times by default, each variant randomly
# jittering EVERY parameter around a known-good reference config (not just
# severity) with its own turbulence/wind field — real visual diversity for
# training, not the same cloud repeated. Pass VARIANT_MODE=random for the old
# single-random-draw-per-unit behaviour instead.
#
# Two tiers of output: OUTPUT_DIR keeps the full per-frame ground truth
# (clean/degraded PNGs + depth/transmission/beta/tau float32 maps) — large,
# local only, never uploaded or pruned. EXPORT_DIR is what actually goes to
# Drive: per unit, one clean.mp4 (the original footage) + per variant a
# sandstorm.mp4 and metadata.json — small, video pairs + metadata, not
# thousands of loose files.
#
# Google Drive upload needs a one-time interactive setup (Google OAuth
# requires a real browser, so this can't be done unattended the first time —
# see the README "Batch Dataset Processing (VisDrone)" section for the exact
# steps). Until that's done, this script still runs fine — upload is just
# skipped with a warning, output stays local.
#
# Recommended first call — smoke-test the full path (download -> depth ->
# degrade -> encode -> upload, all variants) on one short sequence in a few
# minutes, before committing GPU time and disk to the full dataset:
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
VARIANT_MODE="${VARIANT_MODE:-jitter}"
N_VARIANTS="${N_VARIANTS:-3}"
JITTER_FRAC="${JITTER_FRAC:-0.35}"
MAX_UNITS="${MAX_UNITS:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DATA_ROOT="${DATA_ROOT:-data/VisDrone}"
OUTPUT_DIR="${OUTPUT_DIR:-output_visdrone}"
EXPORT_DIR="${EXPORT_DIR:-output_visdrone_drive}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-1WCbhitewaqUKYb9iemvC4_UMGJtglqY0}"
PRUNE_LOCAL="${PRUNE_LOCAL:-0}"

SPLIT_ARGS=()
if [ "$VISDRONE_SPLITS" != "all" ]; then
  read -ra SPLIT_ARR <<< "$VISDRONE_SPLITS"
  SPLIT_ARGS=(--splits "${SPLIT_ARR[@]}")
fi

PRUNE_FLAG=()
if [ "$PRUNE_LOCAL" = "1" ]; then
  PRUNE_FLAG=(--prune_after_upload)
fi

echo "=== [1/4] Installing dependencies ==="
pip install -q -r requirements.txt

if [ -n "$RCLONE_REMOTE" ] && ! command -v rclone >/dev/null 2>&1; then
  echo "--- rclone not found, installing (no sudo needed) ---"
  ( curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip \
    && python -c "import zipfile; zipfile.ZipFile('/tmp/rclone.zip').extractall('/tmp/rclone_extract')" \
    && mkdir -p "$HOME/.local/bin" \
    && cp /tmp/rclone_extract/rclone-*-linux-amd64/rclone "$HOME/.local/bin/rclone" \
    && chmod +x "$HOME/.local/bin/rclone" \
  ) || echo "[warn] rclone install failed -- Drive auto-upload will be skipped this run. See README."
  export PATH="$HOME/.local/bin:$PATH"
  grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

if [ -n "$RCLONE_REMOTE" ] && command -v rclone >/dev/null 2>&1 && ! rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE}:$"; then
  echo "--- rclone remote '$RCLONE_REMOTE' is not configured yet ---"
  echo "    Drive auto-upload will be skipped this run. One-time setup (see README):"
  echo "    run 'rclone config' here, create a remote named '$RCLONE_REMOTE' (type: drive),"
  echo "    answer 'n' to auto config, and use 'rclone authorize \"drive\"' on a machine with"
  echo "    a browser to get the token this prompts you for."
fi

echo "=== [2/4] Downloading + extracting VisDrone ==="
python download_visdrone.py --data_root "$DATA_ROOT" "${SPLIT_ARGS[@]}"

echo "=== [3/4] Running degradation pipeline (GPU if available) ==="
python process_visdrone.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --export_dir "$EXPORT_DIR" \
  --variant_mode "$VARIANT_MODE" \
  --n_variants "$N_VARIANTS" \
  --jitter_frac "$JITTER_FRAC" \
  --rclone_remote "$RCLONE_REMOTE" \
  --drive_folder_id "$DRIVE_FOLDER_ID" \
  --max_units_per_split "$MAX_UNITS" \
  --max_frames_per_unit "$MAX_FRAMES" \
  "${PRUNE_FLAG[@]}" \
  "${SPLIT_ARGS[@]}"

echo "=== [4/4] Done ==="
echo "Raw ground truth (local only): $OUTPUT_DIR"
echo "Video + metadata export:       $EXPORT_DIR"
[ -n "$RCLONE_REMOTE" ] && echo "Synced export to Drive remote '$RCLONE_REMOTE' (folder $DRIVE_FOLDER_ID) if configured."
