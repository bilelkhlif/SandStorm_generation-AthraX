"""
process_visdrone.py
=====================
Batch-apply the physics-based sand/dust degradation pipeline
(sand_dust_pipeline.degrade_video) to an extracted VisDrone2019 DET/VID
dataset (see download_visdrone.py). Built for training-data generation:
maximum visual diversity per input, GPU-fast, auto-synced to Google Drive.

VisDrone ships as image sequences, not video files: each VisDrone-VID
sequence is a folder of numbered frames (sequences/<name>/0000001.jpg, ...),
degraded here as one temporally-coherent unit (the dust field advects
frame-to-frame, exactly like a normal input video would). Each VisDrone-DET
image (images/*.jpg) is a standalone single-frame unit. Nothing in the
read path assumes a video container -- _discover_units() reads numbered
frame files directly.

Variant sampling ("jitter" mode, default)
------------------------------------------
Every unit is degraded --n_variants times (default 3). For each variant,
ALL seven physical parameters (beta_0, g, C_rho_sq, gamma, sigma_0,
atmospheric colour, refresh rate) are independently randomised around a
reference configuration — the values from a GUI run the author confirmed
looks good (app.py, beta_0=0.01, g=0.80, C_rho_sq=0.00189, gamma=0.40,
sigma_0=1.60, A=(0.94,0.85,0.63), refresh_rate=0.10) — within a window
--jitter_frac (default 0.35) wide relative to each parameter's full Table 4
range, clipped back into that range. Each variant also gets its own RNG
seed, so the turbulence/wind field differs too, not just the parameter
values. Pass --variant_mode random for the old behaviour instead: one
random draw per unit from the full Table 4 range, no reference centring.

Two-tier output
-----------------
Raw output (--output_dir, default output_visdrone/) keeps the full
per-frame ground truth exactly as before: clean_rgb/, degraded_rgb/,
depth_maps/, transmission_maps/, beta_maps/, tau_maps/, metadata.json per
(unit, variant) — this is the scientifically complete artifact and stays
local only.

Export output (--export_dir, default output_visdrone_drive/) is the
lightweight bundle that actually gets uploaded: per unit, one shared
clean.mp4 (the original footage, identical across its variants, encoded
once) plus, per variant, sandstorm.mp4 (the degraded footage) and
metadata.json. Single-frame DET units get clean.png/degraded.png instead
of video. This is what --rclone_remote syncs to Drive, not the raw tree —
video pairs + metadata, not thousands of loose PNG/npy files.

Auto-upload to Google Drive
-----------------------------
If --rclone_remote (default "gdrive") is configured and authorised on this
machine, each split's EXPORT bundle is rclone-synced to the Drive folder
identified by --drive_folder_id as soon as that split finishes. If rclone
isn't installed/configured, this is skipped with a warning, not a crash —
see README for the one-time setup (Google OAuth needs a real browser, so it
can't be fully automated from a headless box the first time).
--prune_after_upload deletes each split's local EXPORT copy (not the raw
ground-truth tree) right after a verified successful upload — always safe,
since it's a small derived copy, not the only copy of anything.

Resumable: a (unit, variant) pair whose raw output already has
metadata.json is skipped; if every variant of a unit is already done, frame
loading + depth estimation for that unit is skipped entirely. The export
step independently skips whatever's already staged. Fault-isolated:
failures are logged to <output_dir>/failures.log and don't stop the rest.

Usage
-----
    python process_visdrone.py                          # 3 jittered variants, everything found
    python process_visdrone.py --dry_run                 # just report scope/size
    python process_visdrone.py --variant_mode random      # legacy: 1 random draw/unit
    python process_visdrone.py --n_variants 5 --jitter_frac 0.5
    python process_visdrone.py --splits VisDrone2019-VID-val --max_frames_per_unit 20
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from sand_dust_pipeline import degrade_video
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import sand_dust_pipeline: {exc}\n"
        "Make sure sand_dust_pipeline.py is in the same directory or on PYTHONPATH."
    )

try:
    from process_test_video import _load_midas, _estimate_depth_midas, _encode_output_video
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import process_test_video: {exc}")

_IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Reference config: a GUI run (app.py) confirmed to look like a convincing,
# usable sandstorm on real aerial/drone footage. Jitter mode samples around
# this rather than the full Table 4 range so variants stay "in the
# neighbourhood of a look that's known to work" while still all differing.
_REFERENCE_PARAMS = {
    "beta_0": 0.01000,
    "g": 0.80000,
    "C_rho_sq": 0.00189,
    "gamma": 0.40000,
    "sigma_0": 1.60000,
    "A": [0.9411764740943985, 0.8470588326454163, 0.6274510025978088],
    "rho_refresh_rate": 0.10,
}

# Table 4 physically-valid ranges (mirrors sand_dust_pipeline.sample_parameters).
_PARAM_RANGES = {
    "beta_0": (0.002, 0.02),
    "g": (0.70, 0.90),
    "gamma": (0.20, 0.60),
    "sigma_0": (0.5, 2.0),
    "rho_refresh_rate": (0.0, 1.0),
}
_LOG_C_RHO_SQ_RANGE = (-4.0, -2.0)  # log10(C_rho_sq), C_rho_sq itself spans 1e-4..1e-2


# =========================================================================== #
#  DATASET DISCOVERY  (VisDrone = numbered image files, never a video container)
# =========================================================================== #

def _list_images(d: Path) -> list:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def _discover_units(split_root: Path):
    """Yield (unit_name, [image paths]) for one extracted VisDrone split.

    Auto-detects VID-style ``sequences/<name>/*.jpg`` (multi-frame units, one
    per sub-folder of numbered frames) or DET-style ``images/*.jpg``
    (single-frame units, one per file). Falls back to a recursive scan
    grouped by parent folder for any other layout, so an unrecognised split
    still gets processed instead of silently producing nothing.
    """
    seq_root = split_root / "sequences"
    if seq_root.is_dir():
        for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            imgs = _list_images(seq_dir)
            if imgs:
                yield seq_dir.name, imgs
        return

    img_root = split_root / "images"
    if img_root.is_dir():
        for img in _list_images(img_root):
            yield img.stem, [img]
        return

    found = False
    for sub in sorted(p for p in split_root.rglob("*") if p.is_dir()):
        imgs = _list_images(sub)
        if imgs:
            found = True
            yield sub.relative_to(split_root).as_posix().replace("/", "__"), imgs
    if not found:
        for img in _list_images(split_root):
            yield img.stem, [img]


# =========================================================================== #
#  VARIANT PARAMETER SAMPLING
# =========================================================================== #

def _jittered_params(rng: np.random.Generator, jitter_frac: float) -> dict:
    """Sample one full parameter set, every parameter jittered independently
    around _REFERENCE_PARAMS within jitter_frac of its Table 4 range, clipped
    back into that range. C_rho_sq is jittered in log-space (it spans two
    orders of magnitude, same convention as sample_parameters()).
    """
    def jit(name):
        lo, hi = _PARAM_RANGES[name]
        half = jitter_frac * (hi - lo) / 2.0
        ref = _REFERENCE_PARAMS[name]
        return float(np.clip(rng.uniform(ref - half, ref + half), lo, hi))

    log_lo, log_hi = _LOG_C_RHO_SQ_RANGE
    log_ref = math.log10(_REFERENCE_PARAMS["C_rho_sq"])
    half_log = jitter_frac * (log_hi - log_lo) / 2.0
    C_rho_sq = float(10 ** np.clip(rng.uniform(log_ref - half_log, log_ref + half_log), log_lo, log_hi))

    A_ref = np.array(_REFERENCE_PARAMS["A"], dtype=np.float32)
    A = np.clip(A_ref + rng.uniform(-0.06, 0.06, size=3).astype(np.float32), 0.0, 1.0)

    return {
        "beta_0": jit("beta_0"),
        "g": jit("g"),
        "C_rho_sq": C_rho_sq,
        "gamma": jit("gamma"),
        "sigma_0": jit("sigma_0"),
        "A": A.tolist(),
        "rho_refresh_rate": jit("rho_refresh_rate"),
    }


# =========================================================================== #
#  PER-UNIT PROCESSING  (raw ground truth -- unchanged output format)
# =========================================================================== #

def _variant_targets(unit_out: Path, seed: int, variant_mode: str, n_variants: int, jitter_frac: float) -> list:
    """Compute (label, output_dir, (seed, params)|None) for every variant of
    a unit. Pure function of (seed, mode, n_variants, jitter_frac) -- doesn't
    touch disk or frames, so callers can check what's already done before
    deciding whether to load anything.
    """
    if variant_mode == "random":
        return [("", unit_out, None)]  # params=None -> degrade_video samples internally
    variants = []
    for v in range(n_variants):
        v_seed = seed * 10_000 + v  # unique across (unit, variant), own turbulence/wind field
        v_params = _jittered_params(np.random.default_rng(v_seed), jitter_frac)
        label = f"variant_{v:02d}_beta{v_params['beta_0']:.4f}"
        variants.append((label, unit_out / label, (v_seed, v_params)))
    return variants


def _process_unit(unit_name: str, image_paths: list, split_out: Path, seed: int,
                   max_frames: int, use_gpu: bool, variant_mode: str, n_variants: int,
                   jitter_frac: float) -> list:
    """Produce the full raw ground-truth output for every variant of one
    unit. Returns a list of (label, status) — status is "processed",
    "skipped", or "failed: <reason>".
    """
    unit_out = split_out / unit_name
    variants = _variant_targets(unit_out, seed, variant_mode, n_variants, jitter_frac)

    # If every variant is already done, skip frame loading + depth estimation
    # entirely — the expensive part of a resumed run.
    if all((out / "metadata.json").exists() for _, out, _ in variants):
        return [(label, "skipped") for label, _, _ in variants]

    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    frames_u8 = []
    for p in image_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"    [warn] unreadable, skipping frame: {p}")
            continue
        frames_u8.append(bgr[..., ::-1].copy())  # BGR -> RGB
    if not frames_u8:
        raise RuntimeError("no readable frames in unit")

    depth_frames = [_estimate_depth_midas(f) for f in frames_u8]
    clean_frames = [f.astype(np.float32) / 255.0 for f in frames_u8]

    results = []
    for label, v_out, payload in variants:
        if (v_out / "metadata.json").exists():
            results.append((label, "skipped"))
            continue
        try:
            v_seed, v_params = (seed, None) if payload is None else payload
            degrade_video(
                clean_frames=clean_frames,
                depth_frames=depth_frames,
                output_dir=str(v_out),
                sequence_seed=v_seed,
                params=v_params,
                n_ray_steps=64,
                n_blur_levels=16,
                use_gpu=use_gpu,
            )
            results.append((label, "processed"))
        except Exception as exc:
            results.append((label, f"failed: {exc!r}"))

    return results


# =========================================================================== #
#  EXPORT  (lightweight Drive bundle: original video + sand video + metadata)
# =========================================================================== #

def _encode_from_dir(frame_dir: Path, output_path: Path, fps: float) -> bool:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if len(frames) <= 1:
        return False
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    H, W = first.shape[:2]
    _encode_output_video(degraded_dir=frame_dir, output_path=output_path,
                          fps=fps, width=W, height=H, n_frames=len(frames))
    return output_path.exists()


def _export_unit(unit_name: str, variants: list, unit_export: Path,
                  encode_video: bool, fps: float) -> None:
    """Stage the lightweight Drive bundle for one unit from whatever raw
    output is already on disk (just-produced or from an earlier run) --
    never reprocesses frames. One shared clean.mp4 (identical across
    variants, so encoded once), then per variant: sandstorm.mp4 +
    metadata.json. Single-frame units get clean.png/degraded.png instead of
    video. Skips anything already staged.
    """
    src = next((out for _, out, _ in variants if (out / "metadata.json").exists()), None)
    if src is None:
        return  # nothing finished for this unit yet

    unit_export.mkdir(parents=True, exist_ok=True)
    clean_dir = src / "clean_rgb"
    clean_frames = sorted(clean_dir.glob("frame_*.png"))

    if len(clean_frames) > 1:
        clean_video = unit_export / "clean.mp4"
        if encode_video and not clean_video.exists():
            _encode_from_dir(clean_dir, clean_video, fps)
    elif len(clean_frames) == 1:
        clean_img = unit_export / "clean.png"
        if not clean_img.exists():
            shutil.copy2(clean_frames[0], clean_img)

    for label, v_out, _ in variants:
        if not (v_out / "metadata.json").exists():
            continue
        v_export = unit_export / label if label else unit_export
        v_export.mkdir(parents=True, exist_ok=True)

        meta_dst = v_export / "metadata.json"
        if not meta_dst.exists():
            shutil.copy2(v_out / "metadata.json", meta_dst)

        deg_dir = v_out / "degraded_rgb"
        deg_frames = sorted(deg_dir.glob("frame_*.png"))
        if len(deg_frames) > 1:
            deg_video = v_export / "sandstorm.mp4"
            if encode_video and not deg_video.exists():
                _encode_from_dir(deg_dir, deg_video, fps)
        elif len(deg_frames) == 1:
            deg_img = v_export / "degraded.png"
            if not deg_img.exists():
                shutil.copy2(deg_frames[0], deg_img)


# =========================================================================== #
#  GOOGLE DRIVE AUTO-UPLOAD  (rclone)
# =========================================================================== #

def _rclone_available() -> bool:
    return shutil.which("rclone") is not None


def _rclone_sync(local_dir: Path, remote: str, folder_id: str, dest_subpath: str) -> bool:
    """Copy local_dir's contents into <remote>:<dest_subpath>, rooted at the
    Drive folder identified by folder_id. Never raises -- upload failures are
    logged and treated as non-fatal so a flaky network doesn't cost local
    processing progress. Returns True on success.
    """
    if not local_dir.exists():
        return True
    cmd = [
        "rclone", "copy", str(local_dir), f"{remote}:{dest_subpath}",
        "--drive-root-folder-id", folder_id,
        "--transfers", "8", "--checkers", "8", "-q",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  [upload WARN] rclone exited {result.returncode} for {local_dir}:\n"
                  f"{result.stderr[-2000:]}")
            return False
        return True
    except Exception as exc:
        print(f"  [upload WARN] rclone failed for {local_dir}: {exc!r}")
        return False


# =========================================================================== #
#  SIZE ESTIMATE  (ground-truth .npy maps are the dominant local-disk cost)
# =========================================================================== #

def _estimate_and_report(all_units: list, n_variants: int) -> None:
    """Print a lower-bound disk estimate from actual discovered frame counts
    and the resolution of the first readable frame. Covers the RAW local
    output only (the export bundle is comparatively tiny — compressed
    video, not raw float32 arrays). Excludes PNG previews, the mp4s, and the
    input dataset itself — real local usage will be higher than this floor.
    """
    total_frames = sum(len(imgs) for _, _, imgs in all_units)
    if total_frames == 0:
        print("[visdrone] no frames discovered.")
        return
    total_frame_outputs = total_frames * n_variants

    sample_hw = None
    for _, _, imgs in all_units:
        img = cv2.imread(str(imgs[0]))
        if img is not None:
            sample_hw = img.shape[:2]
            break

    print(f"\n[visdrone] {len(all_units)} unit(s), {total_frames} source frame(s) "
          f"x {n_variants} variant(s) = {total_frame_outputs} degraded frame(s) total")
    if sample_hw:
        H, W = sample_hw
        npy_bytes_per_frame = H * W * 4 * 4  # 4 float32 maps: depth, transmission, beta, tau
        floor_gib = total_frame_outputs * npy_bytes_per_frame / (1024 ** 3)
        print(f"[visdrone] sample resolution ~{W}x{H} -> raw local .npy ground-truth alone "
              f">= {floor_gib:.1f} GiB (PNGs and the source dataset are extra; clean_rgb/ and "
              f"depth_maps/ are also duplicated per variant locally). The exported Drive bundle "
              f"(compressed clean.mp4 + sandstorm.mp4 + metadata.json per unit/variant) is far "
              f"smaller -- that's the only part that leaves this machine.")
    print("[visdrone] narrow scope with --splits / --max_units_per_split / --max_frames_per_unit / "
          "--n_variants if the raw local total is more than your disk/time budget. "
          "--prune_after_upload only ever deletes the small export copy, never the raw tree.\n")


# =========================================================================== #
#  MAIN
# =========================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-degrade an extracted VisDrone2019 DET/VID dataset with the\n"
            "physics-based sand/dust pipeline, for training-data generation.\n"
            "Depth is estimated per frame with Intel/dpt-hybrid-midas, same as\n"
            "process_test_video.py. Raw per-frame ground truth stays local; a\n"
            "lightweight original+sand video (+ metadata) bundle per unit is what\n"
            "gets auto-uploaded to Drive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/VisDrone",
                         help="Root folder of extracted VisDrone splits (default: data/VisDrone)")
    parser.add_argument("--output_dir", default="output_visdrone",
                         help="Root folder for raw per-frame ground-truth output, local only "
                              "(default: output_visdrone)")
    parser.add_argument("--export_dir", default="output_visdrone_drive",
                         help="Root folder for the lightweight video+metadata bundle that gets "
                              "uploaded to Drive (default: output_visdrone_drive)")
    parser.add_argument("--splits", nargs="+", default=None,
                         help="Split folder name(s) under data_root to process (default: all found)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base RNG seed; unit i uses seed+i as the root for its variants' seeds "
                              "(default: 42)")
    parser.add_argument("--variant_mode", choices=["jitter", "random"], default="jitter",
                         help="'jitter' (default): every parameter randomised per variant, each "
                              "independently seeded (own turbulence/wind field too), centred on a "
                              "known-good reference config, spread set by --jitter_frac -- built for "
                              "training-data diversity. 'random': legacy single draw per unit, "
                              "uniform over the full Table 4 range, no centring.")
    parser.add_argument("--n_variants", type=int, default=3,
                         help="Independent degraded variants generated per unit in jitter mode "
                              "(default: 3). Ignored in random mode (always 1).")
    parser.add_argument("--jitter_frac", type=float, default=0.35,
                         help="Jitter window width as a fraction of each parameter's full Table 4 "
                              "range, centred on the reference config (default: 0.35)")
    parser.add_argument("--rclone_remote", default="gdrive",
                         help="rclone remote name to auto-sync the export bundle to after each split "
                              "finishes; empty string disables upload. Requires a one-time "
                              "'rclone config' on this machine authorising Google Drive access -- "
                              "see README. (default: gdrive)")
    parser.add_argument("--drive_folder_id", default="1WCbhitewaqUKYb9iemvC4_UMGJtglqY0",
                         help="Google Drive folder ID to upload into (the id from the folder's URL). "
                              "Default is the project's configured destination folder.")
    parser.add_argument("--prune_after_upload", action="store_true",
                         help="Delete each split's local EXPORT copy (never the raw ground-truth "
                              "tree) right after a verified successful Drive upload. Off by default.")
    parser.add_argument("--max_units_per_split", type=int, default=0,
                         help="Cap units (sequences/images) processed per split; 0 = no limit")
    parser.add_argument("--max_frames_per_unit", type=int, default=0,
                         help="Cap frames processed per sequence; 0 = no limit")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="FPS for the exported mp4s only (VisDrone does not embed a "
                              "per-sequence frame rate); default: 30.0")
    parser.add_argument("--no_video", action="store_true",
                         help="Export metadata (and, for single-frame units, PNGs) without encoding "
                              "mp4s for multi-frame units")
    parser.add_argument("--cpu", action="store_true", help="Force CPU path even if a GPU is available")
    parser.add_argument("--dry_run", action="store_true",
                         help="Discover units and print the scope/size estimate, then exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"[ERROR] data_root not found: {data_root}\n"
                  "Run download_visdrone.py first (or use run_visdrone.sh).")

    n_variants = 1 if args.variant_mode == "random" else max(1, args.n_variants)

    if args.splits:
        split_dirs = [data_root / s for s in args.splits]
        missing = [str(d) for d in split_dirs if not d.is_dir()]
        if missing:
            sys.exit(f"[ERROR] split(s) not found: {missing}")
    else:
        split_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name != "_archives")
    if not split_dirs:
        sys.exit(f"[ERROR] No split directories found under {data_root}")

    print(f"[visdrone] {len(split_dirs)} split(s) to process: {[d.name for d in split_dirs]}")
    print(f"[visdrone] variant_mode={args.variant_mode}  n_variants={n_variants}"
          + (f"  jitter_frac={args.jitter_frac}" if args.variant_mode == "jitter" else ""))

    units_by_split = {}
    for split_dir in split_dirs:
        units = list(_discover_units(split_dir))
        if not units:
            print(f"[warn] no images found under {split_dir}, skipping")
            continue
        if args.max_units_per_split > 0:
            units = units[: args.max_units_per_split]
        if args.max_frames_per_unit > 0:
            units = [(n, imgs[: args.max_frames_per_unit]) for n, imgs in units]
        units_by_split[split_dir] = units

    all_units_flat = [(sd, n, imgs) for sd, us in units_by_split.items() for n, imgs in us]
    _estimate_and_report(all_units_flat, n_variants)
    if args.dry_run:
        return
    if not all_units_flat:
        sys.exit("[ERROR] nothing to process.")

    upload_enabled = bool(args.rclone_remote)
    if upload_enabled and not _rclone_available():
        print("[visdrone] WARNING: --rclone_remote is set but rclone is not installed/on PATH -- "
              "export will NOT be auto-uploaded this run. See README for one-time setup.")
        upload_enabled = False
    elif upload_enabled:
        print(f"[visdrone] auto-upload: {args.rclone_remote}: -> Drive folder {args.drive_folder_id} "
              f"(export bundle, after each split; prune_after_upload={args.prune_after_upload})")

    _load_midas()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "failures.log"

    summary = {}
    global_i = 0
    t0 = time.time()
    for split_dir, units in units_by_split.items():
        split_out = output_root / split_dir.name
        split_export = export_root / split_dir.name
        s = summary.setdefault(split_dir.name, {"processed": 0, "skipped": 0, "failed": 0, "total": 0})

        for unit_name, image_paths in tqdm(units, desc=split_dir.name, unit="unit"):
            seed = args.seed + global_i
            global_i += 1
            try:
                results = _process_unit(
                    unit_name, image_paths, split_out,
                    seed=seed,
                    max_frames=args.max_frames_per_unit,
                    use_gpu=not args.cpu,
                    variant_mode=args.variant_mode,
                    n_variants=n_variants,
                    jitter_frac=args.jitter_frac,
                )
            except Exception as exc:
                s["total"] += n_variants
                s["failed"] += n_variants
                msg = f"{split_dir.name}/{unit_name}: (frame loading) {exc!r}"
                print(f"  [FAIL] {msg}")
                with open(failures_path, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
                continue

            for label, status in results:
                s["total"] += 1
                tag = f"{unit_name}/{label}" if label else unit_name
                if status == "processed":
                    s["processed"] += 1
                elif status == "skipped":
                    s["skipped"] += 1
                else:
                    s["failed"] += 1
                    msg = f"{split_dir.name}/{tag}: {status}"
                    print(f"  [FAIL] {msg}")
                    with open(failures_path, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")

            # Stage the lightweight export bundle from whatever raw output
            # now exists on disk (just-produced or already there).
            try:
                variants = _variant_targets(split_out / unit_name, seed, args.variant_mode,
                                             n_variants, args.jitter_frac)
                _export_unit(unit_name, variants, split_export / unit_name,
                              encode_video=not args.no_video, fps=args.fps)
            except Exception as exc:
                msg = f"{split_dir.name}/{unit_name}: (export) {exc!r}"
                print(f"  [FAIL] {msg}")
                with open(failures_path, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")

        # Split finished -- sync the export bundle now rather than waiting
        # for the whole run, so upload overlaps with the next split's GPU
        # work. The raw ground-truth tree is never uploaded or pruned.
        if upload_enabled:
            print(f"[upload] syncing {split_export} -> {args.rclone_remote}:{split_dir.name} ...")
            ok = _rclone_sync(split_export, args.rclone_remote, args.drive_folder_id, split_dir.name)
            if ok and args.prune_after_upload:
                print(f"[prune] removing local export copy {split_export} after verified upload")
                shutil.rmtree(split_export, ignore_errors=True)
            elif not ok:
                print(f"  [upload WARN] {split_dir.name} export not fully synced -- kept locally "
                      f"regardless of --prune_after_upload; re-run to retry.")

    elapsed = time.time() - t0
    summary["_elapsed_seconds"] = elapsed
    summary["_variant_mode"] = args.variant_mode
    summary["_n_variants"] = n_variants
    with open(output_root / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if upload_enabled:
        print("[upload] final sync (any retries) ...")
        _rclone_sync(export_root, args.rclone_remote, args.drive_folder_id, "")

    print("\n" + "=" * 60)
    print("[visdrone] RUN SUMMARY")
    print(f"  variant_mode: {args.variant_mode}  n_variants: {n_variants}")
    for split_name, s in summary.items():
        if split_name.startswith("_"):
            continue
        print(f"  {split_name:30s} processed={s['processed']:5d}  skipped={s['skipped']:5d}  "
              f"failed={s['failed']:5d}  / {s['total']}")
    print(f"  elapsed    : {elapsed / 3600:.2f} h")
    print(f"  raw output : {output_root.resolve()}  (local only, full ground truth)")
    print(f"  export     : {export_root.resolve()}  (video + metadata bundle)")
    if upload_enabled:
        print(f"  uploaded   : {args.rclone_remote}:  (Drive folder {args.drive_folder_id})")
    if any(s.get("failed") for k, s in summary.items() if not k.startswith("_")):
        print(f"  failures   : {failures_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
