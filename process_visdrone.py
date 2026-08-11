"""
process_visdrone.py
=====================
Batch-apply the physics-based sand/dust degradation pipeline
(sand_dust_pipeline.degrade_video) to an extracted VisDrone2019 DET/VID
dataset (see download_visdrone.py).

Each VisDrone-VID sequence (sequences/<name>/*.jpg) is degraded as one
temporally-coherent unit — the dust field advects frame-to-frame, exactly
like a normal input video. Each VisDrone-DET image (images/*.jpg) is
degraded as an independent single-frame unit (no temporal structure to
exploit, but the same physical model applies).

Every unit gets its own reproducible-but-distinct set of degradation
parameters (`sequence_seed = base_seed + unit_index`), so a batch run
produces varied storm severity/colour/turbulence across the dataset rather
than one condition repeated everywhere ("dataset mode" — see
sand_dust_pipeline.degrade_video docstring).

Resumable: a unit whose output already has metadata.json is skipped.
Fault-isolated: a failure on one unit is logged to <output_dir>/failures.log
and processing continues with the next unit.

Usage
-----
    python process_visdrone.py                          # process everything found
    python process_visdrone.py --dry_run                 # just report scope/size
    python process_visdrone.py --splits VisDrone2019-VID-val --max_frames_per_unit 20
"""

import argparse
import json
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


# =========================================================================== #
#  DATASET DISCOVERY
# =========================================================================== #

def _list_images(d: Path) -> list:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def _discover_units(split_root: Path):
    """Yield (unit_name, [image paths]) for one extracted VisDrone split.

    Auto-detects VID-style ``sequences/<name>/*.jpg`` (multi-frame units, one
    per sub-folder) or DET-style ``images/*.jpg`` (single-frame units, one
    per file). Falls back to a recursive scan grouped by parent folder for
    any other layout, so an unrecognised split still gets processed instead
    of silently producing nothing.
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
#  PER-UNIT PROCESSING
# =========================================================================== #

def _process_unit(unit_name: str, image_paths: list, split_out: Path, seed: int,
                   max_frames: int, use_gpu: bool, encode_video: bool, fps: float) -> tuple:
    if max_frames > 0:
        image_paths = image_paths[:max_frames]

    unit_out = split_out / unit_name
    if (unit_out / "metadata.json").exists():
        return "skipped", len(image_paths)

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

    degrade_video(
        clean_frames=clean_frames,
        depth_frames=depth_frames,
        output_dir=str(unit_out),
        sequence_seed=seed,
        n_ray_steps=64,
        n_blur_levels=16,
        use_gpu=use_gpu,
    )

    if encode_video and len(frames_u8) > 1:
        H, W = frames_u8[0].shape[:2]
        _encode_output_video(
            degraded_dir=unit_out / "degraded_rgb",
            output_path=unit_out / f"{unit_name}_sandstorm.mp4",
            fps=fps, width=W, height=H, n_frames=len(frames_u8),
        )

    return "processed", len(frames_u8)


# =========================================================================== #
#  SIZE ESTIMATE  (ground-truth .npy maps are the dominant cost)
# =========================================================================== #

def _estimate_and_report(all_units: list) -> None:
    """Print a lower-bound disk estimate from actual discovered frame counts
    and the resolution of the first readable frame. Excludes PNG previews,
    the optional mp4 re-encode, and the input dataset itself — real usage
    will be higher than this floor.
    """
    total_frames = sum(len(imgs) for _, _, imgs in all_units)
    if total_frames == 0:
        print("[visdrone] no frames discovered.")
        return

    sample_hw = None
    for _, _, imgs in all_units:
        img = cv2.imread(str(imgs[0]))
        if img is not None:
            sample_hw = img.shape[:2]
            break

    print(f"\n[visdrone] {len(all_units)} unit(s), {total_frames} frame(s) total")
    if sample_hw:
        H, W = sample_hw
        npy_bytes_per_frame = H * W * 4 * 4  # 4 float32 maps: depth, transmission, beta, tau
        floor_gib = total_frames * npy_bytes_per_frame / (1024 ** 3)
        print(f"[visdrone] sample resolution ~{W}x{H} -> raw .npy ground-truth alone "
              f"is >= {floor_gib:.1f} GiB (PNGs, degraded video and the source dataset are extra).")
    print("[visdrone] narrow scope with --splits / --max_units_per_split / --max_frames_per_unit "
          "if that is more than your disk/time budget.\n")


# =========================================================================== #
#  MAIN
# =========================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-degrade an extracted VisDrone2019 DET/VID dataset with the\n"
            "physics-based sand/dust pipeline. Depth is estimated per frame with\n"
            "Intel/dpt-hybrid-midas, same as process_test_video.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/VisDrone",
                         help="Root folder of extracted VisDrone splits (default: data/VisDrone)")
    parser.add_argument("--output_dir", default="output_visdrone",
                         help="Root folder for degraded output (default: output_visdrone)")
    parser.add_argument("--splits", nargs="+", default=None,
                         help="Split folder name(s) under data_root to process (default: all found)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base RNG seed; unit i uses seed+i for distinct, reproducible "
                              "storm parameters per unit (default: 42)")
    parser.add_argument("--max_units_per_split", type=int, default=0,
                         help="Cap units (sequences/images) processed per split; 0 = no limit")
    parser.add_argument("--max_frames_per_unit", type=int, default=0,
                         help="Cap frames processed per sequence; 0 = no limit")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="FPS for the optional preview mp4 only (VisDrone does not embed "
                              "a per-sequence frame rate); default: 30.0")
    parser.add_argument("--no_video", action="store_true",
                         help="Skip re-encoding each sequence's degraded frames into a preview mp4")
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

    all_units = []  # (split_dir, unit_name, image_paths)
    for split_dir in split_dirs:
        units = list(_discover_units(split_dir))
        if not units:
            print(f"[warn] no images found under {split_dir}, skipping")
            continue
        if args.max_units_per_split > 0:
            units = units[: args.max_units_per_split]
        if args.max_frames_per_unit > 0:
            units = [(n, imgs[: args.max_frames_per_unit]) for n, imgs in units]
        for unit_name, imgs in units:
            all_units.append((split_dir, unit_name, imgs))

    _estimate_and_report(all_units)
    if args.dry_run:
        return
    if not all_units:
        sys.exit("[ERROR] nothing to process.")

    _load_midas()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "failures.log"

    summary = {}
    t0 = time.time()
    for i, (split_dir, unit_name, image_paths) in enumerate(tqdm(all_units, desc="visdrone", unit="unit")):
        split_out = output_root / split_dir.name
        s = summary.setdefault(split_dir.name, {"processed": 0, "skipped": 0, "failed": 0, "total": 0})
        s["total"] += 1
        try:
            status, _ = _process_unit(
                unit_name, image_paths, split_out,
                seed=args.seed + i,
                max_frames=args.max_frames_per_unit,
                use_gpu=not args.cpu,
                encode_video=not args.no_video,
                fps=args.fps,
            )
            s["processed" if status == "processed" else "skipped"] += 1
        except Exception as exc:
            s["failed"] += 1
            msg = f"{split_dir.name}/{unit_name}: {exc!r}"
            print(f"  [FAIL] {msg}")
            with open(failures_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    elapsed = time.time() - t0
    summary["_elapsed_seconds"] = elapsed
    with open(output_root / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print("[visdrone] RUN SUMMARY")
    for split_name, s in summary.items():
        if split_name.startswith("_"):
            continue
        print(f"  {split_name:30s} processed={s['processed']:5d}  skipped={s['skipped']:5d}  "
              f"failed={s['failed']:5d}  / {s['total']}")
    print(f"  elapsed : {elapsed / 3600:.2f} h")
    print(f"  output  : {output_root.resolve()}")
    if any(s.get("failed") for k, s in summary.items() if not k.startswith("_")):
        print(f"  failures: {failures_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
