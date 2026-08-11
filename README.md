# SandStorm-Video Generator

Physics-based sand and dust video degradation for autonomous driving datasets.
The pipeline renders temporally coherent degraded video pairs — clean RGB,
degraded RGB, and physical ground-truth maps (optical depth, transmission,
extinction coefficient) — from any input video and a monocular depth estimate.

---

## Table of Contents

1. [Features](#features)
2. [Physical Model](#physical-model)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Batch Dataset Processing (VisDrone)](#batch-dataset-processing-visdrone)
6. [Parameter Reference](#parameter-reference)
7. [Output Structure](#output-structure)
8. [Citation](#citation)
9. [License](#license)

---

## Features

- Kolmogorov turbulence density field with k^{−11/6} amplitude spectrum and log-normal transform
- Beer–Lambert spectral extinction with per-pixel optical depth computed by ray-marching (N=64 steps)
- Mie forward-scattering PSF applied via Gaussian pyramid scale-space interpolation — O(M·H·W)
- Per-pixel multiple-scattering glow using the Henyey–Greenstein phase function
- Temporally coherent sequences via curl-noise advection and controllable `rho_refresh_rate`
- Per-frame MiDaS monocular depth estimation (Intel/dpt-hybrid-midas, ~500 MB, auto-downloaded)
- Streamlit GUI with interactive parameter sliders and real-time progress
- CLI for batch dataset generation
- Raw float32 `.npy` ground-truth maps alongside 16-bit PNG previews

---

## Physical Model

### 1 — Kolmogorov turbulence density field

```
ρ(x) = exp( ρ_G(x) − σ²/2 )      ρ_G ~ N(0, σ²)
```

White noise is shaped by k^{−11/6} in Fourier space (Obukhov–Corrsin inertial
range for a passive scalar) and passed through a log-normal transform to
guarantee a positive-definite concentration field.

### 2 — Extinction and optical depth (ray-marching)

```
β(x)  = β₀ · ρ(x) / ⟨ρ⟩              (Doc B, Eq. 2)
τ(x)  = Σᵢ β(sᵢ) Δs    (N = 64 steps)
t(x)  = exp(−τ(x))                    Beer–Lambert
```

### 3 — Mie PSF (depth-varying Gaussian blur)

```
P(θ) = (1 − g²) / [ 4π (1 + g² − 2g cosθ)^{3/2} ]   (HG phase function)
σ_PSF(x) = σ₀ · τ(x)^{0.6}                            (Doc A, Alg. 1)
```

### 4 — Multiple-scattering glow and final composition

```
I_MS = γ · (J_blur * G_{σ₀√τ}) · (1 − t)             (Doc B, Eq. 11)
I    = J_blur · t  +  A · (1 − t)  +  I_MS            (Doc A, §5.4)
```

---

## Installation

```bash
git clone https://github.com/bilelkhlif/SandStorm_generation-AthraX.git
cd SandStorm_generation-AthraX
pip install -r requirements.txt
```

### Depth model (MiDaS)

On the **first run** the pipeline downloads Intel/dpt-hybrid-midas (~500 MB)
from HuggingFace Hub automatically and caches it in `~/.cache/huggingface/hub`.
No manual download is required.

If you prefer to work offline, place the model files in a folder named
`midas_model/` in the project root:

```
midas_model/
  model.safetensors      (or pytorch_model.bin)
  config.json
  preprocessor_config.json
```

Direct download links:

```
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/model.safetensors
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/config.json
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/preprocessor_config.json
```

If you downloaded `pytorch_model.bin` instead of `model.safetensors`, convert
it once with:

```bash
python -c "
import torch
from safetensors.torch import save_file
state = torch.load('midas_model/pytorch_model.bin', map_location='cpu', weights_only=True)
save_file(state, 'midas_model/model.safetensors')
"
```

---

## Usage

### GUI (recommended)

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a video, adjust parameters, click **Run pipeline**.

### CLI

```bash
python process_test_video.py --input path/to/video.mp4 --output_dir output/
```

All parameters are sampled from Table 4 ranges with `--seed 42` by default.

---

## Batch Dataset Processing (VisDrone)

Applies the pipeline across an entire [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)
DET/VID download instead of one video at a time — built for a cloud GPU box
(e.g. a Lightning AI Studio) where you clone the repo and want one command
to download the dataset and degrade all of it, unattended.

```bash
git clone https://github.com/bilelkhlif/SandStorm_generation-AthraX.git
cd SandStorm_generation-AthraX
bash run_visdrone.sh
```

That installs dependencies (+ rclone if needed), downloads + extracts the 7
official VisDrone2019 splits (DET train/val/test-dev/test-challenge, VID
val/test-dev/test-challenge — ~8.2 GB compressed), degrades every
sequence/image found on GPU, and auto-syncs each split's video export to
Google Drive as soon as it's done. VisDrone ships as numbered image
sequences, never video files — `sequences/<name>/0000001.jpg, 0000002.jpg,
...` — and the pipeline reads that format directly; each VisDrone-VID
sequence is processed as one temporally-coherent clip (the dust field
advects frame-to-frame across its frames, same as a normal input video
would), and each VisDrone-DET image is processed as an independent single
frame. The scripts turn the *output* into actual `.mp4` video (see below) —
the input side never assumes one.

**Randomised variants, for training-data diversity.** By default every unit
is degraded `N_VARIANTS` times (default 3). For each variant, *every*
physical parameter — β₀, g, C²ᵨ, γ, σ₀, atmospheric colour, refresh rate —
is independently randomised, jittered around a reference configuration
(a GUI run confirmed to look convincing on real aerial footage: β₀=0.01,
g=0.80, C²ᵨ=0.00189, γ=0.40, σ₀=1.60, A=(0.94,0.85,0.63), refresh=0.10) by
`JITTER_FRAC` (default 0.35, i.e. 35% of that parameter's full Table 4
range) and clipped back into the physically valid range. Each variant also
gets its own RNG seed, so the turbulence/wind field differs too — different
cloud *shapes*, not just the same cloud dimmed or brightened, which is what
actually gives a training set diversity rather than one look repeated.
Output goes to `<unit>/variant_NN_betaX.XXXX/`. Pass `VARIANT_MODE=random`
for the old behaviour instead — one random draw per unit uniform over the
full Table 4 range, no centring, output directly under `<unit>/`.

The run is resumable and fault-isolated: already-downloaded archives,
already-extracted splits, and already-processed (unit, variant) pairs are
skipped on re-run — if every variant of a unit is already done, its frames
aren't even reloaded/re-depth-estimated — and a failure on one variant is
logged to `output_visdrone/failures.log` without stopping sibling variants
or other units.

### Two-tier output: full ground truth locally, video + metadata to Drive

`degrade_video()`'s full per-frame output (clean/degraded PNGs plus
depth/transmission/beta/tau as float32 `.npy` + 16-bit PNG preview — the
complete scientific ground truth) is written to `OUTPUT_DIR`
(`output_visdrone/`) exactly as documented below. That tree is **local
only** — large, never uploaded, never pruned.

What actually goes to Drive is a separate, lightweight bundle in
`EXPORT_DIR` (`output_visdrone_drive/`), built from that raw output after
each unit finishes: one shared `clean.mp4` per unit (the original footage —
identical across its variants, so it's encoded once, not duplicated), and
per variant a `sandstorm.mp4` (the degraded footage) plus `metadata.json`.
Single-frame DET units get `clean.png` / `degraded.png` instead of video,
since there's nothing to encode. This is "original video, sand video, and
metadata for each video" — not thousands of loose per-frame files.

### Auto-upload to Google Drive

The export bundle is rclone-synced to a Drive folder as soon as each split
finishes (not just at the end — this overlaps upload with the next split's
GPU work). The raw ground-truth tree is never uploaded.
This needs a **one-time interactive setup**, because Google OAuth requires
a real browser and can't be completed unattended on a headless Studio the
first time — after that, every future run uploads automatically with no
further action.

1. On the Studio (`run_visdrone.sh` installs `rclone` itself if missing):
   ```bash
   rclone config
   ```
   Choose `n` (new remote) → name it `gdrive` → type `drive` (Google Drive)
   → leave client_id/secret blank → scope `drive` → leave root_folder_id
   blank (the scripts pass the target folder explicitly) → when asked **"Use
   auto config?" answer `n`** (headless machine, no browser here). It prints
   something like `rclone authorize "drive" "<long token>"`.
2. **On your own machine** (which has a real browser and is logged into the
   Google account that owns/can-edit the target Drive folder), install
   rclone and run the exact command printed above:
   ```bash
   rclone authorize "drive" "<the long token from step 1>"
   ```
   This opens a browser, you approve access, and it prints a result token.
3. Paste that result back into the waiting prompt on the Studio. `rclone
   config` finishes and saves the `gdrive` remote — done permanently for
   this Studio (persists across runs, not just this session).

Until this is done, `run_visdrone.sh` still runs fine — upload is skipped
with a clear warning and output just stays local. Target folder defaults to
[this project's Drive folder](https://drive.google.com/drive/folders/1WCbhitewaqUKYb9iemvC4_UMGJtglqY0)
(override with `DRIVE_FOLDER_ID`); the authorising Google account needs
edit access to it. Pass `PRUNE_LOCAL=1` to delete each split's local
**export** copy right after a *verified* successful upload — always safe,
since it only ever deletes the small derived video bundle, never the raw
ground-truth tree, so nothing irreplaceable is lost. Off by default.

**Check scope before committing to a full run** — the raw local tree is
the large part: the four float32 `.npy` ground-truth maps per frame are
uncompressed, `clean_rgb/` and `depth_maps/` are duplicated per variant too
(each is a fully independent `degrade_video()` output), and a full DET+VID
run at 3 variants can reach the high hundreds of GB *locally*. The Drive
export is much smaller — compressed video, not raw arrays.

```bash
python process_visdrone.py --dry_run
```

This reports discovered unit/frame counts (already multiplied by variant
count) and a local disk-size floor (raw `.npy` maps only — PNGs and the
source dataset add more) without processing anything. Narrow scope with the
flags/env vars below if the raw local total exceeds your Studio's disk or
time budget.

| Env var (`run_visdrone.sh`) | Script flag | Default | Purpose |
|---|---|---|---|
| `VISDRONE_SPLITS` | `--splits` | `all` | Which split(s) to fetch/process, e.g. `"VisDrone2019-VID-val"` |
| `VARIANT_MODE` | `--variant_mode` | `jitter` | `jitter` = every param randomised per variant; `random` = legacy single draw |
| `N_VARIANTS` | `--n_variants` | `3` | Variants generated per unit in jitter mode |
| `JITTER_FRAC` | `--jitter_frac` | `0.35` | Jitter width as a fraction of each param's full Table 4 range |
| `MAX_UNITS` | `--max_units_per_split` | `0` (unlimited) | Cap sequences/images processed per split |
| `MAX_FRAMES` | `--max_frames_per_unit` | `0` (unlimited) | Cap frames processed per sequence |
| `DATA_ROOT` | `--data_root` | `data/VisDrone` | Download/extraction location |
| `OUTPUT_DIR` | `--output_dir` | `output_visdrone` | Raw per-frame ground truth, local only |
| `EXPORT_DIR` | `--export_dir` | `output_visdrone_drive` | Video + metadata bundle — this is what gets uploaded |
| `RCLONE_REMOTE` | `--rclone_remote` | `gdrive` | rclone remote name; empty disables upload |
| `DRIVE_FOLDER_ID` | `--drive_folder_id` | (project folder) | Target Google Drive folder ID |
| `PRUNE_LOCAL` | `--prune_after_upload` | `0` (off) | `1` = delete local **export** copy after a verified upload |

Recommended first call — a full download→depth→degrade→export→upload smoke
test (all variants) on one short sequence, done in a few minutes:

```bash
VISDRONE_SPLITS="VisDrone2019-VID-val" MAX_UNITS=1 MAX_FRAMES=20 bash run_visdrone.sh
```

Raw output (`output_visdrone/`) mirrors the dataset layout with a variant
folder under each unit —
`output_visdrone/<split>/<sequence-or-image-name>/variant_00_betaX.XXXX/` —
each containing `clean_rgb/ degraded_rgb/ depth_maps/ transmission_maps/
beta_maps/ tau_maps/ metadata.json` as described below. (In legacy
`VARIANT_MODE=random` there's no variant folder — output goes directly
under `<unit>/`.) A top-level `output_visdrone/run_summary.json` records
processed/skipped/failed counts per split, counted per variant.

Export output (`output_visdrone_drive/` — same split/unit layout) has, per
unit, `clean.mp4` plus one folder per variant —
`variant_00_betaX.XXXX/{sandstorm.mp4, metadata.json}` — this is what's
synced to Drive.

`download_visdrone.py` and `process_visdrone.py` also run standalone if you
want finer control than the env vars above (`--help` on either for the full
flag list). Note VisDrone-VID-train (7.5 GB) is intentionally not included —
it isn't in the shared folder this integration targets; add its file ID to
`_SPLITS` in `download_visdrone.py` if you need it.

---

## Parameter Reference

| Symbol | Name | Range | Default | Physical meaning |
|--------|------|-------|---------|-----------------|
| β₀ | Mean extinction | 0.002 – 0.020 m⁻¹ | 0.008 | Light attenuation per metre; controls storm opacity |
| g | HG asymmetry | 0.70 – 0.90 | 0.80 | Forward-scattering strength of dust particles |
| C²ᵨ | Turbulence constant | 10⁻⁴ – 10⁻² | 10⁻³ | Spatial patchiness of particle concentration |
| γ | MS glow strength | 0.20 – 0.60 | 0.40 | Intensity of multiple-scattering halo |
| σ₀ | PSF base spread | 0.5 – 2.0 px | 1.0 | Blur radius at τ = 1 |
| **A** | Atmospheric light | RGB colour | #F0D8A0 | Warm dust-tone colour filling occluded regions |
| r | Refresh rate | 0.0 – 1.0 | 0.10 | Temporal correlation: 0 = steady, 1 = per-frame random |
| N | Ray steps | 16 – 128 | 64 | Integration accuracy along each camera ray |

---

## Output Structure

```
output_sandstorm/
├── clean_rgb/              frame_NNNN.png   — 8-bit RGB input
├── degraded_rgb/           frame_NNNN.png   — 8-bit RGB degraded output
├── depth_maps/             frame_NNNN.png   — 16-bit preview
│                           frame_NNNN.png.npy — float32 depth [m]
├── transmission_maps/      frame_NNNN.png + .npy — t ∈ [0, 1]
├── beta_maps/              frame_NNNN.png + .npy — β [m⁻¹]
├── tau_maps/               frame_NNNN.png + .npy — τ (dimensionless)
├── sandstorm_video.mp4     — re-encoded degraded video
└── metadata.json           — all parameters + per-frame normalisation factors
```

### Recovering physical units from PNG previews

Each 16-bit PNG is normalised by its per-frame maximum before saving.
The raw physical value is always in the `.npy` file, or can be recovered:

```python
import numpy as np, cv2, json

meta    = json.load(open("output_sandstorm/metadata.json"))
max_val = meta["frame_map_maxvals"][0]["beta_per_m"]   # m⁻¹
png     = cv2.imread("output_sandstorm/beta_maps/frame_0000.png", cv2.IMREAD_UNCHANGED)
beta    = png / 65535.0 * max_val   # physical β in m⁻¹
```

---

## Citation

```bibtex
@article{khlif2026sandstorm,
  title   = {SandStorm-Video: A Benchmark for Sand and Dust Video Degradation},
  author  = {Khlif, Bilel et al.},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
