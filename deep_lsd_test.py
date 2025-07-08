#!/usr/bin/env python3
"""
DeepLSD ‑ batch inference (CPU‑friendly)
=======================================
Runs DeepLSD on **all** images in an input directory and saves the line‑overlay
visualisations to an output directory.

Defaults are set for your SATELLITE dataset so you can simply:
```bash
python deep_lsd_batch.py          # uses ~90 % of available CPUs
```

Key points
-----------
* **Default input  = /datasets/SATELLITE/crop_smooth**
* **Default output = /datasets/SATELLITE/deepLSD_results_ver1**
* **Default device = CPU**  (pass `--device cuda` only if you really want GPU)
* **Workers**: automatically picks **≈ 90 % of available CPUs** (respects
  `SLURM_CPUS_PER_TASK` when present).
"""
from __future__ import annotations

import argparse
import math
import os
import cv2
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import torch
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines
from tqdm import tqdm

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Utils                                                                   │
# ╰──────────────────────────────────────────────────────────────────────────╯


def _overlay_lines(img, lines, color=(0, 0, 255), thickness=2):
    """Draws line segments returned by DeepLSD onto the image (in‑place)."""
    for x1, y1, x2, y2, _ in lines:  # last value is score
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


# The model will be **global** inside each worker so it is loaded only once
_MODEL = None
_DEVICE = None


def _init_worker(device: str):
    """Initialises global model for each subprocess."""
    global _MODEL, _DEVICE
    _DEVICE = torch.device(device)
    _MODEL = DeepLSD.from_pretrained().to(_DEVICE).eval()


def _process_one(task: Tuple[Path, Path]):
    """Runs DeepLSD on a single image and writes visualisation."""
    in_path, out_path = task
    img_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {in_path}")

    with torch.no_grad():
        result = _MODEL(img_bgr, as_numpy=True)
    lines = result["lines"]  # (N,5)  x1,y1,x2,y2,score

    vis = img_bgr.copy()
    _overlay_lines(vis, lines)
    cv2.imwrite(str(out_path), vis)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Helpers                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯


def _default_workers() -> int:
    """Compute 90 % of available CPU cores (≥1).

    * If running under SLURM and `SLURM_CPUS_PER_TASK` is defined, use that as
      the baseline.
    * Falls back to `os.cpu_count()` otherwise.
    """
    env_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    try:
        total = int(env_cpus) if env_cpus and int(env_cpus) > 0 else None
    except ValueError:
        total = None

    if total is None:
        total = os.cpu_count() or 4

    return max(1, math.floor(total * 0.9))


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Main                                                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯


def main():
    parser = argparse.ArgumentParser(description="Batch DeepLSD inference")
    parser.add_argument("--input_dir", type=Path,
                        default=Path("/datasets/SATELLITE/crop_smooth"),
                        help="Directory with input images (default: %(default)s)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/datasets/SATELLITE/deepLSD_results_ver1"),
                        help="Directory where visualised results are saved (default: %(default)s)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Computation device (default: cpu)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: 90%% of CPUs)")
    parser.add_argument("--ext", default="png",
                        help="Image extension to search for (default: png)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available — aborting.")

    # Decide number of workers (≈90 % CPUs) if not provided
    if args.workers is None or args.workers <= 0:
        args.workers = _default_workers()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(args.input_dir.glob(f"*.{args.ext}")))
    if not img_paths:
        parser.error(f"No '*.{args.ext}' images found in {args.input_dir}")

    tasks = []
    for p in img_paths:
        out_name = p.stem + "_lines.png"
        tasks.append((p, args.output_dir / out_name))

    print(f"Using {args.workers} worker process(es) on {args.device.upper()} …")

    # Multiprocessing pool
    with mp.Pool(processes=args.workers, initializer=_init_worker,
                 initargs=(args.device,)) as pool:
        list(tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)))

    print(f"✔ All done — results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
