#!/usr/bin/env python3
"""
Directory-based DeepLSD Inference with Checkpoint Loading
==========================================================
Apply DeepLSD to all PNG images in a source directory and save the overlaid results
in a target directory using CPU, loading weights from a checkpoint as you provided.

Paths are hardcoded for your SATELLITE dataset:
  INPUT_DIR      = /datasets/SATELLITE/crop_smooth
  OUTPUT_DIR     = /datasets/SATELLITE/deepLSD_ver1
  CHECKPOINT_PATH= /data/nfs/home/stario/DeepLSD/weights/deeplsd_md.tar

Usage:
  python run_lsd.py
"""
import os
from pathlib import Path
import cv2
import torch
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD

# Configuration
INPUT_DIR       = Path("/datasets/SATELLITE/crop_smooth")
OUTPUT_DIR      = Path("/datasets/SATELLITE/deepLSD_ver1")
CHECKPOINT_PATH = Path("/data/nfs/home/stario/DeepLSD/weights/deeplsd_md.tar")
DEVICE          = torch.device("cpu")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model from checkpoint
ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
conf = {
    "detect_lines": True,
    "line_detection_params": {
        "merge": False,
        "filtering": True,
        "grad_thresh": 3,
        "grad_nfa": True,
    }
}
net = DeepLSD(conf)
net.load_state_dict(ckpt["model"])
net = net.to(DEVICE).eval()

# Process each PNG file
for img_path in sorted(INPUT_DIR.glob("*.png")):
    # Read image
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] Could not read {img_path}, skipping.")
        continue
    # Grayscale tensor input
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inp = torch.tensor(gray[None, None], dtype=torch.float32, device=DEVICE) / 255.0
    batch = {"image": inp}

    # Inference
    with torch.no_grad():
        out = net(batch)
    lines = out.get("lines")[0]
    # If tensor, move to CPU and convert to numpy
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()

    # Overlay lines
    vis = img_bgr.copy()
    for x1, y1, x2, y2, _ in lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Save result
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), vis)
    print(f"Processed {img_path.name} -> {out_path.name}")

print("All images processed.")
