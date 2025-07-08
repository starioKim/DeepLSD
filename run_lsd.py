#!/usr/bin/env python3
"""
Directory-based DeepLSD Inference
=================================
Apply DeepLSD to all PNG images in a source directory and save the overlaid results
in a target directory using CPU.

Paths are hardcoded for your SATELLITE dataset:
  INPUT_DIR  = /datasets/SATELLITE/crop_smooth
  OUTPUT_DIR = /datasets/SATELLITE/deepLSD_ver1

Usage:
  python deep_lsd_dir.py
"""
import os
from pathlib import Path
import cv2
import torch
import numpy as np
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD

# Configuration
INPUT_DIR = Path("/datasets/SATELLITE/crop_smooth")
OUTPUT_DIR = Path("/datasets/SATELLITE/deepLSD_ver1")
DEVICE = torch.device("cpu")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize model
net = DeepLSD.from_pretrained().to(DEVICE).eval()

# Process each PNG file
for img_path in sorted(INPUT_DIR.glob("*.png")):
    # Read and preprocess
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] Could not read {img_path}, skipping.")
        continue
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inp = torch.tensor(gray[None, None], dtype=torch.float32, device=DEVICE) / 255.0
    batch = {"image": inp}

    # Run inference
    with torch.no_grad():
        out = net(batch)
    lines = out["lines"][0].cpu().numpy()  # (N,5): x1,y1,x2,y2,score

    # Overlay detected lines
    vis = img_bgr.copy()
    for x1, y1, x2, y2, _ in lines:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Save result with original filename
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), vis)
    print(f"Processed {img_path.name} -> {out_path.name}")

print("All images processed.")
