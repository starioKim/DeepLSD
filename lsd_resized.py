#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import csv

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from pathlib import Path

# ── Model config ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
conf = {
    'detect_lines': True,
    'line_detection_params': {
        'merge': False,
        'filtering': True,
        'grad_thresh': 3,
        'grad_nfa': True,
    }
}

# ── Load the model ─────────────────────────────────────────────────────────────
ckpt_path = 'weights/deeplsd_md.tar'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()

# ── I/O paths & parameters ───────────────────────────────────────────────────
# 데이터셋 경로 변경 (리사이즈된 폴더 사용)
INPUT_DIR   = Path("/datasets/SATELLITE/crop_smooth_resized_maxpool")
OUTPUT_DIR  = Path("/datasets/SATELLITE/deepLSD_ver10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = "/nfs/home/stario/DeepLSD/results/lsd_detected_images_ver10.csv"

BATCH_SIZE  = 32       # batch size
MIN_LENGTH  = 60       # minimum line length (pixels)

# ── Gather image list ─────────────────────────────────────────────────────────
img_paths     = sorted(INPUT_DIR.glob("*.png"))
num_batches   = math.ceil(len(img_paths) / BATCH_SIZE)
detected_files = []

# ── Batch processing with tqdm ────────────────────────────────────────────────
for i in tqdm(range(0, len(img_paths), BATCH_SIZE),
              desc="Processing batches",
              total=num_batches):
    batch_paths = img_paths[i:i + BATCH_SIZE]
    originals   = []
    tensors     = []

    # 1) Batch preprocess: load → gray → tensor
    for p in batch_paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = img_bgr[:, :, ::-1]  # BGR→RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        originals.append(img_bgr)
        t = torch.tensor(gray,
                         dtype=torch.float32,
                         device=device)[None, None] / 255.0
        tensors.append(t)

    batch_input = torch.cat(tensors, dim=0)  # shape: (B,1,H,W)

    # 2) Inference
    with torch.no_grad():
        out = net({'image': batch_input})
        lines_batch = out['lines']  # list of (Ni,2,2) or ndarray

    # 3) Post-process and save
    for img_bgr, lines, p in zip(originals, lines_batch, batch_paths):
        coords = lines.cpu().numpy() if torch.is_tensor(lines) else lines
        long_lines = []
        for (x1, y1), (x2, y2) in coords:
            if math.hypot(x2 - x1, y2 - y1) >= MIN_LENGTH:
                long_lines.append((x1, y1, x2, y2))

        if not long_lines:
            print(f"Skipped (no long lines): {p.name}")
            continue

        vis = img_bgr.copy()
        for x1, y1, x2, y2 in long_lines:
            cv2.line(vis,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (0, 0, 255), 2)

        out_path = OUTPUT_DIR / p.name
        cv2.imwrite(str(out_path), vis)
        detected_files.append(p.name)

# ── Write CSV ────────────────────────────────────────────────────────────────
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename'])
    for fname in detected_files:
        # Path 객체로 분리
        p = Path(fname)
        stem = p.stem.removesuffix("_resized")  # 파이썬 ≥3.9
        writer.writerow([f"{stem}{p.suffix}"])

print(f"✔ CSV written: {CSV_PATH}")

# ── Write TXT ────────────────────────────────────────────────────────────────
txt_path = CSV_PATH.replace('.csv', '.txt')
with open(txt_path, 'w', newline='') as f_txt:
    f_txt.write('filename\n')
    for fname in detected_files:
        p = Path(fname)
        stem = p.stem.removesuffix("_resized")
        f_txt.write(f"{stem}{p.suffix}\n")

print(f"✔ TXT written: {txt_path}")
