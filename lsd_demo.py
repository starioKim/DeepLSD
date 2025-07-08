import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import h5py

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

from pathlib import Path



# Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
conf = {
    'detect_lines': True,  # Whether to detect lines or only DF/AF
    'line_detection_params': {
        'merge': False,  # Whether to merge close-by lines
        'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        'grad_thresh': 3,
        'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    }
}

# Load the model
ckpt = 'weights/deeplsd_md.tar'
ckpt = torch.load(str(ckpt), map_location='cpu', weights_only=False)
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()

INPUT_DIR  = Path("/datasets/SATELLITE/crop_smooth")
OUTPUT_DIR = Path("/datasets/SATELLITE/deepLSD_ver1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 배치 크기 설정
BATCH_SIZE = 8

# 처리할 모든 이미지 목록
img_paths = list(INPUT_DIR.glob("*.png"))

# 배치 단위로 순회
for i in range(0, len(img_paths), BATCH_SIZE):
    batch_paths = img_paths[i:i + BATCH_SIZE]
    originals = []
    batch_tensors = []

    # 1) 배치 전처리: 로드 → RGB→Gray → 텐서화
    for p in batch_paths:
        img = cv2.imread(str(p))[:, :, ::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        originals.append(img)
        t = torch.tensor(gray, dtype=torch.float, device=device)[None, None] / 255.0
        batch_tensors.append(t)

    # (B,1,H,W) 크기로 합치기
    batch_input = torch.cat(batch_tensors, dim=0)

    # 2) 배치 추론
    with torch.no_grad():
        out = net({'image': batch_input})
        lines_batch = out['lines']  # length B 리스트

    # 3) 배치 후처리 및 저장
    for img, lines, p in zip(originals, lines_batch, batch_paths):
        img_bgr = img[:, :, ::-1].copy()
        # lines: Tensor (num_lines, 2, 2)
        for (x1, y1), (x2, y2) in lines.cpu().numpy():
            cv2.line(img_bgr,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (0, 0, 255), 2)

        out_path = OUTPUT_DIR / p.name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved: {out_path}")