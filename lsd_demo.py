import os
import numpy as np
import cv2
import torch
import csv

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
OUTPUT_DIR = Path("/datasets/SATELLITE/deepLSD_ver2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 파라미터
BATCH_SIZE = 1       # 배치 크기
MIN_LENGTH = 150      # 최소 선분 길이 (픽셀)
CSV_PATH   = "/nfs/home/stario/DeepLSD/results/lsd_detected_images_ver2.csv"

# 처리할 모든 이미지 리스트
img_paths = sorted(INPUT_DIR.glob("*.png"))

# 검출된 파일명 저장용
detected_files = []

# 배치 단위 순회
for i in range(0, len(img_paths), BATCH_SIZE):
    batch_paths = img_paths[i:i + BATCH_SIZE]
    originals   = []
    tensors     = []

    # 1) 배치 전처리: 로드 → Gray → Tensor
    for p in batch_paths:
        img = cv2.imread(str(p))[:, :, ::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 원본 gray: (H, W)
        h, w = gray.shape
        scale = 0.25
        new_w, new_h = int(w * scale), int(h * scale)
        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


        originals.append(img)
        t = torch.tensor(gray_small, dtype=torch.float, device=device)[None, None] / 255.0
        tensors.append(t)

    batch_input = torch.cat(tensors, dim=0)  # (B,1,H,W)

    # 2) 배치 추론
    with torch.no_grad():
        out = net({'image': batch_input})
        lines_batch = out['lines']  # 길이 B list of (Ni,2,2) Tensors

    # 3) 후처리: 길이 필터 → 저장 여부 판단 → 이미지 저장 & 리스트에 추가
    for img, lines, p in zip(originals, lines_batch, batch_paths):
        # NumPy array 로 변환
        coords = lines
        long_lines = []
        for (x1, y1), (x2, y2) in coords:
            if np.hypot(x2 - x1, y2 - y1) >= MIN_LENGTH:
                long_lines.append((x1, y1, x2, y2))

        if not long_lines:
            print(f"Skipped (no long lines): {p.name}")
            continue

        # 충분히 긴 선분이 하나라도 있으면 이미지 생성
        img_bgr = img[:, :, ::-1].copy()
        for x1, y1, x2, y2 in long_lines:
            cv2.line(img_bgr,
                     (int(x1/scale), int(y1/scale)),
                     (int(x2/scale), int(y2/scale)),
                     (0, 0, 255), 2)

        out_path = OUTPUT_DIR / p.name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved: {out_path}")

        detected_files.append(p.name)

# 4) CSV 파일로 기록
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename'])
    for fname in detected_files:
        writer.writerow([fname])

print(f"CSV written: {CSV_PATH}")