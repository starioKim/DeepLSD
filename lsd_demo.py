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

# Load an image
img = cv2.imread('/datasets/SATELLITE/crop_smooth/calib_7DT01_OMC2-3_20240323_003605_m400_120.png')[:, :, ::-1]
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Detect (and optionally refine) the lines
inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
with torch.no_grad():
    out = net(inputs)
    pred_lines = out['lines'][0]

# 1) 출력 폴더 정의 및 생성
OUTPUT_DIR = Path("/datasets/SATELLITE/deepLSD_ver1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) RGB→BGR 변환 후 선 그리기
img_bgr = img[:, :, ::-1].copy()
for (x1, y1), (x2, y2) in pred_lines:
    cv2.line(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 3) 동일한 파일명으로 저장
in_fname = Path('../assets/images/example2.png').name  # 원본 파일명
out_path = OUTPUT_DIR / in_fname
cv2.imwrite(str(out_path), img_bgr)
print(f"Saved result to {out_path}")