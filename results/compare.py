import os
import re

# ---------- 1) except_frame.csv 처리 ----------
except_set = set()
with open("except_frame.csv", "r", encoding="utf-8") as f:
    for line in f:
        # 작은따옴표(') 사이에 들어 있는 전체 경로+파일명(.fits) 잡아내기
        m = re.search(r"'([^']+\.fits)'", line)
        if m:
            full_path = m.group(1)                       # path+'.../abc.fits'
            base_no_ext = os.path.splitext(
                os.path.basename(full_path)
            )[0]                                         # 'abc'
            except_set.add(base_no_ext)

# ---------- 2) positive_only_ver4.txt 처리 ----------
with open("lsd_detected_images_ver10.txt", "r", encoding="utf-8") as f:
    positive_list = [
        os.path.splitext(line.strip())[0] #.removesuffix("_resized")                # 'abc.png' → 'abc'
        for line in f
        if line.strip()
    ]

# ---------- 3) 교집합 계산 ----------
in_both = [name for name in positive_list if name in except_set]

# ---------- 4) 결과 ----------
total   = len(positive_list)
overlap = len(in_both)

print(f"모델 결과 총 {total}개 중 {overlap}개가 육안 csv에 포함돼 있습니다.")

# ---------- 3) 교집합 계산 ----------
in_both = [name for name in except_set if name in positive_list]

# ---------- 4) 결과 ----------
total   = len(except_set)
overlap = len(in_both)

print(f"육안 csv 총 {total}개 중 {overlap}개가 모델 결과에 포함돼 있습니다.")

# 필요하면 겹친 파일명들 확인
# print("겹친 파일 목록:", in_both)
