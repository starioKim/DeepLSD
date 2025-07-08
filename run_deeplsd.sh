#!/bin/bash
#SBATCH -J deeplsd                 # 잡 이름
#SBATCH -o out.deeplsd.%j          # 표준 출력 로그 (%x=잡이름, %j=잡ID)
#SBATCH --partition=rtx4090        # GPU 파티션 이름
#SBATCH --gres=gpu:1               # GPU 1개
#SBATCH --nodelist=aisys-gpu01
#SBATCH --cpus-per-task=4          # CPU 코어 4개
#SBATCH --mem=16G                  # 메모리 16GB
#SBATCH --time=12:00:00            # 최대 2시간

python3 lsd_demo.py 