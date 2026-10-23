#!/bin/bash
# run_all_experiments.sh

# 1. 主实验 (默认 Setting 1, 包含所有模块)
bash run_ubuntu.sh kiba 150 64 0.0001 unimol

# 2. 冷启动实验
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --setting 2
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --setting 3
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --setting 4

# 3. 消融实验 (都在默认 setting 1 下做)
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --lambda_phy 0.0
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --no_cross_attention
bash run_ubuntu.sh kiba 150 64 0.0001 unimol --lambda_phy 0.0 --no_cross_attention
