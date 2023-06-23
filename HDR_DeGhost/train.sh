#!/bin/bash

### 作业名
#SBATCH --job-name=hdr

### 节点数
#SBATCH --nodes=1

### 申请CPU数
#SBATCH --ntasks=4

### 申请GPU卡数
#SBATCH --gres=gpu:1

### 程序的执行命令
export LD_LIBRARY_PATH="/opt/app/nvidia/460.91.03/lib32:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/app/nvidia/460.91.03/lib:$LD_LIBRARY_PATH"
export PATH="/opt/app/nvidia/460.91.03/bin:$PATH"
source activate hutao
nvidia-smi
python train.py