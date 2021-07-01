#!/bin/bash
#SBATCH -p P1
#SBATCH -n 1
#SBATCH -J test_k_inoue
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

# 設定の読み込み
module load cuda/10.0

# ジョブを発行したdirに移動
cd $SLURM_SUBMIT_DIR

# 以下，実行したいコマンド
echo "INFO: $PWD on `hostname`"
nvcc--version
nvidia-smi
sleep 10
