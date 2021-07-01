#!/bin/bash
#SBATCH --partition=P1
#SBATCH --job-name=test_k_inoue
#SBATCH --output=result.txt
#SBATCH --ntasks=1
#SBATCH --time=0:10:00

# 設定の読み込み
module load cuda/10.0

# ジョブを発行したdirに移動
cd $SLURM_SUBMIT_DIR

# 以下，実行したいコマンド
echo "INFO: $PWD on `hostname`"
nvcc--version
nvidia-smi

