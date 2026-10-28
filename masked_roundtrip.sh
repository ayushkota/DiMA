#!/bin/bash
#SBATCH --account=es_reddy
#SBATCH --partition=gpu.4h
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:40:00
#SBATCH --output=masked_roundtrip_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dima_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /cluster/home/aykota/DiMA
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m src.diffusion.masked_roundtrip --config_path="src/configs" --n 500 --offline --mask_mode scattered
python -m src.diffusion.masked_roundtrip --config_path="src/configs" --n 500 --offline --mask_mode span
