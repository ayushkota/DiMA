#!/bin/bash
#SBATCH --account=es_reddy
#SBATCH --partition=gpu.4h
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:00:00
#SBATCH --output=reconstruction_check_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dima_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /cluster/home/aykota/DiMA
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m src.preprocessing.reconstruction_check --config_path="src/configs" --n 5000 --offline
