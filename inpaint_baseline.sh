#!/bin/bash
#SBATCH --account=es_reddy
#SBATCH --partition=gpu.4h
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=03:00:00
#SBATCH --output=inpaint_baseline_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dima_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /cluster/home/aykota/DiMA
nvidia-smi --query-gpu=name,memory.total --format=csv

# plain replacement baseline
python -m src.diffusion.inpaint_baseline --config_path="src/configs" --n 500 --offline --tag baseline

# variant: also give the clean framework through the self-conditioning input
python -m src.diffusion.inpaint_baseline --config_path="src/configs" --n 500 --offline --tag baseline_selfcond_fw --selfcond_framework
