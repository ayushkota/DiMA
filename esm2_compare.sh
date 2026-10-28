#!/bin/bash
#SBATCH --account=es_reddy
#SBATCH --partition=gpu.4h
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=01:30:00
#SBATCH --output=esm2_compare_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dima_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /cluster/home/aykota/DiMA
nvidia-smi --query-gpu=name,memory.total --format=csv

# ESM2's own prediction on the same masks (random scattered, random contiguous, CDR-H3)
python -m src.diffusion.esm2_masked_prediction --config_path="src/configs" --offline

# how diverse is DiMA for one masked antibody (4 samples, first 50 antibodies)
python -m src.diffusion.diversity_dima --config_path="src/configs" --offline --mask_mode scattered
python -m src.diffusion.diversity_dima --config_path="src/configs" --offline --mask_mode span
