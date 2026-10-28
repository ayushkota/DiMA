#!/bin/bash
#SBATCH --account=es_reddy
#SBATCH --partition=gpu.24h
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=07:30:00
#SBATCH --output=adapter_train_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dima_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /cluster/home/aykota/DiMA
nvidia-smi --query-gpu=name,memory.total --format=csv

# resumable: if the time limit is hit, submit again and it continues from checkpoints/adapter/<tag>/last.pth
python -m src.diffusion.train_adapter --config_path="src/configs" --offline --resume \
    --tag adapter_v1 --batch_size 16 --accum 4 --max_steps 2000 --time_limit_minutes 420
