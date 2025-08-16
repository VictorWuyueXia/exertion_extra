#!/bin/bash

#SBATCH --job-name=fused_Focal
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50g
#SBATCH --time=02:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fused_Focal_switch_%j.out

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate cardio_env

# python cardio_pause/src/script_cls.py cardio_pause/src/config.yaml
python cardio_pause/src/script_reg.py cardio_pause/src/config.yaml