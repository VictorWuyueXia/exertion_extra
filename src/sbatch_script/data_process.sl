#!/bin/bash

#SBATCH --job-name=data_preprocess
#SBATCH --ntasks=1
#SBATCH --mem=10g
#SBATCH --time=00:30:00
#SBATCH -p general
#SBATCH --output=logs/data_process_%j.out

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate cardio_env


python src/data/segment_audio_label.py
