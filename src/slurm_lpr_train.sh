#!/bin/bash
#SBATCH --job-name=fast_ocr_train
#SBATCH --output=/work/jcorea/logs/lpr_train_%j.out
#SBATCH --error=/work/jcorea/logs/lpr_train_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

cd /work/jcorea
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/jcorea/myenv310

echo "Starting FastOCR training on $(hostname) at $(date)"
python plate_recognition.py
