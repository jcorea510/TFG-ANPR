#!/bin/bash
#SBATCH --job-name=synthetic_data
#SBATCH --output=/work/jcorea/logs/synthetic_data_%j.out
#SBATCH --error=/work/jcorea/logs/synthetic_data_%j.err
#SBATCH --partition=nukwa
#SBATCH --time=02:00:00       # walltime
#SBATCH --ntasks=1

cd /work/jcorea

eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/jcorea/myenv310

echo "Starting synthetic data generation on $(hostname) at $(date)"k
python data_syntesis.py -q 2000 -s dataset/fast_ocr/train/ -a -r

