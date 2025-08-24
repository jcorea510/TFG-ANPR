#!/bin/bash
#SBATCH --job-name=yolov_compare
#SBATCH --output=/work/jcorea/logs/model_compare_%j.out
#SBATCH --error=/work/jcorea/logs/model_compare_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

# Switch to working directory with enough space
cd /work/jcorea

# Make Ultralytics use /work instead of home
export YOLO_CONFIG_DIR=/work/jcorea/.ultralytics_config
mkdir -p $YOLO_CONFIG_DIR

echo "Activating micromamba environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/jcorea/myenv39

echo "Executing YOLO comparation"
python compare_yolo.py
