#!/bin/bash
#SBATCH --job-name=yolov11_plates
#SBATCH --output=/work/jcorea/logs/lpd_train_%j.out
#SBATCH --error=/work/jcorea/logs/lpd_train_%j.err
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
micromamba activate /work/jcorea/myenv310

echo "Executing YOLO training"
python plate_detection.py --train --yolo_model /work/jcorea/yolo11n.pt --dataset_path /work/jcorea/dataset/yolo/data.yaml
