#!/bin/bash
#SBATCH --job-name=yolov11_plates_experiments
#SBATCH --output=/work/jcorea/logs/lpd_train_experiments_unique_%j.out
#SBATCH --error=/work/jcorea/logs/lpd_train_experiments_unique_%j.err
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

echo "Executing YOLO training Experiment 1"
yolo detect train model=models/yolo/yolo11n.pt \
	data=dataset/yolo_unique_1/data.yaml epochs=100 imgsz=640 batch=16 \
	project=runs/trained_yolo/yolo_unique/ name=yolov11_plates_exp1

echo "Executing YOLO training Experiment 2"
yolo detect train model=models/yolo/yolo11n.pt \
	data=dataset/yolo_unique_2/data.yaml epochs=100 imgsz=640 batch=16 \
	project=runs/trained_yolo/yolo_unique/ name=yolov11_plates_exp2

echo "Executing YOLO training Experiment 3"
yolo detect train model=models/yolo/yolo11n.pt \
	data=dataset/yolo_unique_3/data.yaml epochs=100 imgsz=640 batch=16 \
	project=runs/trained_yolo/yolo_unique name=yolov11_plates_exp3

