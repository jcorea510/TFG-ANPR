#!/bin/bash
#SBATCH --job-name=fast_ocr_ablation
#SBATCH --output=/work/jcorea/logs/lpr_train_%j.out
#SBATCH --error=/work/jcorea/logs/lpr_train_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# change the user
cd /work/jcorea

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/jcorea/myenv310

echo "Starting FastOCR training on $(hostname) at $(date)"

fast-plate-ocr train \
	--model-config-file ./models/model_config.yaml \
	--plate-config-file ./models/plate_config.yaml \
	--annotations ./dataset/fast_ocr/train/annotations.csv \
	--val-annotations ./dataset/fast_ocr/valid/annotations.csv \
	--augmentation-path ./models/train_augmentation.yaml \
	--epochs 150 \
	--batch-size 16 \
	--output-dir ./runs/trained_ocr/ \
	--weights-path ./models/weights/cct_xs_v1_global.keras \
	--label-smoothing 0.0 \
	--weight-decay 0.0002 \
	--lr 0.001

echo "All ablation experiments completed at $(date)"
