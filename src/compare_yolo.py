import os
from ultralytics import YOLO

# Dataset config (yaml, not just image folder)
data_yaml = "dataset/yolo/data.yaml"

# Output directory for metrics
out_dir = "runs/eval"
os.makedirs(out_dir, exist_ok=True)

models = {
    "new_model": "runs/yolo_plates/weights/best.pt",
    "old_model": "modelos/yolo/modelo_placas.pt",
    "pretrained": "modelos/yolo/yolo11n.pt",
}

for name, path in models.items():
    print(f"Evaluating {name} ({path})...")
    model = YOLO(path)
    metrics = model.val(
        data=data_yaml,
        project=out_dir,
        name=name,
        save_json=True  # optional: COCO-style results
    )
    print(f"Results for {name}: {metrics}")
