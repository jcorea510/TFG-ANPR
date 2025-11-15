import json
import csv
import wandb
from pathlib import Path

# === Ruta base donde están todos los experimentos ===
# BASE_DIR = Path("runs/trained_ocr/augmentations_ablation_final")
BASE_DIR = Path("runs/trained_ocr/augmentations_optimization/exp_9/2025-09-30_14-32-17")

# --- Buscar experimentos automáticamente ---
experiment_models = {}

# for exp_dir in BASE_DIR.glob("**/"):
#     log_path = exp_dir / "training_log.csv"
#     hypr_path = exp_dir / "hyper_params.json"
#
#     if log_path.exists() and hypr_path.exists():
#         # Nombrar el experimento automáticamente
#         exp_name = exp_dir.parent.name + "__" + exp_dir.name  # ejemplo: ablation_03_no_geometric__2025-10-14_00-19-29
#         experiment_models[exp_name] = {
#             "log_path": log_path,
#             "hypr_config": hypr_path,
#         }

experiment_models = {}
experiment_models["Baseline"] = {
        "log_path": BASE_DIR / "training_log.csv",
        "hypr_config": BASE_DIR / "hyper_params.json",
        }

# --- Helper to load JSON config ---
def load_json_config(path: Path):
    with open(path, "r") as f:
        return json.load(f)

# --- Main loop ---
for exp_name, exp_data in experiment_models.items():
    log_path = exp_data["log_path"]
    hypr_config_path = exp_data["hypr_config"]

    # Cargar config desde JSON y reducir a lo esencial
    cfg_full = load_json_config(hypr_config_path)
    cfg = {
        "exp_name": exp_name,
        "augmentation_path": cfg_full.get("augmentation_path", None),
        "phase": "augmentation_tuning",
    }

    # --- Log to W&B ---
    run = wandb.init(
        project="lpr_augmentation_tuning_best_final",
        name=exp_name,
        config=cfg,
        reinit=True
    )

    metrics = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mrow = {
                "epoch": int(row["epoch"]),
                "train/cat_acc": float(row["cat_acc"]),
                "train/loss": float(row["loss"]),
                "train/plate_acc": float(row["plate_acc"]),
                "train/plate_len_acc": float(row["plate_len_acc"]),
                "train/top_3_k": float(row["top_3_k"]),
                "val/cat_acc": float(row["val_cat_acc"]),
                "val/loss": float(row["val_loss"]),
                "val/plate_acc": float(row["val_plate_acc"]),
                "val/plate_len_acc": float(row["val_plate_len_acc"]),
                "val/top_3_k": float(row["val_top_3_k"]),
            }
            metrics.append(mrow)
            wandb.log(mrow, step=mrow["epoch"])

    # --- Summaries ---
    if metrics:
        last = metrics[-1]

        # Calcular mejores valores por métrica
        best_summary = {}
        for key in metrics[0].keys():
            if key == "epoch":
                continue
            if "loss" in key:
                best_summary[f"best_{key}"] = min(m[key] for m in metrics)  # loss -> menor es mejor
            else:
                best_summary[f"best_{key}"] = max(m[key] for m in metrics)  # acc -> mayor es mejor

        # Agregar finales
        final_summary = {f"final_{k}": v for k, v in last.items() if k != "epoch"}

        wandb.summary.update({
            **best_summary,
            **final_summary,
            "epochs_logged": len(metrics),
        })

    run.finish()
    print(f"[{exp_name}] Logged {len(metrics)} epochs to W&B.")
