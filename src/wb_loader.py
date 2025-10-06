import re
import csv
import wandb
from pathlib import Path

# --- Experiments ---
experiment_models = {
    # Primerly used experiments
    "Exp0": {
        "log_path": Path("logs/lpr_train_240864.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_baseline.csv"),
    },
    "Exp1": {
        "log_path": Path("logs/lpr_train_241158.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp0.csv"),
    },
    "Exp2": {
        "log_path": Path("logs/lpr_train_241171.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp1.csv"),
    },
    "Exp3": {
        "log_path": Path("logs/lpr_train_241172.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp2.csv"),
    },
    "Exp4": {
        "log_path": Path("logs/lpr_train_241173.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp3.csv"),
    },
    # secondary experiments for validation
    "Exp5": {
        "log_path": Path("logs/lpr_train_exp5244238.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp5.csv"),
    },
    "Exp6": {
        "log_path": Path("logs/lpr_train_exp6244242.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp6.csv"),
    },
    "Exp7": {
        "log_path": Path("logs/lpr_train_exp7244246.out"),
        "hypr_config": Path("models/ocr/hyperparams/hypr_exp7.csv"),
    },
}

# --- Regex patterns ---
metrics_pattern = re.compile(
    r"cat_acc:\s*([\d.]+)\s*-\s*loss:\s*([\d.]+)\s*-\s*plate_acc:\s*([\d.]+)\s*-\s*plate_len_acc:\s*([\d.]+)\s*-\s*top_3_k:\s*([\d.]+)"
    r"(?:.*?val_cat_acc:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)\s*-\s*val_plate_acc:\s*([\d.]+)\s*-\s*val_plate_len_acc:\s*([\d.]+)\s*-\s*val_top_3_k:\s*([\d.]+))?"
)
train_only_pattern = re.compile(
    r"cat_acc:\s*([\d.]+)\s*-\s*loss:\s*([\d.]+)\s*-\s*plate_acc:\s*([\d.]+)\s*-\s*plate_len_acc:\s*([\d.]+)\s*-\s*top_3_k:\s*([\d.]+)$"
)
val_only_pattern = re.compile(
    r"val_cat_acc:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)\s*-\s*val_plate_acc:\s*([\d.]+)\s*-\s*val_plate_len_acc:\s*([\d.]+)\s*-\s*val_top_3_k:\s*([\d.]+)"
)
epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")

# --- Helper to load CSV config ---
def load_csv_config(path: Path):
    cfg = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                key, val = row
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val:
                        cfg[key] = float(val)
                    elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                        cfg[key] = int(val)
                    else:
                        cfg[key] = val
                except ValueError:
                    cfg[key] = val
    return cfg


# --- Main loop ---
for exp_name, exp_data in experiment_models.items():
    log_path = exp_data["log_path"]
    hypr_config_path = exp_data["hypr_config"]

    # Load config
    cfg = load_csv_config(hypr_config_path)
    cfg["exp_name"] = exp_name

    # Optional: parse job id from log filename
    m = re.search(r"(\d+)", log_path.name)
    if m:
        cfg["slurm_job_id"] = m.group(1)

    # Parse metrics
    metrics = []
    current_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            em = epoch_pattern.search(line)
            if em:
                current_epoch = int(em.group(1))
                continue

            if current_epoch is None:
                continue

            combined_match = metrics_pattern.search(line)
            if combined_match:
                groups = combined_match.groups()
                train_metrics = {
                    "epoch": current_epoch,
                    "train/cat_acc": float(groups[0]),
                    "train/loss": float(groups[1]),
                    "train/plate_acc": float(groups[2]),
                    "train/plate_len_acc": float(groups[3]),
                    "train/top_3_k": float(groups[4]),
                }

                if groups[5] is not None:
                    train_metrics.update({
                        "val/cat_acc": float(groups[5]),
                        "val/loss": float(groups[6]),
                        "val/plate_acc": float(groups[7]),
                        "val/plate_len_acc": float(groups[8]),
                        "val/top_3_k": float(groups[9]),
                    })
                metrics.append(train_metrics)
                continue

            train_match = train_only_pattern.search(line)
            if train_match:
                cat_acc, loss, plate_acc, plate_len_acc, top_3_k = map(float, train_match.groups())
                metrics.append({
                    "epoch": current_epoch,
                    "train/cat_acc": cat_acc,
                    "train/loss": loss,
                    "train/plate_acc": plate_acc,
                    "train/plate_len_acc": plate_len_acc,
                    "train/top_3_k": top_3_k,
                })
                continue

            val_match = val_only_pattern.search(line)
            if val_match and metrics and metrics[-1]["epoch"] == current_epoch:
                val_cat_acc, val_loss, val_plate_acc, val_plate_len_acc, val_top_3_k = map(float, val_match.groups())
                metrics[-1].update({
                    "val/cat_acc": val_cat_acc,
                    "val/loss": val_loss,
                    "val/plate_acc": val_plate_acc,
                    "val/plate_len_acc": val_plate_len_acc,
                    "val/top_3_k": val_top_3_k,
                })

    # --- Log to W&B ---
    run = wandb.init(
        project="lpr_ablation_analysis_best",
        name=exp_name,
        config=cfg,
        reinit=True
    )

    for mrow in metrics:
        wandb.log(mrow, step=mrow["epoch"])

    # --- Extended Summaries ---
    if metrics:
        last = metrics[-1]

        # 1️⃣ Reunir todas las claves presentes en cualquier epoch
        all_keys = set()
        for m in metrics:
            all_keys.update(m.keys())
        all_keys.discard("epoch")

        # 2️⃣ Rellenar los valores faltantes con None
        for m in metrics:
            for k in all_keys:
                if k not in m:
                    m[k] = None

        # 3️⃣ Calcular mejores valores y la época correspondiente
        best_summary = {}
        for key in all_keys:
            valid_values = [(m["epoch"], m[key]) for m in metrics if m[key] is not None]
            if not valid_values:
                continue

            if "loss" in key:
                best_epoch, best_value = min(valid_values, key=lambda x: x[1])
            else:
                best_epoch, best_value = max(valid_values, key=lambda x: x[1])

            best_summary[f"best_{key}"] = best_value
            best_summary[f"best_{key}_epoch"] = best_epoch

        # 4️⃣ Agregar métricas finales
        final_summary = {f"final_{k}": v for k, v in last.items() if k != "epoch"}

        # 5️⃣ Subir a W&B
        wandb.summary.update({
            **best_summary,
            **final_summary,
            "epochs_logged": len(metrics),
        })

    run.finish()
    print(f"[{exp_name}] Logged {len(metrics)} epochs to W&B.")
    val_epochs = sum(1 for m in metrics if "val/plate_acc" in m)
    print(f"[{exp_name}] Found validation data for {val_epochs}/{len(metrics)} epochs.")

