import re
import csv
import wandb
from pathlib import Path

# --- Experiments ---
experiment_models = {
    "baseline": {
        "log_path": Path("logs/lpr_train_240864.out"),
        "hypr_config": Path("models/ocr/hypr_baseline.csv"),
    },
    "exp0": {
        "log_path": Path("logs/lpr_train_241158.out"),
        "hypr_config": Path("models/ocr/hypr_exp0.csv"),
    },
    "exp1": {
        "log_path": Path("logs/lpr_train_241171.out"),
        "hypr_config": Path("models/ocr/hypr_exp1.csv"),
    },
    "exp2": {
        "log_path": Path("logs/lpr_train_241172.out"),
        "hypr_config": Path("models/ocr/hypr_exp2.csv"),
    },
    "exp3": {
        "log_path": Path("logs/lpr_train_241173.out"),
        "hypr_config": Path("models/ocr/hypr_exp3.csv"),
    },
}

# --- Regex patterns ---
# Combined pattern to match both train and validation metrics on the same line
metrics_pattern = re.compile(
    r"cat_acc:\s*([\d.]+)\s*-\s*loss:\s*([\d.]+)\s*-\s*plate_acc:\s*([\d.]+)\s*-\s*plate_len_acc:\s*([\d.]+)\s*-\s*top_3_k:\s*([\d.]+)"
    r"(?:.*?val_cat_acc:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)\s*-\s*val_plate_acc:\s*([\d.]+)\s*-\s*val_plate_len_acc:\s*([\d.]+)\s*-\s*val_top_3_k:\s*([\d.]+))?"
)

# Fallback patterns for cases where they might still be on separate lines
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
                key = key.strip()  # Remove any whitespace
                val = val.strip()  # Remove any whitespace
                try:
                    # cast to float/int if possible
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
            # Check for epoch information
            em = epoch_pattern.search(line)
            if em:
                current_epoch = int(em.group(1))
                continue

            if current_epoch is None:
                continue

            # Try to match combined metrics (train + validation on same line)
            combined_match = metrics_pattern.search(line)
            if combined_match:
                groups = combined_match.groups()
                
                # Extract training metrics (first 5 groups)
                train_metrics = {
                    "epoch": current_epoch,
                    "train/cat_acc": float(groups[0]),
                    "train/loss": float(groups[1]),
                    "train/plate_acc": float(groups[2]),
                    "train/plate_len_acc": float(groups[3]),
                    "train/top_3_k": float(groups[4]),
                }
                
                # Check if validation metrics are present (groups 5-9)
                if groups[5] is not None:  # val_cat_acc exists
                    train_metrics.update({
                        "val/cat_acc": float(groups[5]),
                        "val/loss": float(groups[6]),
                        "val/plate_acc": float(groups[7]),
                        "val/plate_len_acc": float(groups[8]),
                        "val/top_3_k": float(groups[9]),
                    })
                
                metrics.append(train_metrics)
                continue

            # Fallback: try to match training-only metrics
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

            # Fallback: try to match validation-only metrics and update last entry
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
        project="lpr_ablation_analysis",
        name=exp_name,
        config=cfg,
        reinit=True
    )

    # Log metrics per epoch
    for mrow in metrics:
        wandb.log(mrow, step=mrow["epoch"])

    # Summaries
    if metrics:
        best_val = max((m.get("val/plate_acc", 0.0) for m in metrics), default=0.0)
        last = metrics[-1]
        wandb.summary.update({
            "best_val_plate_acc": best_val,
            "final_val_plate_acc": last.get("val/plate_acc", None),
            "final_train_plate_acc": last.get("train/plate_acc", None),
            "final_train_loss": last.get("train/loss", None),
            "final_val_loss": last.get("val/loss", None),
            "epochs_logged": len(metrics),
        })

    run.finish()
    print(f"[{exp_name}] Logged {len(metrics)} epochs to W&B.")
    
    # Debug information
    val_epochs = sum(1 for m in metrics if "val/plate_acc" in m)
    print(f"[{exp_name}] Found validation data for {val_epochs}/{len(metrics)} epochs.")
