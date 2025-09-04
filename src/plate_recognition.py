import os
import subprocess
import yaml
import pathlib
import random

# --- Search space ---

plate_space = {
    "img_height": [32, 64, 96],
    "img_width": [64, 128, 192],
    "image_color_mode": ["rgb", "grayscale"],
}

model_space = {
    "transformer_encoder.activation": ["relu", "gelu"],
    "transformer_encoder.mlp_dropout": [0.05, 0.1, 0.15, 0.2],
    "transformer_encoder.attention_dropout": [0.05, 0.1, 0.15, 0.2],
    "transformer_encoder.layers": [1, 2, 3, 4],
    "transformer_encoder.projection_dim": [16, 32, 64],
}

train_space = {
    "batch_size": [16, 32, 64],
    "learning_rate": [0.0002, 0.0004, 0.0006, 0.0008, 0.0010],
    "weight_decay": [0.0002, 0.0004, 0.0006, 0.0008, 0.0010],
}

# --- Base configs ---
MODEL_CONFIG = pathlib.Path("./models/ocr/cct_xs_v1_global_model_config.yaml")
PLATE_CONFIG = pathlib.Path("./models/ocr/cct_xs_v1_global_plate_config.yaml")

OUTPUT_DIR = pathlib.Path("./runs/trained_ocr/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = "./logs"

def modify_and_save_config(base_path, updates: dict, suffix: int) -> pathlib.Path:
    """Copy base yaml, apply updates, and save as new file."""
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    # Nested update logic
    for key, value in updates.items():
        if "." in key:  # nested keys like transformer_encoder.mlp_dropout
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d.get(p, {})
            d[parts[-1]] = value
        else:
            config[key] = value

    out_path = base_path.parent / f"{base_path.stem}_exp{suffix}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f)
    return out_path

def run_training(model_config: pathlib.Path, plate_config: pathlib.Path, params: dict, index: int):
    """Launch training subprocess."""
    cmd = [
        "fast-plate-ocr", "train",
        "--model-config-file", str(model_config),
        "--plate-config-file", str(plate_config),
        "--annotations", "./dataset/fast_ocr/train/annotations.csv",
        "--val-annotations", "./dataset/fast_ocr/valid/annotations.csv",
        "--epochs", "150",
        "--batch-size", str(params["batch_size"]),
        "--output-dir", str(OUTPUT_DIR),
        "--weights-path", "./models/ocr/cct_xs_v1_global.keras",
        "--weight-decay", str(params["weight_decay"]),
        "--lr", str(params["learning_rate"]),
    ]
    print(">>> Running:", " ".join(cmd))

    logfile = os.path.join(LOGS_DIR, f"train_{index}.out")
    with open(logfile, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    # subprocess.run(cmd, check=True)

if __name__ == "__main__":
    N_RUNS = 4  # number of random experiments
    random.seed(42)  # reproducibility

    all_space = {**plate_space, **model_space, **train_space}

    for i in range(N_RUNS):
        params = {k: random.choice(v) for k, v in all_space.items()}

        suffix = i
        plate_cfg = modify_and_save_config(PLATE_CONFIG, params, suffix)
        model_cfg = modify_and_save_config(MODEL_CONFIG, params, suffix)

        run_training(model_cfg, plate_cfg, params, i)
