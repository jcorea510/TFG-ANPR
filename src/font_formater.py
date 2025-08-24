import os
import shutil
import argparse
import re

# Plate type mapping
PLATE_TYPE_MAP = {
    r"^256-": "tec_vehicle",
    r"^m\d": "motorbike",
    r"^cl-": "lightweight_charger",
    r"^d-": "disabled",
    r"^vh-": "historic_vehicle",
    # Split private_vehicle into two separate types
    r"^[a-z]{3}-\d{3}$": "private_vehicle_7char",  # lll-nnn
    r"^\d{6}$": "private_vehicle_6char",           # nnnnnn
}

# Color rules
GREEN_PLATES = [r"^cl-", r"^256-", r"^vh-"]

def get_plate_type(plate_number):
    for pattern, plate_type in PLATE_TYPE_MAP.items():
        if re.match(pattern, plate_number, re.IGNORECASE):
            return plate_type
    return "private_vehicle_unknown"

def get_color(plate_number):
    for pattern in GREEN_PLATES:
        if re.match(pattern, plate_number, re.IGNORECASE):
            return "green"
    return "blue"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(src_dir, dest_dir):
    cropped_dir = os.path.join(src_dir, "cropped_chars")
    structure_dir = os.path.join(dest_dir, "structure")

    for plate_folder in os.listdir(cropped_dir):
        plate_path = os.path.join(cropped_dir, plate_folder)
        if not os.path.isdir(plate_path):
            continue
        
        plate_number = plate_folder.replace("_aligned", "")
        color = get_color(plate_number)
        plate_type = get_plate_type(plate_number)

        # Handle files inside each plate folder
        for file_name in os.listdir(plate_path):
            file_path = os.path.join(plate_path, file_name)

            if file_name.lower().startswith("template") and file_name.lower().endswith(".jpeg"):
                templates_path = os.path.join(structure_dir, "templates", plate_type)
                ensure_dir(templates_path)
                count = len([f for f in os.listdir(templates_path) if f.lower().endswith(".jpeg")]) + 1
                shutil.copy(file_path, os.path.join(templates_path, f"Template_{count}.jpeg"))

            elif file_name.lower() == "place_holder.txt":
                placeholder_path = os.path.join(structure_dir, "place_holder", plate_type)
                ensure_dir(placeholder_path)
                count = len([f for f in os.listdir(placeholder_path) if f.lower().endswith(".txt")]) + 1
                shutil.copy(file_path, os.path.join(placeholder_path, f"Template_{count}.txt"))

            elif file_name.lower().endswith(".jpeg"):
                char_name = os.path.splitext(file_name)[0].split("_")[0]
                char_dir = os.path.join(structure_dir, "chars", color, char_name)
                ensure_dir(char_dir)
                count = len([f for f in os.listdir(char_dir) if f.lower().endswith(".jpeg")]) + 1
                shutil.copy(file_path, os.path.join(char_dir, f"{char_name}_{count}.jpeg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize cropped plate characters into structured dataset")
    parser.add_argument("src", help="Source root directory (where cropped_chars/ is located)")
    parser.add_argument("dest", help="Destination root directory for new structure")
    args = parser.parse_args()

    main(args.src, args.dest)
