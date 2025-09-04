from PIL import Image
import argparse
import numpy as np
if not hasattr(np, "float"):
    np.float = float
import augly.image as imaugs
from plate_generator import generate_plate
import uuid

import os
import random

def get_random_pattern():
    patterns = [
        "265-nn",
        "265-nnn",
        "cl-nnnnnn",
        "d-nnn",
        # "vh-nn",
        # "vh-nnn",
        "m-nnnnnn",
        "m-nnn/nnn",
        "lll-nnn",
        "nnnnnn",
    ]
    return random.sample(patterns, 1)[0]

def augmentData(pil_image):
    # STEP 1: Pad the original image with a larger transparent background
    padding = 100  # adjust as needed
    padded_width = pil_image.width + 2 * padding
    padded_height = pil_image.height + 2 * padding

    padded_image = Image.new("RGBA", (padded_width, padded_height), (255, 255, 255, 0))
    padded_image.paste(pil_image, (padding, padding))

    # STEP 2: Apply augmentations on the padded image
    AUGMENTATIONS = [
        imaugs.RandomBlur(min_radius=0, max_radius=2, p=0.4),
        imaugs.RandomNoise(mean=0, var=0.001, seed=42, p=0.75),
        imaugs.RandomBrightness(min_factor=0.7, max_factor=1.3, p=0.8),
        imaugs.PerspectiveTransform(sigma=10, p=1),
        imaugs.RandomPixelization(min_ratio=0.6, max_ratio=1.0, p=0.6),
        imaugs.RandomRotation(min_degrees=-20, max_degrees=20, p=0.7),
        imaugs.RandomAspectRatio(min_ratio=0.5, max_ratio=1, p=0.9),
        imaugs.ColorJitter(brightness_factor=random.uniform(0.7, 1.3), contrast_factor=random.uniform(0.7, 1.3), p=0.3),
        imaugs.Grayscale(mode="average", p=0.1),
        imaugs.PerspectiveTransform(sigma=random.randint(10, 60), dx=random.uniform(0.1, 0.6), dy=random.uniform(0.1, 0.6), p=0.2)
    ]

    TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
    return TRANSFORMS(padded_image)

########################### INITIALIZE PARAMETERS #########################

parser = argparse.ArgumentParser(prog="data synthesis", description="Allow to create synthetic images of costa rican plates")
parser.add_argument("-q", "--quantity", default=20, type=int, help="Amount of images to generate")
parser.add_argument("-s", "--save_directory", default="dataset/license_plates",
                    type=str, help="Directory to save generated images")
parser.add_argument("-a", "--augmented_data", action="store_true", help="Generated augmented data")
parser.add_argument("-r", "--augmented_plates", action="store_true", help="Generated augmented images")

args = parser.parse_args()

quantityOfImages:int = args.quantity
save2Directory:str = ""

os.makedirs(args.save_directory, exist_ok=True)
os.makedirs(os.path.join(args.save_directory, "images"), exist_ok=True)

########################## MAIN PROGRAM ###################################
for index in range(args.quantity):
    newImage, chars = generate_plate(plate_code=get_random_pattern(), mode="exact", augment=args.augmented_plates)
    if args.augmented_data:
        newImage = augmentData(newImage)
    unique_id = str(uuid.uuid4())[:8]
    plate_name = f"synthetic_plate_{unique_id}.png"
    newImage.save(os.path.join(args.save_directory, "images", plate_name))
    with open(os.path.join(args.save_directory, "annotations.csv"), "a") as f:
        f.write(f"images/{plate_name},{chars}\n")

