import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
import numpy as np

# --- CONFIG ---
CHAR_COLORS = {
    "265": "green",
    "d": "blue",
    "m": "blue",
    "vh": "red",
    "cl": "red",
    "lll": "blue",
    "n": "blue"  # fallback
}

PATTERNS = {
    "265-nn":     "nn",
    "265-nnn":    "nnn",
    "cl-nnnnnn":  "nnnnnn",
    "d-nnn":      "nnn",
    "vh-nn":      "nn",
    "vh-nnn":     "nnn",
    "m-nnnnnn":   "nnnnnn",
    "m-nnn/nnn":  "nnnnnn",
    "lll-nnn":    "lll-nnn",
    "nnnnnn":     "nnnnnn"
}

CODE_TO_FOLDER = {
    "265-nnn": "tec_vehicle_3char", #if similar names exists order is important, because it checks startwith "string"
    "265-nn": "tec_vehicle_2char",
    "d": "disabled",
    "cl": "lightweight_charger",
    "m": "motorbike",
    "vh": "historic_vehicle",
    "lll": "private_vehicle_7char",  # Adjust if different
    "n": "private_vehicle_6char"
}

CODE_EXTRACT = {
    "265-nn":     "265_",
    "265-nnn":    "265_",
    "cl-nnnnnn":  "CL_",
    "d-nnn":      "D_",
    "vh-nn":      "VH_",
    "vh-nnn":     "VH_",
    "m-nnnnnn":   "M_",
    "m-nnn/nnn":  "M_",
    "lll-nnn":    "",
    "nnnnnn":     ""
}

BASE_DIR = "font/results/structure"

# Global default
DEFAULT_CHAR_SIZE = (50, 70)
MOTORBIKE_CHAR_SIZE = (48, 42)

# --- HELPERS ---
def _get_char_size(plate_code):
    if plate_code.startswith("m-"):
        return MOTORBIKE_CHAR_SIZE
    return DEFAULT_CHAR_SIZE

def _load_positions(filepath):
    with open(filepath, "r") as f:
        return [tuple(map(int, line.strip().split(","))) for line in f if line.strip()]

def _pick_color(plate_code):
    for code, color in CHAR_COLORS.items():
        if plate_code.startswith(code):
            return color
    return "red"

def _pattern_to_chars(pattern):
    seq = []
    for ch in pattern:
        if ch == "n":
            seq.append(str(random.randint(0, 9)))
        elif ch == "l":
            seq.append(chr(random.randint(65, 90)))
        elif ch == "-":
            seq.append('-')
        else:
            seq.append(ch)
    return seq

def _code_extractor(pattern):
    return CODE_EXTRACT[pattern]

def _apply_position_mode(mode, positions):
    if mode == "exact":
        return positions
    elif mode == "jitter":
        return [(x + random.randint(-2, 2), y + random.randint(-2, 2)) for x, y in positions]
    elif mode == "random":
        cx = sum(x for x, _ in positions) // len(positions)
        cy = sum(y for _, y in positions) // len(positions)
        return [(cx + random.randint(-10, 10), cy + random.randint(-10, 10)) for _ in positions]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _get_folder_for_code(plate_code):
    for code, folder in CODE_TO_FOLDER.items():
        if plate_code.startswith(code):
            return folder
    raise ValueError(f"No folder mapping for plate code: {plate_code}")

def paste_centered(base_img: Image.Image, char_img: Image.Image, center_xy):
    """Paste char_img onto base_img so that center_xy (x,y) corresponds to char_img center.
       Handles clipping and preserves transparency.
    """
    bx, by = center_xy
    bw, bh = base_img.size
    cw, ch = char_img.size

    # top-left coordinates to paste
    px = int(round(bx - cw/2))
    py = int(round(by - ch/2))

    # compute overlapping region
    left = max(px, 0)
    top  = max(py, 0)
    right = min(px + cw, bw)
    bottom = min(py + ch, bh)

    if left >= right or top >= bottom:
        # nothing to paste (char fully outside plate)
        return

    # crop portion of char image to paste
    cx1 = left - px
    cy1 = top  - py
    cx2 = cx1 + (right - left)
    cy2 = cy1 + (bottom - top)
    char_crop = char_img.crop((cx1, cy1, cx2, cy2))

    # paste using alpha if present
    if char_crop.mode in ("RGBA", "LA") or ("transparency" in char_crop.info):
        base_img.paste(char_crop, (left, top), char_crop)
    else:
        base_img.paste(char_crop, (left, top))

# --- augmentations ---
def recolor_char(img, color, augmented=False):
    """Recolor a blue char image into target color (blue/green)."""
    
    # assume chars are pure blue (0,0,255). Replace with target color.
    arr = np.array(img)
    blue_mask = (arr[:, :, 2] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 1] < 100)

    if color == "green":
        arr[blue_mask] = [88, 124, 88, 255]

    elif color == "red":
        arr[blue_mask] = [155, 0, 0, 255]
    else:
        arr[blue_mask] = [92, 107, 172, 255]

    img = Image.fromarray(arr)
    
    if augmented:
        if color == "green":
            img = paint_with_texture(char_img=img, color=(88, 124, 88))
        elif color == "red":
            img = paint_with_texture(char_img=img, color=(155, 0, 0))
        else:
            img = paint_with_texture(char_img=img, color=(92, 107, 172))
    return img

def paint_with_texture(char_img, color=(0, 0, 255)):
    arr = np.array(char_img)  # RGBA
    mask = arr[:, :, 3] > 0   # alpha channel as mask

    h, w = mask.shape
    texture = np.ones((h, w, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

    # Add random darker/lighter zones
    for _ in range(5):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(
            texture,
            (x, y),
            np.random.randint(10, 20),
            (
                np.random.randint(max(color[0]-20,0), min(color[0]+20,255)),
                np.random.randint(max(color[1]-20,0), min(color[1]+20,255)),
                np.random.randint(max(color[2]-20,0), min(color[2]+20,255))
            ),
            -1
        )

    # Apply texture only where mask is True
    arr[mask, :3] = texture[mask]

    return Image.fromarray(arr, mode="RGBA")

def degrade_edges(char_img):
    arr = np.array(char_img.convert("L"))
    arr = cv2.erode(arr, np.ones((2,2), np.uint8), iterations=np.random.randint(0,2))
    arr = cv2.dilate(arr, np.ones((2,2), np.uint8), iterations=np.random.randint(0,2))
    return Image.fromarray(arr).convert("RGBA")

def degrade(img):
    """Apply random degradation (blur, noise, contrast)."""
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.5))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.0))
    return img

def add_screws(plate_img):
    screw_dir = os.path.join(BASE_DIR, "chars", "blue", "screw")
    screw_files = [f for f in os.listdir(screw_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not screw_files:
        return plate_img  # no hay tornillos disponibles

    W, H = plate_img.size
    regions = [(0, W//4), (3*W//4, W)]  # primera y cuarta región


    for region_idx, (x_start, x_end) in enumerate(regions):
        if random.random() < 0.7:
            screw_file = random.choice(screw_files)
            screw_img = Image.open(os.path.join(screw_dir, screw_file)).convert("RGBA")

            screw_scale = random.uniform(0.05, 0.10)
            screw_size = (int(W * screw_scale), int(H * screw_scale))
            screw_img = screw_img.resize(screw_size, Image.LANCZOS)

            # base en borde izquierdo o derecho de la región
            if region_idx == 0:
                x = x_start + 20  # un poco alejado del borde
            else:
                x = x_end - screw_size[0] - 20

            y = H // 2

            # jitter pequeño
            x += random.randint(-5, 5)
            y += random.randint(-3, 3)

            # asegurar dentro del marco
            x = max(0, min(x, W - screw_size[0] - 5))
            y = max(0, min(y, H - screw_size[1] - 5))

            plate_img.alpha_composite(screw_img, (x, y))
    
    return plate_img

# --- MAIN ---
def generate_plate(plate_code, mode="exact", augment=False):
    folder = _get_folder_for_code(plate_code)

    template_path_dir = os.path.join(BASE_DIR, "templates", folder)
    placeholder_path_dir = os.path.join(BASE_DIR, "place_holder", folder)
    chars_path_dir = os.path.join(BASE_DIR, "chars", "blue")  # always neutral

    # pick template
    template_files = [f for f in os.listdir(template_path_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    if not template_files:
        raise FileNotFoundError(f"No templates in {template_path_dir}")
    template_file = random.choice(template_files)
    template_path = os.path.join(template_path_dir, template_file)

    # load template
    plate_img = Image.open(template_path).convert("RGBA")
    if augment:
        plate_img = degrade(plate_img)

    # load positions
    placeholder_file = os.path.splitext(template_file)[0] + ".txt"
    placeholder_path = os.path.join(placeholder_path_dir, placeholder_file)
    positions = _load_positions(placeholder_path)

    # characters
    chars = _pattern_to_chars(PATTERNS[plate_code])
    code_for_plate = _code_extractor(plate_code)

    # code_for_plate = 
    if len(chars) != len(positions):
        raise ValueError(f"Mismatch: {len(chars)} vs {len(positions)} positions with code: {plate_code}")

    positions = _apply_position_mode(mode, positions)
    color = _pick_color(plate_code)  # "blue" or "green"

    for (char, (x, y)) in zip(chars, positions):
        char_dir = os.path.join(chars_path_dir, char)
        char_files = [f for f in os.listdir(char_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        if not char_files:
            raise FileNotFoundError(f"No chars in {char_dir}")
        char_file = random.choice(char_files)
        char_path = os.path.join(char_dir, char_file)

        char_img = Image.open(char_path).convert("RGBA")
        char_img = recolor_char(char_img, color, augment)
        if augment:
            # char_img = degrade_edges(char_img)
            char_img = degrade(char_img)

        # resize
        char_size = _get_char_size(plate_code)
        if char_img.size != char_size:
            char_img = char_img.resize(char_size, Image.LANCZOS)

        paste_centered(plate_img, char_img, (x, y))

    # if augment:
    plate_img = add_screws(plate_img)
    
    chars = "".join(chars)
    chars = chars.replace("-", "")
    plate_id = f"{code_for_plate}{chars}"
    return plate_img, plate_id

# --- Example usage ---
if __name__ == "__main__":
    plate = generate_plate(plate_code="nnnnnn", mode="exact", augment=False)
    plate.show()
