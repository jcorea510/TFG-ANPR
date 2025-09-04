import pathlib
import keras
import numpy as np
import cv2

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities import utils
from fast_plate_ocr.train.utilities.utils import postprocess_model_output


class PlateRecognizer:
    def __init__(self, model_path: str, plate_config_file: str, low_conf_thresh: float = 0.35):
        # Load config
        self.plate_config = load_plate_config_from_yaml(plate_config_file)
        # Load keras model
        self.model = utils.load_keras_model(model_path, self.plate_config)
        self.low_conf_thresh = low_conf_thresh


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize + pad NumPy array to match training setup."""
        h, w = self.plate_config.img_height, self.plate_config.img_width

        # Keep aspect ratio
        if self.plate_config.keep_aspect_ratio:
            scale = min(w / image.shape[1], h / image.shape[0])
            new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((h, w, 3), self.plate_config.padding_color, dtype=np.uint8)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = resized
            image = canvas
        else:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert color mode if needed
        if self.plate_config.image_color_mode == "grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, -1)

        return image

    def recognize(self, plate_img: np.ndarray) -> tuple[str, np.ndarray]:
        """Run OCR on a single plate image."""
        img_proc = self.preprocess(plate_img)
        x = np.expand_dims(img_proc, 0)
        prediction = self.model(x, training=False)
        prediction = keras.ops.stop_gradient(prediction).numpy()
        plate, probs = postprocess_model_output(
            prediction=prediction,
            alphabet=self.plate_config.alphabet,
            max_plate_slots=self.plate_config.max_plate_slots,
            vocab_size=self.plate_config.vocabulary_size,
        )
        return plate, probs


if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2

    test_dir = pathlib.Path("dataset/fast_ocr/test")

    recognizer = PlateRecognizer(
        model_path="runs/trained_ocr/2025-08-31_20-21-11/ckpt-epoch_52-acc_0.914.keras",
        plate_config_file="models/ocr/cct_xs_v1_global_plate_config.yaml",
    )

    for img_path in test_dir.glob("*.jpg"):
        yolo_plate_img = cv2.imread(img_path)  # replace with YOLO output
        plate_text, probs = recognizer.recognize(yolo_plate_img)
        yolo_plate_img = cv2.putText(
                yolo_plate_img,
                f"{plate_text}, (min conf={probs.min():.2f})",
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
                False
        )
        cv2.imshow("Plate recognition", yolo_plate_img) 
        print(f"{img_path.name} -> {plate_text} (min conf={probs.min():.2f})")
        cv2.waitKey(0)  # wait for a key press before moving to next image
        cv2.destroyAllWindows()

