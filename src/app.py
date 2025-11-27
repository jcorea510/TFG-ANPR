from ultralytics import YOLO
import cv2
import numpy as np
import keras
import argparse
import time
import csv
import os
from datetime import datetime

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


def format_license_plate(plate_text: str) -> str:
    """Format license plate text according to specified rules."""
    # Remove trailing underscores (padding) first
    plate_text = plate_text.rstrip("_")
    
    # Check if it starts with vehicle type prefixes (256, CL, M, D, etc.)
    vehicle_prefixes = ["256_", "CL_", "M_", "D_"]
    for prefix in vehicle_prefixes:
        if plate_text.startswith(prefix):
            # Replace underscore after prefix with space
            plate_text = plate_text.replace("_", " ", 1)
            return plate_text
    
    # Check if plate has underscore in the middle with 3 letters at start
    # Pattern: ABC_123
    if len(plate_text) >= 4 and plate_text[:3].isalpha() and "_" in plate_text[3:]:
        # Replace underscore with space
        plate_text = plate_text.replace("_", " ")
    
    return plate_text


def store_plate_in_database(plate_text: str, db_file: str = "license_plates_db.csv"):
    """
    Store license plate in CSV database if not already registered.
    Returns True if plate was newly added, False if already exists.
    """
    # Check if file exists and read existing plates
    existing_plates = set()
    file_exists = os.path.exists(db_file)
    
    if file_exists:
        try:
            with open(db_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_plates.add(row.get("license_plate", ""))
        except Exception as e:
            print(f"Error reading database: {e}")
            # Continue anyway to create/fix the file
    
    # Check if plate already exists
    if plate_text in existing_plates:
        return False  # Plate already registered
    
    # Add new plate with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write to CSV in append mode
    with open(db_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists or os.path.getsize(db_file) == 0:
            writer.writerow(["time", "license_plate"])
        
        writer.writerow([timestamp, plate_text])
    
    print(f"✓ New plate registered: {plate_text}")
    return True


def process_frame(frame, yolo_model, recognizer, font_params, db_file):
    """Process a single frame for license plate detection and recognition"""
    font, fontScale, color, color_outline, thickness, thickness_outline = font_params
    
    detection_results = yolo_model(frame)
    plates_detected = False
    
    for result in detection_results:
        boxes = result.boxes
        if boxes is not None:
            plates_detected = True
            for box in boxes: 
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                image2recognize = frame[int(y1):int(y2), int(x1):int(x2)]
                org = (int(x1), int(y1))
                
                plate_text, probs = recognizer.recognize(image2recognize)
                
                # Format the license plate
                formatted_plate = format_license_plate(plate_text)
                
                # Store in database
                store_plate_in_database(formatted_plate, db_file)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Draw text with outline effect
                cv2.putText(frame, f"{formatted_plate}", org, font, fontScale, 
                           color_outline, thickness_outline, cv2.LINE_AA, False)
                cv2.putText(frame, f"{formatted_plate}", org, font, fontScale, 
                           color, thickness, cv2.LINE_AA, False)
    
    return frame, plates_detected


def image_mode(image_path, yolo_model, recognizer, font_params, db_file):
    """Process a single image"""
    image2predict = cv2.imread(image_path)
    if image2predict is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image2predict = cv2.resize(image2predict, (1080, 720))
    processed_frame, plates_detected = process_frame(image2predict, yolo_model, recognizer, font_params, db_file)
    
    # Add detection status
    status_text = "Plates Detected!" if plates_detected else "No Plates Detected"
    status_color = (0, 255, 0) if plates_detected else (0, 0, 255)
    cv2.putText(processed_frame, status_text, (10, 30), font_params[0], 0.8, status_color, 2)
    
    cv2.imshow("Plate recognition", processed_frame)
    while True:
        key = cv2.waitKey(0)
        print(f"Key pressed: {key}")
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


def video_mode(video_path, yolo_model, recognizer, font_params, db_file):
    """Process video file or webcam"""
    # Open video file or webcam (0 for default camera)
    if video_path.lower() == 'webcam' or video_path == '0':
        cap = cv2.VideoCapture(0)
        print("Using webcam...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    
    frame_count = 0
    plates_detected_count = 0
    total_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        start_time = time.time()
        frame_count += 1
        frame = cv2.resize(frame, (1080, 720))
        processed_frame, plates_detected = process_frame(frame, yolo_model, recognizer, font_params, db_file)
        
        if plates_detected:
            plates_detected_count += 1

        frame_time = time.time() - start_time
        total_time += frame_time
        fps = 1.0 / frame_time if frame_time > 0.0 else 0.0

        print(f"Frame {frame_count}: {frame_time*1000:.1f} ms ({fps:.2f} FPS) | Plate detected: {plates_detected}")
        cv2.putText(processed_frame, f"{fps:.1f} FPS", (950, 60), font_params[0], 0.6, (255, 255, 255), 2)

        # Add status information to frame
        status_text = "PLATE DETECTED!" if plates_detected else "Scanning..."
        status_color = (0, 255, 0) if plates_detected else (0, 255, 255)
        cv2.putText(processed_frame, status_text, (10, 30), font_params[0], 0.8, status_color, 2)
        
        # Add frame counter and detection stats
        info_text = f"Frame: {frame_count} | Detections: {plates_detected_count}"
        cv2.putText(processed_frame, info_text, (10, 60), font_params[0], 0.5, (255, 255, 255), 1)
        
        # Show detection indicator (red dot when plate detected)
        indicator_color = (0, 255, 0) if plates_detected else (0, 0, 255)
        cv2.circle(processed_frame, (1050, 30), 15, indicator_color, -1)
        
        cv2.imshow("Plate recognition - Video Mode", processed_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):  # Pause/unpause
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('s'):  # Save current frame
            filename = f"detection_frame_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Total frames: {frame_count}, Frames with plates: {plates_detected_count}")

    if frame_count > 0:
        avg_time = total_time / frame_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\n--- Desempeño promedio ---")
        print(f"Frames procesados: {frame_count}")
        print(f"Latencia promedio por cuadro: {avg_time*1000:.1f} ms")
        print(f"Velocidad promedio: {avg_fps:.2f} FPS")
        print(f"Frames con detección de placa: {plates_detected_count} ({(plates_detected_count/frame_count)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('mode', choices=['image', 'video'], 
                       help='Processing mode: image or video')
    parser.add_argument('--input', '-i', required=True,
                       help='Input file path (image/video) or "webcam" for camera')
    parser.add_argument('--yolo-model', default='models/best_yolo.pt',
                       help='Path to YOLO model (default: models/yolo/best.pt)')
    parser.add_argument('--ocr-model', default='models/best_ocr.keras',
                       help='Path to OCR model (default: models/best_ocr.keras)')
    parser.add_argument('--config', default='models/plate_config.yaml',
                       help='Path to plate config file (default: models/plate_config.yaml)')
    parser.add_argument('--database', default='license_plates_db.csv',
                       help='Path to CSV database file (default: license_plates_db.csv)')
    
    args = parser.parse_args()
    
    # Font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 255, 0)
    color_outline = (0, 0, 0)
    thickness = 1
    thickness_outline = 2
    font_params = (font, fontScale, color, color_outline, thickness, thickness_outline)
    
    # Initialize models
    print("Loading YOLO model...")
    yolo_model = YOLO(args.yolo_model)
    
    print("Loading OCR model...")
    recognizer = PlateRecognizer(
        model_path=args.ocr_model,
        plate_config_file=args.config,
    )
    
    # Process based on mode
    if args.mode == 'image':
        print(f"Processing image: {args.input}")
        image_mode(args.input, yolo_model, recognizer, font_params, args.database)
    else:  # video mode
        print(f"Processing video: {args.input}")
        video_mode(args.input, yolo_model, recognizer, font_params, args.database)
        
    print("Done!")
