from ultralytics import YOLO
import cv2
import numpy as np
import keras
import argparse

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

def process_frame(frame, yolo_model, recognizer, font_params):
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
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # Draw text with outline effect
                cv2.putText(frame, f"{plate_text}", org, font, fontScale, 
                           color_outline, thickness_outline, cv2.LINE_AA, False)
                cv2.putText(frame, f"{plate_text}", org, font, fontScale, 
                           color, thickness, cv2.LINE_AA, False)
    
    return frame, plates_detected

def image_mode(image_path, yolo_model, recognizer, font_params):
    """Process a single image"""
    image2predict = cv2.imread(image_path)
    if image2predict is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image2predict = cv2.resize(image2predict, (1080, 720))
    processed_frame, plates_detected = process_frame(image2predict, yolo_model, recognizer, font_params)
    
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

def video_mode(video_path, yolo_model, recognizer, font_params):
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        frame_count += 1
        frame = cv2.resize(frame, (1080, 720))
        processed_frame, plates_detected = process_frame(frame, yolo_model, recognizer, font_params)
        
        if plates_detected:
            plates_detected_count += 1
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('mode', choices=['image', 'video'], 
                       help='Processing mode: image or video')
    parser.add_argument('--input', '-i', required=True,
                       help='Input file path (image/video) or "webcam" for camera')
    parser.add_argument('--yolo-model', default='models/yolo/best.pt',
                       help='Path to YOLO model (default: models/yolo/best.pt)')
    parser.add_argument('--ocr-model', default='models/ocr/ckpt-epoch_52-acc_0.914.keras',
                       help='Path to OCR model (default: models/ocr/ckpt-epoch_52-acc_0.914.keras)')
    parser.add_argument('--config', default='models/ocr/plate_config.yaml',
                       help='Path to plate config file (default: models/ocr/plate_config.yaml)')
    
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
        image_mode(args.input, yolo_model, recognizer, font_params)
    else:  # video mode
        print(f"Processing video: {args.input}")
        video_mode(args.input, yolo_model, recognizer, font_params)
        
    print("Done!")
