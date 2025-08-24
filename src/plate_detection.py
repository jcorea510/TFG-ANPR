import os
import argparse
from ultralytics import YOLO
import cv2

parser = argparse.ArgumentParser(
    prog="yolo",
    description="Allow to train, val, and predict based in a yolo trained model",
)

parser.add_argument("-y", "--yolo_model", type=str, help="Path to a Yolo model's path", default="yolo11n.pt")
parser.add_argument("-d", "--dataset_path", type=str, help="Path to file dataset.yaml", default="dataset/yolo/data.yaml")
parser.add_argument("-t", "--train", action="store_true", help="Fine tunne the given model")
parser.add_argument("-v", "--validation", action="store_true", help="Performs validation metrics for the given model")
parser.add_argument("-p", "--prediction", type=str, help="Performs prediction for a given image")

args = parser.parse_args()

model_path = args.yolo_model
if not os.path.exists(model_path):
     raise ValueError(f"Model {model_path} not found")
model = YOLO(model_path)  # load a pretrained model (recommended for training)

if args.train:
    # Train the model
    results = model.train(
            data=args.dataset_path,
            epochs=100,
            imgsz=640,
            project="runs",
            name="yolo_plates",
            exist_ok=True,
            )

if args.validation:
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

if args.prediction != "":
    # Predict with the model
    results = model(args.prediction)  # predict on an image

    # Get the plotted image
    results[0].save("predicted_image.jpg")

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

