import sys
import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import argparse
import cv2

parser = argparse.ArgumentParser(prog="App test", description="Test ANPR via video camera")
parser.add_argument("--list_cameras", action="store_true", help="Print number of available cameras")
parser.add_argument("-c", "--camera_index", type=int, default=0, help="Index of camera to use")

args = parser.parse_args()

all_camera_idx_available = []
for camera_idx in range(10):
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        all_camera_idx_available.append(camera_idx)
        cap.release()

if len(all_camera_idx_available) == 0:
    raise ValueError("No available cameras")

if args.list_cameras:
    print("\nList of available cameras:\n")
    for camera_idx in all_camera_idx_available:
        print(f'Camera index available: {camera_idx}')
    print()

################ MAIN ##################
camera_index = args.camera_index if "-c" in sys.argv or "--camera_index" in sys.argv else all_camera_idx_available[0]

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
