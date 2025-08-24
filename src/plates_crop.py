import os
import glob
import argparse
import cv2
import numpy as np
from math import hypot

WINDOW_NAME = "Plate Annotator (click 4 points, r=undo, q=accept)"

# Globals used by mouse callback
_orig_img = None
_points_orig = []  # points in original-image coords
_disp_w = None
_disp_h = None


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def mouse_cb(event, x, y, flags, param):
    """Mouse callback — x,y are coordinates in the displayed (possibly resized) window.
    We map them back to original image coords and store them in _points_orig.
    """
    global _orig_img, _points_orig, _disp_w, _disp_h
    if _orig_img is None:
        return

    h_orig, w_orig = _orig_img.shape[:2]
    # Defensive defaults for display size
    if not _disp_w or not _disp_h:
        disp_w, disp_h = w_orig, h_orig
    else:
        disp_w, disp_h = _disp_w, _disp_h

    # Map displayed coords back to original image coords
    # (imshow stretches image to fit window; we assume linear scale)
    orig_x = int(x * (w_orig / disp_w))
    orig_y = int(y * (h_orig / disp_h))

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_points_orig) < 8:  # allow a few more clicks and use the first 4
            _points_orig.append((orig_x, orig_y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # right-click = undo last
        if _points_orig:
            _points_orig.pop()


def draw_overlay(display_img):
    """Draw current polygon, points, indexes and rules on the display image."""
    global _points_orig, _orig_img, _disp_w, _disp_h
    if _orig_img is None:
        return display_img
    h_orig, w_orig = _orig_img.shape[:2]
    disp_w, disp_h = display_img.shape[1], display_img.shape[0]

    # Draw points and small index labels
    for i, (ox, oy) in enumerate(_points_orig):
        dx = int(ox * (disp_w / w_orig))
        dy = int(oy * (disp_h / h_orig))
        cv2.circle(display_img, (dx, dy), 6, (0, 255, 0), -1)
        cv2.putText(display_img, str(i + 1), (dx + 6, dy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If 2+ points draw connecting lines
    if len(_points_orig) >= 2:
        pts = [ (int(x*(disp_w/w_orig)), int(y*(disp_h/h_orig))) for (x,y) in _points_orig ]
        for i in range(len(pts)-1):
            cv2.line(display_img, pts[i], pts[i+1], (200,200,0), 1)

    # If 4 or more points: pick the last 4 or the first 4? We'll use the first 4
    if len(_points_orig) >= 4:
        pts4 = np.array(_points_orig[:4], dtype="float32")
        rect = order_points(pts4.copy())
        # map ordered rect to display coords
        rect_disp = [(int(rect[i][0]*(disp_w/w_orig)), int(rect[i][1]*(disp_h/h_orig))) for i in range(4)]
        # polygon
        cv2.polylines(display_img, [np.array(rect_disp, dtype=np.int32)], True, (0,128,255), 2)

        # RULE LINES: draw top edge extended across window and left edge extended
        (tl, tr, br, bl) = rect_disp
        # top edge line through tl-tr
        x1, y1 = tl; x2, y2 = tr
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            x_start, x_end = 0, disp_w - 1
            y_start = int(m * x_start + b)
            y_end   = int(m * x_end   + b)
            cv2.line(display_img, (x_start, y_start), (x_end, y_end), (0,255,255), 1, lineType=cv2.LINE_AA)
        # left edge line through tl-bl
        x1, y1 = tl; x2, y2 = bl
        if y2 != y1:
            m = (x2 - x1) / (y2 - y1)  # inverse slope (x as function of y)
            b = x1 - m * y1
            y_start, y_end = 0, disp_h - 1
            x_start = int(m * y_start + b)
            x_end   = int(m * y_end   + b)
            cv2.line(display_img, (x_start, y_start), (x_end, y_end), (0,255,255), 1, lineType=cv2.LINE_AA)

    # instructions
    cv2.putText(display_img, "Left click: add point  |  r: undo  |  q: accept & save",
                (10, max(20, int(16*(disp_h/300)))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(display_img, f"Points: {len(_points_orig)}", (10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2, cv2.LINE_AA)

    return display_img


def process_image(img_path, out_dir, target_w=440, target_h=140):
    """Interactive annotation and processing for one image."""
    global _orig_img, _points_orig, _disp_w, _disp_h
    _points_orig = []

    basename = os.path.splitext(os.path.basename(img_path))[0]
    aligned_dir = os.path.join(out_dir, "aligned_plates")
    os.makedirs(aligned_dir, exist_ok=True)

    _orig_img = cv2.imread(img_path)
    if _orig_img is None:
        print(f"[ERR] Cannot read {img_path}")
        return

    h_orig, w_orig = _orig_img.shape[:2]

    # Create a resizable window and set mouse callback
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(WINDOW_NAME, _orig_img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    print(f"[INFO] Annotate: {basename} — click 4 corners (any order). Press 'q' to accept, 'r' to undo last point.")

    # Main loop: let user click and resize window as they want
    while True:
        # try to get current window size to scale the display image and map clicks correctly
        try:
            wx, wy, ww, wh = cv2.getWindowImageRect(WINDOW_NAME)
            # getWindowImageRect may return 0 for size before first show; fallback:
            if ww <= 0 or wh <= 0:
                ww, wh = w_orig, h_orig
        except Exception:
            # Older OpenCV or not available: use original size
            ww, wh = w_orig, h_orig

        _disp_w, _disp_h = ww, wh

        # Resize for display
        disp_img = cv2.resize(_orig_img, (ww, wh), interpolation=cv2.INTER_LINEAR)
        disp_img = draw_overlay(disp_img)
        cv2.imshow(WINDOW_NAME, disp_img)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            # Undo last
            if _points_orig:
                _points_orig.pop()
        elif key == ord('q'):
            if len(_points_orig) >= 4:
                break
            else:
                print("[WARN] Need at least 4 points (r to undo).")
        # don't close on other keys; ignore

    # After user accepted, compute warp and save
    # Use the first 4 points provided
    pts = np.array(_points_orig[:4], dtype="float32")
    rect = order_points(pts.copy())

    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    aligned = cv2.warpPerspective(_orig_img, M, (target_w, target_h))

    aligned_path = os.path.join(aligned_dir, f"{basename}_aligned.jpg")
    cv2.imwrite(aligned_path, aligned)
    print(f"[INFO] Saved aligned plate: {aligned_path}")

def main():
    parser = argparse.ArgumentParser(description="Interactive plate corner picker + cropper.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .jpeg images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find .jpeg or jpg files (case-insensitive)
    patterns = [os.path.join(in_dir, "*.jpeg"), os.path.join(in_dir, "*.JPEG"),
                os.path.join(in_dir, "*.jpg"), os.path.join(in_dir, "*.JPG")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)

    if not files:
        print("[ERROR] No JPEG files found in input directory.")
        return

    for path in files:
        try:
            process_image(path, out_dir)
        except Exception as e:
            print(f"[ERROR] Failed processing {path}: {e}")

    print("[DONE] All images processed.")


if __name__ == "__main__":
    main()
