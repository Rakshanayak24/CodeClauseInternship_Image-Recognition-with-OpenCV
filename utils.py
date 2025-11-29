import cv2
import numpy as np
from pathlib import Path

def ensure_output_dir():
    out = Path("output")
    out.mkdir(exist_ok=True)
    return out

def detect_red_object(frame, min_area=500):
    """
    Detects red-colored objects using HSV thresholding.
    Returns bounding boxes list [(x,y,w,h), ...] and mask used.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # red can wrap around 180 hue, so combine two ranges
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append((x,y,w,h))
    return boxes, mask

def detect_faces(frame):
    """
    Detects faces using Haar cascade (if available). Returns boxes or raises informative error.
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except Exception as e:
        raise RuntimeError("cv2.data.haarcascades not available in this OpenCV build.") from e
    if not Path(cascade_path).exists():
        raise RuntimeError(f"Haar cascade file not found at {cascade_path}.")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    boxes = [(int(x),int(y),int(w),int(h)) for (x,y,w,h) in faces]
    return boxes
