import argparse
import cv2
from pathlib import Path
import time
from utils import ensure_output_dir, detect_red_object, detect_faces

OUTPUT = ensure_output_dir()

def draw_boxes(frame, boxes, label="Object", color=(0,255,0)):
    for i,(x,y,w,h) in enumerate(boxes):
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_frame(frame, detect_mode):
    if detect_mode == "color":
        boxes, mask = detect_red_object(frame)
        draw_boxes(frame, boxes, label="RedObj", color=(0,255,0))
        return frame, mask
    elif detect_mode == "face":
        try:
            boxes = detect_faces(frame)
        except Exception as e:
            # propagate for calling code to show friendly message
            raise
        draw_boxes(frame, boxes, label="Face", color=(255,0,0))
        return frame, None
    else:
        raise ValueError("Unknown detect mode: choose 'color' or 'face'")

def run_webcam(detect_mode, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open webcam. Exiting.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            out_frame, mask = process_frame(frame, detect_mode)
        except Exception as e:
            print("Detection error:", e)
            break
        cv2.imshow("Live", out_frame)
        if mask is not None:
            cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image(path, detect_mode):
    path = Path(path)
    frame = cv2.imread(str(path))
    if frame is None:
        print("Unable to read image:", path)
        return
    try:
        out_frame, mask = process_frame(frame, detect_mode)
    except Exception as e:
        print("Detection error:", e)
        return
    timestamp = int(time.time())
    outpath = OUTPUT / f"result_{timestamp}.jpg"
    cv2.imwrite(str(outpath), out_frame)
    print("Result saved to", outpath)
    cv2.imshow("Result", out_frame)
    if mask is not None:
        cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_sample_image():
    # create a simple image with a red circle (PIL)
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640,480), (200,200,200))
    draw = ImageDraw.Draw(im)
    # draw a red circle (object)
    draw.ellipse((220,140,420,340), fill=(200,30,30))
    p = Path("images")
    p.mkdir(exist_ok=True)
    sample = p / "sample.jpg"
    im.save(sample)
    print("Generated sample image at", sample)
    return sample

def main():
    parser = argparse.ArgumentParser(description="Simple Image Recognition (OpenCV)")
    parser.add_argument("--mode", choices=["webcam","image","sample"], default="sample", help="Run mode")
    parser.add_argument("--input", help="Input image path for image mode")
    parser.add_argument("--detect", choices=["color","face"], default="color", help="Detection mode")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()
    if args.mode == "webcam":
        try:
            run_webcam(args.detect, cam_index=args.cam)
        except Exception as e:
            print("Error running webcam:", e)
    elif args.mode == "image":
        if not args.input:
            print("Provide --input for image mode.")
            return
        run_image(args.input, args.detect)
    elif args.mode == "sample":
        sample = generate_sample_image()
        run_image(sample, args.detect)

if __name__ == "__main__":
    main()
