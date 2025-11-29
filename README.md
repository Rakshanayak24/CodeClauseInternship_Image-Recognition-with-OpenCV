# Simple Image Recognition Project (OpenCV)

## Aim
Build a simple image recognition system using OpenCV (Python). This project demonstrates:
- Color-based object detection (detect a red object)
- Optional Haar cascade face detection (if `opencv-python` includes Haar cascades on your system)
- How to run using webcam or image files

## Technologies
- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- Pillow (only for generating sample image)

## Files
- `main.py` — main runnable script
- `utils.py` — helper functions
- `requirements.txt` — pip requirements
- `images/sample.jpg` — generated sample image (contains a red circle)

## Usage (VS Code / Terminal)
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:

- Run webcam color detection (detects red objects):
  ```bash
  python main.py --mode webcam --detect color
  ```

- Run webcam face detection (if Haar cascade available on your OpenCV installation):
  ```bash
  python main.py --mode webcam --detect face
  ```

- Run on an image file (saved results are written to `output/`):
  ```bash
  python main.py --mode image --input images/sample.jpg --detect color
  ```

- Generate and use the sample image:
  ```bash
  python main.py --mode sample --detect color
  ```

Press `q` in the display window to quit. Processed images are saved to `output/` folder.

## Notes
- Face detection uses OpenCV's Haar cascade file path (`cv2.data.haarcascades`). If your `opencv-python` installation doesn't include cascades, face detection will show an informative error.
- This is a simple educational project to get hands-on with OpenCV image processing basics.

Enjoy! 🎯
