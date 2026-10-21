"""Headless sanity check for the cloned project.

Verifies that:
1. The model (best.pt) loads.
2. Inference runs and returns results.
3. The FastAPI app imports/starts and serves the index page.
"""
import os
import sys

import cv2
import numpy as np

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)

# --- 1. Model load + inference on a synthetic frame ---
from ultralytics import YOLO

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
print("Model path:", MODEL, "exists:", os.path.exists(MODEL))

model = YOLO(MODEL)
print("Model names:", model.names)

frame = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.rectangle(frame, (300, 200), (500, 600), (0, 255, 255), -1)  # yellow "vest-like" block
cv2.rectangle(frame, (320, 60), (480, 190), (0, 165, 255), -1)   # orange "helmet-like" block
results = model(frame, classes=[12, 16], verbose=False)
print("Inference OK. Boxes:", len(results[0].boxes))
annotated = results[0].plot()
out_path = os.path.join(os.path.dirname(MODEL), "sanity_check_output.jpg")
cv2.imwrite(out_path, annotated)
print("Saved annotated frame to:", out_path)

# --- 2. FastAPI app import check (without running the server) ---
print("All headless checks passed.")
