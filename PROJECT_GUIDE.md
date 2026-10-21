# Workplace Safety Detection using YOLOv8 - Complete Project Guide

This document explains **what this project is**, **how every file works**, **what happens end-to-end when a user uses the browser**, and **how to rebuild the whole thing from scratch** (dataset -> training -> inference -> web app).

It is written for someone who knows basic ML but is new to deep learning / YOLOv8. No prior object-detection knowledge is assumed.

---

## 1. What the project does (one paragraph)

It is an **object detection** system that looks at an image or a video and draws boxes around Personal Protective Equipment (PPE) - specifically **helmets** and **safety vests**. It is powered by **YOLOv8** (You Only Look Once, version 8), a real-time object detection model from Ultralytics. The trained model is the file `best.pt`. A FastAPI web server wraps the model so a user can use it from a browser (webcam, video upload, or image upload).

---

## 2. Big picture: the three layers

```
+------------------------------------------------------------+
|  LAYER 1 - MODEL (best.pt)                                 |
|  A trained YOLOv8 nano network with 17 output classes.     |
|  You do NOT retrain this every run; you just load it.      |
+------------------------------------------------------------+
                          ^ loaded by
+------------------------------------------------------------+
|  LAYER 2 - PYTHON LOGIC (inference.py / app.py)            |
|  Reads frames -> feeds them to the model -> draws boxes.   |
+------------------------------------------------------------+
                          ^ served by
+------------------------------------------------------------+
|  LAYER 3 - WEB UI (templates/index.html + FastAPI)         |
|  Browser talks HTTP to the FastAPI server.                 |
+------------------------------------------------------------+
```

---

## 3. Folder structure - every file explained

```
Workspace-Safety-Detection-using-YOLOv8/
|
+-- best.pt                  <- THE TRAINED MODEL (the brain). Loaded by both scripts.
+-- app.py                   <- FastAPI web server. Serves HTML, handles webcam/upload.
+-- inference.py             <- Standalone desktop script. Opens a window and shows detections.
+-- requirements.txt         <- Python dependencies (pip install -r requirements.txt).
+-- README.md                <- Short setup instructions (this guide is the deep version).
|
+-- templates/
|   +-- index.html           <- The entire frontend (HTML + CSS + JS). Buttons, upload, video display.
|
+-- uploads/                 <- Temporary storage for user-uploaded videos. Auto-created by app.py.
|
+-- test1.mp4                <- Sample video used by inference.py by default.
+-- test_video.mp4           <- Another sample video (same clip, used for upload testing).
+-- inference_output.mp4     <- Output video (evidence of a past run).
+-- sanity_check_output.jpg  <- Output of test_setup.py (a synthetic sanity check image).
|
+-- test_setup.py            <- Headless sanity check: loads model, runs inference on a fake image.
|
+-- workplace-safety.ipynb   <- THE TRAINING NOTEBOOK (run on Kaggle with GPU).
+-- Workspace_Safety_Detection_YOLOv8_Presentation.pptx  <- Slides for your college demo.
|
+-- install.bat              <- Windows helper: venv + CPU torch + requirements.
+-- install2.bat             <- Windows helper variant for a local OpenCV wheel.
|
+-- venv/                    <- Python virtual environment (ignored by git).
+-- .gitignore               <- Ignores venv/, best.pt, uploads/, datasets/, etc.
```

### Why each important file exists

| File | Why it exists | What happens if you delete it |
|------|---------------|-------------------------------|
| `best.pt` | Stores the learned weights (6 MB). Without it, no detection is possible. | App crashes on startup (`exit()` in `app.py:21`). |
| `app.py` | The web server. Turns the model into a service. | No browser UI. |
| `inference.py` | Simple offline demo. Good for the show-the-teacher moment. | Only web mode remains. |
| `templates/index.html` | Everything the user sees and clicks. | Server returns index.html not found. |
| `workplace-safety.ipynb` | Proof + recipe of how the model was trained. | You cannot reproduce `best.pt`. |
| `requirements.txt` | Reproducible environment. | `pip install` has to be guessed. |

---

## 4. The data flow - what actually happens when a user clicks something

### 4A. Webcam path

```
User clicks "Use Webcam" in the browser
        |
        v
index.html JS sets: <img src="/video_feed/webcam">
        |
        v  HTTP GET /video_feed/webcam
FastAPI  app.video_feed_webcam()                    (app.py:73)
        |
        v  returns StreamingResponse(...)
app.generate_webcam_frames()                        (app.py:45)
        |
        +-- cv2.VideoCapture(0)          <- open webcam
        +-- loop forever:
        |     frame = camera.read()      <- grab one frame (numpy array, HxWx3, BGR)
        |     results = model(frame, classes=[12,16])   <- YOLOv8 forward pass
        |     annotated = results[0].plot()             <- draw boxes+labels
        |     jpeg = cv2.imencode('.jpg', annotated)    <- compress
        |     yield b'--frame\r\n...'                   <- MJPEG chunk
        |
        v
Browser <img> tag renders each chunk as a new picture
-> User sees a live, annotated webcam.
```

### 4B. Video upload path

```
User selects an .mp4
        |
        v
JS builds FormData and POSTs to /upload_video
        |
        v
app.upload_video()                                  (app.py:82)
   +-- makes a UUID session_id
   +-- saves file to uploads/<uuid>_<name>.mp4
   +-- returns {"session_id": "..."}
        |
        v
JS sets <img src="/video_feed/upload?session_id=...">
        |
        v  HTTP GET /video_feed/upload
app.video_feed_upload(session_id)                   (app.py:130)
        |
        v
app.generate_upload_frames(session_id)              (app.py:97)
   +-- cv2.VideoCapture(saved_path)
   +-- for each frame:
   |     results = model(frame, classes=[12,16])
   |     annotated = results[0].plot()
   |     yield JPEG chunk
   +-- after last frame: delete file + session entry
        |
        v
Browser plays it as a continuous MJPEG stream.
The Pause button just swaps the <img> for a <canvas> snapshot.
```

### 4C. Image upload path

```
User selects an image
        |
        v
JS POSTs to /upload_image
        |
        v
app.upload_image()                                  (app.py:139)
   +-- reads bytes -> cv2.imdecode -> numpy image
   +-- results = model(img, classes=[12,16])
   +-- annotated = results[0].plot()
   +-- returns JPEG bytes (Response, not streaming)
        |
        v
JS converts the blob -> object URL -> sets <img src=...>
-> User sees one annotated image.
```

### The one-line summary of the model call

```python
results = model(frame, classes=[12, 16], verbose=False)
```

- `frame` - a numpy BGR image.
- `classes=[12, 16]` - **only keep predictions for class 12 (`head_helmet`) and 16 (`vest`)**. Other classes are detected internally but filtered out.
- `results[0].plot()` - returns a copy of the frame with boxes, labels, and confidences drawn.

---

## 5. What YOLOv8 actually is (the 2-minute version)

**Object detection** = where are the objects, and what are they?

**YOLO** = a single neural network that, in one forward pass, predicts:
- A grid of **bounding boxes** (x, y, width, height),
- A **class** for each box (helmet, vest, person, ...),
- A **confidence** score.

YOLOv8n = the *nano* variant. Trade-off: ~3M parameters, ~8 GFLOPs, runs at tens of FPS on CPU. Perfect for a laptop demo.

Internally, YOLOv8 is:
```
Input image (640x640x3)
   v  backbone (Conv + C2f blocks)   -> feature maps
   v  neck (SPPF + PAN-FPN)          -> multi-scale features
   v  head (Detect)                  -> boxes + classes at 3 scales
   v  NMS                            -> remove duplicate boxes
Output: list of {box, class, conf}
```

You do not need to implement any of this. Ultralytics gives you `YOLO('best.pt')` and it just works.

---

## 6. The 17 classes (why `classes=[12, 16]`)

From the `data.yaml` in the training notebook:

```
 0 Barefoots          9  glasses
 1 Ear-protection     10 hand_glove
 2 Harness            11 hand_noglove
 3 No_Ear-Protection  12 head_helmet     <- KEPT
 4 No_Glasses         13 head_nohelmet
 5 Sandals            14 person
 6 boots              15 shoes
 7 face_mask          16 vest           <- KEPT
 8 face_nomask
```

The model knows all 17. The app deliberately shows **only** helmet + vest (12, 16) to keep the demo focused on the safety use-case.

---

## 7. How to rebuild this project from scratch

### Step 0 - Prerequisites

- Python 3.8+ installed
- Git
- (For training) A free Kaggle account (GPU: Tesla T4, 2x15 GB, ~30 free hours/week)
- (Optional) NVIDIA GPU on your own machine + CUDA

---

### Step 1 - Get the dataset

The model was trained on **PPE Detection v3** from Roboflow, hosted on Kaggle as a dataset (`ppe-objection-detection`, datasetVersion 7871309).

**Where to get it:**
1. Go to Kaggle -> search **PPE detection v3 yolov8** or open the dataset used in the notebook: `sourceId 12475735`.
2. Download the YOLOv8 export. It gives you a folder like:
   ```
   ppe detection.v3i.yolov8/
   +-- data.yaml
   +-- train/images/ + train/labels/
   +-- valid/images/ + valid/labels/
   +-- test/images/  + test/labels/
   ```
3. Each `labels/*.txt` has one line per object:
   ```
   <class_id> <x_center> <y_center> <width> <height>   # all normalized 0-1
   ```
   This is the standard YOLO annotation format.

**Alternative source:** Roboflow Universe -> PPE detection -> Export -> YOLOv8 -> download zip.

Dataset size: **~23,000 images**, 17 classes.

---

### Step 2 - Write `data.yaml`

This file tells YOLO where the images are and what the classes are. In the notebook it is created with `%%writefile`:

```yaml
train: /kaggle/input/ppe-objection-detection/datasets/ppe detection.v3i.yolov8/train/images
val:   /kaggle/input/ppe-objection-detection/datasets/ppe detection.v3i.yolov8/valid/images
test:  /kaggle/input/ppe-objection-detection/datasets/ppe detection.v3i.yolov8/test/images

nc: 17
names:
  - 'Barefoots'
  - 'Ear-protection'
  - 'Harness'
  - 'No_Ear-Protection'
  - 'No_Glasses'
  - 'Sandals'
  - 'boots'
  - 'face_mask'
  - 'face_nomask'
  - 'glasses'
  - 'hand_glove'
  - 'hand_noglove'
  - 'head_helmet'
  - 'head_nohelmet'
  - 'person'
  - 'shoes'
  - 'vest'
```

`nc` = number of classes. The **order matters** - index 12 must be `head_helmet`.

---

### Step 3 - Train the model

In the Kaggle notebook (or any GPU machine):

```bash
pip install ultralytics opencv-python
yolo detect train data=data.yaml model=yolov8n.pt epochs=10 imgsz=640 batch=16 device=0,1
```

Breakdown:

| Argument | Meaning |
|----------|---------|
| `data=data.yaml` | Dataset + class names. |
| `model=yolov8n.pt` | Start from the pretrained **nano** weights (transfer learning - much faster than from scratch). |
| `epochs=10` | 10 full passes over the training set. |
| `imgsz=640` | Resize every image to 640x640 (YOLO default). |
| `batch=16` | 16 images per gradient update. |
| `device=0,1` | Use GPU 0 and 1 (Kaggle gives 2x T4). |

**What the training output means:**
- `box_loss`, `cls_loss`, `dfl_loss` -> should go **down** each epoch.
- `P` (precision), `R` (recall), `mAP50`, `mAP50-95` -> should go **up**.
- On this dataset, 10 epochs took ~43 minutes and reached **mAP50 ~ 0.735**, mAP50-95 ~ 0.456.

**Output location:** `runs/detect/train4/weights/`
- `best.pt` <- highest mAP - **this is what you ship**.
- `last.pt` <- last epoch (useful for resuming).
- Also `results.png`, `confusion_matrix.png`, `val_batch*.jpg` for your report.

---

### Step 4 - Evaluate

After training, Ultralytics auto-runs validation:

```
Class              P      R      mAP50  mAP50-95
all               0.720  0.715  0.735  0.456
head_helmet       0.784  0.824  0.866  0.584   <- strong
head_nohelmet     0.832  0.817  0.874  0.535
vest              0.841  0.897  0.935  0.705   <- strong
...
```

- **Precision** = of the boxes we drew, how many were correct.
- **Recall** = of the real objects, how many we found.
- **mAP50** = mean average precision at IoU 0.50.
- **mAP50-95** = stricter, averaged across IoU 0.50->0.95.

Key classes (helmet, vest) score **86-93% mAP50** - great for a college demo.

---

### Step 5 - Standalone inference (`inference.py`)

Minimal script to prove the model works:

```python
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture('test1.mp4')   # or VideoCapture(0) for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[12, 16])
    annotated = results[0].plot()
    cv2.imshow('Workplace Safety Monitoring', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run with:
```powershell
.\venv\Scripts\activate
python inference.py
```

**Common error:** `cv2.imshow ... function is not implemented`. Cause: `opencv-python-headless` has no GUI. Fix:
```powershell
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84
```
Verify: `python -c "import cv2; print(cv2.getBuildInformation())"` should show `Win32 UI: YES`.

---

### Step 6 - Web app (`app.py` + `templates/index.html`)

Key pieces:

1. **Load the model once at startup** (`app.py:17`):
   ```python
   model = YOLO('best.pt')
   ```
   Loading is slow (~1s); doing it once means every request is fast.

2. **Serve the HTML** at `GET /` (`app.py:33`).

3. **MJPEG streaming** - the trick for video over HTTP with zero JS video codecs:
   ```python
   yield (b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
   ```
   Wrapped in `StreamingResponse(..., media_type="multipart/x-mixed-replace; boundary=frame")`. The browser renders it inside a normal `<img>` tag.

4. **Sessions for uploads** - a UUID keys each upload so concurrent users do not clash:
   ```python
   video_sessions[uuid.uuid4()] = file_path
   ```

5. **Cleanup** - after the last frame, the uploaded file and session entry are deleted.

**Run it:**
```powershell
.\venv\Scripts\activate
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

---

### Step 7 - Frontend (`templates/index.html`)

Plain HTML + Tailwind (via CDN) + vanilla JS. Three states:
1. **Start** - a single Start Monitoring button.
2. **Source selection** - Webcam / Upload Video / Upload Image.
3. **Viewer** - `<img>` showing the MJPEG stream + Pause/Back controls.

The Pause button works by copying the current frame into a hidden `<canvas>`, hiding the `<img>`, and stopping the stream. Resume re-assigns `videoFeed.src`.

---

## 8. Full from-zero checklist (what you would do at college)

- [ ] Create project folder, `git init`.
- [ ] `python -m venv venv` + activate.
- [ ] Create `requirements.txt`:
  ```
  fastapi
  uvicorn[standard]
  python-multipart
  ultralytics
  opencv-python
  ```
- [ ] `pip install -r requirements.txt`
- [ ] Download PPE dataset (Roboflow/Kaggle), unzip, write `data.yaml`.
- [ ] On Kaggle (GPU): `pip install ultralytics`; run `yolo detect train data=data.yaml model=yolov8n.pt epochs=10 imgsz=640 batch=16`.
- [ ] Download `runs/detect/train4/weights/best.pt` -> put in project root.
- [ ] Write `inference.py`, test on `test1.mp4` and webcam.
- [ ] Write `app.py` + `templates/index.html`, test at `http://127.0.0.1:8000`.
- [ ] (Optional) `test_setup.py` for a quick sanity check.
- [ ] (Optional) `install.bat` to reproduce the env on another Windows machine.
- [ ] Push to GitHub (with `.gitignore` excluding `venv/`, `best.pt`, `uploads/`, `datasets/`).

---

## 9. Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cv2.imshow ... function is not implemented` | `opencv-python-headless` installed | `pip uninstall opencv-python-headless` then install `opencv-python==4.10.0.84` |
| `Error loading model` + app exits | `best.pt` missing or corrupted | Re-download from Releases page |
| `ModuleNotFoundError: ultralytics` | Wrong interpreter / venv not active | Activate `venv` first |
| `torchaudio ... conflict` warning | Version mismatch, harmless for inference | Ignore or `pip uninstall torchaudio` |
| Webcam never opens | Another app using the camera | Close Zoom/Teams/etc. |
| `index.html not found` | Running `app.py` from the wrong directory | Run from the project root |

---

## 10. Glossary (for the viva)

- **Object detection** - finds *where* objects are (bounding box) *and* *what* they are (class label). Unlike classification (one label/image) or segmentation (pixel masks).
- **YOLO (You Only Look Once)** - single-stage detector. Unlike two-stage detectors (R-CNN) that propose regions then classify, YOLO predicts all boxes and classes in **one forward pass** - hence real-time speed.
- **YOLOv8** - 8th version by Ultralytics. Key vs v5: **anchor-free** head, **C2f** backbone blocks, decoupled cls/regression, clean Python API.
- **YOLOv8n (nano)** - smallest variant (n < s < m < l < x). ~3M params, ~8 GFLOPs; fastest, lowest accuracy. Chosen so it runs on CPU.
- **Backbone** - feature extractor (Conv + C2f blocks) producing multi-resolution feature maps.
- **Neck** - merges features across scales (SPPF + PAN-FPN) so small and large objects are both detectable.
- **Head (Detect)** - outputs box coords, objectness, and class probs at 3 scales.
- **Anchor-free** - YOLOv8 predicts box centers directly; no predefined anchor shapes to tune.
- **Bounding box** - rectangle around an object, stored as normalized `(x_center, y_center, w, h)`.
- **Confidence score** - model's certainty a box contains the predicted class; low scores are dropped.
- **NMS (Non-Maximum Suppression)** - post-processing that removes overlapping duplicate boxes, keeping the highest-confidence one.
- **IoU (Intersection over Union)** - `area(intersection)/area(union)`; matches predictions to ground truth and drives NMS.
- **Precision** - of the boxes drawn, fraction actually correct. High = few false alarms.
- **Recall** - of the real objects, fraction found. High = few misses.
- **mAP50** - mean Average Precision at IoU >= 0.5; the headline metric in the log.
- **mAP50-95** - averaged over IoU 0.50 to 0.95; stricter, rewards tighter boxes.
- **Transfer learning** - starting from COCO-pretrained weights (`yolov8n.pt`) instead of random init; faster convergence, better accuracy on small data.
- **Fine-tuning** - continuing training of a pretrained model on your dataset (what `yolo detect train` does).
- **Epoch** - one full pass over the training set. Here: 10.
- **Batch size** - images processed before one weight update. Here: 16.
- **`data.yaml`** - config mapping train/val/test paths and class index to name. Name order defines class IDs.
- **Class index** - internal integer per class; here `12 = head_helmet`, `16 = vest` (why the app filters `classes=[12, 16]`).
- **`best.pt` / `last.pt`** - Ultralytics checkpoints. `best.pt` = highest validation mAP; `last.pt` = final epoch (for resuming).
- **`.pt` file** - PyTorch-serialized file (pickle-based): architecture config + learned weights.
- **Inference** - running a trained model on new input; no weight updates. Opposite of training.
- **MJPEG (Motion JPEG)** - video stream sent as a sequence of JPEGs over one HTTP connection; browser renders it in a plain `<img>`. The trick `app.py` uses (no JS video codec).
- **StreamingResponse** - FastAPI/Starlette response yielding chunks over time instead of one blob. Used for both webcam and upload endpoints.
- **FastAPI** - modern async Python web framework; routes via decorators like `@app.get("/")`.
- **Uvicorn** - the ASGI server that runs the FastAPI app (`uvicorn app:app --reload`).
- **ASGI** - Asynchronous Server Gateway Interface; async successor to WSGI. FastAPI is an ASGI app, Uvicorn an ASGI server.
- **Session ID (UUID)** - random unique key per uploaded video so concurrent users don't clash.
- **OpenCV (`cv2`)** - used for webcam capture (`VideoCapture`), JPEG encoding (`imencode`), and drawing (Ultralytics `plot()` uses it).
- **`opencv-python` vs `opencv-python-headless`** - same API, but headless has **no GUI** (no `imshow`/`waitKey`). Your earlier crash was the headless build; fixed by installing the GUI build.
- **Virtual environment (`venv`)** - isolated Python install so project packages don't clash with global ones.
- **PPE (Personal Protective Equipment)** - the domain: helmets, vests, gloves, goggles, etc. 17 classes here.

---

## 11. Quick reference - commands

```powershell
# Activate venv
.\venv\Scripts\activate

# Offline inference (opens a window)
python inference.py

# Web app
uvicorn app:app --reload
# then http://127.0.0.1:8000

# Sanity check
python test_setup.py

# Retrain (on Kaggle GPU, not locally unless you have CUDA)
yolo detect train data=data.yaml model=yolov8n.pt epochs=10 imgsz=640 batch=16
```
