# ExamShield AI — N4020 Compatibility Patch

## Problem

Intel Celeron N4020 **lacks AVX and AVX2** instruction sets.  
The original stack crashes at import time with `Illegal instruction (SIGILL)` because:

| Package | Why it crashes |
|---|---|
| `intel-tensorflow 2.11` | Compiled with `-mavx2` — emits AVX2 instructions at startup |
| `mediapipe 1.0.1` | Ships a pre-built TFLite runtime with AVX2 intrinsics |
| `deepface` | Pulls in TensorFlow as a hard dependency |

## Solution — Stack Replacement

| Old (broken) | New (N4020 safe) | Reason |
|---|---|---|
| TensorFlow/Keras YOLOv3 graph | **OpenCV DNN** `readNetFromDarknet` | OpenCV DNN is pure C++/SSE — no AVX |
| `deepface` | **`face_recognition`** (dlib) | dlib only needs SSE4.1 — N4020 has it |
| `mediapipe` FaceMesh (gaze) | **dlib** `shape_predictor_68` + OpenCV | Same dlib dep, zero AVX requirement |
| `mediapipe` FaceMesh (landmarks) | **dlib** `shape_predictor_68` | Same dlib dep, zero AVX requirement |

All business logic is preserved.  `get_frame()` returns the identical dict.  
`app.py` only needs two small changes (handled by `apply_patch.py`).

---

## Files in this patch

```
examshield_fix/
├── requirements.txt                              ← full replacement
├── apply_patch.py                                ← patches app.py in-place
├── PATCH_NOTES.md                                ← this file
└── features/
    └── tracking/
        ├── face/
        │   ├── camera.py          ← full replacement (was TF/Keras YOLOv3)
        │   ├── face_detector.py   ← updated (auto-download + CPU backend pin)
        │   └── face_landmarks.py  ← full replacement (was mediapipe)
        └── gaze/
            └── gaze_tracking.py   ← full replacement (was mediapipe)
```

---

## Installation

### Step 1 — Copy patch files into the project

```bash
cd ~/Documents/ExamShieldAi

# Copy all patched files (overwrite originals)
cp -r /path/to/examshield_fix/features ./
cp /path/to/examshield_fix/requirements.txt ./
cp /path/to/examshield_fix/apply_patch.py  ./
```

### Step 2 — Remove the broken packages

```bash
pip uninstall -y \
    tensorflow \
    intel-tensorflow \
    tensorflow-estimator \
    tensorflow-io-gcs-filesystem \
    keras \
    mediapipe \
    deepface
```

### Step 3 — Install the compatible stack

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` builds from source the first time (~3–5 minutes on N4020).  
> Ensure `cmake` and `build-essential` are installed:
> ```bash
> sudo apt install -y cmake build-essential
> ```

### Step 4 — Patch app.py

```bash
python apply_patch.py
```

This creates `app.py.bak` (safe backup) then patches two spots in `app.py`.

### Step 5 — Download YOLO assets (one-time)

`yolov3.weights` (~236 MB) auto-downloads on the **first** proctoring request.  
To pre-download manually:

```bash
# Config + class names (tiny, fast)
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg \
     -O model_defination/yolov3.cfg

wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names \
     -O model_defination/coco.names

# Weights (~236 MB)
wget https://pjreddie.com/media/files/yolov3.weights \
     -O model_defination/yolov3.weights
```

### Step 6 — Run

```bash
python app.py
```

---

## Manual app.py patch (if apply_patch.py fails)

Open `app.py` and make two edits:

### Edit 1 — Line ~30, replace the import

```python
# REMOVE this line:
from deepface import DeepFace

# ADD this line:
import face_recognition
```

### Edit 2 — Lines ~728–737, replace the verify block

```python
# REMOVE:
            # Convert to grayscale for DeepFace
            captured_cv_gray = cv2.cvtColor(captured_cv_color, cv2.COLOR_BGR2GRAY)
            stored_cv_gray = cv2.cvtColor(stored_cv, cv2.COLOR_BGR2GRAY)

            # Perform face verification
            result = DeepFace.verify(captured_cv_gray, stored_cv_gray, enforce_detection=True)
            print(result)

            if result['verified']:

# ADD:
            # face_recognition (dlib, SSE4.1 only — works on N4020)
            captured_rgb  = cv2.cvtColor(captured_cv_color, cv2.COLOR_BGR2RGB)
            stored_rgb    = cv2.cvtColor(stored_cv,         cv2.COLOR_BGR2RGB)

            captured_encs = face_recognition.face_encodings(captured_rgb)
            stored_encs   = face_recognition.face_encodings(stored_rgb)

            if not captured_encs or not stored_encs:
                raise ValueError("No face detected in one or both images.")

            match  = face_recognition.compare_faces(
                [stored_encs[0]], captured_encs[0], tolerance=0.55
            )
            result = {"verified": match[0]}
            print(result)

            if result['verified']:
```

---

## Verifying the fix

```bash
# Should print "OK" with no SIGILL / ImportError
python -c "
import cv2
import face_recognition
import dlib
from features.tracking.face.camera import get_frame
from features.tracking.gaze.gaze_tracking import GazeTracking
print('OK — all imports successful, no AVX required')
"
```
