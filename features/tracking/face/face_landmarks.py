"""
features/tracking/face/face_landmarks.py
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement — removes mediapipe dependency.

Compatibility fix for Intel Celeron N4020 (no AVX / no AVX2):

  BEFORE  →  mediapipe FaceMesh (468 landmarks)
              → ships AVX2-compiled TFLite, crashes on N4020

  AFTER   →  dlib shape_predictor_68_face_landmarks (68 landmarks)
              → SSE4.1 only, works on N4020

Public API: unchanged
    model = get_landmark_model()
    marks = detect_marks(img, model, face_bbox)
        → np.ndarray  shape (68, 2)  dtype int32
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import cv2
import dlib
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Locate dlib predictor dat
# ─────────────────────────────────────────────────────────────────────────────

def _predictor_path() -> str:
    """
    Return the filesystem path to shape_predictor_68_face_landmarks.dat.

    Checks (in order):
      1. model_defination/ directory
      2. face_recognition_models package
      3. working directory
    """
    candidates = [
        Path("model_defination/shape_predictor_68_face_landmarks.dat"),
        Path("shape_predictor_68_face_landmarks.dat"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    try:
        import face_recognition_models
        import os
        dat_path = os.path.join(
            os.path.dirname(face_recognition_models.__file__),
            "models",
            "shape_predictor_68_face_landmarks.dat"
        )
        if os.path.exists(dat_path):
            return dat_path
    except ImportError:
        pass

    raise FileNotFoundError(
        "shape_predictor_68_face_landmarks.dat not found.\n"
        "Place the shape_predictor_68_face_landmarks.dat file in model_defination/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_landmark_model() -> dlib.shape_predictor:
    """
    Return a dlib shape_predictor loaded from the 68-point dat file.

    This is the drop-in replacement for the mediapipe model loader.
    The returned object is passed unchanged to detect_marks().
    """
    return dlib.shape_predictor(_predictor_path())


def detect_marks(
    img:   np.ndarray,
    model: dlib.shape_predictor,
    face:  list[int],
) -> np.ndarray:
    """
    Detect 68 facial landmarks in *img* within the given face bounding box.

    Parameters
    ----------
    img   : BGR frame  (numpy ndarray, H × W × 3)
    model : dlib.shape_predictor returned by get_landmark_model()
    face  : [x1, y1, x2, y2] as returned by find_faces() in face_detector.py

    Returns
    -------
    marks : np.ndarray  shape (68, 2)  dtype int32
        Each row is [x, y] pixel coordinate of the corresponding landmark.
        Landmark ordering follows the dlib / iBUG 300-W convention.

    Notes
    -----
    The original mediapipe implementation returned 468 landmarks mapped to
    a subset used by camera.py (indices 30, 8, 36, 45, 48, 54 for solvePnP).
    dlib's 68-point model uses the identical indices for those same anatomical
    points, so the head-pose code in camera.py works without modification.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = (int(v) for v in face)
    rect  = dlib.rectangle(x1, y1, x2, y2)
    shape = model(gray, rect)

    marks = np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(68)],
        dtype=np.int32,
    )
    return marks
