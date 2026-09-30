"""
features/tracking/face/camera.py
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the TensorFlow/Keras YOLOv3 proctoring pipeline.

Compatibility fix for Intel Celeron N4020 (no AVX / no AVX2):

  BEFORE  →  builds YOLOv3 Keras graph at import time
              → TF 2.11 emits AVX2 instructions → SIGILL crash on N4020

  AFTER   →  loads YOLOv3 via cv2.dnn.readNetFromDarknet
              → OpenCV DNN backend is pure C++ / SSE2 / SSE4.1
              → zero TensorFlow dependency, zero AVX requirement

Folder structure: unchanged
Public API:       get_frame(img_data: str) -> dict   (identical contract)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import wget
from PIL import Image

from .face_detector import find_faces, get_face_detector
from .face_landmarks import detect_marks, get_landmark_model
from features.tracking.gaze.gaze_tracking import GazeTracking

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("model_defination")

YOLO_WEIGHTS     = MODEL_DIR / "yolov3.weights"
YOLO_CFG         = MODEL_DIR / "yolov3.cfg"
YOLO_COCO_NAMES  = MODEL_DIR / "coco.names"

YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLO_CFG_URL     = (
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
)
COCO_NAMES_URL   = (
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
)

# ─────────────────────────────────────────────────────────────────────────────
# Detection thresholds
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.40
NMS_THRESHOLD  = 0.40

# COCO class indices
COCO_PERSON     = 0
COCO_CELL_PHONE = 67

# Head-pose deviation limits (degrees)
PITCH_THRESH = 15.0   # up / down
YAW_THRESH   = 25.0   # left / right

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (lazy-loaded on first frame)
# ─────────────────────────────────────────────────────────────────────────────

_net:            cv2.dnn.Net | None = None
_coco_classes:   list[str]          = []
_face_detector                      = None
_landmark_model                     = None
_gaze                               = GazeTracking()


# ─────────────────────────────────────────────────────────────────────────────
# Asset auto-download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_if_missing(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s  (%s) …", label, url)
    wget.download(url, str(dest))
    log.info("  → saved to %s", dest)


def _ensure_yolo_assets() -> None:
    _download_if_missing(YOLO_CFG_URL,     YOLO_CFG,        "yolov3.cfg")
    _download_if_missing(COCO_NAMES_URL,   YOLO_COCO_NAMES, "coco.names")
    # Weights are large (~236 MB) — download last so cfg/names errors surface first
    _download_if_missing(YOLO_WEIGHTS_URL, YOLO_WEIGHTS,    "yolov3.weights (~236 MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton getters
# ─────────────────────────────────────────────────────────────────────────────

def _get_net() -> cv2.dnn.Net:
    global _net
    if _net is not None:
        return _net

    _ensure_yolo_assets()

    net = cv2.dnn.readNetFromDarknet(str(YOLO_CFG), str(YOLO_WEIGHTS))
    # Force CPU/OpenCV backend — no CUDA, no AVX
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _net = net
    log.info("YOLOv3 loaded via OpenCV DNN (no TensorFlow, no AVX).")
    return _net


def _get_coco_classes() -> list[str]:
    global _coco_classes
    if _coco_classes:
        return _coco_classes
    _ensure_yolo_assets()
    with open(YOLO_COCO_NAMES, "r") as fh:
        _coco_classes = [ln.strip() for ln in fh.readlines()]
    return _coco_classes


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = get_face_detector()
    return _face_detector


def _get_landmark_model():
    global _landmark_model
    if _landmark_model is None:
        _landmark_model = get_landmark_model()
    return _landmark_model


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv3 via OpenCV DNN  (replaces the Keras graph build)
# ─────────────────────────────────────────────────────────────────────────────

def _yolo_detect(frame: np.ndarray) -> tuple[int, int]:
    """
    Run YOLOv3 on *frame* using OpenCV DNN.

    Returns
    -------
    (person_count, phone_count)
    """
    net     = _get_net()
    h, w    = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0, size=(416, 416),
        swapRB=True, crop=False,
    )
    net.setInput(blob)

    # Resolve output layer names (works across OpenCV 4.x versions)
    all_layers   = net.getLayerNames()
    unconnected  = net.getUnconnectedOutLayers()
    if isinstance(unconnected[0], (list, np.ndarray)):
        out_layers = [all_layers[i[0] - 1] for i in unconnected]
    else:
        out_layers = [all_layers[i - 1] for i in unconnected]

    outs = net.forward(out_layers)

    boxes:       list[list[int]] = []
    confidences: list[float]     = []
    class_ids:   list[int]       = []

    for out in outs:
        for det in out:
            scores    = det[5:]
            class_id  = int(np.argmax(scores))
            conf      = float(scores[class_id])

            if conf < CONF_THRESHOLD:
                continue
            # Keep only the classes we need
            if class_id not in (COCO_PERSON, COCO_CELL_PHONE):
                continue

            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            confidences.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD,
    )

    person_count = phone_count = 0
    if len(indices) > 0:
        for idx in np.array(indices).flatten():
            cid = class_ids[idx]
            if cid == COCO_PERSON:
                person_count += 1
            elif cid == COCO_CELL_PHONE:
                phone_count  += 1

    return person_count, phone_count


# ─────────────────────────────────────────────────────────────────────────────
# Head-pose via solvePnP (replaces mediapipe pose model)
# ─────────────────────────────────────────────────────────────────────────────

# Generic 3-D face model (mm), matched to dlib 68-point indices
_MODEL_3D = np.array([
    (   0.0,    0.0,    0.0),   # 30 — nose tip
    (   0.0, -330.0,  -65.0),   #  8 — chin
    (-225.0,  170.0, -135.0),   # 36 — left eye corner
    ( 225.0,  170.0, -135.0),   # 45 — right eye corner
    (-150.0, -150.0, -125.0),   # 48 — left mouth corner
    ( 150.0, -150.0, -125.0),   # 54 — right mouth corner
], dtype=np.float64)

_LANDMARK_IDX = [30, 8, 36, 45, 48, 54]


def _head_pose(marks: np.ndarray, frame_shape: tuple) -> tuple[float, float]:
    """
    Estimate (pitch_deg, yaw_deg) from 68-point landmark array.
    Returns (0.0, 0.0) on failure.
    """
    h, w = frame_shape[:2]
    image_points = marks[_LANDMARK_IDX].astype(np.float64)

    focal     = float(w)
    cam_mat   = np.array(
        [[focal, 0.0, w / 2.0],
         [0.0, focal, h / 2.0],
         [0.0,   0.0,     1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_3D, image_points, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1])   # pitch, yaw


# ─────────────────────────────────────────────────────────────────────────────
# Public API  — identical contract to the original camera.py
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Public API  — identical contract to the original camera.py
# ─────────────────────────────────────────────────────────────────────────────

def get_frame(img_data: str) -> dict:
    """
    Process one Base64-encoded JPEG webcam frame from the proctoring client.

    Parameters
    ----------
    img_data : str
        Raw Base64 string (with or without the ``data:image/jpeg;base64,``
        prefix) as sent by the browser AJAX call.

    Returns
    -------
    dict with keys:
        person_status           int   0 = alone,      1 = multiple people
        phone_detection         int   0 = no phone,   1 = phone visible
        user_movements_updown   int   0 = forward,    1 = up/down deviation
        user_movements_lr       int   0 = forward,    1 = left/right deviation
        user_movements_eyes     int   0 = eyes open,  1 = blinking / not seen
        img_log                 str   Base64 JPEG of the (annotated) frame
        jpg_as_text             str   Base64 JPEG of the (annotated) frame (matches app.py)
    """
    result: dict = {
        "person_status":         0,
        "phone_detection":       0,
        "user_movements_updown": 0,
        "user_movements_lr":     0,
        "user_movements_eyes":   0,
        "img_log":               img_data,
        "jpg_as_text":           img_data,  # Added to prevent KeyError in app.py
    }

    # ── 1. Decode frame ───────────────────────────────────────────────────────
    try:
        raw = img_data.split(",", 1)[1] if "," in img_data else img_data
        img_bytes = base64.b64decode(raw)
        pil_img   = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        log.warning("get_frame: decode failed — %s", exc)
        return result

    # ── 2. YOLOv3 — person count + phone detection (OpenCV DNN) ──────────────
    try:
        n_persons, n_phones   = _yolo_detect(frame)
        result["person_status"]   = 1 if n_persons > 1 else 0
        result["phone_detection"] = 1 if n_phones  > 0 else 0
    except Exception as exc:
        log.warning("get_frame: YOLO failed — %s", exc)

    # ── 3. Gaze tracking (dlib-based, replaces mediapipe) ────────────────────
    try:
        _gaze.refresh(frame)

        if _gaze.is_blinking():
            result["user_movements_eyes"] = 1

        if _gaze.is_right() or _gaze.is_left():
            result["user_movements_lr"] = 1

    except Exception as exc:
        log.warning("get_frame: gaze failed — %s", exc)

    # ── 4. Head pose — up/down via solvePnP + dlib landmarks ─────────────────
    try:
        face_model  = _get_face_detector()
        lm_model    = _get_landmark_model()
        faces       = find_faces(frame, face_model)

        if faces:
            marks        = detect_marks(frame, lm_model, faces[0])
            pitch, yaw   = _head_pose(marks, frame.shape)

            if abs(pitch) > PITCH_THRESH:
                result["user_movements_updown"] = 1
            if abs(yaw) > YAW_THRESH:
                result["user_movements_lr"]     = 1

    except Exception as exc:
        log.warning("get_frame: head-pose failed — %s", exc)

    # ── 5. Re-encode annotated frame ──────────────────────────────────────────
    try:
        _, buf     = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64        = base64.b64encode(buf).decode("utf-8")
        encoded_img = "data:image/jpeg;base64," + b64
        result["img_log"]     = encoded_img
        result["jpg_as_text"] = encoded_img  # Populate both expected keys
    except Exception as exc:
        log.warning("get_frame: encode failed — %s", exc)

    return result