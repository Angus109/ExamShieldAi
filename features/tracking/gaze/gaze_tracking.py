"""
features/tracking/gaze/gaze_tracking.py
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement — removes mediapipe dependency.

Compatibility fix for Intel Celeron N4020 (no AVX / no AVX2):

  BEFORE  →  mediapipe 1.0.1 ships pre-compiled TFLite with AVX2 intrinsics
              → SIGILL crash on N4020

  AFTER   →  dlib shape_predictor_68  (SSE4.1 only — N4020 has it)
              + OpenCV pupil detection (pure C++/SSE)
              → zero AVX requirement

Public API: unchanged
    gaze = GazeTracking()
    gaze.refresh(frame)
    gaze.is_blinking()       → bool
    gaze.is_right()          → bool
    gaze.is_left()           → bool
    gaze.is_center()         → bool
    gaze.horizontal_ratio()  → float | None
    gaze.vertical_ratio()    → float | None  (reserved, always None for now)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import dlib
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locate dlib's bundled 68-point predictor dat file
# (installed as part of the face_recognition package)
# ─────────────────────────────────────────────────────────────────────────────

def _predictor_path() -> str:
    try:
        import face_recognition_models
        return face_recognition_models.pose_predictor_model_location()
    except ImportError:
        pass

    # Fallback: scan common locations
    candidates = [
        Path("model_defination/shape_predictor_68_face_landmarks.dat"),
        Path("shape_predictor_68_face_landmarks.dat"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "shape_predictor_68_face_landmarks.dat not found. "
        "Install face_recognition (pip install face_recognition) "
        "or place the .dat file in model_defination/."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GazeTracking
# ─────────────────────────────────────────────────────────────────────────────

class GazeTracking:
    """
    Lightweight gaze tracker built on dlib 68-point landmarks.

    Replaces the mediapipe FaceMesh implementation with an approach that
    requires only SSE4.1 (available on Celeron N4020).

    Algorithm
    ---------
    1. Detect face with dlib HOG detector.
    2. Predict 68 landmarks with shape_predictor_68_face_landmarks.
    3. Compute Eye Aspect Ratio (EAR) for blink detection.
    4. Threshold the eye ROI to find the pupil centroid and derive a
       normalised horizontal ratio  (0 = far-left, 1 = far-right).
    """

    # EAR below this → blinking
    EAR_BLINK_THRESH: float = 0.20

    # Horizontal ratio boundaries
    GAZE_LEFT_THRESH:  float = 0.40   # ratio ≤ 0.40 → looking right in image
    GAZE_RIGHT_THRESH: float = 0.60   # ratio ≥ 0.60 → looking left  in image

    # dlib 68-point eye index ranges
    _RIGHT_EYE_IDX: list[int] = list(range(36, 42))
    _LEFT_EYE_IDX:  list[int] = list(range(42, 48))

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._detector  = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(_predictor_path())
        self._reset()

    def _reset(self) -> None:
        self._h_ratio: float | None = None
        self._v_ratio: float | None = None
        self._blinking: bool        = False

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _ear(pts: np.ndarray) -> float:
        """
        Eye Aspect Ratio  (Soukupová & Čech, CVWW 2016).
        pts : (6, 2) array of eye landmark coordinates.
        """
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return float((A + B) / (2.0 * C + 1e-6))

    @staticmethod
    def _pupil_ratio(eye_pts: np.ndarray, gray: np.ndarray) -> float | None:
        """
        Normalised horizontal pupil position within the eye bounding box.

        Returns
        -------
        float in [0, 1]  — 0 = far left edge, 1 = far right edge
        None             — if the pupil cannot be localised
        """
        xs, ys = eye_pts[:, 0], eye_pts[:, 1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Isolate dark (pupil) pixels
        _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0.0:
            return None

        cx    = M["m10"] / M["m00"]
        ratio = cx / (roi.shape[1] + 1e-6)
        return float(np.clip(ratio, 0.0, 1.0))

    # ── public API ────────────────────────────────────────────────────────────

    def refresh(self, frame: np.ndarray) -> None:
        """
        Analyse one BGR frame.  Must be called before any is_* / ratio method.
        """
        self._reset()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self._detector(gray, 0)
        if not dets:
            return

        # Use the largest detected face
        det   = max(dets, key=lambda d: d.area())
        shape = self._predictor(gray, det)

        pts = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float64,
        )

        r_eye = pts[self._RIGHT_EYE_IDX]
        l_eye = pts[self._LEFT_EYE_IDX]

        avg_ear = (self._ear(r_eye) + self._ear(l_eye)) / 2.0
        self._blinking = avg_ear < self.EAR_BLINK_THRESH

        if not self._blinking:
            r_ratio = self._pupil_ratio(r_eye.astype(np.int32), gray)
            l_ratio = self._pupil_ratio(l_eye.astype(np.int32), gray)
            valid   = [r for r in (r_ratio, l_ratio) if r is not None]
            if valid:
                self._h_ratio = float(np.mean(valid))

    def is_blinking(self) -> bool:
        """True when EAR is below blink threshold."""
        return self._blinking

    def horizontal_ratio(self) -> float | None:
        """
        Normalised horizontal gaze ratio (0 = left, 1 = right).
        None if no face / pupil was detected this frame.
        """
        return self._h_ratio

    def vertical_ratio(self) -> float | None:
        """Reserved — always None in this implementation."""
        return self._v_ratio

    def is_right(self) -> bool:
        """True when gaze is directed to the right side of the frame."""
        r = self._h_ratio
        return r is not None and r <= self.GAZE_LEFT_THRESH

    def is_left(self) -> bool:
        """True when gaze is directed to the left side of the frame."""
        r = self._h_ratio
        return r is not None and r >= self.GAZE_RIGHT_THRESH

    def is_center(self) -> bool:
        """True when gaze is roughly centered."""
        r = self._h_ratio
        return r is not None and self.GAZE_LEFT_THRESH < r < self.GAZE_RIGHT_THRESH
