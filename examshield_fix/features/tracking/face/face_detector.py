"""
features/tracking/face/face_detector.py
─────────────────────────────────────────────────────────────────────────────
Unchanged from the original — included in the patch zip so the folder is
self-contained and to add the model auto-download guard.

Uses OpenCV DNN with a Caffe ResNet-10 SSD face detector (no TensorFlow,
no AVX requirement — has always been compatible with N4020).

Public API (unchanged):
    model = get_face_detector()
    faces = find_faces(img, model)   → list of [x1, y1, x2, y2]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import wget

MODEL_DIR   = Path("model_defination")

CAFFE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
CAFFE_PROTO = MODEL_DIR / "deploy.prototxt"

TF_MODEL    = MODEL_DIR / "opencv_face_detector_uint8.pb"
TF_CONFIG   = MODEL_DIR / "opencv_face_detector.pbtxt"

# Remote URLs for the Caffe model (small, ~10 MB)
_CAFFE_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_CAFFE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)


def _download_if_missing(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import logging
    logging.getLogger(__name__).info("Downloading %s …", label)
    wget.download(url, str(dest))


def get_face_detector(
    modelFile:  str | None = None,
    configFile: str | None = None,
    quantized:  bool       = False,
) -> cv2.dnn.Net:
    """
    Return an OpenCV DNN face-detection network.

    Parameters
    ----------
    quantized : bool
        If True, load the uint8-quantised TF protobuf model.
        If False (default), load the Caffe ResNet-10 SSD model.
    """
    if quantized:
        mf = modelFile  or str(TF_MODEL)
        cf = configFile or str(TF_CONFIG)
        model = cv2.dnn.readNetFromTensorflow(mf, cf)
    else:
        mf = modelFile  or str(CAFFE_MODEL)
        cf = configFile or str(CAFFE_PROTO)

        # Auto-download Caffe model if missing
        if not Path(mf).exists():
            _download_if_missing(_CAFFE_MODEL_URL, CAFFE_MODEL, "face detector model")
        if not Path(cf).exists():
            _download_if_missing(_CAFFE_PROTO_URL, CAFFE_PROTO, "face detector prototxt")

        model = cv2.dnn.readNetFromCaffe(cf, mf)

    # Always use CPU backend — no AVX, no CUDA needed
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model


def find_faces(img: np.ndarray, model: cv2.dnn.Net) -> list[list[int]]:
    """
    Detect all faces in *img*.

    Returns
    -------
    list of [x1, y1, x2, y2]  — one entry per face, confidence > 0.5
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    model.setInput(blob)
    res   = model.forward()
    faces: list[list[int]] = []

    for i in range(res.shape[2]):
        confidence = float(res[0, 0, i, 2])
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype(int).tolist())

    return faces
