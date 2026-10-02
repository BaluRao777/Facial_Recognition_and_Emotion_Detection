"""Face detection (OpenCV YuNet) and cropping shared by training and inference."""

import os
import logging
import urllib.request

import cv2
import numpy as np

from src.config import (
    BACKBONE,
    FACE_DETECTOR_PATH,
    FACE_DETECTOR_URL,
    FACE_DETECTOR_SCORE_THRESHOLD,
    IMAGE_SIZE,
)

logger = logging.getLogger(__name__)

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        if not os.path.isfile(FACE_DETECTOR_PATH):
            os.makedirs(os.path.dirname(FACE_DETECTOR_PATH), exist_ok=True)
            logger.info("Downloading face detector to %s", FACE_DETECTOR_PATH)
            urllib.request.urlretrieve(FACE_DETECTOR_URL, FACE_DETECTOR_PATH)
        _detector = cv2.FaceDetectorYN.create(
            FACE_DETECTOR_PATH, "", (320, 320), FACE_DETECTOR_SCORE_THRESHOLD, 0.3, 5000
        )
    return _detector


def detect_faces(image_bgr: np.ndarray) -> list:
    """Return face boxes as (x, y, w, h) ints, largest first."""
    detector = _get_detector()
    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    boxes = [tuple(int(v) for v in face[:4]) for face in faces]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def crop_face(image_bgr: np.ndarray, box, margin: float = 0.0) -> np.ndarray:
    """Square crop centred on the box, enlarged by `margin`, clipped to the image."""
    x, y, w, h = box
    side = max(w, h) * (1.0 + margin)
    cx, cy = x + w / 2.0, y + h / 2.0
    img_h, img_w = image_bgr.shape[:2]
    left = int(max(0, round(cx - side / 2)))
    top = int(max(0, round(cy - side / 2)))
    right = int(min(img_w, round(cx + side / 2)))
    bottom = int(min(img_h, round(cy + side / 2)))
    return image_bgr[top:bottom, left:right]


def central_face_box(image_bgr: np.ndarray):
    """Box of the detected face closest to the image centre, or None."""
    boxes = detect_faces(image_bgr)
    if not boxes:
        return None
    h, w = image_bgr.shape[:2]
    return min(
        boxes,
        key=lambda b: (b[0] + b[2] / 2 - w / 2) ** 2 + (b[1] + b[3] / 2 - h / 2) ** 2,
    )


def scale_pixels(x: np.ndarray) -> np.ndarray:
    """Scale 0-255 RGB pixels to what the ImageNet backbone expects."""
    if BACKBONE == "efficientnetb0":
        return x  # EfficientNet rescales internally
    return x / 127.5 - 1.0


def prepare_emotion_input(face_bgr: np.ndarray) -> np.ndarray:
    """Emotion model is trained on greyscale faces (FER2013)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    return scale_pixels(rgb)


def prepare_face_input(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return scale_pixels(rgb)
