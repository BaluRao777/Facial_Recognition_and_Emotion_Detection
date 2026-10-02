"""Shared inference: detect faces, then predict emotion and identity for each."""

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import (
    EMOTION_MODEL_PATH,
    FACE_MODEL_PATH,
    EMOTION_ENCODER_PATH,
    FACE_ENCODER_PATH,
    IMAGE_SIZE,
    FACE_CROP_MARGIN,
    EMOTION_CROP_MARGIN,
    UNKNOWN_FACE_THRESHOLD,
    INFERENCE_XLA,
)
from src.face_utils import (
    crop_face,
    detect_faces,
    prepare_emotion_input,
    prepare_face_input,
)


class _Classifier:
    def __init__(self, model_path: str, encoder_path: str):
        self.model = load_model(model_path, compile=False)
        with open(encoder_path, "rb") as f:
            self.classes = pickle.load(f).classes_
        signature = [tf.TensorSpec([None, *IMAGE_SIZE, 3], tf.float32)]
        self._forward = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=signature,
            jit_compile=INFERENCE_XLA,
        )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self._forward(tf.constant(batch)).numpy()


class FaceEmotionPredictor:
    """Loads the trained models once and analyses BGR frames."""

    def __init__(self, emotion: bool = True, face: bool = True):
        self.emotion = (
            _Classifier(EMOTION_MODEL_PATH, EMOTION_ENCODER_PATH)
            if emotion and os.path.isfile(EMOTION_MODEL_PATH)
            else None
        )
        self.face = (
            _Classifier(FACE_MODEL_PATH, FACE_ENCODER_PATH)
            if face and os.path.isfile(FACE_MODEL_PATH)
            else None
        )
        if self.emotion is None and self.face is None:
            raise FileNotFoundError(
                "No trained models found in src/checkpoints/. Run `python main.py` first."
            )

    def predict_probabilities(self, image_bgr: np.ndarray, whole_image_fallback=False):
        """
        Returns (boxes, emotion_probs, face_probs); the prob arrays are None when
        that model is not loaded. With whole_image_fallback, an image with no
        detected face is treated as one already-cropped face.
        """
        boxes = detect_faces(image_bgr)
        if not boxes and whole_image_fallback:
            h, w = image_bgr.shape[:2]
            boxes = [(0, 0, w, h)]
            margins = (0.0, 0.0)
        else:
            margins = (EMOTION_CROP_MARGIN, FACE_CROP_MARGIN)
        if not boxes:
            return [], None, None

        emotion_probs = face_probs = None
        if self.emotion is not None:
            batch = np.stack(
                [prepare_emotion_input(crop_face(image_bgr, b, margins[0])) for b in boxes]
            )
            emotion_probs = self.emotion.predict(batch)
        if self.face is not None:
            batch = np.stack(
                [prepare_face_input(crop_face(image_bgr, b, margins[1])) for b in boxes]
            )
            face_probs = self.face.predict(batch)
        return boxes, emotion_probs, face_probs

    def describe(self, emotion_probs=None, face_probs=None) -> dict:
        """Turn one face's probability vectors into labels."""
        result = {}
        if emotion_probs is not None:
            idx = int(np.argmax(emotion_probs))
            result["emotion"] = str(self.emotion.classes[idx])
            result["emotion_confidence"] = float(emotion_probs[idx])
        if face_probs is not None:
            idx = int(np.argmax(face_probs))
            confidence = float(face_probs[idx])
            known = confidence >= UNKNOWN_FACE_THRESHOLD
            result["identity"] = str(self.face.classes[idx]) if known else "Unknown"
            result["identity_confidence"] = confidence
        return result

    def analyze(self, image_bgr: np.ndarray, whole_image_fallback=False) -> list:
        boxes, emotion_probs, face_probs = self.predict_probabilities(
            image_bgr, whole_image_fallback
        )
        results = []
        for i, box in enumerate(boxes):
            result = self.describe(
                None if emotion_probs is None else emotion_probs[i],
                None if face_probs is None else face_probs[i],
            )
            result["box"] = box
            results.append(result)
        return results
