"""Run inference on a single image."""

import argparse

import cv2

from src.inference import FaceEmotionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to face image")
    parser.add_argument("--emotion-only", action="store_true")
    parser.add_argument("--face-only", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Could not read image: {args.image}")

    predictor = FaceEmotionPredictor(
        emotion=not args.face_only, face=not args.emotion_only
    )
    # Images with no detectable face (e.g. tight FER2013 crops) are used whole.
    for result in predictor.analyze(image, whole_image_fallback=True):
        if "emotion" in result:
            print(f"Emotion: {result['emotion']} ({result['emotion_confidence']:.2%})")
        if "identity" in result:
            print(f"Identity: {result['identity']} ({result['identity_confidence']:.2%})")


if __name__ == "__main__":
    main()
