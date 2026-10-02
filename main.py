"""
Facial Recognition + Emotion Detection
Entry point — train and evaluate separate models (GPU-ready for Ubuntu + NVIDIA).
"""

import argparse
import logging
import sys

from src.gpu_utils import print_device_summary
from src.train import train_emotion_model, train_face_model, train_model
from src.evaluate import evaluate_model, evaluate_task
from src.utils import plot_histories, plot_history


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate face recognition and emotion detection models."
    )
    parser.add_argument(
        "--task",
        choices=["all", "emotion", "face", "eval", "gpu-check"],
        default="all",
        help="What to run (default: all = train both + evaluate)",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Rebuild train/val/test splits (ignore cache)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate (models must exist in src/checkpoints/)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train, skip evaluation",
    )
    args = parser.parse_args()

    if args.task == "gpu-check":
        print_device_summary()
        return

    if args.task == "eval":
        evaluate_model()
        return

    if args.skip_train:
        if not args.skip_eval:
            evaluate_model()
        return

    if args.task == "all":
        emotion_history, face_history = train_model(
            force_rebuild_splits=args.rebuild_splits
        )
        plot_histories(emotion_history, face_history)
        if not args.skip_eval:
            evaluate_model()
    elif args.task == "emotion":
        history = train_emotion_model(force_rebuild_splits=args.rebuild_splits)
        plot_history(history, "Emotion", "emotion_training_history.png")
        if not args.skip_eval:
            evaluate_task("emotion")
    elif args.task == "face":
        history = train_face_model(force_rebuild_splits=args.rebuild_splits)
        plot_history(history, "Face", "face_training_history.png")
        if not args.skip_eval:
            evaluate_task("face")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(1)
