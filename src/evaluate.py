"""Evaluate face and emotion models on held-out test splits."""

import os
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from src.config import (
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    EMOTION_MODEL_PATH,
    FACE_MODEL_PATH,
    EMOTION_ENCODER_PATH,
    FACE_ENCODER_PATH,
    IMAGE_SIZE,
    BATCH_SIZE_EMOTION,
    BATCH_SIZE_FACE,
)
from src.data_generator import ImageSequence
from src.dataset import build_or_load_splits
from src.gpu_utils import setup_gpu

logger = logging.getLogger(__name__)


def _load_encoder(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_generator(model, generator):
    y_true = []
    y_pred = []
    for i in range(len(generator)):
        x, y = generator[i]
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y, axis=1))
    return np.array(y_true), np.array(y_pred)


def _safe_classification_report(y_true, y_pred, target_names, zero_division=0):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[target_names[i] for i in labels],
        zero_division=zero_division,
    )


def _plot_confusion(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def evaluate_task(task: str) -> dict:
    setup_gpu(use_mixed_precision=False)

    if task == "emotion":
        model_path = EMOTION_MODEL_PATH
        encoder_path = EMOTION_ENCODER_PATH
        batch_size = BATCH_SIZE_EMOTION
    else:
        model_path = FACE_MODEL_PATH
        encoder_path = FACE_ENCODER_PATH
        batch_size = BATCH_SIZE_FACE

    if not os.path.isfile(model_path):
        logger.error("Model not found: %s. Train first.", model_path)
        return {}

    data = build_or_load_splits(task)
    splits, encoder = data["splits"], data["encoder"]

    if os.path.isfile(encoder_path):
        encoder = _load_encoder(encoder_path)

    test_gen = ImageSequence(
        *splits["test"],
        encoder,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    model = load_model(model_path)
    y_true, y_pred = _predict_generator(model, test_gen)
    acc = accuracy_score(y_true, y_pred)

    report = _safe_classification_report(
        y_true, y_pred, encoder.classes_.tolist()
    )
    label_ids = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"{task}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)

    plot_names = [encoder.classes_[i] for i in label_ids]
    if len(plot_names) <= 30:
        _plot_confusion(
            cm,
            plot_names,
            f"{task.title()} Confusion Matrix (test)",
            os.path.join(RESULTS_DIR, f"{task}_confusion_matrix.png"),
        )
    else:
        logger.info(
            "Skipping confusion matrix plot (%d classes). See %s",
            len(plot_names),
            report_path,
        )

    print(f"\n{'=' * 60}")
    print(f"{task.upper()} — Test accuracy: {acc:.4f}")
    print(f"{'=' * 60}")
    print(report)
    print(f"Report saved to {report_path}")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def evaluate_model():
    """Evaluate both models on test sets."""
    results = {}
    results["emotion"] = evaluate_task("emotion")
    results["face"] = evaluate_task("face")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_model()
