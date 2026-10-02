"""Plotting and helpers."""

import os
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def plot_history(history, title: str, save_name: str):
    if history is None:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    acc_key = "accuracy" if "accuracy" in hist else None
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else None
    if acc_key:
        axes[0].plot(hist[acc_key], label="train")
        if val_acc_key:
            axes[0].plot(hist[val_acc_key], label="val")
        axes[0].set_title(f"{title} — Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

    loss_key = "loss"
    if loss_key in hist:
        axes[1].plot(hist[loss_key], label="train")
        if "val_loss" in hist:
            axes[1].plot(hist["val_loss"], label="val")
        axes[1].set_title(f"{title} — Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved plot: {path}")


def plot_histories(emotion_history, face_history):
    plot_history(emotion_history, "Emotion", "emotion_training_history.png")
    plot_history(face_history, "Face", "face_training_history.png")
