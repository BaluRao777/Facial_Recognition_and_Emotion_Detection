"""Train face and emotion models separately."""

import os
import pickle
import logging

import numpy as np

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import (
    CHECKPOINTS_DIR,
    BATCH_SIZE_EMOTION,
    BATCH_SIZE_FACE,
    EPOCHS,
    FINE_TUNE_AT_EPOCH,
    UNFREEZE_BACKBONE_LAYERS,
    FINE_TUNE_LR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    USE_MIXED_PRECISION,
    EMOTION_MODEL_PATH,
    FACE_MODEL_PATH,
    EMOTION_ENCODER_PATH,
    FACE_ENCODER_PATH,
    IMAGE_SIZE,
    USE_AUGMENTATION,
    USE_CLASS_WEIGHTS,
    DATA_LOADER_WORKERS,
)
from src.data_generator import ImageSequence
from src.dataset import build_or_load_splits
from src.gpu_utils import setup_gpu
from src.model import build_classifier, unfreeze_backbone

logger = logging.getLogger(__name__)


def _callbacks(model_path: str, log_name: str, model=None):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    cbs = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="max",
        ),
        ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
        ),
        TensorBoard(
            log_dir=os.path.join(CHECKPOINTS_DIR, "logs", log_name),
            histogram_freq=0,
        ),
    ]
    return cbs


def _fit(model, train_gen, val_gen, callbacks):
    """Train the head with a frozen backbone, then fine-tune the backbone."""
    head_epochs = min(FINE_TUNE_AT_EPOCH, EPOCHS)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=head_epochs,
        callbacks=callbacks,
    )
    if head_epochs >= EPOCHS:
        return history

    unfreeze_backbone(model, UNFREEZE_BACKBONE_LAYERS, FINE_TUNE_LR)
    print(f"\n>>> Fine-tuning backbone (epoch {head_epochs}), lr={FINE_TUNE_LR}\n")
    fine_tune_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=head_epochs,
        callbacks=callbacks,
    )
    for key, values in fine_tune_history.history.items():
        fine_tune_history.history[key] = history.history.get(key, []) + values
    return fine_tune_history


def _save_encoder(encoder, path: str):
    with open(path, "wb") as f:
        pickle.dump(encoder, f)


def _class_weights(labels, encoder):
    """Sqrt-inverse-frequency weights (mean 1 over samples) so rare classes are not ignored."""
    if not USE_CLASS_WEIGHTS:
        return None
    counts = np.bincount(encoder.transform(labels), minlength=len(encoder.classes_))
    weights = np.sqrt(counts.sum() / (len(counts) * np.maximum(counts, 1)))
    return weights / np.average(weights, weights=counts)


def _train(task: str, batch_size: int, model_path: str, encoder_path: str,
           force_rebuild_splits: bool = False):
    setup_gpu(USE_MIXED_PRECISION)
    data = build_or_load_splits(task, force_rebuild=force_rebuild_splits)
    splits, encoder = data["splits"], data["encoder"]

    train_gen = ImageSequence(
        *splits["train"],
        encoder,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        augment=USE_AUGMENTATION,
        class_weights=_class_weights(splits["train"][1], encoder),
        workers=DATA_LOADER_WORKERS,
    )
    val_gen = ImageSequence(
        *splits["val"], encoder, batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=False
    )

    num_classes = len(encoder.classes_)
    logger.info("Training %s model for %d classes", task, num_classes)
    model = build_classifier(num_classes, name=f"{task}_classifier")
    model.summary()

    # Saved first so a checkpoint from an interrupted run is still usable.
    _save_encoder(encoder, encoder_path)
    history = _fit(model, train_gen, val_gen, _callbacks(model_path, task))

    logger.info("%s model saved to %s", task.title(), model_path)
    return history


def train_emotion_model(force_rebuild_splits: bool = False):
    return _train(
        "emotion", BATCH_SIZE_EMOTION, EMOTION_MODEL_PATH, EMOTION_ENCODER_PATH,
        force_rebuild_splits,
    )


def train_face_model(force_rebuild_splits: bool = False):
    return _train(
        "face", BATCH_SIZE_FACE, FACE_MODEL_PATH, FACE_ENCODER_PATH,
        force_rebuild_splits,
    )


def train_model(force_rebuild_splits: bool = False):
    """Train both models (used by main.py). Returns (emotion_history, face_history)."""
    print("=" * 60)
    print("Training EMOTION model")
    print("=" * 60)
    emotion_history = train_emotion_model(force_rebuild_splits)

    print("=" * 60)
    print("Training FACE model")
    print("=" * 60)
    face_history = train_face_model(force_rebuild_splits)

    return emotion_history, face_history
