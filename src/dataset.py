"""Dataset indexing, face-identity filtering, and stratified splits."""

import os
import pickle
import logging
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    FACE_DATASET_PATH,
    FACE_CROPS_PATH,
    CROP_FACES,
    FACE_CROP_MARGIN,
    FACE_CROP_SAVE_SIZE,
    EMOTION_DATASET_PATH,
    OPTIONAL_FACE_PATH,
    OPTIONAL_EMOTION_PATH,
    MIN_IMAGES_PER_PERSON,
    MAX_FACE_IDENTITIES,
    MIN_SAMPLES_PER_CLASS,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    SPLITS_PATH,
    CHECKPOINTS_DIR,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def _scan_class_folder(directory: str) -> List[Tuple[str, str]]:
    """Return list of (image_path, class_label) from folder-per-class layout."""
    samples = []
    if not os.path.isdir(directory):
        return samples

    for label in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if _is_image(fname):
                samples.append((os.path.join(class_dir, fname), label))
    return samples


def collect_emotion_samples(
    extra_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    dirs = [EMOTION_DATASET_PATH]
    if extra_dirs:
        dirs.extend(extra_dirs)
    elif os.path.isdir(OPTIONAL_EMOTION_PATH):
        dirs.append(OPTIONAL_EMOTION_PATH)

    samples = []
    for d in dirs:
        found = _scan_class_folder(d)
        logger.info("Emotion: %d images from %s", len(found), d)
        samples.extend(found)
    return samples


def _cropped_face_path(path: str, person: str) -> str:
    """Crop the detected face (cached on disk) so training matches live inference."""
    import cv2
    from src.face_utils import central_face_box, crop_face

    out_path = os.path.join(
        FACE_CROPS_PATH, person, os.path.splitext(os.path.basename(path))[0] + ".jpg"
    )
    if os.path.isfile(out_path):
        return out_path

    image = cv2.imread(path)
    if image is None:
        return path
    box = central_face_box(image)
    if box is None:
        # No detection: assume a roughly centred face (LFW layout)
        h, w = image.shape[:2]
        box = (w // 4, h // 4, w // 2, h // 2)
    face = crop_face(image, box, FACE_CROP_MARGIN)
    face = cv2.resize(face, (FACE_CROP_SAVE_SIZE, FACE_CROP_SAVE_SIZE))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, face)
    return out_path


def collect_face_samples_filtered(
    min_images: int = MIN_IMAGES_PER_PERSON,
    max_identities: int = MAX_FACE_IDENTITIES,
    extra_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Keep only identities with >= min_images photos.
    Cap to max_identities (most images first) so softmax training is feasible.
    """
    dirs = [FACE_DATASET_PATH]
    if extra_dirs:
        dirs.extend(extra_dirs)
    elif os.path.isdir(OPTIONAL_FACE_PATH):
        dirs.append(OPTIONAL_FACE_PATH)

    by_person: defaultdict = defaultdict(list)
    for d in dirs:
        for path, label in _scan_class_folder(d):
            by_person[label].append(path)

    eligible = [(p, paths) for p, paths in by_person.items() if len(paths) >= min_images]
    eligible.sort(key=lambda x: len(x[1]), reverse=True)
    eligible = eligible[:max_identities]

    samples = []
    for person, paths in eligible:
        for path in paths:
            if CROP_FACES:
                path = _cropped_face_path(path, person)
            samples.append((path, person))

    logger.info(
        "Face: %d images, %d identities (min_images=%d, max_identities=%d)",
        len(samples),
        len(eligible),
        min_images,
        max_identities,
    )
    return samples


def _drop_rare_classes(
    samples: List[Tuple[str, str]], min_count: int
) -> List[Tuple[str, str]]:
    from collections import Counter

    counts = Counter(label for _, label in samples)
    kept = [s for s in samples if counts[s[1]] >= min_count]
    dropped = len(counts) - len({s[1] for s in kept})
    if dropped:
        logger.info(
            "Dropped %d classes with fewer than %d samples", dropped, min_count
        )
    return kept


def stratified_split(
    samples: List[Tuple[str, str]],
    val_ratio: float = VALIDATION_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
    min_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> dict:
    samples = _drop_rare_classes(samples, min_per_class)
    if not samples:
        raise ValueError("No samples left after filtering rare classes.")

    paths = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": (train_paths.tolist(), train_labels.tolist()),
        "val": (val_paths.tolist(), val_labels.tolist()),
        "test": (test_paths.tolist(), test_labels.tolist()),
    }


def fit_label_encoder(labels: List[str]) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


def save_splits(splits: dict, path: str = SPLITS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    logger.info("Saved dataset splits to %s", path)


def load_splits(path: str = SPLITS_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_or_load_splits(task: str, force_rebuild: bool = False) -> dict:
    """
    task: 'emotion' | 'face'
    Returns splits dict with train/val/test and fitted encoder in metadata.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    cache_path = os.path.join(
        CHECKPOINTS_DIR, f"{task}_splits.pkl"
    )

    if not force_rebuild and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Loaded cached %s splits from %s", task, cache_path)
        return data

    if task == "emotion":
        samples = collect_emotion_samples()
    elif task == "face":
        samples = collect_face_samples_filtered()
    else:
        raise ValueError(f"Unknown task: {task}")

    if not samples:
        raise RuntimeError(f"No samples found for task '{task}'. Check data/ folders.")

    splits = stratified_split(samples)
    train_labels = splits["train"][1]
    encoder = fit_label_encoder(train_labels)

    data = {"splits": splits, "encoder": encoder, "task": task}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Built and cached %s splits (%d train samples)", task, len(train_labels))
    return data
