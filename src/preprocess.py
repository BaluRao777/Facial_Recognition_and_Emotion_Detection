"""Legacy helpers — training uses batch generators in data_generator.py."""

import os
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.config import IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(directory):
    """Load all images from a class-folder dataset into memory (prefer generators for training)."""
    images = []
    labels = []
    if not os.path.exists(directory):
        logger.error("Directory not found: %s", directory)
        return np.array([]), np.array([])

    for label_dir in sorted(os.listdir(directory)):
        dir_path = os.path.join(directory, label_dir)
        if not os.path.isdir(dir_path):
            continue
        for image_file in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_file)
            try:
                image = load_img(image_path, target_size=IMAGE_SIZE)
                images.append(img_to_array(image))
                labels.append(label_dir)
            except Exception as e:
                logger.warning("Skip %s: %s", image_path, e)

    logger.info("Loaded %d images from %s", len(images), directory)
    return np.array(images), np.array(labels)
