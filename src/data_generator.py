"""Keras Sequence — loads images in batches (no full-RAM preload)."""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder

from src.config import (
    AUGMENT_ROTATION_DEGREES,
    AUGMENT_MIN_CROP_SCALE,
    AUGMENT_BRIGHTNESS,
    AUGMENT_CONTRAST,
)
from src.face_utils import scale_pixels


def augment_image(img: Image.Image) -> Image.Image:
    """Random flip, rotation, crop-zoom/shift, brightness and contrast."""
    size = img.size

    if np.random.rand() < 0.5:
        img = ImageOps.mirror(img)

    angle = np.random.uniform(-AUGMENT_ROTATION_DEGREES, AUGMENT_ROTATION_DEGREES)
    img = img.rotate(angle, resample=Image.BILINEAR)

    scale = np.random.uniform(AUGMENT_MIN_CROP_SCALE, 1.0)
    crop_w, crop_h = int(size[0] * scale), int(size[1] * scale)
    left = np.random.randint(0, size[0] - crop_w + 1)
    top = np.random.randint(0, size[1] - crop_h + 1)
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize(
        size, Image.BILINEAR
    )

    img = ImageEnhance.Brightness(img).enhance(
        np.random.uniform(1 - AUGMENT_BRIGHTNESS, 1 + AUGMENT_BRIGHTNESS)
    )
    img = ImageEnhance.Contrast(img).enhance(
        np.random.uniform(1 - AUGMENT_CONTRAST, 1 + AUGMENT_CONTRAST)
    )
    return img


class ImageSequence(Sequence):
    def __init__(
        self,
        paths,
        labels,
        label_encoder: LabelEncoder,
        batch_size: int = 32,
        image_size=(128, 128),
        shuffle: bool = True,
        augment: bool = False,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.paths = np.asarray(paths)
        self.labels = np.asarray(labels)
        self.encoder = label_encoder
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        # Optional per-class weights (indexed by encoded label); adds sample weights
        self.class_weights = class_weights
        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_paths = self.paths[batch_idx]
        batch_labels = self.labels[batch_idx]

        x = np.zeros((len(batch_idx), *self.image_size, 3), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            img = load_img(path, target_size=self.image_size)
            if self.augment:
                img = augment_image(img)
            x[i] = scale_pixels(img_to_array(img))

        y_encoded = self.encoder.transform(batch_labels)
        y = to_categorical(y_encoded, num_classes=len(self.encoder.classes_))
        if self.class_weights is not None:
            return x, y, self.class_weights[y_encoded].astype(np.float32)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
