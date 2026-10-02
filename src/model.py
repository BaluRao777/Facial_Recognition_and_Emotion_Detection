"""Classifier builders with frozen ImageNet backbone + small trainable head."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import CategoricalCrossentropy

from src.config import BACKBONE, LEARNING_RATE, LABEL_SMOOTHING


def _build_backbone(input_shape=(128, 128, 3)):
    if BACKBONE == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0

        base = EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        from tensorflow.keras.applications import MobileNetV2

        base = MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    base.trainable = False
    return base


def build_classifier(num_classes: int, name: str = "classifier") -> Model:
    """Single-task classifier: backbone -> GAP -> Dense -> softmax."""
    base = _build_backbone()
    x = GlobalAveragePooling2D(name=f"{name}_gap")(base.output)
    x = Dropout(0.4, name=f"{name}_dropout")(x)
    x = Dense(256, activation="relu", name=f"{name}_dense")(x)
    outputs = Dense(
        num_classes, activation="softmax", name="predictions", dtype="float32"
    )(x)
    model = Model(inputs=base.input, outputs=outputs, name=name)
    _compile(model, LEARNING_RATE)
    return model


def _compile(model: Model, lr: float):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )


def unfreeze_backbone(model: Model, num_layers=30, lr: float = 1e-4):
    """Unfreeze last N layers of the backbone (all if num_layers is None)."""
    # The backbone is inlined into the model graph, so its layers are the
    # model's own layers minus the classification head.
    head_prefix = f"{model.name}_"
    backbone_layers = [
        layer
        for layer in model.layers
        if not layer.name.startswith(head_prefix) and layer.name != "predictions"
    ]
    if num_layers is not None:
        backbone_layers = backbone_layers[-num_layers:]
    for layer in backbone_layers:
        layer.trainable = True
    _compile(model, lr)
