import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "src", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FACE_DATASET_PATH = os.path.join(DATA_DIR, "face_dataset")
# Detected-face crops of the face dataset (built automatically, safe to delete)
FACE_CROPS_PATH = os.path.join(DATA_DIR, "face_dataset_cropped")
EMOTION_DATASET_PATH = os.path.join(DATA_DIR, "emotion_dataset")

FACE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "face_model.keras")
EMOTION_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "emotion_model.keras")
FACE_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "face_label_encoder.pkl")
EMOTION_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "emotion_label_encoder.pkl")
SPLITS_PATH = os.path.join(CHECKPOINTS_DIR, "dataset_splits.pkl")

# ---------------------------------------------------------------------------
# Image / model
# ---------------------------------------------------------------------------
IMAGE_SIZE = (128, 128)
BACKBONE = "mobilenetv2"  # mobilenetv2 | efficientnetb0

# ---------------------------------------------------------------------------
# Face recognition (filtered subset — viable closed-set classification)
# ---------------------------------------------------------------------------
MIN_IMAGES_PER_PERSON = 10
MAX_FACE_IDENTITIES = 500
MIN_SAMPLES_PER_CLASS = 10  # required for stratified train/val/test

# ---------------------------------------------------------------------------
# Face detection (OpenCV YuNet) — used to crop faces for training and inference
# ---------------------------------------------------------------------------
FACE_DETECTOR_PATH = os.path.join(
    BASE_DIR, "model_weights", "face_detection_yunet_2023mar.onnx"
)
FACE_DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_DETECTOR_SCORE_THRESHOLD = 0.6
CROP_FACES = True  # train the face model on detected-face crops
FACE_CROP_MARGIN = 0.2  # extra border around the box for the identity model
EMOTION_CROP_MARGIN = 0.1  # FER2013 faces are framed slightly wider than the box
FACE_CROP_SAVE_SIZE = 160
UNKNOWN_FACE_THRESHOLD = 0.5  # below this confidence the identity is "Unknown"
INFERENCE_XLA = False  # XLA-compiled inference; measured slower overall here (recompilation stalls)

# ---------------------------------------------------------------------------
# Training (tuned for NVIDIA RTX 6000 Ada / 16GB+ RAM / Ubuntu)
# ---------------------------------------------------------------------------
BATCH_SIZE_EMOTION = 64
BATCH_SIZE_FACE = 32
EPOCHS = 60  # maximum; early stopping usually ends sooner
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
FINE_TUNE_AT_EPOCH = 5
UNFREEZE_BACKBONE_LAYERS = None  # None = fine-tune the whole backbone

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 3
USE_MIXED_PRECISION = True
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True  # sqrt-inverse-frequency weights for rare classes

# ---------------------------------------------------------------------------
# Data augmentation (training split only)
# ---------------------------------------------------------------------------
USE_AUGMENTATION = True
AUGMENT_ROTATION_DEGREES = 15
AUGMENT_MIN_CROP_SCALE = 0.8  # random crop of 80-100% of the image, resized back
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST = 0.2
DATA_LOADER_WORKERS = 8

# ---------------------------------------------------------------------------
# Optional extra datasets (drop folders under data/optional_*)
# ---------------------------------------------------------------------------
OPTIONAL_EMOTION_PATH = os.path.join(DATA_DIR, "optional_emotion")
OPTIONAL_FACE_PATH = os.path.join(DATA_DIR, "optional_face")
