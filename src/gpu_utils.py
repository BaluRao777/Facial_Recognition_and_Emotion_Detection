"""Configure TensorFlow for NVIDIA GPU (CUDA) on Ubuntu/Linux."""

import glob
import importlib.util
import os
import logging
import sys

logger = logging.getLogger(__name__)


def ensure_cuda_library_path() -> None:
    """
    Put the pip-installed CUDA libraries (tensorflow[and-cuda]) on LD_LIBRARY_PATH.

    The dynamic loader only reads LD_LIBRARY_PATH at process start, so when the
    directories are missing the interpreter is re-executed once with them set.
    Must run before TensorFlow is imported.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return

    lib_dirs = []
    for location in spec.submodule_search_locations:
        lib_dirs.extend(sorted(glob.glob(os.path.join(location, "*", "lib"))))

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    missing = [d for d in lib_dirs if d not in current]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(missing + current)
    os.execv(sys.executable, sys.orig_argv)


def setup_gpu(use_mixed_precision: bool = True) -> list:
    """
    Enable GPU memory growth and optional mixed precision.
    Returns list of detected physical GPU names (empty if CPU-only).
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        logger.info("Found %d GPU(s): %s", len(gpus), [g.name for g in gpus])
    else:
        logger.warning(
            "No GPU detected. Training will use CPU. "
            "On Ubuntu install: pip install tensorflow[and-cuda] and NVIDIA drivers."
        )

    if use_mixed_precision and gpus:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled (mixed_float16).")

    return [g.name for g in gpus]


def print_device_summary():
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))
