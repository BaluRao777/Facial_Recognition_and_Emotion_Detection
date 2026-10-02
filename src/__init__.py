from src.gpu_utils import ensure_cuda_library_path

# Runs before any module in this package imports TensorFlow.
ensure_cuda_library_path()
