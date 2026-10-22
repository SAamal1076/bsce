"""Device / dtype helpers.

Default device is CUDA when available, otherwise CPU (as requested: tests and
fixes run on CPU here, full training runs on the user's GPU).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (must be set before CUDA starts)
import torch  # noqa: E402

REAL = {"float32": torch.float32, "float64": torch.float64}
CPLX = {"float32": torch.complex64, "float64": torch.complex128}


def deterministic() -> None:
    """Same inputs + same hardware/software -> bit-identical training. Operations without a deterministic GPU
    kernel only warn (warn_only) instead of stopping the run."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:                                    # very old torch
        torch.use_deterministic_algorithms(True)


def full_precision_cuda() -> None:
    """Never let float32 matmuls run in TF32 (10-bit mantissa) on Ampere+ GPUs: TF32 would
    make float32 results differ by ~1e-3 instead of ~1e-7 from the CPU / original code."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def get_device(name: str = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise torch.device(name)."""
    if name in (None, "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[bsce] requested device '{name}' but CUDA is not available -> using CPU")
        return torch.device("cpu")
    if name.startswith("cuda"):
        full_precision_cuda()
    deterministic()
    return torch.device(name)


def dtypes(precision: str = "float32"):
    """Return (real_dtype, complex_dtype) for 'float32' or 'float64'."""
    if precision not in REAL:
        raise ValueError(f"precision must be one of {list(REAL)}")
    return REAL[precision], CPLX[precision]


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"
