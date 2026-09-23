"""Hardware detection and numerical-precision configuration.

Profiles the runtime (A100/H100/L4 vs T4/V100 vs CPU), enables TF32
matmuls on Ampere+, selects the mixed-precision dtype, and reports a
structured summary used by the CLI and the notebook.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def configure_runtime_environment(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Detect and optimise the execution environment.

    * CPU runtime -> plain fp32, no TF32.
    * Ampere+ (A100/L4/H100, compute capability >= 8) -> TF32 on, bf16 autocast.
    * Turing/Volta (T4/V100) -> fp16 autocast with GradScaler.

    Args:
        config: optional system config with ``device`` (``"auto"`` or
            explicit) and ``tf32`` override.
    """
    config = config or {}
    requested_device = str(config.get("device", "auto"))

    info: Dict[str, Any] = {
        "device": "cpu",
        "device_name": "CPU",
        "precision": "fp32",
        "tf32_enabled": False,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "mixed_precision_dtype": None,
        "num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        if requested_device == "auto":
            device_idx = 0
        else:
            try:
                device_idx = int(requested_device.split(":")[1])
            except (IndexError, ValueError):
                device_idx = 0
        info["device"] = f"cuda:{device_idx}"
        info["device_name"] = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
        major_cc, _ = torch.cuda.get_device_capability(device_idx)

        if major_cc >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            info["tf32_enabled"] = True
            info["precision"] = "bf16"
            info["mixed_precision_dtype"] = "bfloat16"
        else:
            info["precision"] = "fp16"
            info["mixed_precision_dtype"] = "float16"

        torch.backends.cudnn.benchmark = True
        free, total = torch.cuda.mem_get_info(device_idx)
        info["gpu_free_memory_gb"] = round(free / 1024 ** 3, 2)
    elif requested_device.startswith("cuda"):
        logger.warning(
            "CUDA device requested but torch.cuda.is_available() is False; "
            "falling back to CPU."
        )

    if config.get("num_workers") is not None:
        info["num_workers"] = int(config["num_workers"])

    return info


def set_determinism(seed: int, deterministic: bool = False) -> None:
    """Configure global determinism knobs."""
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def gpu_summary() -> str:
    """One-line human-readable GPU summary for logs."""
    if not torch.cuda.is_available():
        return "CPU-only runtime"
    return (
        f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
    )


__all__ = ["configure_runtime_environment", "set_determinism", "gpu_summary"]
