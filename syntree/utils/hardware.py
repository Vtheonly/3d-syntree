"""Hardware detection and numerical-precision configuration.

Profiles the runtime (RTX 6000 Ada / A100 / H100 vs T4/V100 vs CPU),
applies the TF32 matmul setting from ``system.tf32`` (default: auto -
enabled on compute capability >= 8), selects the mixed-precision dtype
(``bf16`` on Ampere+ which has fp32 dynamic range and needs no loss
scaling; ``fp16`` + GradScaler on Turing-class GPUs), and reports a
structured summary used by the CLI and the notebook.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def resolve_amp_dtype(requested: str, device: "torch.device") -> Optional[torch.dtype]:
    """Resolve the autocast dtype for a requested precision setting.

    Args:
        requested: ``"fp32"`` (or anything else -> no autocast), ``"fp16"``
            or ``"bf16"``.
        device: target device (autocast only applies on CUDA).

    Returns:
        ``torch.bfloat16`` / ``torch.float16`` for the AMP settings on a CUDA
        device, otherwise ``None`` (run fp32). ``bf16`` requires compute
        capability >= 8 (RTX 6000 Ada cc 8.9, A100, H100, L4, ...); on older
        GPUs it falls back to fp16 with a warning so a misconfigured T4
        session cannot silently produce bf16 emulation paths.
    """
    requested = str(requested).lower()
    if requested not in ("fp16", "bf16"):
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if requested == "fp16":
        return torch.float16

    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        major_cc, _ = torch.cuda.get_device_capability(index)
        bf16_ok = major_cc >= 8
    except Exception:  # pragma: no cover - defensive
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_ok:
        return torch.bfloat16
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:  # pragma: no cover
        name = "CUDA device"
    logger.warning(
        "mixed_precision='bf16' requested but %s (compute capability < 8) "
        "lacks native bfloat16; falling back to float16 autocast with "
        "loss scaling.",
        name,
    )
    return torch.float16


def _apply_tf32_setting(tf32_request: Any, device_idx: int) -> bool:
    """Apply the system.tf32 setting for a CUDA device.

    ``None``/absent -> auto (enabled iff compute capability >= 8, matching
    the historical behaviour); ``True`` -> force on (warn if the hardware
    cannot honour it); ``False`` -> force off (full fp32 matmuls, useful
    for bitwise reproducibility checks).
    """
    major_cc, _ = torch.cuda.get_device_capability(device_idx)
    supported = major_cc >= 8

    if tf32_request is None:
        enabled = supported
    else:
        enabled = bool(tf32_request)
        if enabled and not supported:
            logger.warning(
                "system.tf32=true requested but compute capability %d.%d "
                "has no TF32 tensor cores; matmuls stay fp32.",
                major_cc,
                torch.cuda.get_device_capability(device_idx)[1],
            )
            enabled = False

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    return enabled


def configure_runtime_environment(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Detect and optimise the execution environment.

    * CPU runtime -> plain fp32, no TF32.
    * Ampere+ (RTX 6000 Ada / A100 / L4 / H100, cc >= 8) -> TF32 per
      ``system.tf32`` (default on), bf16 autocast.
    * Turing/Volta (T4/V100) -> fp16 autocast with GradScaler.

    Args:
        config: optional system config with ``device`` (``"auto"`` or
            explicit) and ``tf32`` override (true / false / absent = auto).
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
        major_cc, minor_cc = torch.cuda.get_device_capability(device_idx)
        info["compute_capability"] = f"{major_cc}.{minor_cc}"

        # RTX 6000 Ada (cc 8.9, 96 GB) detection: informational, used by the
        # notebook / CLI banner to confirm the target accelerator profile.
        name = info["device_name"].lower()
        info["rtx6000_ada"] = "rtx 6000" in name or (
            major_cc == 8 and minor_cc == 9 and "rtx" in name
        )

        # Honour system.tf32 explicitly (None/absent = auto on Ampere+).
        info["tf32_enabled"] = _apply_tf32_setting(
            config.get("tf32"), device_idx
        )

        # Resolve the autocast dtype from the requested setting with a
        # hardware fallback (bf16 needs cc >= 8).
        requested_mp = str(config.get("mixed_precision", "")).lower()
        if not requested_mp:
            requested_mp = "bf16" if major_cc >= 8 else "fp16"
        amp_dtype = resolve_amp_dtype(requested_mp, torch.device(info["device"]))
        info["requested_mixed_precision"] = requested_mp
        if amp_dtype is not None:
            info["mixed_precision_dtype"] = str(amp_dtype).replace("torch.", "")
            info["precision"] = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
            info["amp_dtype"] = str(amp_dtype)
        else:
            info["precision"] = "fp32"

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


def free_vram_bytes(device: torch.device | None = None) -> int:
    """Currently free VRAM in bytes (0 on CPU)."""
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info(
        torch.cuda.current_device() if device is None else device.index or 0
    )
    return int(free)


__all__ = [
    "configure_runtime_environment",
    "resolve_amp_dtype",
    "set_determinism",
    "gpu_summary",
    "free_vram_bytes",
]
