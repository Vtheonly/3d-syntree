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


# ---------------------------------------------------------------------------
# GPU auto-scaling
# ---------------------------------------------------------------------------
# (min free VRAM [GB], model overrides). Tiers respect the architecture
# constraints hidden_dim == synthon_embedding_dim and hidden_dim % heads == 0.
MODEL_TIERS = (
    (12.0, {"hidden_dim": 256, "num_equivariant_layers": 8,
            "num_attention_heads": 8, "synthon_embedding_dim": 256}),
    (8.0,  {"hidden_dim": 192, "num_equivariant_layers": 6,
            "num_attention_heads": 6, "synthon_embedding_dim": 192}),
    (4.0,  {"hidden_dim": 160, "num_equivariant_layers": 5,
            "num_attention_heads": 4, "synthon_embedding_dim": 160}),
    (0.0,  {"hidden_dim": 128, "num_equivariant_layers": 4,
            "num_attention_heads": 4, "synthon_embedding_dim": 128}),
)


def scale_model_config(model_cfg: Dict[str, Any], free_gb: float | None = None) -> Dict[str, Any]:
    """Upscale the model architecture for the detected GPU tier.

    A pure function (the input dict is never mutated) that maps the amount of
    free VRAM to a larger ``hidden_dim`` / layer / head count so that big
    GPUs (T4 16 GB, A100 40 GB, ...) do not idle at ~1 GB of usage. Tier
    boundaries: >=12 GB free -> 256/8/8, >=8 -> 192/6/6, >=4 -> 160/5/4,
    otherwise the portable 128/4/4 default.

    Args:
        model_cfg: the ``config["model"]`` block.
        free_gb: free VRAM in GiB; when None, queried from the active CUDA
            device (CPU runtimes fall back to the smallest tier).

    Returns:
        A new model config dict with the tier overrides applied. Keys the
        caller set explicitly that are NOT part of a tier (e.g.
        ``cutoff_radius``, ``dropout``) are preserved unchanged.
    """
    if free_gb is None:
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / 1024 ** 3
        else:
            free_gb = 0.0

    for min_gb, overrides in MODEL_TIERS:
        if free_gb >= min_gb:
            scaled = dict(model_cfg)
            scaled.update(overrides)
            return scaled
    return dict(model_cfg)  # pragma: no cover - unreachable (0.0 tier)


def free_vram_bytes(device: torch.device | None = None) -> int:
    """Currently free VRAM in bytes (0 on CPU)."""
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info(
        torch.cuda.current_device() if device is None else device.index or 0
    )
    return int(free)


def autotune_batch_size(
    probe_fn,
    dataset_size: int,
    start_batch: int = 16,
    max_batch: int = 8192,
    target_bytes: int | None = None,
    target_fraction: float = 0.85,
    device: torch.device | None = None,
) -> "Dict[str, float]":
    """Empirically find the largest batch size that fills the GPU to a target.

    Runs ``probe_fn(batch_size)`` (one forward+backward pass on a real batch,
    as supplied by the trainer) for exponentially growing batch sizes,
    measuring the peak reserved CUDA memory of each probe with
    ``torch.cuda.max_memory_reserved``. After the first overshoot the search
    binary-refines between the last good and first bad candidate, so the
    final batch typically lands within a few percent of the target.

    The caller is responsible for making ``probe_fn`` side-effect free with
    respect to training state (no optimizer step, RNG snapshots around the
    whole call).

    Args:
        probe_fn: callable ``(batch_size) -> None`` executing one
            forward+backward on device.
        dataset_size: hard cap (a batch larger than the dataset is pointless).
        start_batch: smallest candidate (also the fallback on failure).
        max_batch: configuration cap for the search space.
        target_bytes: absolute VRAM budget; defaults to
            ``free VRAM * target_fraction``.
        target_fraction: fraction of currently free VRAM to fill.
        device: cuda device (probing is skipped on CPU).

    Returns:
        Dict with ``batch_size`` and ``peak_bytes`` (0 on CPU / failure,
        meaning "keep the configured batch").
    """
    result = {"batch_size": float(max(1, start_batch)), "peak_bytes": 0.0}
    if not torch.cuda.is_available():
        return result
    if device is not None and device.type != "cuda":
        return result

    if target_bytes is None:
        target_bytes = int(free_vram_bytes() * float(target_fraction))
    if target_bytes <= 0:
        return result

    hard_cap = max(1, min(int(max_batch), int(dataset_size)))

    def _measure(candidate: int) -> int:
        """Peak reserved bytes for one probe; -1 on OOM."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            probe_fn(candidate)
            peak = int(torch.cuda.max_memory_reserved())
        except torch.cuda.OutOfMemoryError:
            peak = -1
        finally:
            torch.cuda.empty_cache()
        return peak

    def _fits(candidate: int) -> "tuple[bool, int]":
        peak = _measure(candidate)
        return (peak >= 0 and peak <= target_bytes), max(peak, 0)

    # Phase 1: exponential doubling.
    best, best_peak = max(1, int(start_batch)), 0
    lo, hi = best, None  # search bracket once doubling overshoots
    candidate = best
    while candidate <= hard_cap:
        fits, peak = _fits(candidate)
        if fits:
            best, best_peak = candidate, peak
            lo = candidate
            candidate *= 2
        else:
            hi = candidate
            break

    # Phase 2: binary refinement inside (lo, hi).
    if hi is not None and hi - lo > max(1, lo // 16):
        left, right = lo, hi
        while right - left > max(1, left // 32):
            mid = (left + right) // 2
            if mid <= left or mid >= right:
                break
            fits, peak = _fits(mid)
            if fits:
                left, best, best_peak = mid, mid, peak
            else:
                right = mid

    result["batch_size"] = float(best)
    result["peak_bytes"] = float(best_peak)
    return result


__all__ = [
    "configure_runtime_environment",
    "set_determinism",
    "gpu_summary",
    "scale_model_config",
    "autotune_batch_size",
    "free_vram_bytes",
    "MODEL_TIERS",
]
