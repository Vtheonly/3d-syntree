"""Fail-closed hardware precision engine — pure FP32 on the supported T4s.

Clean-room V2 implementation of issue #10 MATH-04 (constitution #45 §5
"Exclusive hardware and numerical precision policy"; decision D-V2-005
"no mixed-precision machinery in V2 by construction").

**Why this module exists (the #10 motivation — legacy "Singularity 3").**
On Google Colab / Kaggle Tesla T4 GPUs (compute capability 7.5) the
pre-V2 trainer ran FP16 with a dynamic ``GradScaler``.  When gradients
underflowed, the scale factor collapsed ``2^16 -> 1.84e-40 -> 0.0`` and
the unscaled updates corrupted the model weights to NaN — an
unrecoverable training collapse.  The V2 constitution therefore mandates
**pure FP32** on the supported T4 environments: no FP16, no BF16, no
autocast, no ``GradScaler``, no silent precision fallback (#45 §5.1).
This module is the *enforcement layer*: it probes (or accepts a simulated)
device, applies the strict hardware gate, and returns an auditable
:class:`RuntimeConfig` that the trainer (#26 TRAIN-01) must consume.

**What was deliberately NOT ported from the legacy module.**  The legacy
``syntree/utils/hardware.py`` (pre-reset, forensic history only per
D-V2-001) contained ``resolve_amp_dtype`` (fp16/bf16 selection with
GradScaler-implied loss scaling), auto-TF32-on-Ampere, and
``cudnn.benchmark = True``.  None of it enters V2: the clean-room branch
has no AMP machinery at all (D-V2-005), and the historical
``scale_model_config()`` / ``autotune_batch_size()`` helpers named by the
issue body never entered the clean room — there is nothing to remove,
and ``tests/test_fp32_enforcement.py`` enforces at AST level that no
``GradScaler`` / ``autocast`` / ``torch.amp`` reference ever reappears
anywhere under ``src/syntree``.

**The strict hardware gate (issue-pinned).**

* ``device.type == "cuda"`` and ``compute_capability < 8.0`` (the Tesla T4
  class, cc 7.5): any non-FP32 ``mixed_precision`` request is rejected
  with a ``FAIL-CLOSED`` warning citing the T4 GradScaler-collapse
  history; the configuration executes in **pure FP32** with autocast
  disabled and no ``GradScaler`` in the training loop.
* ``compute_capability >= 8.0`` (A100/H100/...): those devices are
  *outside the supported hardware scope* (#45 §5/§29.2 — only Colab/Kaggle
  Tesla T4s are supported); the configuration still enforces pure FP32
  (the constitution's production mandate is device-independent) and
  records an unsupported-runtime warning so training never silently
  proceeds on non-T4 hardware.
* CPU: the sanctioned development/verification environment (D-V2-008:
  FP32 production + FP64 local verification); mixed-precision requests
  are rejected with the same fail-closed warning.

**TF32 policy (#45 §5.1).**  TF32 is disabled by default and may be
enabled only with an explicit, measured, documented justification
(``tf32_justified=True``) on hardware that actually has TF32 tensor cores
(compute capability >= 8.0).  The supported Tesla T4 (cc 7.5) has no
TF32 units, so ``tf32_enabled`` is structurally ``False`` on every
supported production runtime.  Global ``torch.backends`` flags are only
ever touched for a *real* probed CUDA device — simulated
:class:`DeviceInfo` inputs (used by the tests and by future profiling
issues) never mutate global state.

**Standard logit masks (issue-pinned constants).**  The categorical
policy heads (#24 MODEL-04) mask illegal actions with
``LOGIT_MASK_FILL = -1e4`` while every *legal* logit must stay above
``LOGIT_MASK_LEGAL_ABOVE = -1e3`` — a guaranteed 9,000-nat separation so
a masked action can never survive ``log_softmax`` and a legal action can
never be confused with a masked one.  :func:`apply_logit_mask` enforces
both sides fail-closed.  The *loss-pass target unmasking invariant* is
:func:`unmask_target_actions` — the issue-pinned ``scatter`` that zeroes
the mask entry at the supervised target action, so the taken action's
log-probability is always computable; :func:`validate_target_actions`
additionally rejects supervised targets that were illegal to begin with
(constitution §8 / #45 §9.6: never silently accept an invalid action).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

__all__ = [
    "LOGIT_MASK_FILL",
    "LOGIT_MASK_LEGAL_ABOVE",
    "TESLA_T4_COMPUTE_CAPABILITY",
    "DeviceInfo",
    "RuntimeConfig",
    "HardwareError",
    "probe_device",
    "configure_runtime_environment",
    "apply_logit_mask",
    "unmask_target_actions",
    "validate_target_actions",
]

logger = logging.getLogger(__name__)

#: Fill value for masked (illegal) logits in every categorical head (#10).
LOGIT_MASK_FILL = -1e4

#: Legal logits must stay strictly above this bound (#10), guaranteeing
#: ``LOGIT_MASK_FILL`` is at least 9,000 below any legal logit so masked
#: actions cannot survive log_softmax.
LOGIT_MASK_LEGAL_ABOVE = -1e3

#: The only supported production GPU: NVIDIA Tesla T4, compute capability 7.5
#: (#45 §5 — Google Colab T4 and Kaggle T4).
TESLA_T4_COMPUTE_CAPABILITY = (7, 5)

#: Compute-capability floor for TF32 tensor cores (Ampere and newer).
_TF32_CAPABILITY_FLOOR = (8, 0)

#: Accepted mixed-precision request tokens.  ``"fp32"`` is the only one the
#: gate ever honours; every other token is rejected fail-closed.
_VALID_PRECISION_REQUESTS = ("fp32", "fp16", "bf16", "amp", "mixed")


class HardwareError(ValueError):
    """Fail-closed hardware/precision configuration error."""


@dataclass(frozen=True)
class DeviceInfo:
    """An immutable hardware description of one accelerator device.

    Constructed by :func:`probe_device` for real runtimes, or directly by
    callers that need to *simulate* hardware (the #10 acceptance tests
    simulate a Tesla T4; #43 OPS-03 will reuse this for profiling).
    Simulated infos never cause global ``torch.backends`` mutation.
    """

    device_type: str                     # "cuda" | "cpu"
    name: str                            # e.g. "Tesla T4" | "CPU"
    index: Optional[int] = None          # CUDA device index
    capability: Optional[Tuple[int, int]] = None  # (major, minor) cc
    total_vram_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.device_type not in ("cuda", "cpu"):
            raise HardwareError(
                f"unsupported device type {self.device_type!r} (only 'cuda'/'cpu')"
            )
        if self.device_type == "cuda":
            if self.capability is None:
                raise HardwareError("a CUDA DeviceInfo requires a compute capability")
            if not (isinstance(self.capability, tuple) and len(self.capability) == 2):
                raise HardwareError(
                    f"compute capability must be a (major, minor) tuple, "
                    f"got {self.capability!r}"
                )
        if self.device_type == "cpu" and self.capability is not None:
            raise HardwareError("a CPU DeviceInfo carries no compute capability")

    @property
    def is_tesla_t4(self) -> bool:
        return self.device_type == "cuda" and self.capability == TESLA_T4_COMPUTE_CAPABILITY

    @property
    def capability_string(self) -> Optional[str]:
        if self.capability is None:
            return None
        return f"{self.capability[0]}.{self.capability[1]}"


def probe_device(device: Optional[torch.device] = None) -> DeviceInfo:
    """Probe the *actual* runtime device (never a nominal specification).

    ``None`` resolves to ``cuda`` when available, otherwise ``cpu``.  The
    returned :class:`DeviceInfo` records the device type, name, index,
    compute capability, and total VRAM as reported by the live runtime
    (#45 §28.4: measure, do not infer from device names alone).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cpu":
        return DeviceInfo(device_type="cpu", name="CPU", index=None, capability=None)
    if not torch.cuda.is_available():
        raise HardwareError(
            f"cannot probe CUDA device {device!r}: torch.cuda.is_available() is False"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        device_type="cuda",
        name=props.name,
        index=index,
        capability=(props.major, props.minor),
        total_vram_bytes=int(props.total_memory),
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """The auditable outcome of the strict hardware gate.

    The trainer (#26 TRAIN-01) reads this configuration: when
    ``precision == "fp32"`` (always, in production) it must construct no
    ``torch.autocast`` context and no ``GradScaler`` — the
    ``grad_scaler``/``autocast_enabled`` fields are therefore pinned to
    ``False`` and exist to make the contract machine-checkable.
    """

    device: str
    device_name: str
    compute_capability: Optional[str]
    requested_mixed_precision: str
    precision: str                       # enforced — "fp32" in production
    autocast_enabled: bool               # always False in V2
    grad_scaler: bool                    # always False in V2
    tf32_enabled: bool
    supported_runtime: bool
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    total_vram_bytes: Optional[int] = None
    torch_version: str = ""
    cuda_available: bool = False

    def summary(self) -> str:
        requested_note = (
            f" (requested {self.requested_mixed_precision})"
            if self.requested_mixed_precision != self.precision
            else ""
        )
        tf32_note = (
            f", tf32={'on' if self.tf32_enabled else 'off'}"
            if self.device.startswith("cuda")
            else ""
        )
        scope_note = "SUPPORTED" if self.supported_runtime else "OUTSIDE SUPPORTED T4 SCOPE"
        return (
            f"{self.device_name} (cc {self.compute_capability}, {self.device}) — "
            f"precision={self.precision}{requested_note}{tf32_note}, {scope_note}"
        )


def _reject_precision_request(info: DeviceInfo, requested: str, warnings: List[str]) -> None:
    """Emit the fail-closed rejection warning for a non-FP32 request."""
    if info.device_type == "cuda" and info.capability is not None and info.capability[0] < 8:
        message = (
            f"FAIL-CLOSED: mixed_precision={requested!r} requested on CUDA compute "
            f"capability {info.capability_string} (< 8.0, Tesla T4 class). The pre-V2 "
            f"T4 GradScaler collapse (scale 2^16 -> 1.84e-40 -> 0.0, weights -> NaN) "
            f"is exactly this configuration; the V2 constitution (#45 §5.1) mandates "
            f"pure FP32. Rejecting the request and executing in pure FP32 with "
            f"autocast disabled and no GradScaler in the training loop."
        )
    elif info.device_type == "cuda":
        message = (
            f"FAIL-CLOSED: mixed_precision={requested!r} requested, but the V2 "
            f"production path is pure FP32 on every device (#45 §5.1). Furthermore "
            f"compute capability {info.capability_string} is outside the supported "
            f"Tesla T4 scope (#45 §5). Rejecting the request and executing in pure "
            f"FP32; report the device — do not silently train on unsupported hardware."
        )
    else:
        message = (
            f"FAIL-CLOSED: mixed_precision={requested!r} requested on a CPU runtime. "
            f"V2 is pure FP32 by construction (D-V2-005); mixed precision is not a "
            f"valid configuration anywhere in V2. Rejecting the request and executing "
            f"in pure FP32."
        )
    logger.warning(message)
    warnings.append(message)


def configure_runtime_environment(
    device: Optional[torch.device] = None,
    *,
    mixed_precision: str = "fp32",
    tf32: bool = False,
    tf32_justified: bool = False,
    device_info: Optional[DeviceInfo] = None,
) -> RuntimeConfig:
    """Apply the strict hardware gate and return the runtime configuration.

    Parameters
    ----------
    device:
        Target device (``None`` = auto: CUDA when available, else CPU).
        Ignored when ``device_info`` supplies a simulated device.
    mixed_precision:
        Requested precision.  ``"fp32"`` is the only honoured request; any
        other token (``"fp16"``/``"bf16"``/``"amp"``/``"mixed"``) is
        rejected with a ``FAIL-CLOSED`` warning and the configuration
        executes in pure FP32 (issue #10 acceptance).
    tf32:
        Request TF32 tensor-core matmuls.  Requires an explicit measured
        justification (``tf32_justified``, #45 §5.1) *and* hardware with
        TF32 units (cc >= 8.0 — the supported Tesla T4 has none), else it
        stays disabled with a warning.
    tf32_justified:
        attests that enabling TF32 is explicitly justified, measured, and
        documented for this runtime.
    device_info:
        Explicit (possibly *simulated*) device description.  When given,
        no real probing happens and **no global ``torch.backends`` state
        is mutated** — this is the test/profiling path.

    Returns
    -------
    RuntimeConfig
        The frozen, auditable configuration (``precision`` is always
        ``"fp32"``; ``autocast_enabled``/``grad_scaler`` are always
        ``False``).
    """
    warnings: List[str] = []

    # ---- resolve the device ------------------------------------------------
    cuda_available = torch.cuda.is_available()
    if device_info is not None:
        info = device_info
        probed = False
    else:
        probed = True
        if device is None and cuda_available:
            device = torch.device("cuda")
        try:
            info = probe_device(device)
        except HardwareError:
            if device is not None and torch.device(device).type == "cuda":
                message = (
                    "CUDA device requested but torch.cuda.is_available() is False; "
                    "falling back to the CPU development runtime (no GPU claims made)."
                )
                logger.warning(message)
                warnings.append(message)
                info = DeviceInfo(device_type="cpu", name="CPU")
            else:
                raise

    # ---- validate the request ---------------------------------------------
    requested = str(mixed_precision).lower().strip()
    if requested not in _VALID_PRECISION_REQUESTS:
        raise HardwareError(
            f"mixed_precision={mixed_precision!r} is not a recognised request "
            f"(expected one of {_VALID_PRECISION_REQUESTS}); refuse to guess"
        )

    # ---- the strict gate: production is pure FP32, everywhere ---------------
    if requested != "fp32":
        _reject_precision_request(info, requested, warnings)
    if info.device_type == "cuda" and not info.is_tesla_t4:
        message = (
            f"runtime device {info.name!r} (compute capability "
            f"{info.capability_string}) is outside the supported hardware scope "
            f"(Google Colab / Kaggle Tesla T4 only, #45 §5). The configuration "
            f"still enforces pure FP32; report the device and obtain an authorized "
            f"T4 runtime before GPU training."
        )
        logger.warning(message)
        warnings.append(message)

    # ---- TF32 policy (#45 §5.1) ---------------------------------------------
    tf32_enabled = False
    if tf32:
        if info.device_type != "cuda":
            message = "TF32 requested on a non-CUDA runtime; TF32 stays disabled."
            logger.warning(message)
            warnings.append(message)
        elif not tf32_justified:
            message = (
                "TF32 requested without an explicit justification. TF32 may only be "
                "enabled when justified, measured, and documented (#45 §5.1); "
                "staying disabled (pure FP32 matmuls)."
            )
            logger.warning(message)
            warnings.append(message)
        elif (info.capability or (0, 0)) < _TF32_CAPABILITY_FLOOR:
            message = (
                f"TF32 requested with justification, but compute capability "
                f"{info.capability_string} has no TF32 tensor cores (Ampere+ only); "
                f"TF32 stays disabled. The supported Tesla T4 (cc 7.5) never has TF32."
            )
            logger.warning(message)
            warnings.append(message)
        else:
            tf32_enabled = True

    # Apply global backends ONLY for a real probed CUDA device (simulated
    # DeviceInfo inputs must never mutate global state).
    if probed and info.device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    device_str = f"cuda:{info.index}" if info.device_type == "cuda" else "cpu"
    return RuntimeConfig(
        device=device_str,
        device_name=info.name,
        compute_capability=info.capability_string,
        requested_mixed_precision=requested,
        precision="fp32",
        autocast_enabled=False,
        grad_scaler=False,
        tf32_enabled=tf32_enabled,
        supported_runtime=(info.is_tesla_t4 or info.device_type == "cpu"),
        warnings=tuple(warnings),
        total_vram_bytes=info.total_vram_bytes,
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
    )


# ---------------------------------------------------------------------------
# Standard logit masks (issue #10 pinned constants + enforcement)
# ---------------------------------------------------------------------------


def apply_logit_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked (illegal) action logits with ``LOGIT_MASK_FILL``.

    Parameters
    ----------
    logits:
        ``[B, A]`` (or any shape) raw action logits.
    mask:
        Boolean tensor, ``True = masked/illegal`` (the same convention as
        the decoder's ``key_padding_mask``).  Must match ``logits.shape``.

    Returns
    -------
    torch.Tensor
        A new tensor with illegal slots set to ``LOGIT_MASK_FILL``; legal
        logits are returned unchanged.

    Fail-closed behaviour
    ---------------------
    * non-boolean masks, shape mismatches, and non-finite *legal* logits
      raise :class:`HardwareError`;
    * every **legal** logit must lie strictly above
      ``LOGIT_MASK_LEGAL_ABOVE`` — a poisoned logit that could compete
      with the fill value fails loudly instead of silently;
    * a row with **no** legal action raises (that state must be handled
      by the STOP head before a categorical mask is ever applied, #45
      §8.4).
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise HardwareError("logits and mask must be torch tensors")
    if mask.dtype != torch.bool:
        raise HardwareError(f"the logit mask must be boolean, got dtype {mask.dtype}")
    if logits.shape != mask.shape:
        raise HardwareError(
            f"logit mask shape {tuple(mask.shape)} does not match logits "
            f"{tuple(logits.shape)}"
        )
    if logits.numel() == 0:
        raise HardwareError("apply_logit_mask received empty logits (fail-closed)")
    if logits.dim() == 0:
        raise HardwareError("logits must have at least one dimension")
    legal = ~mask
    if logits.dim() >= 1 and not bool(legal.any(dim=-1).all()):
        raise HardwareError(
            "at least one row has no legal action: the STOP head must terminate "
            "such states before a categorical mask is applied (#45 §8.4)"
        )
    legal_values = logits[legal]
    if not bool(torch.isfinite(legal_values).all()):
        raise HardwareError("legal logits contain non-finite values (fail-closed)")
    if bool((legal_values <= LOGIT_MASK_LEGAL_ABOVE).any()):
        raise HardwareError(
            f"a legal logit is at or below LOGIT_MASK_LEGAL_ABOVE "
            f"({LOGIT_MASK_LEGAL_ABOVE}): poisoned logits would compete with the "
            f"mask fill value {LOGIT_MASK_FILL} (fail-closed)"
        )
    return logits.masked_fill(mask, LOGIT_MASK_FILL)


def unmask_target_actions(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss-pass target-unmasking invariant (issue #10 pinned scatter).

    Implements exactly the pinned semantics::

        mask.scatter_(dim=-1, index=target.unsqueeze(-1),
                      src=torch.zeros_like(mask))

    as a **pure function** (a new boolean mask is returned; the input is
    never mutated — same semantics, safer for auditing).  After the call,
    ``out[b, target[b]]`` is ``False`` (legal/unmasked) for every row, so
    the supervised action's log-probability is always computable in the
    loss pass; every other entry is bit-identical to the input.

    Parameters
    ----------
    mask:
        Boolean ``[B, A]`` mask, ``True = masked/illegal``.
    target:
        Integer ``[B]`` tensor of supervised action indices in ``[0, A)``.
    """
    if not isinstance(mask, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise HardwareError("mask and target must be torch tensors")
    if mask.dtype != torch.bool:
        raise HardwareError(f"the mask must be boolean, got dtype {mask.dtype}")
    if mask.ndim != 2:
        raise HardwareError(f"the mask must be [B, A], got {tuple(mask.shape)}")
    if target.ndim != 1 or target.shape[0] != mask.shape[0]:
        raise HardwareError(
            f"target must be [B] with B={mask.shape[0]}, got {tuple(target.shape)}"
        )
    if target.dtype.is_floating_point:
        raise HardwareError(f"target indices must be integer, got dtype {target.dtype}")
    if mask.shape[1] == 0:
        raise HardwareError("unmask_target_actions received an empty action axis")
    if bool((target < 0).any()) or bool((target >= mask.shape[1]).any()):
        raise HardwareError(
            "target action indices out of bounds for the mask's action axis "
            "(fail-closed)"
        )
    out = mask.clone()
    out.scatter_(
        dim=-1,
        index=target.unsqueeze(-1),
        src=torch.zeros_like(out),
    )
    return out


def validate_target_actions(mask: torch.Tensor, target: torch.Tensor) -> None:
    """Strict supervision check: every target action must already be legal.

    Complements :func:`unmask_target_actions` for callers that refuse to
    *heal* an invalid supervised target (constitution §8; #45 §9.6 "never
    silently accept an unmatched or mismatched trajectory"): raises
    :class:`HardwareError` listing the offending rows when any
    ``mask[b, target[b]]`` is ``True``.
    """
    if mask.dtype != torch.bool:
        raise HardwareError(f"the mask must be boolean, got dtype {mask.dtype}")
    if mask.ndim != 2:
        raise HardwareError(f"the mask must be [B, A], got {tuple(mask.shape)}")
    if target.ndim != 1 or target.shape[0] != mask.shape[0]:
        raise HardwareError(
            f"target must be [B] matching the mask rows, got {tuple(target.shape)}"
        )
    offending = mask.gather(1, target.unsqueeze(-1)).squeeze(-1)
    rows = torch.nonzero(offending, as_tuple=False).flatten().tolist()
    if rows:
        raise HardwareError(
            f"supervised target actions are masked/illegal in rows {rows}: "
            f"refusing to unmask an invalid action (fail-closed, #45 §9.6)"
        )
