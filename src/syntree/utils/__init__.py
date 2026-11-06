"""V2 utility layer: the fail-closed hardware precision engine (#10).

Public surface (issue #10 MATH-04; constitution #45 §5; decision
D-V2-005):

* :mod:`syntree.utils.hardware` — device probing, the strict pure-FP32
  hardware gate (``configure_runtime_environment``), the standard logit
  mask constants (``LOGIT_MASK_FILL`` / ``LOGIT_MASK_LEGAL_ABOVE``), the
  pinned loss-pass target-unmasking invariant, and strict supervised
  target validation.
"""

from __future__ import annotations

from syntree.utils.hardware import (
    LOGIT_MASK_FILL,
    LOGIT_MASK_LEGAL_ABOVE,
    TESLA_T4_COMPUTE_CAPABILITY,
    DeviceInfo,
    HardwareError,
    RuntimeConfig,
    apply_logit_mask,
    configure_runtime_environment,
    probe_device,
    unmask_target_actions,
    validate_target_actions,
)

__all__ = [
    "LOGIT_MASK_FILL",
    "LOGIT_MASK_LEGAL_ABOVE",
    "TESLA_T4_COMPUTE_CAPABILITY",
    "DeviceInfo",
    "HardwareError",
    "RuntimeConfig",
    "apply_logit_mask",
    "configure_runtime_environment",
    "probe_device",
    "unmask_target_actions",
    "validate_target_actions",
]
