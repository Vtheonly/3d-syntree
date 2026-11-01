"""Fail-closed FP32 hardware precision engine — issue #10 MATH-04 acceptance.

Pins the contracts of :mod:`syntree.utils.hardware`:

* **acceptance 1**: configuring on a *simulated* Tesla T4 (compute
  capability 7.5) forces ``precision == "fp32"`` and disables
  ``GradScaler`` (and autocast) — no real GPU is required or claimed;
* **acceptance 2**: requesting ``mixed_precision="fp16"`` on compute
  capability 7.5 logs a ``FAIL-CLOSED`` warning and the configuration
  still executes in pure FP32 (pinned test-file name
  ``tests/test_fp32_enforcement.py``);
* the strict gate on every other device class (cc < 8 non-T4, cc >= 8
  non-T4, CPU) — pure FP32 everywhere, unsupported-scope reporting per
  #45 §29.2;
* the TF32 policy (#45 §5.1): off by default, requires justification,
  structurally impossible on the supported Tesla T4;
* no global ``torch.backends`` mutation from *simulated* device info;
* the issue-pinned logit-mask constants, :func:`apply_logit_mask`
  enforcement, the pinned target-unmasking ``scatter`` invariant, and the
  strict supervised-target validation;
* the AST-level ban on ``GradScaler`` / ``autocast`` / ``torch.amp``
  anywhere under ``src/syntree`` — the "completely remove GradScaler"
  clause enforced as a regression gate.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest
import torch

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

# ---------------------------------------------------------------------------
# Simulated devices (the CPU dev sandbox must never claim GPU results)
# ---------------------------------------------------------------------------

SIM_T4 = DeviceInfo(
    device_type="cuda",
    name="Tesla T4",
    index=0,
    capability=(7, 5),
    total_vram_bytes=15_735_569_920,
)
SIM_V100 = DeviceInfo(device_type="cuda", name="Tesla V100", capability=(7, 0))
SIM_K80 = DeviceInfo(device_type="cuda", name="Tesla K80", capability=(3, 7))
SIM_A100 = DeviceInfo(device_type="cuda", name="A100", capability=(8, 0))
SIM_H100 = DeviceInfo(device_type="cuda", name="H100", capability=(9, 0))
SIM_CPU = DeviceInfo(device_type="cpu", name="CPU")


# ---------------------------------------------------------------------------
# 1. The strict hardware gate — issue #10 acceptance criteria
# ---------------------------------------------------------------------------


class TestStrictHardwareGate:
    def test_simulated_tesla_t4_forces_fp32_and_disables_grad_scaler(self):
        # Acceptance criterion 1: simulated (or real) Tesla T4 -> pure FP32,
        # no GradScaler, no autocast.
        config = configure_runtime_environment(device_info=SIM_T4)
        assert config.precision == "fp32"
        assert config.grad_scaler is False
        assert config.autocast_enabled is False
        assert config.supported_runtime is True
        assert config.compute_capability == "7.5"
        assert config.device == "cuda:0"
        assert config.device_name == "Tesla T4"

    def test_fp16_request_on_cc75_logs_fail_closed_warning_and_runs_fp32(self, caplog):
        # Acceptance criterion 2 (pinned file name: tests/test_fp32_enforcement.py).
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(
                device_info=SIM_T4, mixed_precision="fp16"
            )
        assert config.precision == "fp32"
        assert config.grad_scaler is False
        assert config.autocast_enabled is False
        assert config.requested_mixed_precision == "fp16"
        warnings_text = " ".join(config.warnings)
        assert "FAIL-CLOSED" in warnings_text
        assert "fp16" in warnings_text
        assert "pure FP32" in warnings_text
        # The warning was actually logged (not merely recorded on the config).
        assert any(
            "FAIL-CLOSED" in record.message and "fp16" in record.message
            for record in caplog.records
        )

    @pytest.mark.parametrize("precision_request", ["fp16", "bf16", "amp", "mixed"])
    def test_every_amp_request_is_rejected_fail_closed_on_t4(self, precision_request, caplog):
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(
                device_info=SIM_T4, mixed_precision=precision_request
            )
        assert config.precision == "fp32"
        assert config.grad_scaler is False
        assert any("FAIL-CLOSED" in record.message for record in caplog.records)

    def test_fp32_request_on_t4_is_silent_and_supported(self, caplog):
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(device_info=SIM_T4)
        assert config.precision == "fp32"
        assert config.warnings == ()
        assert config.supported_runtime is True

    def test_cc_below_8_gate_is_pinned(self, caplog):
        # The issue's literal gate: cc < 8.0 (V100, K80) -> forced FP32 with
        # the T4-class GradScaler-collapse warning.
        for sim in (SIM_V100, SIM_K80):
            with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
                config = configure_runtime_environment(
                    device_info=sim, mixed_precision="fp16"
                )
            assert config.precision == "fp32"
            assert config.grad_scaler is False
            assert any(
                "GradScaler collapse" in record.message for record in caplog.records
            )

    def test_cc_at_or_above_8_still_fp32_but_reports_unsupported_scope(self, caplog):
        # A100/H100 are outside the supported T4 scope (#45 §29.2): pure FP32
        # is still enforced (constitution §5.1 is device-independent), and
        # the configuration reports the unsupported runtime instead of
        # silently training.
        for sim in (SIM_A100, SIM_H100):
            with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
                config = configure_runtime_environment(device_info=sim)
            assert config.precision == "fp32"
            assert config.grad_scaler is False
            assert config.supported_runtime is False
            assert any(
                "outside the supported hardware scope" in record.message
                for record in caplog.records
            )

    def test_cpu_runtime_forces_fp32_on_amp_request(self, caplog):
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(
                device_info=SIM_CPU, mixed_precision="bf16"
            )
        assert config.precision == "fp32"
        assert config.supported_runtime is True  # CPU is the sanctioned dev env
        assert any("FAIL-CLOSED" in record.message for record in caplog.records)

    def test_unrecognised_precision_request_fails_closed(self):
        with pytest.raises(HardwareError):
            configure_runtime_environment(device_info=SIM_T4, mixed_precision="fp64-amp")

    def test_grad_scaler_is_never_enabled_for_any_configuration(self):
        for sim in (SIM_T4, SIM_V100, SIM_A100, SIM_CPU):
            for request in ("fp32", "fp16", "bf16"):
                config = configure_runtime_environment(
                    device_info=sim, mixed_precision=request
                )
                assert config.grad_scaler is False
                assert config.autocast_enabled is False
                assert config.precision == "fp32"

    def test_runtime_config_is_frozen_and_auditable(self):
        config = configure_runtime_environment(device_info=SIM_T4)
        with pytest.raises(Exception):  # frozen dataclass
            config.precision = "fp16"
        assert isinstance(config.warnings, tuple)
        assert config.torch_version == str(torch.__version__)
        assert config.summary().startswith("Tesla T4")

    def test_real_probe_on_cpu_sandbox(self):
        # The dev sandbox truthfully reports a CPU runtime (no GPU claims).
        info = probe_device()
        assert info.device_type == "cpu"
        config = configure_runtime_environment()
        assert config.precision == "fp32"
        assert config.device == "cpu"
        assert config.supported_runtime is True

    def test_cuda_request_without_cuda_warns_and_falls_back(self, caplog):
        if torch.cuda.is_available():
            pytest.skip("a real CUDA runtime exists; the fallback path is moot")
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(device=torch.device("cuda"))
        assert config.device == "cpu"
        assert config.precision == "fp32"
        assert any("falling back" in record.message for record in caplog.records)

    def test_tesla_t4_capability_constant_is_pinned(self):
        assert TESLA_T4_COMPUTE_CAPABILITY == (7, 5)

    def test_device_info_validation(self):
        with pytest.raises(HardwareError):
            DeviceInfo(device_type="tpu", name="TPU")
        with pytest.raises(HardwareError):
            DeviceInfo(device_type="cuda", name="?", capability=None)
        with pytest.raises(HardwareError):
            DeviceInfo(device_type="cpu", name="CPU", capability=(7, 5))
        assert SIM_T4.is_tesla_t4 is True
        assert SIM_A100.is_tesla_t4 is False
        assert SIM_CPU.capability_string is None


# ---------------------------------------------------------------------------
# 2. TF32 policy (#45 §5.1)
# ---------------------------------------------------------------------------


class TestTF32Policy:
    def test_tf32_disabled_by_default(self):
        config = configure_runtime_environment(device_info=SIM_T4)
        assert config.tf32_enabled is False

    def test_tf32_requires_explicit_justification(self, caplog):
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(device_info=SIM_A100, tf32=True)
        assert config.tf32_enabled is False
        assert any("justification" in record.message for record in caplog.records)

    def test_tf32_structurally_impossible_on_supported_t4(self, caplog):
        # The Tesla T4 (cc 7.5) has no TF32 tensor cores at all.
        with caplog.at_level(logging.WARNING, logger="syntree.utils.hardware"):
            config = configure_runtime_environment(
                device_info=SIM_T4, tf32=True, tf32_justified=True
            )
        assert config.tf32_enabled is False
        assert any("no TF32 tensor cores" in record.message for record in caplog.records)

    def test_tf32_on_ampere_with_justification(self):
        config = configure_runtime_environment(
            device_info=SIM_A100, tf32=True, tf32_justified=True
        )
        assert config.tf32_enabled is True

    def test_simulated_device_info_never_mutates_global_backends(self):
        matmul_before = torch.backends.cuda.matmul.allow_tf32
        cudnn_before = torch.backends.cudnn.allow_tf32
        configure_runtime_environment(
            device_info=SIM_A100, tf32=True, tf32_justified=True
        )
        configure_runtime_environment(device_info=SIM_T4, mixed_precision="fp16")
        assert torch.backends.cuda.matmul.allow_tf32 == matmul_before
        assert torch.backends.cudnn.allow_tf32 == cudnn_before


# ---------------------------------------------------------------------------
# 3. Standard logit masks (issue-pinned constants + enforcement)
# ---------------------------------------------------------------------------


class TestLogitMasks:
    def test_constants_are_pinned(self):
        assert LOGIT_MASK_FILL == -1e4
        assert LOGIT_MASK_LEGAL_ABOVE == -1e3
        assert LOGIT_MASK_FILL < LOGIT_MASK_LEGAL_ABOVE

    def test_apply_logit_mask_fills_illegal_slots(self):
        logits = torch.tensor([[0.5, -0.2, 1.0], [2.0, 0.0, -0.5]])
        mask = torch.tensor([[False, True, False], [True, False, False]])
        out = apply_logit_mask(logits, mask)
        assert out[0, 1] == LOGIT_MASK_FILL
        assert out[1, 0] == LOGIT_MASK_FILL
        assert out[0, 0] == 0.5 and out[0, 2] == 1.0
        assert out[1, 1] == 0.0 and out[1, 2] == -0.5
        # the input is untouched (pure function)
        assert logits[0, 1] == -0.2

    def test_masked_actions_never_survive_softmax(self):
        logits = torch.tensor([[LOGIT_MASK_LEGAL_ABOVE + 1.0, 3.0]])
        mask = torch.tensor([[True, False]])
        probs = torch.softmax(apply_logit_mask(logits, mask), dim=-1)
        assert probs[0, 0] < 1e-5
        assert probs[0, 1] > 1 - 1e-5

    def test_apply_logit_mask_fail_closed(self):
        logits = torch.tensor([[1.0, 2.0]])
        with pytest.raises(HardwareError):
            apply_logit_mask(logits, torch.tensor([[0, 1]], dtype=torch.int64))
        with pytest.raises(HardwareError):
            apply_logit_mask(logits, torch.tensor([[False, True], [False, False]]))
        with pytest.raises(HardwareError):
            apply_logit_mask(torch.tensor([[float("nan"), 2.0]]), torch.tensor([[False, True]]))
        with pytest.raises(HardwareError):
            # a legal logit at the boundary competes with the fill value
            apply_logit_mask(
                torch.tensor([[LOGIT_MASK_LEGAL_ABOVE, 2.0]]),
                torch.tensor([[False, True]]),
            )
        with pytest.raises(HardwareError):
            # no legal action in the row: STOP must handle such states
            apply_logit_mask(torch.tensor([[1.0, 2.0]]), torch.tensor([[True, True]]))
        with pytest.raises(HardwareError):
            apply_logit_mask(torch.zeros(0, 2), torch.zeros(0, 2, dtype=torch.bool))

    def test_logit_mask_works_with_policy_scale_logits(self):
        # Typical #24-scale logits (small values around 0) pass validation.
        logits = torch.randn(8, 128)
        mask = torch.rand(8, 128) < 0.3
        mask[:, 0] = False  # every row keeps at least one legal action
        out = apply_logit_mask(logits, mask)
        assert torch.equal(out[mask], torch.full_like(out[mask], LOGIT_MASK_FILL))
        assert torch.equal(out[~mask], logits[~mask])


class TestTargetUnmaskingInvariant:
    def test_pinned_scatter_semantics(self):
        # The exact issue-pinned snippet, executed in-place on a clone,
        # must produce the identical result as the pure utility.
        torch.manual_seed(0)
        mask = torch.rand(6, 10) < 0.5
        pristine = mask.clone()
        target = torch.randint(0, 10, (6,))
        pinned = mask.clone()
        pinned.scatter_(
            dim=-1,
            index=target.unsqueeze(-1),
            src=torch.zeros_like(pinned),
        )
        out = unmask_target_actions(mask, target)
        assert torch.equal(out, pinned)
        # the invariant: the taken action is never masked afterwards
        assert not out.gather(1, target.unsqueeze(-1)).any()
        # the input mask is never mutated (pure function)
        assert torch.equal(mask, pristine)

    def test_unmasking_heals_accidentally_masked_targets(self):
        mask = torch.ones(3, 5, dtype=torch.bool)  # everything masked
        mask[0, 1] = False
        target = torch.tensor([1, 2, 3])
        out = unmask_target_actions(mask, target)
        assert not out.gather(1, target.unsqueeze(-1)).any()
        # non-target entries unchanged
        assert out[0, 0] and out[1, 0] and out[2, 4]

    def test_unmask_target_actions_fail_closed(self):
        mask = torch.zeros(2, 4, dtype=torch.bool)
        with pytest.raises(HardwareError):
            unmask_target_actions(mask, torch.tensor([[0, 1]]))
        with pytest.raises(HardwareError):
            unmask_target_actions(mask, torch.tensor([0, 5]))
        with pytest.raises(HardwareError):
            unmask_target_actions(mask, torch.tensor([-1, 0]))
        with pytest.raises(HardwareError):
            unmask_target_actions(mask, torch.tensor([0.0, 1.0]))
        with pytest.raises(HardwareError):
            unmask_target_actions(
                torch.zeros(2, 4, dtype=torch.int64), torch.tensor([0, 1])
            )
        with pytest.raises(HardwareError):
            unmask_target_actions(torch.zeros(2, 0, dtype=torch.bool), torch.tensor([0, 1]))

    def test_validate_target_actions_strict_mode(self):
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[1, 2] = True
        validate_target_actions(mask, torch.tensor([0, 0]))  # both legal
        with pytest.raises(HardwareError, match="rows \\[1\\]"):
            validate_target_actions(mask, torch.tensor([0, 2]))

    def test_unmask_then_softmax_loss_is_computable(self):
        # The end-to-end purpose of the invariant: the supervised action's
        # log-probability survives the masked log_softmax.
        logits = torch.tensor([[0.3, -0.4, 1.2, 0.1]])
        mask = torch.tensor([[False, True, True, True]])  # only action 0 legal
        target = torch.tensor([0])
        healed = unmask_target_actions(mask, target)
        masked_logits = apply_logit_mask(logits, healed)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        assert torch.isfinite(log_probs[0, 0])
        assert log_probs[0, 0] == pytest.approx(0.0, abs=1e-6)  # sole legal action


# ---------------------------------------------------------------------------
# 4. "Completely remove GradScaler" — AST enforcement across src/syntree
# ---------------------------------------------------------------------------


class TestNoAMPMachineryAnywhere:
    def test_no_grad_scaler_autocast_or_amp_in_src(self):
        # Issue #10: GradScaler is completely removed from the training
        # loop; the clean-room has none anywhere, and this test keeps it
        # that way (constitution §5.1 / D-V2-005).
        package_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "syntree"
        files = sorted(package_root.rglob("*.py"))
        assert files, "the syntree package must exist"
        offenders = []
        for path in files:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "attr", None) or getattr(func, "id", None)
                    if name in ("GradScaler", "autocast"):
                        offenders.append(f"{path.name}:{node.lineno} {name}()")
                if isinstance(node, ast.Attribute):
                    if (
                        getattr(node.value, "id", None) == "torch"
                        and node.attr == "amp"
                    ):
                        offenders.append(f"{path.name}:{node.lineno} torch.amp")
        assert not offenders, f"banned AMP machinery found: {offenders}"

    def test_no_fp16_bf16_dtype_construction_in_src(self):
        package_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "syntree"
        offenders = []
        for path in sorted(package_root.rglob("*.py")):
            if path.name == "hardware.py":
                continue  # owns the request-token strings, no dtype usage
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if node.attr in ("float16", "bfloat16", "half"):
                        offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
        assert not offenders, f"banned reduced-precision dtype usage: {offenders}"

    def test_legacy_helpers_are_absent(self):
        # scale_model_config()/autotune_batch_size() must never reappear.
        import syntree.utils.hardware as hw

        assert not hasattr(hw, "scale_model_config")
        assert not hasattr(hw, "autotune_batch_size")
        assert not hasattr(hw, "resolve_amp_dtype")
