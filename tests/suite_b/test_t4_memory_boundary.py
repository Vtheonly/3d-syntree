"""Suite B / 4 — the Tesla T4 memory-boundary profile (issue #33 TEST-02).

The issue-pinned battery: **profile the micro-batch execution (B = 12,
accumulation = 6, N_pocket <= 700) and verify peak VRAM <= 9.5 GB**.

This battery has two deliberately distinct parts:

1. **CPU-runnable profile contracts** (execute everywhere): the frozen
   Profile-A constants (micro-batch 12, gradient accumulation 6 —
   effective batch 72 — pocket cap 700), the config-file/profile
   equality, the full-policy parameter-memory arithmetic (the VRAM
   floor), and the honest absence of GPU telemetry off CUDA (#45 §28.4
   — host RAM and GPU VRAM are separate pools, never inferred from one
   another).
2. **The hardware measurement** (executes ONLY on the authorized Tesla
   T4 runtime, #36/#43): the B=12 x cap-700 full-profile
   forward/backward with peak reserved VRAM asserted against the 9.5 GB
   budget.  On any other environment it skips *loudly* with the precise
   reason — the constitution (#45 §5.2/§29.2) forbids profiling on
   non-T4 devices or reporting a nominal specification as a
   measurement, and a CPU sandbox can never establish GPU behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.engine.trainer import (  # noqa: E402
    ResilientTrainer,
    TrainerProfile,
    gpu_memory_telemetry,
    load_profile,
)

#: The issue-pinned VRAM budget of the Tesla T4 profile (#45 s5.2).
T4_VRAM_BUDGET_GIB = 9.5

#: Profile A constants (issue-pinned, #26/#45 s5.2).
PROFILE_A_MICRO_BATCH = 12
PROFILE_A_ACCUMULATION = 6
PROFILE_A_POCKET_CAP = 700


# ---------------------------------------------------------------------------
# Part 1 — the CPU-runnable profile contracts
# ---------------------------------------------------------------------------


class TestProfileAContracts:
    def test_profile_a_constants_are_issue_pinned(self):
        profile = TrainerProfile()
        assert profile.micro_batch_size == PROFILE_A_MICRO_BATCH
        assert profile.gradient_accumulation_steps == PROFILE_A_ACCUMULATION
        assert profile.effective_batch_size == 72
        assert profile.max_pocket_atoms == PROFILE_A_POCKET_CAP
        assert profile.precision == "fp32"

    def test_frozen_config_file_matches_profile_a(self):
        profile = load_profile("configs/train_colab_t4.json")
        assert profile == TrainerProfile()

    def test_the_vram_budget_is_the_issue_pinned_value(self):
        assert T4_VRAM_BUDGET_GIB == 9.5

    def test_full_policy_parameter_memory_is_the_documented_floor(self):
        """The parameter block is the irreducible VRAM floor (< 50 MB).

        Exact arithmetic on the frozen profile: 10,880,719 parameters at
        FP32 = 43.5 MB (decision D-V2-019) — verified without any GPU.
        """
        from syntree.models.policy import SynTreePolicy

        torch.manual_seed(0)
        policy = SynTreePolicy()
        count = sum(parameter.numel() for parameter in policy.parameters())
        memory_mib = count * 4 / 1024 ** 2
        assert count == 10_880_719
        assert memory_mib < 50.0
        assert memory_mib < T4_VRAM_BUDGET_GIB * 1024  # far below the budget

    def test_gpu_telemetry_is_honestly_absent_off_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("this check asserts the CPU dev-sandbox contract")
        telemetry = gpu_memory_telemetry()
        assert telemetry["gpu_name"] is None
        assert telemetry["gpu_peak_reserved_mib"] is None
        assert telemetry["gpu_total_mib"] is None
        # host fields never leak into the GPU section
        assert not any(key.startswith("host") for key in telemetry)


# ---------------------------------------------------------------------------
# Part 2 — the hardware-gated measurement (authorized Tesla T4 only)
# ---------------------------------------------------------------------------


def _t4_device() -> torch.device:
    """The authorized Tesla T4 device, or the precise refusal reason."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA runtime (CPU sandbox)")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    if not name.strip().upper().startswith("TESLA T4"):
        raise RuntimeError(
            f"non-T4 GPU allocated ({name}) — project policy (#45 s29.2) "
            "forbids profiling or training on it"
        )
    return device


def _profile_a_batch_state(device: torch.device) -> SimpleNamespace:
    """The B=12 x cap-700 Profile-A micro-batch state (synthetic)."""
    gen = torch.Generator().manual_seed(2026)

    def positions(count: int, spread: float) -> torch.Tensor:
        return (
            (torch.rand((count, 3), generator=gen) * 2.0 - 1.0) * spread
        ).to(device=device, dtype=torch.float32)

    pocket_counts = [PROFILE_A_POCKET_CAP] * PROFILE_A_MICRO_BATCH
    ligand_counts = [8] * PROFILE_A_MICRO_BATCH
    pocket_pos, pocket_batch, ligand_pos, ligand_batch = [], [], [], []
    for graph, (n_p, n_l) in enumerate(zip(pocket_counts, ligand_counts)):
        pocket_pos.append(positions(n_p, 14.0))
        pocket_batch.append(torch.full((n_p,), graph, dtype=torch.long, device=device))
        ligand_pos.append(positions(n_l, 2.0))
        ligand_batch.append(torch.full((n_l,), graph, dtype=torch.long, device=device))
    # CPU generator + .to(device): keeps one deterministic RNG stream and
    # mirrors the positions() helper. A CUDA-side torch.randint with a CPU
    # generator raises "Expected a 'cuda' device type for generator" (first
    # surfaced on the authorized T4, S-V2-16).
    z_pocket = torch.randint(1, 9, (sum(pocket_counts),), generator=gen).to(device)
    z_ligand = torch.randint(1, 9, (sum(ligand_counts),), generator=gen).to(device)
    return SimpleNamespace(
        pocket_pos=torch.cat(pocket_pos),
        pocket_z=z_pocket,
        pocket_charge=torch.zeros(sum(pocket_counts), device=device),
        pocket_batch=torch.cat(pocket_batch),
        ligand_pos=torch.cat(ligand_pos),
        ligand_z=z_ligand,
        ligand_charge=torch.zeros(sum(ligand_counts), device=device),
        ligand_batch=torch.cat(ligand_batch),
    )


class TestT4MemoryBoundary:
    def test_profile_a_micro_batch_peak_vram_within_budget(self):
        """B=12, accumulation 6, N_pocket<=700: peak reserved VRAM <= 9.5 GB.

        Executes ONLY on the authorized Tesla T4 runtime (#36/#43); on
        every other environment it fails loudly into the documented skip
        — never a silent pass, never a nominal-specification claim.
        """
        try:
            device = _t4_device()
        except RuntimeError as reason:
            pytest.skip(
                f"T4 memory boundary requires the authorized Tesla T4 runtime "
                f"(#36 gate): {reason}"
            )
        from syntree.models.policy import SynTreePolicy

        torch.manual_seed(0)
        policy = SynTreePolicy().to(device=device, dtype=torch.float32)
        policy.train()
        state = _profile_a_batch_state(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        backbone = policy.backbone(state)
        num_pocket = backbone.num_pocket
        ligand_batch = backbone.node_batch[num_pocket:]
        ligand_pos = backbone.node_pos[num_pocket:]
        # one handle row per ligand atom: the maximal Profile-A pool shape
        handle_atom = torch.arange(
            ligand_batch.numel(), device=device, dtype=torch.long
        )
        handle_graph = ligand_batch
        handle_pos = ligand_pos
        # CPU generator + .to(device) — same pattern as the fixture above
        # (S-V2-16: a CPU generator is illegal for device='cuda' sampling).
        handle_descriptors = torch.randn(
            (handle_atom.numel(), 64),
            generator=torch.Generator().manual_seed(1),
            dtype=torch.float32,
        ).to(device)
        handle_out = policy.handle_head(
            backbone, handle_descriptors, handle_pos, handle_graph, handle_atom
        )
        # The decoder consumes (backbone, handle_output) — the exact
        # composition of SynTreePolicy.forward_state (policy.py §growth-step
        # encoding); fixed when the gated battery first executed on the
        # authorized T4 (S-V2-16).
        decoder_out = policy.decoder.forward_state(backbone, handle_out)
        stop_out = policy.forward_stop(decoder_out.context)
        loss = (
            decoder_out.context.sum()
            + handle_out.log_probs.sum()
            + stop_out.log_probs.sum()
        )
        # #25's CUDA criterion pinned into the gated battery: the full
        # SynTreePolicy forward executes on the authorized T4 with zero
        # NaNs/Infs in every head's outputs (asserted on-device, measured
        # — never a nominal claim; S-V2-16 close-out).
        assert torch.isfinite(decoder_out.context).all().item(), (
            "non-finite decoder context on CUDA (#25 criterion)"
        )
        assert torch.isfinite(handle_out.log_probs).all().item(), (
            "non-finite handle log-probabilities on CUDA (#25 criterion)"
        )
        assert torch.isfinite(stop_out.log_probs).all().item(), (
            "non-finite stop log-probabilities on CUDA (#25 criterion)"
        )
        assert torch.isfinite(loss).item(), "non-finite loss on CUDA (#25 criterion)"
        loss.backward()

        torch.cuda.synchronize(device)
        peak_reserved_mib = gpu_memory_telemetry(device)["gpu_peak_reserved_mib"]
        peak_allocated_mib = gpu_memory_telemetry(device)["gpu_peak_allocated_mib"]
        assert peak_reserved_mib is not None  # measured, not nominal
        peak_reserved_gib = peak_reserved_mib / 1024.0
        print(
            f"\n[T4-profile] B=12 x N_pocket=700: peak reserved "
            f"{peak_reserved_gib:.3f} GiB / allocated {peak_allocated_mib} MiB "
            f"(budget {T4_VRAM_BUDGET_GIB} GiB)"
        )
        assert peak_reserved_gib <= T4_VRAM_BUDGET_GIB, (
            f"peak reserved VRAM {peak_reserved_gib:.3f} GiB exceeds the "
            f"{T4_VRAM_BUDGET_GIB} GiB Profile-A budget — the profile must be "
            "re-engineered, never silently accepted (#45 s5.2)"
        )
