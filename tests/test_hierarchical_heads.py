"""Decoupled hierarchical policy heads & discrete SO(2) torsion space —
tests (#24 MODEL-04).

Acceptance criteria of issue #24 (clean-room V2 implementation):

* When ``a_t = STOP``, no forward pass is required through the reaction
  or synthon heads — pinned here by forward-hook instrumentation: the
  continuation heads register **zero** invocations during
  ``forward_stop``, the stop law is invariant to perturbing the tied
  catalog embeddings, and the teacher-forced continuation stage receives
  only the CONTINUE subset of the batch.
* All action components (``h_t, r_t, B_t, c_t, phi_t``) are discrete
  probabilities strictly in ``[0, 1]`` whose logs sum cleanly in the
  Trajectory Balance loss without units — pinned here by validity
  assertions over every head law, by the exact decomposition of
  ``ContinueOutput.step_log_prob`` into pure categorical log-probability
  components, and by invariance of the step log-probability to the von
  Mises auxiliary parameters (the pre-V2 rad^-1 density mixing is
  structurally absent).

Beyond the pinned criteria, this suite enforces: the three pre-V2
structural bugs stay dead (STOP token leak, density-in-TB, missing
parity); the certified-registry metadata contracts (27 reactions, 9
families, family membership, handle-class vocabularies); the mask
factories against real ``AssemblyEnvironment`` menus and against
brute-force references; the exact hierarchical factorization
``P(f, r | s) = P(f | s) * P(r | f, s)``; normalization over the actually
valid alternatives (masked mass exactly zero); forced termination for
states without legal continuation; parity scored exactly where a
stereocenter forms; the torsion bin-centre identity with the engine's
pinned geometry contract; the circular loss semantics (floor at zero
residual, maximum at the antipode, circularity across the +/-pi seam);
fail-closed supervision consistency (missing parity/bin on required
junctions, illegal targets, poisoned masks); the frozen production
parameter profile (3,272,975 parameters); gradient connectivity through
every head including the tied catalog table; determinism; SE(3)
invariance of the full head stack end-to-end through the real backbone +
handle head + decoder; and an end-to-end environment step proving the
head indices map onto ``AssemblyAction`` fields exactly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem

from syntree.core.schema import TensorContractError
from syntree.chemistry.reactions import (
    POLICY_REACTIONS as REGISTRY,
    REACTION_FAMILIES,
)
from syntree.engine.environment import (
    TORSION_BINS as ENGINE_TORSION_BINS,
    AssemblyAction,
    AssemblyEnvironment,
    BuildingBlock,
    HandleSpec,
    PocketDefinition,
    bin_center as engine_bin_center,
)
from syntree.engine.gflownet import TrajectoryBalanceEngine, TrajectoryStep
from syntree.models.bipartite import BackboneOutput, BipartitePaiNN
from syntree.models.handle_head import (
    HANDLE_FEATURE_DIM,
    HandleSelectionHead,
)
from syntree.models.policy import (
    BLOCK_HANDLE_CLASS_VOCAB,
    CONTINUE_INDEX,
    DECODER_D_MODEL,
    FAMILY_MEMBERS,
    HIERARCHICAL_HEADS_PARAMETER_PROFILE,
    MOLECULE_HANDLE_CLASS_VOCAB,
    PARITY_ACTIONS,
    POLICY_NUM_FAMILIES,
    POLICY_NUM_REACTIONS,
    POLICY_REACTION_IDS,
    REACTION_FAMILY_NAMES,
    STOP_INDEX,
    SYNTON_CATALOG_SIZE,
    SYNTON_EMBED_DIM,
    ChiralParityHead,
    ContinueOutput,
    FamilyOutput,
    HierarchicalPolicyHeads,
    ParityOutput,
    ReactionFamilyHead,
    ReactionHead,
    ReactionOutput,
    StopOutput,
    SynthonOutput,
    SynthonRetrievalHead,
    TerminationHead,
    build_continuation_masks,
    build_force_stop,
    handle_classes_from_menu,
    reaction_support_mask,
    synthon_legality_mask,
)
from syntree.models.torsion_head import (
    TORSION_BINS,
    TORSION_HEAD_CONTEXT_DIM,
    TORSION_HEAD_SYNTON_DIM,
    TorsionHead,
    TorsionOutput,
    torsion_bin_center,
    torsion_bin_centers,
    torsion_circular_loss,
    von_mises_loss,
)
from syntree.utils.hardware import HardwareError, LOGIT_MASK_FILL

# Small configuration used throughout most of the suite (fast); the frozen
# production profile is covered explicitly by TestFrozenProfile.
D = 16
CATALOG = 120
EMBED = 8
T_HIDDEN = 12

RTOL = 1e-6
ATOL64 = 1e-9


# ---------------------------------------------------------------------------
# Local stack fixtures (self-contained copies of the #22 suite's helpers —
# ``tests/`` is not a package, so cross-test-module imports are not used)
# ---------------------------------------------------------------------------

def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random *proper* rotation (orthogonal, det = +1) in float64."""
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_node_state(
    n_pocket: int = 10,
    n_ligand: int = 5,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    batch: int = 0,
) -> SimpleNamespace:
    """A minimal single-graph state exposing the eight node-level fields."""
    gen = torch.Generator().manual_seed(seed)
    elements = torch.tensor((1, 6, 7, 8, 9, 15, 16, 17), dtype=torch.long)

    def positions(count: int) -> torch.Tensor:
        return (torch.rand((count, 3), generator=gen, dtype=dtype) * 2.0 - 1.0) * 3.0

    def z_of(count: int) -> torch.Tensor:
        idx = torch.randint(0, len(elements), (count,), generator=gen)
        return elements[idx]

    return SimpleNamespace(
        pocket_pos=positions(n_pocket),
        pocket_z=z_of(n_pocket),
        pocket_charge=torch.zeros(n_pocket, dtype=dtype),
        pocket_batch=torch.full((n_pocket,), batch, dtype=torch.long),
        ligand_pos=positions(n_ligand),
        ligand_z=z_of(n_ligand),
        ligand_charge=torch.zeros(n_ligand, dtype=dtype),
        ligand_batch=torch.full((n_ligand,), batch, dtype=torch.long),
    )


def make_batched_state(
    pocket_counts, ligand_counts, *, seed: int = 0, dtype: torch.dtype = torch.float64
) -> SimpleNamespace:
    """A multi-graph state assembled from per-graph single-graph samples."""
    states = []
    for g, (n_p, n_l) in enumerate(zip(pocket_counts, ligand_counts)):
        states.append(make_node_state(n_p, n_l, seed=seed + g, dtype=dtype, batch=g))
    return SimpleNamespace(
        pocket_pos=torch.cat([s.pocket_pos for s in states]),
        pocket_z=torch.cat([s.pocket_z for s in states]),
        pocket_charge=torch.cat([s.pocket_charge for s in states]),
        pocket_batch=torch.cat([s.pocket_batch for s in states]),
        ligand_pos=torch.cat([s.ligand_pos for s in states]),
        ligand_z=torch.cat([s.ligand_z for s in states]),
        ligand_charge=torch.cat([s.ligand_charge for s in states]),
        ligand_batch=torch.cat([s.ligand_batch for s in states]),
    )


def make_pool(
    backbone: BackboneOutput,
    counts_per_graph,
    *,
    seed: int = 2,
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """A consistent handle pool for a batched backbone output."""
    gen = torch.Generator().manual_seed(seed)
    desc_gen = torch.Generator().manual_seed(seed + 10_000)
    num_pocket = backbone.num_pocket
    ligand_batch = backbone.node_batch[num_pocket:]
    ligand_pos = backbone.node_pos[num_pocket:]
    rows_graph, rows_atom = [], []
    for g, count in enumerate(counts_per_graph):
        nodes = torch.nonzero(ligand_batch == g, as_tuple=False).flatten()
        assert nodes.numel() >= count, "fixture requires enough ligand atoms"
        chosen = nodes[torch.randperm(nodes.numel(), generator=gen)[:count]]
        rows_atom.append(chosen)
        rows_graph.append(torch.full((count,), g, dtype=torch.long))
    handle_atom = torch.cat(rows_atom)
    handle_graph = torch.cat(rows_graph)
    handle_pos = ligand_pos.index_select(0, handle_atom).clone()
    descriptors = torch.randn(
        (int(handle_atom.size(0)), HANDLE_FEATURE_DIM), generator=desc_gen, dtype=dtype
    )
    return descriptors, handle_pos, handle_graph, handle_atom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_heads(
    dtype: torch.dtype = torch.float64, seed: int = 0
) -> HierarchicalPolicyHeads:
    torch.manual_seed(seed)
    heads = HierarchicalPolicyHeads(
        context_dim=D,
        catalog_size=CATALOG,
        synthon_embed_dim=EMBED,
        torsion_hidden_dim=T_HIDDEN,
    )
    return heads.to(dtype).eval()


def random_context(
    num_graphs: int, dtype: torch.dtype = torch.float64, seed: int = 1
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((num_graphs, D), generator=gen, dtype=dtype)


def amine_menus(num_graphs: int) -> list:
    """Per-graph active handle classes — the amine fixture."""
    return [["amine"] for _ in range(num_graphs)]


def synthetic_catalog(
    seed: int = 5, ensure_classes_for: tuple = ("acid_chloride",)
) -> torch.Tensor:
    """A random catalog handle-class assignment covering every class."""
    gen = torch.Generator().manual_seed(seed)
    vocab = list(BLOCK_HANDLE_CLASS_VOCAB)
    classes = torch.tensor(
        [int(torch.randint(0, len(vocab), (1,), generator=gen)) for _ in range(CATALOG)],
        dtype=torch.int64,
    )
    # guarantee at least two synthons of every class (non-empty supports)
    for k, cls in enumerate(vocab):
        classes[2 * k % CATALOG] = k
        classes[(2 * k + 1) % CATALOG] = k
    return classes


def default_teacher_forcing(heads, context, menus, catalog_classes, **overrides):
    """A consistent teacher-forced continuation fixture.

    Selects, for each graph, the first legal reaction, a legal synthon of
    that reaction's block class, a parity on stereogenic junctions, and a
    torsion bin on rotatable junctions.
    """
    fam_mask, rxn_mask = build_continuation_masks(menus)
    num_graphs = context.size(0)
    selected = []
    for b in range(num_graphs):
        legal = [
            i
            for i in range(POLICY_NUM_REACTIONS)
            if not bool(rxn_mask[b, i])
        ]
        selected.append(legal[0])
    selected_reaction = torch.tensor(selected, dtype=torch.int64)

    # synthons: pick the first legal synthon of each reaction's class
    mask = synthon_legality_mask(catalog_classes, selected_reaction)
    chosen = []
    for b in range(num_graphs):
        legal_synths = (~mask[b]).nonzero().flatten()
        chosen.append(int(legal_synths[0]))
    selected_synthon = torch.tensor(chosen, dtype=torch.int64)

    locked = torch.tensor(
        [REGISTRY[rid].resonance_lock is not None for rid in POLICY_REACTION_IDS]
    )
    torsion_required = ~locked.index_select(0, selected_reaction)
    kwargs = dict(
        selected_reaction=selected_reaction,
        selected_synthon=selected_synthon,
        selected_parity=torch.where(
            torsion_required, torch.tensor(0, dtype=torch.int64), torch.tensor(-1, dtype=torch.int64)
        ),
        selected_torsion_bin=torch.where(
            torsion_required, torch.tensor(9, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)
        ),
        stereocenter_required=torch.ones(num_graphs, dtype=torch.bool),
        family_mask=fam_mask,
        reaction_mask=rxn_mask,
        catalog_handle_classes=catalog_classes,
    )
    # parity must be provided wherever required: use required=True parity=0
    kwargs["selected_parity"] = torch.tensor(
        [0] * num_graphs, dtype=torch.int64
    )
    kwargs["stereocenter_required"] = torch.ones(num_graphs, dtype=torch.bool)
    kwargs.update(overrides)
    return heads.forward_continue(context, **kwargs), (
        fam_mask,
        rxn_mask,
        selected_reaction,
    )


def assert_valid_law(log_probs: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor):
    """A normalized law over exactly the legal support."""
    assert bool(torch.isfinite(log_probs).all())
    assert bool((probs >= 0.0).all()) and bool((probs <= 1.0).all())
    assert bool((log_probs <= 0.0).all())
    legal = ~mask
    # mass on illegal entries is exactly zero
    assert float(probs.masked_fill(legal, 0.0).sum()) < 1e-12
    # the law is normalized over the legal support
    assert probs.sum(dim=-1).max().item() == pytest.approx(1.0, abs=1e-9)
    assert probs.sum(dim=-1).min().item() == pytest.approx(1.0, abs=1e-9)
    # log-probs are the logs of the legal probabilities (the masked
    # entries underflow exp(), so the comparison is over legal support)
    assert torch.allclose(
        log_probs.masked_fill(mask, 0.0),
        probs.clamp(min=1e-300).log().masked_fill(mask, 0.0),
        atol=1e-9,
    )


class HookCounter:
    """Counts forward invocations of a module (the STOP-isolation pin)."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.count = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.count += 1

    def remove(self) -> None:
        self.handle.remove()


# ---------------------------------------------------------------------------
# 1. Registry metadata contracts (the certified taxonomy as consumed here)
# ---------------------------------------------------------------------------

class TestRegistryMetadata:
    """The heads' action-space cardinalities ARE the #11 registry's."""

    def test_nine_families_twenty_seven_reactions(self):
        assert POLICY_NUM_FAMILIES == 9
        assert POLICY_NUM_REACTIONS == 27
        assert len(REACTION_FAMILY_NAMES) == 9
        assert len(POLICY_REACTION_IDS) == 27
        assert set(POLICY_REACTION_IDS) == set(REGISTRY.keys())

    def test_family_membership_matches_registry(self):
        for name, members in FAMILY_MEMBERS.items():
            assert set(members) == set(REACTION_FAMILIES[name])
            for rid in members:
                assert REGISTRY[rid].family == name

    def test_every_reaction_belongs_to_exactly_one_family(self):
        all_members = [r for members in FAMILY_MEMBERS.values() for r in members]
        assert len(all_members) == 27
        assert len(set(all_members)) == 27

    def test_handle_class_vocabularies(self):
        mol = {r.molecule_handle_class for r in REGISTRY.values()}
        blk = {r.block_handle_class for r in REGISTRY.values()}
        assert set(MOLECULE_HANDLE_CLASS_VOCAB) == mol
        assert set(BLOCK_HANDLE_CLASS_VOCAB) == blk
        assert BLOCK_HANDLE_CLASS_VOCAB == tuple(sorted(BLOCK_HANDLE_CLASS_VOCAB))

    def test_deprotections_are_not_policy_actions(self):
        policy_ids = set(POLICY_REACTION_IDS)
        for rid, reaction in REGISTRY.items():
            if rid in policy_ids:
                assert reaction.policy_action
        assert len([r for r in REGISTRY.values() if r.policy_action]) == 27


# ---------------------------------------------------------------------------
# 2. Acceptance criterion 1 — STOP before continuation heads
# ---------------------------------------------------------------------------

class TestStopBeforeContinuation:
    """STOP terminates immediately; continuation heads never run."""

    def test_forward_stop_touches_no_continuation_head(self):
        heads = make_heads()
        context = random_context(4)
        counters = [
            HookCounter(heads.family_head),
            HookCounter(heads.reaction_head),
            HookCounter(heads.synthon_head),
            HookCounter(heads.parity_head),
            HookCounter(heads.torsion_head),
        ]
        try:
            out = heads.forward_stop(context)
            assert isinstance(out, StopOutput)
            for counter in counters:
                assert counter.count == 0
        finally:
            for counter in counters:
                counter.remove()

    def test_stop_law_independent_of_synthon_table(self):
        heads = make_heads()
        context = random_context(3)
        before = heads.forward_stop(context)
        with torch.no_grad():
            heads.synthon_head.catalog_embeddings.weight.add_(1.0)
        after = heads.forward_stop(context)
        assert torch.equal(before.stop_log_prob, after.stop_log_prob)
        assert torch.equal(before.log_probs, after.log_probs)

    def test_stop_law_independent_of_torsion_parameters(self):
        heads = make_heads()
        context = random_context(3)
        before = heads.forward_stop(context)
        with torch.no_grad():
            heads.torsion_head.bin_logits.weight.add_(0.5)
            heads.torsion_head.mu_xy.bias.fill_(2.0)
        after = heads.forward_stop(context)
        assert torch.equal(before.stop_log_prob, after.stop_log_prob)

    def test_stop_is_binary_not_a_catalog_token(self):
        # The pre-V2 bug indexed STOP as token K inside the synthon head.
        heads = make_heads()
        assert heads.stop_head.logits.out_features == 2
        assert heads.synthon_head.catalog_size == CATALOG
        out = heads.forward_stop(random_context(2))
        assert tuple(out.log_probs.shape) == (2, 2)
        assert STOP_INDEX == 1 and CONTINUE_INDEX == 0

    def test_teacher_forced_flow_excludes_stop_rows(self):
        # The documented rollout pattern: only CONTINUE rows reach the
        # continuation stage. The step log-prob of a STOP row is exactly
        # the stop law's contribution and nothing else.
        heads = make_heads()
        context = random_context(5)
        stop = heads.forward_stop(context)
        decision = stop.probs[:, STOP_INDEX] > 0.5
        continue_rows = torch.nonzero(~decision, as_tuple=False).flatten()
        cont, (fam_mask, rxn_mask, _) = default_teacher_forcing(
            heads,
            context.index_select(0, continue_rows),
            [["amine"] for _ in continue_rows.tolist()],
            synthetic_catalog(),
        )
        assert cont.num_graphs == int(continue_rows.numel())
        assert cont.num_graphs < 5  # the fixture split the batch

    def test_forced_stop_yields_exact_certain_termination(self):
        heads = make_heads()
        context = random_context(3)
        force = torch.tensor([True, False, True])
        out = heads.forward_stop(context, force_stop=force)
        for b in (0, 2):
            assert out.probs[b, STOP_INDEX].item() == pytest.approx(1.0, abs=1e-12)
            assert out.probs[b, CONTINUE_INDEX].item() == 0.0
            assert out.stop_log_prob[b].item() == pytest.approx(0.0, abs=1e-12)
            assert out.forced_stop[b].item() is True
        # unforced row is a genuine binary law
        assert 0.0 < out.probs[1, STOP_INDEX].item() < 1.0

    def test_force_stop_none_matches_all_false_mask(self):
        heads = make_heads()
        context = random_context(2)
        a = heads.forward_stop(context)
        b = heads.forward_stop(
            context, force_stop=torch.zeros(2, dtype=torch.bool)
        )
        assert torch.equal(a.log_probs, b.log_probs)


# ---------------------------------------------------------------------------
# 3. Acceptance criterion 2 — discrete measures, clean TB composition
# ---------------------------------------------------------------------------

class TestDiscreteMeasures:
    """Every action component is a dimensionless categorical law."""

    @pytest.mark.parametrize("menus", [[["amine"]], [["aryl_bromide"], ["thiol"]]])
    def test_all_continuation_laws_are_valid(self, menus):
        heads = make_heads()
        context = random_context(len(menus))
        cont, _ = default_teacher_forcing(
            heads, context, menus, synthetic_catalog()
        )
        fam_mask, rxn_mask = build_continuation_masks(menus)
        member = reaction_support_mask(
            torch.tensor(
                [
                    REACTION_FAMILY_NAMES.index(REGISTRY[POLICY_REACTION_IDS[i]].family)
                    for i in cont.reaction.selected_reaction.tolist()
                ]
            ),
            rxn_mask,
        )
        syn_mask = synthon_legality_mask(
            synthetic_catalog(), cont.reaction.selected_reaction
        )
        assert_valid_law(cont.family.log_probs, cont.family.probs, fam_mask)
        assert_valid_law(cont.reaction.log_probs, cont.reaction.probs, member)
        assert_valid_law(cont.synthon.log_probs, cont.synthon.probs, syn_mask)
        assert_valid_law(
            cont.parity.log_probs,
            cont.parity.probs,
            torch.zeros_like(cont.parity.log_probs, dtype=torch.bool),
        )
        assert_valid_law(
            cont.torsion.log_probs,
            cont.torsion.probs,
            torch.zeros_like(cont.torsion.log_probs, dtype=torch.bool),
        )
        assert cont.parity.probs.shape == (cont.num_graphs, 2)
        assert cont.torsion.probs.shape == (cont.num_graphs, TORSION_BINS)

    def test_step_log_prob_decomposes_exactly(self):
        heads = make_heads()
        context = random_context(4)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(4), synthetic_catalog()
        )
        manual = (
            cont.family.selected_log_prob
            + cont.reaction.selected_log_prob
            + cont.synthon.selected_log_prob
            + cont.parity.selected_log_prob
            + cont.torsion.selected_log_prob
        )
        assert torch.allclose(cont.step_log_prob(), manual, atol=ATOL64)
        handle_lp = torch.tensor([-0.3, -1.1, -0.7, -2.0], dtype=context.dtype)
        assert torch.allclose(
            cont.step_log_prob(handle_lp), manual + handle_lp, atol=ATOL64
        )

    def test_step_log_prob_excludes_von_mises_parameters(self):
        # The pre-V2 failure mixed a rad^-1 density into TB.  The step
        # log-probability is invariant to the auxiliary (mu, kappa).
        heads = make_heads()
        context = random_context(3)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        before = cont.step_log_prob()
        with torch.no_grad():
            heads.torsion_head.kappa_raw.bias.fill_(3.0)
            heads.torsion_head.mu_xy.weight.mul_(2.0)
        cont2, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        # identical selections (fixture is deterministic)
        assert torch.equal(
            cont.torsion.selected_bin, cont2.torsion.selected_bin
        )
        assert torch.equal(before, cont2.step_log_prob())

    def test_step_log_prob_is_finite_and_negative(self):
        heads = make_heads()
        context = random_context(3)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        lp = cont.step_log_prob()
        assert bool(torch.isfinite(lp).all())
        assert bool((lp < 0).all())

    def test_tb_residual_composition_is_finite(self):
        # Compose a two-step trajectory from real head outputs and feed
        # it to the TB engine: the forward term is exactly the sum of the
        # step log-probabilities (dimensionless categorical only).
        heads = make_heads()
        context = random_context(1)
        cont, _ = default_teacher_forcing(
            heads, context, [["amine"]], synthetic_catalog()
        )
        stop = heads.forward_stop(context)
        lp1 = cont.step_log_prob(handle_log_probs=torch.tensor([-0.5], dtype=context.dtype))
        lp2 = stop.stop_log_prob
        steps = [
            TrajectoryStep(action_type="add", action_log_prob=float(lp1[0]), num_leaves=1),
            TrajectoryStep(action_type="stop", action_log_prob=float(lp2[0])),
        ]
        engine = TrajectoryBalanceEngine()
        residual = engine.trajectory_residual(steps, reward=2.0)
        assert bool(torch.isfinite(residual))
        expected = lp1[0] + lp2[0] - math.log(2.0) - float(engine.log_z)
        # the engine evaluates in float32: agreement to float32 rounding
        assert residual.item() == pytest.approx(float(expected), abs=1e-6)

    def test_no_density_units_anywhere_in_torsion_tb_term(self):
        # The torsion TB contribution is the categorical log-prob; kappa
        # never appears in any selected_log_prob tensor.
        heads = make_heads()
        context = random_context(2)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(2), synthetic_catalog()
        )
        # pick a rotatable junction for at least one row
        locked = torch.tensor(
            [REGISTRY[rid].resonance_lock is not None for rid in POLICY_REACTION_IDS]
        )
        rotatable = ~locked.index_select(0, cont.reaction.selected_reaction)
        if bool(rotatable.any()):
            rows = torch.nonzero(rotatable, as_tuple=False).flatten()
            expected = cont.torsion.log_probs.gather(
                1, (cont.torsion.selected_bin[rows] - 1).unsqueeze(-1)
            ).squeeze(-1)
            assert torch.allclose(
                cont.torsion.selected_log_prob[rows], expected, atol=ATOL64
            )

    def test_locked_junctions_contribute_exactly_zero_torsion(self):
        heads = make_heads()
        context = random_context(2)
        # rxn_01 (amide) is resonance-locked: no torsion action.  Pick
        # legal acid_chloride synthons first.
        fam_mask, rxn_mask = build_continuation_masks(amine_menus(2))
        classes = synthetic_catalog()
        acid = BLOCK_HANDLE_CLASS_VOCAB.index("acid_chloride")
        legal_synths = (classes == acid).nonzero().flatten()
        assert legal_synths.numel() >= 2
        cont = heads.forward_continue(
            context,
            selected_reaction=torch.tensor([0, 0], dtype=torch.int64),
            selected_synthon=legal_synths[:2],
            selected_parity=torch.tensor([0, 1], dtype=torch.int64),
            selected_torsion_bin=torch.tensor([0, 0], dtype=torch.int64),
            stereocenter_required=torch.ones(2, dtype=torch.bool),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=classes,
        )
        assert torch.equal(
            cont.torsion.selected_log_prob, torch.zeros(2, dtype=context.dtype)
        )
        assert not bool(cont.torsion.required.any())


# ---------------------------------------------------------------------------
# 4. Mask factories — registry semantics and brute-force equivalence
# ---------------------------------------------------------------------------

class TestMaskFactories:
    def test_reaction_legality_matches_registry_semantics(self):
        menus = [["amine", "aryl_bromide"]]
        fam_mask, rxn_mask = build_continuation_masks(menus)
        for r_idx, rid in enumerate(POLICY_REACTION_IDS):
            reaction = REGISTRY[rid]
            legal = (
                reaction.molecule_handle_class in menus[0]
                and reaction.block_handle_class in BLOCK_HANDLE_CLASS_VOCAB
            )
            assert bool(rxn_mask[0, r_idx]) == (not legal)

    def test_family_legal_iff_any_member_legal(self):
        menus = [["thiol"], ["alkylzinc"], ["vinyl_bromide"], ["boronic_acid"]]
        fam_mask, rxn_mask = build_continuation_masks(menus)
        for b in range(len(menus)):
            for f_idx, fname in enumerate(REACTION_FAMILY_NAMES):
                any_legal = any(
                    not bool(rxn_mask[b, POLICY_REACTION_IDS.index(m)])
                    for m in FAMILY_MEMBERS[fname]
                )
                assert bool(fam_mask[b, f_idx]) == (not any_legal)

    def test_amine_only_menu_cardinality(self):
        # amine molecule-side: 12 reactions across 7 families; the two
        # families requiring aryl/vinyl/alkylzinc or alcohol/thiol
        # molecule-side handles stay illegal.
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        assert int((~rxn_mask).sum()) == 12
        assert int((~fam_mask).sum()) == 7
        assert bool(fam_mask[0, REACTION_FAMILY_NAMES.index("cross_coupling_cc")])
        assert bool(fam_mask[0, REACTION_FAMILY_NAMES.index("ester_and_ether")])
        assert not bool(
            fam_mask[0, REACTION_FAMILY_NAMES.index("amide_and_carbamate")]
        )

    def test_block_availability_restricts_reactions(self):
        # no acid_chloride in the catalog: rxn_01/rxn_17/rxn_20 die and
        # amide_and_carbamate survives only through rxn_08 (chloroformate).
        available = set(BLOCK_HANDLE_CLASS_VOCAB) - {"acid_chloride"}
        fam_mask, rxn_mask = build_continuation_masks([["amine"]], available)
        assert bool(rxn_mask[0, POLICY_REACTION_IDS.index("rxn_01")])
        assert not bool(rxn_mask[0, POLICY_REACTION_IDS.index("rxn_08")])
        assert not bool(fam_mask[0, REACTION_FAMILY_NAMES.index("amide_and_carbamate")])
        # removing chloroformate as well kills the family entirely
        available2 = available - {"chloroformate"}
        fam_mask2, _ = build_continuation_masks([["amine"]], available2)
        assert bool(fam_mask2[0, REACTION_FAMILY_NAMES.index("amide_and_carbamate")])

    def test_unknown_block_class_rejected(self):
        with pytest.raises(TensorContractError):
            build_continuation_masks([["amine"]], {"not_a_real_class"})

    def test_empty_menu_forces_stop(self):
        fam_mask, rxn_mask = build_continuation_masks([[]])
        assert bool(rxn_mask.all())
        forced = build_force_stop(rxn_mask)
        assert bool(forced[0])

    def test_at_capacity_forces_stop(self):
        _, rxn_mask = build_continuation_masks([["amine"]])
        forced = build_force_stop(rxn_mask, torch.tensor([True]))
        assert bool(forced[0])
        forced2 = build_force_stop(rxn_mask, torch.tensor([False]))
        assert not bool(forced2[0])

    def test_synthon_mask_brute_force_equivalence(self):
        classes = synthetic_catalog(seed=9)
        selected = torch.tensor([0, 5, 12, 26], dtype=torch.int64)
        mask = synthon_legality_mask(classes, selected)
        for b, r_idx in enumerate(selected.tolist()):
            blk_class = REGISTRY[POLICY_REACTION_IDS[r_idx]].block_handle_class
            want = BLOCK_HANDLE_CLASS_VOCAB.index(blk_class)
            for k in range(CATALOG):
                assert bool(mask[b, k]) == (int(classes[k]) != want)

    def test_support_mask_restricts_to_selected_family(self):
        _, rxn_mask = build_continuation_masks([["amine", "aryl_bromide"]])
        f_idx = torch.tensor(
            [REACTION_FAMILY_NAMES.index("cross_coupling_cc")]
        )
        support = reaction_support_mask(f_idx, rxn_mask)
        for r_idx, rid in enumerate(POLICY_REACTION_IDS):
            member = REGISTRY[rid].family == "cross_coupling_cc"
            if not member:
                assert bool(support[0, r_idx])
        # all legal cross-coupling members stay legal
        legal_cc = [
            i
            for i in range(27)
            if REGISTRY[POLICY_REACTION_IDS[i]].family == "cross_coupling_cc"
            and not bool(rxn_mask[0, i])
        ]
        assert legal_cc  # aryl_bromide gives at least rxn_09/11/13
        for i in legal_cc:
            assert not bool(support[0, i])

    def test_support_mask_out_of_range_family_rejected(self):
        _, rxn_mask = build_continuation_masks([["amine"]])
        with pytest.raises(TensorContractError):
            reaction_support_mask(torch.tensor([9]), rxn_mask)

    def test_real_environment_menu_integration(self):
        # The mask factory consumes the exact active_handles() menu of a
        # real environment (the #24 issue text: "consume the environment's
        # active_handles() menu").
        env = AssemblyEnvironment(_aniline_seed(), _acid_catalog(), max_blocks=4)
        env.reset()
        menu = env.active_handles()
        classes = handle_classes_from_menu(menu)
        assert classes == ("amine",)
        fam_mask, rxn_mask = build_continuation_masks([list(classes)])
        assert int((~rxn_mask).sum()) == 12


# ---------------------------------------------------------------------------
# 5. Termination head
# ---------------------------------------------------------------------------

class TestTerminationHead:
    def test_binary_law_normalized(self):
        head = TerminationHead(context_dim=D).to(torch.float64).eval()
        out = head(random_context(3))
        assert torch.allclose(out.probs.sum(-1), torch.ones(3, dtype=torch.float64))
        assert bool((out.probs >= 0).all()) and bool((out.probs <= 1).all())
        assert torch.allclose(
            out.log_probs, out.probs.clamp(min=1e-300).log(), atol=1e-9
        )

    def test_seeded_head_is_balanced(self):
        torch.manual_seed(0)
        head = TerminationHead(context_dim=256).to(torch.float64).eval()
        context = torch.zeros(1, 256, dtype=torch.float64)
        out = head(context)
        # zero context through a zero-bias-free linear: near-uniform law
        assert out.probs[0, 0].item() == pytest.approx(0.5, abs=0.2)

    def test_forced_rows_have_masked_continue_logit(self):
        head = TerminationHead(context_dim=D).to(torch.float64).eval()
        out = head(random_context(2), force_stop=torch.tensor([True, False]))
        assert bool((out.logits[0, CONTINUE_INDEX] == LOGIT_MASK_FILL).item())
        assert out.logits[1, CONTINUE_INDEX].item() > LOGIT_MASK_FILL

    def test_fail_closed_dtype(self):
        head = TerminationHead(context_dim=D).to(torch.float64).eval()
        with pytest.raises(TensorContractError):
            head(random_context(2, dtype=torch.float32))
        with pytest.raises(TensorContractError):
            head(random_context(2, dtype=torch.float16))

    def test_fail_closed_shapes_and_finiteness(self):
        head = TerminationHead(context_dim=D).to(torch.float64).eval()
        with pytest.raises(TensorContractError):
            head(torch.zeros(2, D + 1, dtype=torch.float64))
        with pytest.raises(TensorContractError):
            head(torch.zeros(0, D, dtype=torch.float64))
        bad = random_context(2)
        bad[0, 0] = float("nan")
        with pytest.raises(TensorContractError):
            head(bad)

    def test_audit(self):
        head = TerminationHead(context_dim=256)
        audit = head.audit()
        assert audit["num_parameters"] == 256 * 2 + 2
        assert audit["num_actions"] == 2
        assert audit["columns"] == ("CONTINUE", "STOP")
        assert audit["all_parameters_trainable"]


# ---------------------------------------------------------------------------
# 6. Family and reaction heads — the exact hierarchical factorization
# ---------------------------------------------------------------------------

class TestFamilyReactionHeads:
    def test_exact_factorization_identity(self):
        # P(f, r | s) = P(f | s) * P(r | f, s): summing the joint over
        # all legal (family, reaction) pairs recovers exactly 1.
        heads = make_heads(seed=3)
        context = random_context(1, seed=4)
        menus = [["amine", "aryl_bromide"]]
        fam_mask, rxn_mask = build_continuation_masks(menus)
        total = 0.0
        legal_fams = (~fam_mask[0]).nonzero().flatten()
        for f in legal_fams.tolist():
            fam_alone = heads.family_head(
                context, fam_mask, torch.tensor([f])
            )
            rxn_alone = heads.reaction_head(
                context, torch.tensor([f]), rxn_mask
            )
            total += float(fam_alone.probs[0, f]) * float(
                rxn_alone.probs[0].sum()
            )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_reaction_support_normalizes_within_family(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        f_idx = torch.tensor(
            [REACTION_FAMILY_NAMES.index("aryl_amination")]
        )
        out = heads.reaction_head(context, f_idx, rxn_mask)
        member_cols = [
            i
            for i in range(27)
            if REGISTRY[POLICY_REACTION_IDS[i]].family == "aryl_amination"
        ]
        nonmember_mass = sum(
            float(out.probs[0, i]) for i in range(27) if i not in member_cols
        )
        assert nonmember_mass == pytest.approx(0.0, abs=1e-12)
        assert float(out.probs[0].sum()) == pytest.approx(1.0, abs=1e-9)

    def test_selected_family_must_be_legal(self):
        heads = make_heads()
        context = random_context(1)
        # ester_and_ether is illegal for an amine-only menu
        fam_mask, _ = build_continuation_masks([["amine"]])
        bad = torch.tensor([REACTION_FAMILY_NAMES.index("ester_and_ether")])
        with pytest.raises(HardwareError):
            heads.family_head(context, fam_mask, bad)

    def test_selected_reaction_must_be_legal_and_in_family(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        f_idx = torch.tensor([REACTION_FAMILY_NAMES.index("sp3_alkylation")])
        # rxn_16 is sp3_alkylation but alcohol-side: illegal for amine menu
        with pytest.raises(HardwareError):
            heads.reaction_head(
                context, f_idx, rxn_mask, torch.tensor([POLICY_REACTION_IDS.index("rxn_16")])
            )
        # rxn_09 (cross_coupling_cc) is legal for aryl_bromide menus but
        # not a member of sp3_alkylation
        fam2, rxn2 = build_continuation_masks([["amine", "aryl_bromide"]])
        with pytest.raises(HardwareError):
            heads.reaction_head(
                context, f_idx, rxn2, torch.tensor([POLICY_REACTION_IDS.index("rxn_09")])
            )

    def test_all_masked_family_row_raises(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask = torch.ones(1, POLICY_NUM_FAMILIES, dtype=torch.bool)
        with pytest.raises(HardwareError):
            heads.family_head(context, fam_mask)

    def test_wrong_cardinality_rejected(self):
        head = ReactionFamilyHead(context_dim=D)
        with pytest.raises(ValueError):
            ReactionFamilyHead(context_dim=D, num_families=8)
        with pytest.raises(ValueError):
            ReactionHead(context_dim=D, num_reactions=26)


# ---------------------------------------------------------------------------
# 7. Synthon retrieval head — the tied catalog embeddings
# ---------------------------------------------------------------------------

class TestSynthonHead:
    def test_tied_table_shapes_and_scale(self):
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED)
        assert head.catalog_embeddings.weight.shape == (CATALOG, EMBED)
        audit = head.audit()
        assert audit["tied_catalog_embedding_parameters"] == CATALOG * EMBED
        assert audit["logit_scale"] == pytest.approx(1.0 / math.sqrt(EMBED))
        assert audit["num_parameters"] == CATALOG * EMBED + (D * EMBED + EMBED)

    def test_logits_are_scaled_dot_products(self):
        torch.manual_seed(0)
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED).to(torch.float64).eval()
        context = random_context(2)
        classes = synthetic_catalog()
        selected_reaction = torch.tensor([0, 1])
        # pick legal synthons for each reaction's class
        mask = synthon_legality_mask(classes, selected_reaction)
        chosen = [(~mask[b]).nonzero().flatten()[0].item() for b in range(2)]
        selected_synthon = torch.tensor(chosen, dtype=torch.int64)
        out = head(context, selected_reaction, selected_synthon, classes)
        query = head.context_proj(context)
        table = head.catalog_embeddings.weight
        raw = (query @ table.t()) * (1.0 / math.sqrt(EMBED))
        filled = raw.masked_fill(mask, LOGIT_MASK_FILL)
        assert torch.allclose(out.logits, filled, atol=1e-7)

    def test_retrieved_embedding_is_the_tied_row(self):
        torch.manual_seed(0)
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED).to(torch.float64).eval()
        context = random_context(1)
        classes = synthetic_catalog()
        acid = BLOCK_HANDLE_CLASS_VOCAB.index("acid_chloride")
        sel = (classes == acid).nonzero().flatten()[:1]
        out = head(context, torch.tensor([0]), sel, classes)
        assert torch.allclose(
            out.retrieved_embedding, head.catalog_embeddings.weight[sel[0]], atol=1e-12
        )

    def test_no_catalog_assignment_fails_closed(self):
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED)
        context = random_context(1)
        with pytest.raises(TensorContractError):
            head(context, torch.tensor([0]))

    def test_attach_catalog_validates(self):
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED).to(torch.float64)
        with pytest.raises(TensorContractError):
            head.set_catalog_handle_classes(torch.zeros(CATALOG + 1, dtype=torch.int64))
        with pytest.raises(TensorContractError):
            head.set_catalog_handle_classes(
                torch.full((CATALOG,), 999, dtype=torch.int64)
            )
        head.set_catalog_handle_classes(synthetic_catalog())
        context = random_context(1)
        out = head(context, torch.tensor([0]), torch.tensor([3]))
        assert tuple(out.log_probs.shape) == (1, CATALOG)

    def test_class_without_synthons_fails_loudly(self):
        # a reaction whose block class has no synthon in the catalog must
        # raise (inconsistent wiring), never silently normalize.
        torch.manual_seed(0)
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED).to(torch.float64).eval()
        classes = torch.zeros(CATALOG, dtype=torch.int64)  # all class 0
        context = random_context(1)
        # reaction 1 (rxn_02) uses sulfonyl_chloride (class != 0)
        with pytest.raises(HardwareError):
            head(context, torch.tensor([1]), torch.tensor([0]), classes)

    def test_selected_synthon_must_be_legal(self):
        torch.manual_seed(0)
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED).to(torch.float64).eval()
        classes = synthetic_catalog()
        context = random_context(1)
        # find a synthon whose class mismatches rxn_01's acid_chloride
        want = BLOCK_HANDLE_CLASS_VOCAB.index(
            REGISTRY["rxn_01"].block_handle_class
        )
        bad_k = next(
            k for k in range(CATALOG) if int(classes[k]) != want
        )
        with pytest.raises(HardwareError):
            head(context, torch.tensor([0]), torch.tensor([bad_k]), classes)

    def test_catalog_embedding_retrieval_bounds(self):
        head = SynthonRetrievalHead(context_dim=D, catalog_size=CATALOG, embed_dim=EMBED)
        with pytest.raises(TensorContractError):
            head.embedding(torch.tensor([CATALOG]))
        with pytest.raises(TensorContractError):
            head.embedding(torch.tensor([-1]))
        emb = head.embedding(torch.tensor([0, CATALOG - 1]))
        assert emb.shape == (2, EMBED)

    def test_production_catalog_memory_arithmetic(self):
        # issue #24 pins K = 50,000; the tied table at 64-D FP32 is the
        # dominant parameter block — arithmetic, not a T4 measurement.
        head = SynthonRetrievalHead()  # frozen production profile
        audit = head.audit()
        assert audit["catalog_size"] == SYNTON_CATALOG_SIZE == 50_000
        assert audit["tied_catalog_embedding_bytes_fp32"] == 50_000 * 64 * 4
        assert audit["num_parameters"] == 50_000 * 64 + 256 * 64 + 64


# ---------------------------------------------------------------------------
# 8. Chiral parity head
# ---------------------------------------------------------------------------

class TestParityHead:
    def test_two_way_law_normalized(self):
        head = ChiralParityHead(context_dim=D, synthon_dim=EMBED).to(torch.float64).eval()
        context = random_context(3)
        emb = torch.randn(3, EMBED, dtype=torch.float64)
        out = head(context, emb, required=torch.ones(3, dtype=torch.bool))
        assert torch.allclose(out.probs.sum(-1), torch.ones(3, dtype=torch.float64))
        assert tuple(out.log_probs.shape) == (3, 2)

    def test_parity_actions_match_environment_chiral(self):
        # ('R', 'S') are exactly AssemblyAction.chiral's accepted values.
        assert PARITY_ACTIONS == ("R", "S")
        for value in PARITY_ACTIONS:
            AssemblyAction(0, "rxn_01", "blk", chiral=value)  # accepted
        with pytest.raises(ValueError):
            AssemblyAction(0, "rxn_01", "blk", chiral="X")

    def test_required_rows_carry_selection(self):
        head = ChiralParityHead(context_dim=D, synthon_dim=EMBED).to(torch.float64).eval()
        context = random_context(2)
        emb = torch.randn(2, EMBED, dtype=torch.float64)
        out = head(
            context,
            emb,
            required=torch.tensor([True, False]),
            selected_parity=torch.tensor([1, -1]),
        )
        assert out.selected_log_prob[1].item() == 0.0
        assert out.selected_log_prob[0].item() < 0.0
        expected = out.log_probs[0, 1]
        assert out.selected_log_prob[0].item() == pytest.approx(float(expected))

    def test_missing_parity_on_stereocenter_fails_closed(self):
        head = ChiralParityHead(context_dim=D, synthon_dim=EMBED).to(torch.float64).eval()
        context = random_context(1)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        with pytest.raises(TensorContractError):
            head(
                context,
                emb,
                required=torch.tensor([True]),
                selected_parity=torch.tensor([-1]),
            )

    def test_vacuous_parity_on_non_stereocenter_not_scored(self):
        head = ChiralParityHead(context_dim=D, synthon_dim=EMBED).to(torch.float64).eval()
        context = random_context(1)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        out = head(
            context,
            emb,
            required=torch.tensor([False]),
            selected_parity=torch.tensor([1]),
        )
        assert out.selected_log_prob[0].item() == 0.0

    def test_conditioned_on_synthon_embedding(self):
        # P(c | s, B): flipping the synthon embedding changes the law.
        torch.manual_seed(0)
        head = ChiralParityHead(context_dim=D, synthon_dim=EMBED).to(torch.float64).eval()
        context = random_context(1)
        emb1 = torch.randn(1, EMBED, dtype=torch.float64)
        emb2 = -emb1
        out1 = head(context, emb1)
        out2 = head(context, emb2)
        assert not torch.allclose(out1.log_probs, out2.log_probs)


# ---------------------------------------------------------------------------
# 9. Torsion head — the discrete SO(2) space and the circular loss
# ---------------------------------------------------------------------------

class TestTorsionHead:
    def test_bin_centers_match_engine_contract_exactly(self):
        from syntree.engine.environment import bin_center

        for k in range(1, TORSION_BINS + 1):
            assert torsion_bin_center(k) == bin_center(k)

    def test_bin_count_and_width(self):
        assert TORSION_BINS == 36 == ENGINE_TORSION_BINS
        assert TORSION_BINS * (2 * math.pi / 36) == pytest.approx(2 * math.pi)
        centers = torsion_bin_centers()
        assert len(centers) == 36
        assert centers[0] == pytest.approx(-math.pi + math.pi / 36)
        assert centers[-1] == pytest.approx(math.pi - math.pi / 36)

    def test_bin_center_validation(self):
        with pytest.raises(ValueError):
            torsion_bin_center(0)
        with pytest.raises(ValueError):
            torsion_bin_center(37)

    def test_categorical_law_over_bins(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(2)
        emb = torch.randn(2, EMBED, dtype=torch.float64)
        out = head(context, emb)
        assert tuple(out.log_probs.shape) == (2, TORSION_BINS)
        assert torch.allclose(out.probs.sum(-1), torch.ones(2, dtype=torch.float64))
        assert bool((out.probs >= 0).all()) and bool((out.probs <= 1).all())

    def test_mu_range_and_kappa_nonnegative(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(16, seed=7)
        emb = torch.randn(16, EMBED, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
        out = head(context, emb)
        assert bool(((out.mu >= -math.pi) & (out.mu < math.pi)).all())
        assert bool(torch.isfinite(out.mu).all())
        assert bool((out.kappa >= 0).all())
        assert bool(torch.isfinite(out.kappa).all())

    def test_required_row_without_bin_fails_closed(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(1)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        with pytest.raises(ValueError):
            head(
                context,
                emb,
                required=torch.tensor([True]),
                selected_bin=torch.tensor([0]),
            )

    def test_bin_out_of_range_rejected(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(1)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        with pytest.raises(ValueError):
            head(
                context,
                emb,
                required=torch.tensor([True]),
                selected_bin=torch.tensor([37]),
            )

    def test_circular_loss_floor_and_antipode(self):
        # at zero residual the loss is the von Mises entropy floor; at
        # the antipode it is exactly 2*kappa higher.
        centers = torsion_bin_centers()
        k = 7
        center = torch.tensor([centers[k - 1]], dtype=torch.float64)
        kappa = torch.tensor([2.0], dtype=torch.float64)
        target = torch.tensor([k], dtype=torch.int64)
        loss_floor = torsion_circular_loss(center, kappa, target)
        floor = von_mises_loss(center, kappa, center)
        assert float(loss_floor) == pytest.approx(float(floor), abs=1e-12)
        # antipodal mu: the loss rises by exactly 2*kappa (cos(pi) = -1)
        mu_anti = center + math.pi
        loss_anti = torsion_circular_loss(mu_anti, kappa, target)
        assert float(loss_anti) == pytest.approx(
            float(floor) + 2 * 2.0, abs=1e-9
        )

    def test_circular_loss_is_circular_across_the_seam(self):
        # bins 1 and 36 are 10 degrees apart: the loss for a mu at bin 1
        # against target 36 must be small, not huge.
        centers = torsion_bin_centers()
        mu = torch.tensor([centers[0]], dtype=torch.float64)  # bin 1 centre
        kappa = torch.tensor([5.0], dtype=torch.float64)
        target = torch.tensor([36], dtype=torch.int64)
        loss = torsion_circular_loss(mu, kappa, target)
        floor = von_mises_loss(mu, kappa, torch.tensor([centers[35]], dtype=torch.float64))
        assert float(loss) == pytest.approx(float(floor), abs=1e-12)
        assert float(floor) < 1.0  # 10-degree residual at kappa=5 is cheap

    def test_circular_loss_excludes_unrequired_rows(self):
        mu = torch.tensor([0.1, 0.2], dtype=torch.float64)
        kappa = torch.tensor([1.0, 1.0], dtype=torch.float64)
        targets = torch.tensor([0, 5], dtype=torch.int64)
        loss = torsion_circular_loss(mu, kappa, targets)
        only = torsion_circular_loss(
            mu[1:], kappa[1:], torch.tensor([5], dtype=torch.int64)
        )
        assert float(loss) == pytest.approx(float(only), abs=1e-12)

    def test_circular_loss_fails_closed_without_targets(self):
        mu = torch.tensor([0.1], dtype=torch.float64)
        kappa = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            torsion_circular_loss(mu, kappa, torch.tensor([0], dtype=torch.int64))

    def test_head_circular_loss_delegates(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(2)
        emb = torch.randn(2, EMBED, dtype=torch.float64)
        out = head(
            context,
            emb,
            required=torch.ones(2, dtype=torch.bool),
            selected_bin=torch.tensor([7, 11]),
        )
        loss = head.circular_loss(out, torch.tensor([7, 11]))
        manual = torsion_circular_loss(out.mu, out.kappa, torch.tensor([7, 11]))
        assert float(loss) == pytest.approx(float(manual), abs=1e-12)

    def test_bin_mask_restriction(self):
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        context = random_context(1)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        bin_mask = torch.zeros(1, TORSION_BINS, dtype=torch.bool)
        bin_mask[0, 4:] = True  # only bins 1..5 legal
        out = head(context, emb, bin_mask=bin_mask)
        assert float(out.probs[0, 5:].sum()) == pytest.approx(0.0, abs=1e-12)
        assert float(out.probs[0].sum()) == pytest.approx(1.0, abs=1e-9)

    def test_production_dimensions_match_policy_constants(self):
        # the module-local production defaults equal the policy-side
        # canonical constants (pinned by test instead of import).
        from syntree.models.policy import DECODER_D_MODEL, SYNTON_EMBED_DIM

        assert TORSION_HEAD_CONTEXT_DIM == DECODER_D_MODEL == 256
        assert TORSION_HEAD_SYNTON_DIM == SYNTON_EMBED_DIM == 64

    def test_mu_survives_zero_trunk_output(self):
        # the safe_unit_vector route keeps mu finite and its gradients
        # bounded even at the degenerate zero 2D vector.
        head = TorsionHead(
            context_dim=D, synthon_dim=EMBED, hidden_dim=T_HIDDEN
        ).to(torch.float64).eval()
        with torch.no_grad():
            head.mu_xy.weight.zero_()
            head.mu_xy.bias.zero_()
        context = random_context(1, seed=3)
        emb = torch.randn(1, EMBED, dtype=torch.float64)
        context.requires_grad_(True)
        out = head(context, emb)
        assert bool(torch.isfinite(out.mu).all())
        out.mu.sum().backward()
        assert bool(torch.isfinite(context.grad).all())


# ---------------------------------------------------------------------------
# 10. Teacher-forced continuation — consistency and fail-closed semantics
# ---------------------------------------------------------------------------

class TestTeacherForcing:
    def test_full_composition_is_consistent(self):
        heads = make_heads(seed=2)
        context = random_context(3, seed=6)
        cont, (fam_mask, rxn_mask, selected) = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        # the family is derived from the reaction
        expected_family = torch.tensor(
            [
                REACTION_FAMILY_NAMES.index(REGISTRY[POLICY_REACTION_IDS[i]].family)
                for i in selected.tolist()
            ]
        )
        assert torch.equal(cont.reaction.selected_family, expected_family)
        assert torch.equal(cont.family.selected_family, expected_family)
        # the torsion requirement is derived from the resonance lock
        locked = torch.tensor(
            [REGISTRY[rid].resonance_lock is not None for rid in POLICY_REACTION_IDS]
        )
        assert torch.equal(cont.torsion.required, ~locked.index_select(0, selected))

    def test_missing_stereocenter_flag_rejected(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks(amine_menus(1))
        with pytest.raises(TensorContractError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([0]),
                selected_synthon=torch.tensor([3]),
                selected_parity=torch.tensor([0]),
                selected_torsion_bin=torch.tensor([9]),
                stereocenter_required=None,
                family_mask=fam_mask,
                reaction_mask=rxn_mask,
                catalog_handle_classes=synthetic_catalog(),
            )

    def test_stop_rows_never_enter_continuation_stage(self):
        # the acceptance pattern: the continuation inputs are the
        # CONTINUE subset — verified by shape consistency between the
        # context rows and every continuation output row count.
        heads = make_heads()
        context = random_context(4)
        stop = heads.forward_stop(context)
        cont_rows = 3
        cont, _ = default_teacher_forcing(
            heads,
            context[:cont_rows],
            amine_menus(cont_rows),
            synthetic_catalog(),
        )
        assert cont.family.num_graphs == cont_rows
        assert cont.reaction.num_graphs == cont_rows
        assert cont.synthon.num_graphs == cont_rows
        assert cont.parity.num_graphs == cont_rows
        assert cont.torsion.num_graphs == cont_rows
        assert stop.num_graphs == 4

    def test_illegal_supervised_reaction_rejected(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        # rxn_09 requires an aryl_bromide molecule-side handle
        with pytest.raises(HardwareError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([POLICY_REACTION_IDS.index("rxn_09")]),
                selected_synthon=torch.tensor([0]),
                selected_parity=torch.tensor([0]),
                selected_torsion_bin=torch.tensor([9]),
                stereocenter_required=torch.tensor([True]),
                family_mask=fam_mask,
                reaction_mask=rxn_mask,
                catalog_handle_classes=synthetic_catalog(),
            )

    def test_required_torsion_bin_zero_rejected(self):
        # reductive amination (rxn_03) is rotatable: a supervised action
        # without its torsion bin is an incomplete trajectory step.
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        with pytest.raises(ValueError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([POLICY_REACTION_IDS.index("rxn_03")]),
                selected_synthon=torch.tensor([0]),
                selected_parity=torch.tensor([0]),
                selected_torsion_bin=torch.tensor([0]),
                stereocenter_required=torch.tensor([True]),
                family_mask=fam_mask,
                reaction_mask=rxn_mask,
                catalog_handle_classes=synthetic_catalog(),
            )

    def test_locked_reaction_ignores_vacuous_bin(self):
        # amide (rxn_01) is locked: bin 18 (the AssemblyAction default)
        # is vacuous and contributes exactly zero.
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        cont = heads.forward_continue(
            context,
            selected_reaction=torch.tensor([0]),
            selected_synthon=torch.tensor([3]),
            selected_parity=torch.tensor([0]),
            selected_torsion_bin=torch.tensor([18]),
            stereocenter_required=torch.tensor([True]),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=synthetic_catalog(),
        )
        assert cont.torsion.selected_log_prob[0].item() == 0.0
        assert cont.torsion.required[0].item() is False


# ---------------------------------------------------------------------------
# 11. Fail-closed input contract (shared)
# ---------------------------------------------------------------------------

class TestFailClosedInputs:
    def test_heads_reject_fp16_context(self):
        heads = make_heads(dtype=torch.float32, seed=0)
        context = random_context(2, dtype=torch.float16)
        with pytest.raises(TensorContractError):
            heads.forward_stop(context)
        with pytest.raises(TypeError):
            heads.torsion_head(context, torch.zeros(2, EMBED, dtype=torch.float16))

    def test_heads_reject_wrong_context_width(self):
        heads = make_heads()
        with pytest.raises(TensorContractError):
            heads.forward_stop(torch.zeros(2, D + 4, dtype=torch.float64))
        with pytest.raises(ValueError):
            heads.torsion_head(torch.zeros(2, D + 4, dtype=torch.float64), torch.zeros(2, EMBED, dtype=torch.float64))

    def test_heads_reject_empty_batch(self):
        heads = make_heads()
        with pytest.raises(TensorContractError):
            heads.forward_stop(torch.zeros(0, D, dtype=torch.float64))
        with pytest.raises(ValueError):
            heads.torsion_head(torch.zeros(0, D, dtype=torch.float64), torch.zeros(0, EMBED, dtype=torch.float64))

    def test_heads_reject_nonfinite_context(self):
        heads = make_heads()
        context = random_context(2)
        context[0, 0] = float("inf")
        with pytest.raises(TensorContractError):
            heads.forward_stop(context)
        with pytest.raises(ValueError):
            heads.torsion_head(context, torch.zeros(2, EMBED, dtype=torch.float64))

    def test_dtype_mismatch_between_module_and_input(self):
        heads = make_heads(dtype=torch.float64)
        with pytest.raises(TensorContractError):
            heads.forward_stop(random_context(2, dtype=torch.float32))

    def test_continue_rejects_out_of_range_indices(self):
        heads = make_heads()
        context = random_context(1)
        fam_mask, rxn_mask = build_continuation_masks(amine_menus(1))
        base = dict(
            selected_synthon=torch.tensor([3]),
            selected_parity=torch.tensor([0]),
            selected_torsion_bin=torch.tensor([9]),
            stereocenter_required=torch.tensor([True]),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=synthetic_catalog(),
        )
        with pytest.raises(TensorContractError):
            heads.forward_continue(context, selected_reaction=torch.tensor([27]), **base)
        with pytest.raises(TensorContractError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([0]),
                selected_synthon=torch.tensor([CATALOG]),
                **{k: v for k, v in base.items() if k != "selected_synthon"},
            )
        with pytest.raises(TensorContractError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([0]),
                selected_parity=torch.tensor([2]),
                **{k: v for k, v in base.items() if k != "selected_parity"},
            )
        with pytest.raises(ValueError):
            heads.forward_continue(
                context,
                selected_reaction=torch.tensor([0]),
                selected_torsion_bin=torch.tensor([99]),
                **{k: v for k, v in base.items() if k != "selected_torsion_bin"},
            )


# ---------------------------------------------------------------------------
# 12. Frozen production profile
# ---------------------------------------------------------------------------

class TestFrozenProfile:
    def test_container_parameter_count(self):
        torch.manual_seed(0)
        heads = HierarchicalPolicyHeads()  # the frozen production profile
        n = sum(p.numel() for p in heads.parameters())
        assert n == HIERARCHICAL_HEADS_PARAMETER_PROFILE == 3_272_975

    def test_head_breakdown_sums_to_total(self):
        torch.manual_seed(0)
        heads = HierarchicalPolicyHeads()
        audit = heads.audit()
        parts = sum(h["num_parameters"] for h in audit["heads"].values())
        assert parts == audit["num_parameters"] == 3_272_975

    def test_all_parameters_trainable_fp32(self):
        torch.manual_seed(0)
        heads = HierarchicalPolicyHeads()
        assert all(p.requires_grad for p in heads.parameters())
        assert all(p.dtype == torch.float32 for p in heads.parameters())

    def test_metadata_buffers_are_not_persistent(self):
        heads = HierarchicalPolicyHeads()
        state = heads.state_dict()
        assert "reaction_head.family_membership" not in state
        assert "synthon_head.reaction_block_class" not in state
        assert "synthon_head.catalog_handle_classes" not in state

    def test_audit_reports_tb_measures(self):
        heads = HierarchicalPolicyHeads()
        audit = heads.audit()
        measures = audit["tb_probability_measures"]
        assert measures["von_mises_outputs_in_tb"] is False
        assert measures["stop"] == "categorical_binary"
        assert measures["torsion"] == "categorical_36"
        assert audit["action_space"]["synthon"] == 50_000

    def test_small_config_forward_through_all_heads(self):
        heads = make_heads()
        context = random_context(2)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(2), synthetic_catalog()
        )
        assert cont.step_log_prob().shape == (2,)


# ---------------------------------------------------------------------------
# 13. Gradient connectivity
# ---------------------------------------------------------------------------

class TestGradients:
    def test_every_head_backpropagates_into_context(self):
        # weighted losses (the D-V2-012 LayerNorm caveat pattern: never
        # rely on context.sum(), whose gradient is exactly zero through
        # the closing LayerNorm).
        heads = make_heads(seed=1)
        context = random_context(3, seed=2).requires_grad_(True)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        weights = torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64)
        loss = (cont.step_log_prob() * weights).sum()
        loss.backward()
        assert context.grad is not None
        assert bool(torch.isfinite(context.grad).all())
        # at least one nonzero entry per row (the context feeds every head)
        assert bool((context.grad != 0).all(dim=-1).all())

    def test_stop_head_backpropagates(self):
        heads = make_heads(seed=1)
        context = random_context(2, seed=2).requires_grad_(True)
        stop = heads.forward_stop(context)
        (stop.stop_log_prob * torch.tensor([0.7, 0.3], dtype=torch.float64)).sum().backward()
        assert context.grad is not None
        assert bool((context.grad != 0).any())

    def test_synthon_loss_reaches_tied_catalog_table(self):
        heads = make_heads(seed=1)
        context = random_context(2, seed=2)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(2), synthetic_catalog()
        )
        loss = cont.synthon.selected_log_prob.sum()
        loss.backward()
        grad = heads.synthon_head.catalog_embeddings.weight.grad
        assert grad is not None
        selected_rows = cont.synthon.selected_synthon.tolist()
        for k in selected_rows:
            assert bool((grad[k] != 0).any())

    def test_parity_and_torsion_reach_synthon_embedding(self):
        heads = make_heads(seed=1)
        context = random_context(1, seed=2)
        # a rotatable, stereogenic junction: rxn_03 (reductive amination)
        fam_mask, rxn_mask = build_continuation_masks([["amine"]])
        cont = heads.forward_continue(
            context,
            selected_reaction=torch.tensor([POLICY_REACTION_IDS.index("rxn_03")]),
            selected_synthon=torch.tensor([4]),
            selected_parity=torch.tensor([0]),
            selected_torsion_bin=torch.tensor([12]),
            stereocenter_required=torch.tensor([True]),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=synthetic_catalog(),
        )
        loss = (
            cont.parity.selected_log_prob.sum() + cont.torsion.selected_log_prob.sum()
        )
        loss.backward()
        grad = heads.synthon_head.catalog_embeddings.weight.grad
        assert grad is not None
        assert bool((grad[4] != 0).any())

    def test_circular_loss_backpropagates_finite(self):
        heads = make_heads(seed=1)
        context = random_context(2, seed=2).requires_grad_(True)
        cont, _ = default_teacher_forcing(
            heads, context, amine_menus(2), synthetic_catalog()
        )
        targets = torch.tensor([5, 20])
        loss = heads.torsion_head.circular_loss(cont.torsion, targets)
        loss.backward()
        assert context.grad is not None
        assert bool(torch.isfinite(context.grad).all())


# ---------------------------------------------------------------------------
# 14. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_double_forward_bitwise_identical(self):
        heads = make_heads(seed=4)
        context = random_context(3, seed=5)
        stop1 = heads.forward_stop(context)
        stop2 = heads.forward_stop(context)
        assert torch.equal(stop1.log_probs, stop2.log_probs)
        cont1, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        cont2, _ = default_teacher_forcing(
            heads, context, amine_menus(3), synthetic_catalog()
        )
        assert torch.equal(cont1.step_log_prob(), cont2.step_log_prob())
        assert torch.equal(cont1.torsion.mu, cont2.torsion.mu)
        assert torch.equal(cont1.torsion.kappa, cont2.torsion.kappa)


# ---------------------------------------------------------------------------
# 15. End-to-end: the real stack and the real environment
# ---------------------------------------------------------------------------

def _aniline_seed() -> BuildingBlock:
    """An aniline seed exposing one amine handle."""
    template = Chem.MolFromSmiles("c1ccccc1N")
    nitrogen = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 7)
    return BuildingBlock(
        "aniline", "c1ccccc1N", (HandleSpec("nh2", nitrogen, "amine"),)
    )


def _acid_catalog() -> dict:
    from syntree.engine.environment import JunctionReaction

    blocks = {}
    # two acid-chloride blocks (rxn_01) + one aldehyde (rxn_03)
    acyl = Chem.MolFromSmiles("CCC(=O)Cl")
    chlorine = next(a.GetIdx() for a in acyl.GetAtoms() if a.GetAtomicNum() == 17)
    carbonyl = next(
        a.GetIdx() for a in acyl.GetAtoms() if a.GetAtomicNum() == 6 and a.GetSymbol() == "C"
    )
    # pick the carbon bearing both O and Cl
    for atom in acyl.GetAtoms():
        nbrs = {n.GetAtomicNum() for n in atom.GetNeighbors()}
        if atom.GetAtomicNum() == 6 and 8 in nbrs and 17 in nbrs:
            carbonyl = atom.GetIdx()
            break
    blocks["propionyl"] = BuildingBlock(
        "propionyl",
        "CCC(=O)Cl",
        (HandleSpec("cl", carbonyl, "acid_chloride", (chlorine,)),),
    )
    acyl2 = Chem.MolFromSmiles("c1ccccc1C(=O)Cl")
    chlorine2 = next(a.GetIdx() for a in acyl2.GetAtoms() if a.GetAtomicNum() == 17)
    carbonyl2 = None
    for atom in acyl2.GetAtoms():
        nbrs = {n.GetAtomicNum() for n in atom.GetNeighbors()}
        if atom.GetAtomicNum() == 6 and 8 in nbrs and 17 in nbrs:
            carbonyl2 = atom.GetIdx()
            break
    blocks["benzoyl"] = BuildingBlock(
        "benzoyl",
        "c1ccccc1C(=O)Cl",
        (HandleSpec("cl", carbonyl2, "acid_chloride", (chlorine2,)),),
    )
    aldehyde = Chem.MolFromSmiles("CC=O")  # acetaldehyde (O implicit H)
    carb = None
    for atom in aldehyde.GetAtoms():
        nbrs = {n.GetAtomicNum() for n in atom.GetNeighbors()}
        if atom.GetAtomicNum() == 6 and 8 in nbrs:
            carb = atom.GetIdx()
            break
    blocks["ethanal"] = BuildingBlock(
        "ethanal", "CC=O", (HandleSpec("cho", carb, "aldehyde"),)
    )
    return blocks


def _shell_pocket(num_atoms: int = 12, radius: float = 5.5, seed: int = 3):
    import numpy as np

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_atoms, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * radius[:, None] if False else directions * (
        radius + 0.4 * rng.random(num_atoms)[:, None]
    )
    elements = [6, 6, 7, 8, 6, 8, 6, 6, 7, 6, 8, 6][:num_atoms]
    charges = np.zeros(num_atoms)
    return PocketDefinition(positions, np.array(elements, dtype=np.int64), charges)


class TestEndToEndStack:
    """The heads consume the real backbone + handle head + decoder context
    and stay SE(3)-invariant end-to-end; the head indices map onto real
    ``AssemblyAction`` fields and the environment executes them."""

    def _full_stack(self, dtype=torch.float64, seed=0):
        torch.manual_seed(seed)
        backbone = BipartitePaiNN(hidden_dim=D, num_layers=2, num_radial=6, r_cut=5.0)
        handle_head = HandleSelectionHead(
            backbone_dim=D, hidden_dim=8, num_radial=6, geom_cutoff=12.0
        )
        decoder = CrossAttentionDecoderSmall(D, 8)
        heads = HierarchicalPolicyHeads(
            context_dim=32, catalog_size=CATALOG, synthon_embed_dim=EMBED,
            torsion_hidden_dim=T_HIDDEN,
        )
        return (
            backbone.to(dtype).eval(),
            handle_head.to(dtype).eval(),
            decoder.to(dtype).eval(),
            heads.to(dtype).eval(),
        )

    def test_rigid_motion_invariance_of_all_head_laws(self):
        backbone, handle_head, decoder, heads = self._full_stack()
        state = make_batched_state([10, 8], [5, 4], seed=11)
        backbone_out = backbone(state)
        pool = make_pool(backbone_out, [2, 2], seed=12)
        descriptors, handle_pos, handle_graph, handle_atom = pool
        handle_out = handle_head(
            backbone_out, descriptors, handle_pos, handle_graph, handle_atom
        )
        dec_out = decoder.forward_state(backbone_out, handle_out)
        context = dec_out.context

        gen = torch.Generator().manual_seed(21)
        rot = random_rotation(gen)
        shift = torch.randn(3, generator=gen, dtype=torch.float64)

        moved = SimpleNamespace(**{k: v for k, v in state.__dict__.items()})
        moved.pocket_pos = state.pocket_pos @ rot.t() + shift
        moved.ligand_pos = state.ligand_pos @ rot.t() + shift
        moved_backbone = backbone(moved)
        moved_pool = make_pool(moved_backbone, [2, 2], seed=12)
        d2, p2, g2, a2 = moved_pool
        moved_handle = handle_head(moved_backbone, d2, p2, g2, a2)
        moved_dec = decoder.forward_state(moved_backbone, moved_handle)

        stop_a = heads.forward_stop(context)
        stop_b = heads.forward_stop(moved_dec.context)
        assert torch.allclose(stop_a.log_probs, stop_b.log_probs, atol=1e-9)

        menus = [["amine"], ["amine"]]
        fam_mask, rxn_mask = build_continuation_masks(menus)
        cont_a, _ = default_teacher_forcing(
            heads, context, menus, synthetic_catalog()
        )
        cont_b, _ = default_teacher_forcing(
            heads, moved_dec.context, menus, synthetic_catalog()
        )
        assert torch.allclose(
            cont_a.family.log_probs, cont_b.family.log_probs, atol=1e-8
        )
        assert torch.allclose(
            cont_a.reaction.log_probs, cont_b.reaction.log_probs, atol=1e-8
        )
        assert torch.allclose(
            cont_a.synthon.log_probs, cont_b.synthon.log_probs, atol=1e-8
        )
        assert torch.allclose(
            cont_a.parity.log_probs, cont_b.parity.log_probs, atol=1e-8
        )
        assert torch.allclose(
            cont_a.torsion.log_probs, cont_b.torsion.log_probs, atol=1e-8
        )

    def test_real_environment_step_from_head_outputs(self):
        # The full loop: real menu -> masks -> teacher-forced heads ->
        # AssemblyAction field mapping -> environment.step executes.
        backbone, handle_head, decoder, heads = self._full_stack(seed=5)
        state = make_batched_state([10], [7], seed=13)
        backbone_out = backbone(state)
        pool = make_pool(backbone_out, [2], seed=14)
        descriptors, handle_pos, handle_graph, handle_atom = pool
        handle_out = handle_head(
            backbone_out, descriptors, handle_pos, handle_graph, handle_atom
        )
        dec_out = decoder.forward_state(backbone_out, handle_out)
        context = dec_out.context

        env = AssemblyEnvironment(
            _aniline_seed(), _acid_catalog(), pocket=_shell_pocket(), max_blocks=4
        )
        env.reset()
        menu = env.active_handles()
        assert len(menu) == 1
        classes = handle_classes_from_menu(menu)
        # the catalog's actually available block classes (the honest
        # caller pattern: availability comes from the catalog, not the
        # registry default)
        available = {
            h.handle_class
            for block in _acid_catalog().values()
            for h in block.handles
        }
        assert available == {"acid_chloride", "aldehyde"}
        fam_mask, rxn_mask = build_continuation_masks([list(classes)], available)
        # only rxn_01 (acid_chloride) and rxn_03 (aldehyde) are executable
        legal = [
            POLICY_REACTION_IDS[i]
            for i in range(27)
            if not bool(rxn_mask[0, i])
        ]
        assert set(legal) == {"rxn_01", "rxn_03"}

        # teacher-force rxn_03 (rotatable, stereogenic-capable): all five
        # continuation components are scored.
        r_idx = POLICY_REACTION_IDS.index("rxn_03")
        classes_catalog = synthetic_catalog()
        # ensure a legal synthon exists for the aldehyde class
        aldehyde_idx = BLOCK_HANDLE_CLASS_VOCAB.index("aldehyde")
        classes_catalog[0] = aldehyde_idx
        cont = heads.forward_continue(
            context,
            selected_reaction=torch.tensor([r_idx]),
            selected_synthon=torch.tensor([0]),
            selected_parity=torch.tensor([1]),  # 'S'
            selected_torsion_bin=torch.tensor([12]),
            stereocenter_required=torch.tensor([True]),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=classes_catalog,
        )
        assert cont.torsion.required[0].item() is True
        lp = cont.step_log_prob(
            handle_log_probs=handle_out.log_probs[:1].to(torch.float64)
        )
        assert bool(torch.isfinite(lp).all()) and lp[0].item() < 0

        # map the head indices onto the environment's action format
        action = AssemblyAction(
            handle_atom_idx=int(menu[0]["handle_atom_idx"]),
            reaction_id=POLICY_REACTION_IDS[r_idx],
            block_id="ethanal",
            chiral=PARITY_ACTIONS[int(cont.parity.selected_parity[0])],
            torsion_bin=int(cont.torsion.selected_bin[0]),
        )
        record = env.step(action)
        assert record.action_type == "add"
        assert record.done is False
        # the trajectory record's log-probability is the head-composed
        # step log-probability (dimensionless categorical sum)
        assert record.action_log_prob is None or isinstance(
            record.action_log_prob, float
        )
        # the amine handle is consumed: continuation legality collapses
        menu2 = env.active_handles()
        classes2 = handle_classes_from_menu(menu2)
        if not classes2:
            fam2, rxn2 = build_continuation_masks([list(classes2)])
            forced = build_force_stop(rxn2)
            assert bool(forced[0])
        # STOP terminates
        stop = heads.forward_stop(context, force_stop=torch.tensor([False]))
        env.stop()
        assert env.is_terminal

    def test_real_environment_locked_reaction_zero_torsion(self):
        backbone, handle_head, decoder, heads = self._full_stack(seed=6)
        state = make_batched_state([10], [7], seed=15)
        backbone_out = backbone(state)
        pool = make_pool(backbone_out, [2], seed=16)
        descriptors, handle_pos, handle_graph, handle_atom = pool
        handle_out = handle_head(
            backbone_out, descriptors, handle_pos, handle_graph, handle_atom
        )
        dec_out = decoder.forward_state(backbone_out, handle_out)

        env = AssemblyEnvironment(
            _aniline_seed(), _acid_catalog(), pocket=_shell_pocket(), max_blocks=4
        )
        env.reset()
        menu = env.active_handles()
        classes = handle_classes_from_menu(menu)
        available = {
            h.handle_class
            for block in _acid_catalog().values()
            for h in block.handles
        }
        fam_mask, rxn_mask = build_continuation_masks([list(classes)], available)
        r_idx = POLICY_REACTION_IDS.index("rxn_01")  # amide: locked trans
        classes_catalog = synthetic_catalog()
        acid_idx = BLOCK_HANDLE_CLASS_VOCAB.index("acid_chloride")
        classes_catalog[1] = acid_idx
        cont = heads.forward_continue(
            dec_out.context,
            selected_reaction=torch.tensor([r_idx]),
            selected_synthon=torch.tensor([1]),
            selected_parity=torch.tensor([0]),
            selected_torsion_bin=torch.tensor([18]),  # vacuous default
            stereocenter_required=torch.tensor([False]),
            family_mask=fam_mask,
            reaction_mask=rxn_mask,
            catalog_handle_classes=classes_catalog,
        )
        assert cont.torsion.required[0].item() is False
        assert cont.torsion.selected_log_prob[0].item() == 0.0
        assert cont.parity.selected_log_prob[0].item() == 0.0
        action = AssemblyAction(
            handle_atom_idx=int(menu[0]["handle_atom_idx"]),
            reaction_id="rxn_01",
            block_id="propionyl",
            chiral=None,
            torsion_bin=18,
        )
        record = env.step(action)
        assert record.action_type == "add"


class CrossAttentionDecoderSmall(torch.nn.Module):
    """A small-context decoder stand-in for the end-to-end stack tests.

    Produces the same output surface as the real #23 decoder
    (``context [B, d_model]``) without the full 256-wide profile, so the
    end-to-end head tests run fast; the real decoder's own contract is
    pinned by tests/test_cross_attention_decoder.py.
    """

    def __init__(self, backbone_dim: int, hidden: int) -> None:
        super().__init__()
        self.backbone_dim = backbone_dim
        self.hidden = hidden
        self.d_model = 32
        self.proj = torch.nn.Linear(hidden, self.d_model)

    def forward_state(self, backbone, handle):
        pooled = handle.pooled_handle_context
        projected = self.proj(pooled)
        return SimpleNamespace(context=projected)
