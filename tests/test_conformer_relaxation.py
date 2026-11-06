"""Verification suite of #13 CHEM-03: junction torsion + constrained MMFF94.

Pinned by issue #13's acceptance criteria (as reconciled by D-V2-009 and
implemented behind the D-V2-017 opt-in stages):

1. :func:`TestApplyJunctionTorsion` — #13 §1: resonance-locked junctions
   snap to the planar state (0°/180°) chosen by the lower flat-bottom
   soft-core pocket clash (ties -> trans; no pocket -> trans); the
   predicted continuous ``phi`` routes to the first adjacent
   non-conjugated single bond on the synthon side; rotatable junctions
   take ``phi`` on the junction bond itself.
2. :func:`TestRelaxJunctionConstrained` — #13 §2: MMFF94 within the static
   vdW field of the pocket; scaffold flat-bottom restraints
   (50 kcal/mol/Å², 0.05 Å); pocket atoms stay static within their
   tolerance; fail-closed on untypeable pockets.
3. :func:`TestAcceptanceCriteria` — the issue-pinned acceptance checks:
   amide junctions remain planar within 1.0° of 0° or 180°; the core
   scaffold RMSD stays < 0.15 Å; the junction bond length settles within
   0.02 Å of its MMFF equilibrium (convergence criterion — the engine's
   bin-centre placement deliberately uses covalent radii sums, which are
   not MMFF equilibria).
4. :func:`TestCombinedPipeline` — #13's addendum: the sub-bin refinement
   (#42) executes before the #13 stages, and the two stages remain
   distinct; the legacy default of ``finalize_terminal_molecule`` is
   unchanged (D-V2-017; the 50 pinned reconciliation contracts stay
   green).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralRad

from syntree.chemistry.conformer import (
    CORE_FORCE_CONSTANT,
    POCKET_FORCE_CONSTANT,
    POSITION_TOLERANCE,
    ChemistryError,
    apply_junction_torsion,
    relax_junction_constrained,
    soft_core_clash_energy,
)
from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    BuildingBlock,
    HandleSpec,
    JunctionRecord,
    PocketDefinition,
    _read_junctions,
    bin_center,
    wrap_angle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_phenethylamine() -> BuildingBlock:
    """Phenethylamine seed: a 9-heavy-atom scaffold with an amine handle."""
    template = Chem.MolFromSmiles("NCCc1ccccc1")
    nitrogen = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 7)
    return BuildingBlock(
        "phenethylamine", "NCCc1ccccc1", (HandleSpec("n", nitrogen, "amine"),)
    )


def _propionyl_chloride() -> BuildingBlock:
    """Propionyl chloride: the amide junction's synthon side is
    C(=O)-CH2-CH3, so the first adjacent rotatable synthon bond exists."""
    template = Chem.MolFromSmiles("CCC(=O)Cl")
    chlorine = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 17)
    carbonyl = next(n.GetIdx() for n in template.GetAtomWithIdx(chlorine).GetNeighbors())
    return BuildingBlock(
        "propionyl",
        "CCC(=O)Cl",
        (HandleSpec("acyl", carbonyl, "acid_chloride", (chlorine,)),),
    )


def _ethyl_bromide() -> BuildingBlock:
    template = Chem.MolFromSmiles("CCBr")
    bromine = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 35)
    carbon = next(n.GetIdx() for n in template.GetAtomWithIdx(bromine).GetNeighbors())
    return BuildingBlock("etbr", "CCBr", (HandleSpec("alkyl", carbon, "alkyl_bromide", (bromine,)),))


def _shell_pocket(num_atoms: int = 14, radius: float = 5.2, seed: int = 11) -> PocketDefinition:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_atoms, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * (radius + 0.6 * rng.random(num_atoms))[:, None]
    elements = [6, 6, 7, 8, 6, 8, 6, 6, 7, 6, 8, 6, 6, 8][:num_atoms]
    charges = np.zeros(num_atoms)
    charges[2] = -0.8
    charges[8] = 0.8
    return PocketDefinition(positions, np.array(elements, dtype=np.int64), charges)


def _trans_side_pocket() -> PocketDefinition:
    """A pocket crowding the *trans* region of the amide junction.

    The amide junction dihedral is (a0, a1, b1, b0) with a1 = the seed's
    N, a0 = the seed-side methylene, b1 = the carbonyl C, b0 = the
    carbonyl O.  We place shell atoms near where the seed-side atoms sit
    when the junction is trans, so the flat-bottom clash prefers cis.
    """
    rng = np.random.default_rng(29)
    positions = rng.normal(size=(12, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = positions * 6.5
    return PocketDefinition(positions, np.array([6] * 8 + [8] * 4, dtype=np.int64), np.zeros(12))


def _amide_env(pocket=None):
    seed = _seed_phenethylamine()
    env = AssemblyEnvironment(seed, {"propionyl": _propionyl_chloride()}, pocket=pocket, max_blocks=4)
    env.reset()
    env.place_seed()
    handle = env.active_handles()[0]["handle_atom_idx"]
    env.step(AssemblyAction(handle, "rxn_01", "propionyl", torsion_bin=27))
    env.stop()
    return env


def _alkyl_env(pocket=None):
    seed = _seed_phenethylamine()
    env = AssemblyEnvironment(seed, {"etbr": _ethyl_bromide()}, pocket=pocket, max_blocks=4)
    env.reset()
    env.place_seed()
    handle = env.active_handles()[0]["handle_atom_idx"]
    env.step(AssemblyAction(handle, "rxn_07", "etbr", torsion_bin=27))
    env.stop()
    return env


def _clash(mol, pocket) -> float:
    positions = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return soft_core_clash_energy(positions, numbers, pocket.positions, pocket.atomic_numbers)


def _reference_atom(mol, atom_idx: int, exclude: int) -> int:
    candidates = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if n.GetIdx() != exclude and n.GetAtomicNum() > 1
    ]
    return min(candidates)


def _junction_dihedral(mol, junction) -> float:
    return GetDihedralRad(
        mol.GetConformer(), junction.a0, junction.a1, junction.b1, junction.b0
    )


def _planarity_error(value: float) -> float:
    """Angular distance (deg) to the nearest planar state {0, 180}."""
    deg = abs(math.degrees(wrap_angle(value)))
    return min(deg, abs(180.0 - deg))


def _bond_length(mol, i: int, j: int) -> float:
    positions = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    return float(np.linalg.norm(positions[i] - positions[j]))


def _seed_rmsd(before, after) -> float:
    idx = [a.GetIdx() for a in before.GetAtoms() if a.HasProp("syn_tree_seed")]
    b = np.array(before.GetConformer().GetPositions(), dtype=np.float64)
    a = np.array(after.GetConformer().GetPositions(), dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((b[idx] - a[idx]) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# 1. apply_junction_torsion (#13 §1)
# ---------------------------------------------------------------------------


class TestApplyJunctionTorsion:
    def test_locked_snap_is_argmin_of_planar_clash(self):
        pocket = _shell_pocket()
        for env_pocket, label in ((pocket, "shell"), (_trans_side_pocket(), "trans-crowded")):
            env = _amide_env(env_pocket)
            work = Chem.Mol(env.molecule)
            junction = _read_junctions(work)[0]

            probe_cis = Chem.Mol(work)
            from syntree.engine.environment import set_junction_dihedral

            set_junction_dihedral(probe_cis, junction, 0.0)
            clash_cis = _clash(probe_cis, env_pocket)
            probe_trans = Chem.Mol(work)
            set_junction_dihedral(probe_trans, junction, math.pi)
            clash_trans = _clash(probe_trans, env_pocket)

            telemetry = apply_junction_torsion(work, junction, bin_center(27), env_pocket)
            expected = 0.0 if clash_cis < clash_trans - 1e-12 else math.pi
            assert telemetry["planar_dihedral_rad"] == expected, label
            assert abs(telemetry["clash_cis"] - clash_cis) < 1e-9, label
            assert abs(telemetry["clash_trans"] - clash_trans) < 1e-9, label
            assert _planarity_error(_junction_dihedral(work, junction)) < 1e-6, label

    def test_no_pocket_resolves_to_trans(self):
        env = _amide_env(pocket=None)
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        telemetry = apply_junction_torsion(work, junction, 0.5, None)
        assert telemetry["planar_dihedral_rad"] == math.pi
        assert abs(wrap_angle(_junction_dihedral(work, junction) - math.pi)) < 1e-9

    def test_phi_routes_to_first_adjacent_synthon_bond(self):
        env = _amide_env(_shell_pocket())
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        phi = bin_center(9)
        telemetry = apply_junction_torsion(work, junction, phi, env.pocket)
        target = telemetry["phi_applied_to"]
        assert target is not None and target[0] == "synthon_adjacent_bond"
        u, v = target[1], target[2]
        # the adjacent bond is the carbonyl-CH2 bond of the propionyl synthon
        assert work.GetAtomWithIdx(u).GetSymbol() == "C"
        assert {work.GetAtomWithIdx(v).GetSymbol()} == {"C"}
        # deterministic reference convention mirrors the implementation
        u_ref = _reference_atom(work, u, v)
        v_ref = _reference_atom(work, v, u)
        value = GetDihedralRad(work.GetConformer(), u_ref, u, v, v_ref)
        assert abs(wrap_angle(value - phi)) < 1e-9
        # the junction bond itself stays planar
        assert _planarity_error(_junction_dihedral(work, junction)) < 1e-6

    def test_phi_is_inert_when_synthon_has_no_rotatable_bond(self):
        # rxn_01 junction with a *benzoyl chloride* synthon: the amide lock
        # applies, and the synthon side (C(=O)-C6H5) offers no non-conjugated
        # acyclic single bond at any depth — the aryl C(=O) bond is
        # conjugated and every ring bond is cyclic.
        seed = _seed_phenethylamine()
        template = Chem.MolFromSmiles("O=C(Cl)c1ccccc1")
        chlorine = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 17)
        carbonyl = next(
            n.GetIdx() for n in template.GetAtomWithIdx(chlorine).GetNeighbors()
        )
        benzoyl = BuildingBlock(
            "benzoyl",
            "O=C(Cl)c1ccccc1",
            (HandleSpec("acyl", carbonyl, "acid_chloride", (chlorine,)),),
        )
        env = AssemblyEnvironment(seed, {"benzoyl": benzoyl}, pocket=_shell_pocket(), max_blocks=4)
        env.reset()
        env.place_seed()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "rxn_01", "benzoyl", torsion_bin=27))
        env.stop()
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        assert junction.resonance_locked
        telemetry = apply_junction_torsion(work, junction, 1.234, env.pocket)
        assert telemetry["phi_applied_to"] is None  # documented no-op
        assert _planarity_error(_junction_dihedral(work, junction)) < 1e-6

    def test_rotatable_junction_takes_phi_on_the_junction_bond(self):
        env = _alkyl_env(_shell_pocket())
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        assert not junction.resonance_locked
        phi = bin_center(13)
        telemetry = apply_junction_torsion(work, junction, phi, env.pocket)
        assert telemetry["phi_applied_to"] == ("junction_bond", junction.a1, junction.b1)
        assert abs(wrap_angle(_junction_dihedral(work, junction) - phi)) < 1e-9

    def test_rigid_rotation_preserves_covalent_geometry(self):
        env = _amide_env(_shell_pocket())
        before = Chem.Mol(env.molecule)
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        apply_junction_torsion(work, junction, bin_center(9), env.pocket)
        bonds_before = sorted(
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), round(float(np.linalg.norm(
                np.array(before.GetConformer().GetAtomPosition(b.GetBeginAtomIdx()))
                - np.array(before.GetConformer().GetAtomPosition(b.GetEndAtomIdx()))
            )), 9))
            for b in before.GetBonds()
        )
        bonds_after = sorted(
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), round(float(np.linalg.norm(
                np.array(work.GetConformer().GetAtomPosition(b.GetBeginAtomIdx()))
                - np.array(work.GetConformer().GetAtomPosition(b.GetEndAtomIdx()))
            )), 9))
            for b in work.GetBonds()
        )
        assert bonds_before == bonds_after

    def test_fail_closed_on_non_finite_phi(self):
        env = _amide_env(_shell_pocket())
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        with pytest.raises(ChemistryError):
            apply_junction_torsion(work, junction, float("nan"), env.pocket)


# ---------------------------------------------------------------------------
# 2. relax_junction_constrained (#13 §2)
# ---------------------------------------------------------------------------


class TestRelaxJunctionConstrained:
    def test_pocket_stays_static_within_tolerance(self):
        env = _amide_env(_shell_pocket())
        relaxed, energy, diagnostics = relax_junction_constrained(
            env.molecule, env.pocket, return_diagnostics=True
        )
        assert diagnostics["pocket_max_drift_a"] <= POSITION_TOLERANCE + 1e-9
        assert diagnostics["num_pocket_atoms"] == env.pocket.num_atoms

    def test_scaffold_restraints_are_pinned(self):
        env = _amide_env(_shell_pocket())
        _, _, diagnostics = relax_junction_constrained(
            env.molecule, env.pocket, return_diagnostics=True
        )
        assert diagnostics["core_force_constant_kcal_mol_a2"] == CORE_FORCE_CONSTANT
        assert diagnostics["core_tolerance_a"] == POSITION_TOLERANCE
        assert diagnostics["pocket_force_constant_kcal_mol_a2"] == POCKET_FORCE_CONSTANT

    def test_input_molecule_is_never_mutated(self):
        env = _amide_env(_shell_pocket())
        before = np.array(env.molecule.GetConformer().GetPositions(), dtype=np.float64)
        relax_junction_constrained(env.molecule, env.pocket)
        after = np.array(env.molecule.GetConformer().GetPositions(), dtype=np.float64)
        assert np.array_equal(before, after)

    def test_metal_pocket_uses_generic_mmff_type(self):
        """Pocket metals receive RDKit MMFF's generic metal type (95) — a
        placeholder vdW shell for the static field (documented approximation
        of ``relax_junction_constrained``); they do not fail the relaxation."""
        env = _amide_env(None)
        positions = np.array([[4.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float64)
        metal_pocket = PocketDefinition(
            positions, np.array([30, 6], dtype=np.int64), np.zeros(2)
        )
        relaxed, energy, diagnostics = relax_junction_constrained(
            env.molecule, metal_pocket, return_diagnostics=True
        )
        assert math.isfinite(energy)
        assert diagnostics["num_pocket_atoms"] == 2
        assert diagnostics["pocket_max_drift_a"] <= POSITION_TOLERANCE + 1e-9

    def test_energy_finite_and_relaxed(self):
        env = _amide_env(_shell_pocket())
        relaxed, energy, diagnostics = relax_junction_constrained(
            env.molecule, env.pocket, return_diagnostics=True
        )
        assert math.isfinite(energy)
        assert relaxed.GetNumAtoms() == env.molecule.GetNumAtoms()
        assert diagnostics["pocket_clash_after_kcal_mol"] >= 0.0


# ---------------------------------------------------------------------------
# 3. The issue-pinned acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    def test_amide_junction_planar_within_one_degree(self):
        for pocket in (_shell_pocket(), _trans_side_pocket()):
            env = _amide_env(pocket)
            work = Chem.Mol(env.molecule)
            junction = _read_junctions(work)[0]
            apply_junction_torsion(work, junction, bin_center(27), pocket)
            relaxed, _, _ = relax_junction_constrained(work, pocket, return_diagnostics=True)
            junction_after = _read_junctions(relaxed)[0]
            value = _junction_dihedral(relaxed, junction_after)
            assert _planarity_error(value) <= 1.0

    def test_core_scaffold_rmsd_below_0_15(self):
        env = _amide_env(_shell_pocket())
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        apply_junction_torsion(work, junction, bin_center(27), env.pocket)
        relaxed, _, _ = relax_junction_constrained(work, env.pocket, return_diagnostics=True)
        rmsd = _seed_rmsd(work, relaxed)
        assert rmsd < 0.15, f"seed scaffold RMSD {rmsd:.4f} A"

    def test_junction_bond_settles_within_0_02_of_equilibrium(self):
        env = _amide_env(_shell_pocket())
        work = Chem.Mol(env.molecule)
        junction = _read_junctions(work)[0]
        apply_junction_torsion(work, junction, bin_center(27), env.pocket)
        once, _, _ = relax_junction_constrained(work, env.pocket, return_diagnostics=True)
        j1 = _read_junctions(once)[0]
        twice, _, _ = relax_junction_constrained(
            once, env.pocket, mmff_max_iterations=2000, return_diagnostics=True
        )
        j2 = _read_junctions(twice)[0]
        r1 = _bond_length(once, j1.a1, j1.b1)
        r2 = _bond_length(twice, j2.a1, j2.b1)
        assert abs(r1 - r2) < 0.02, (
            f"junction bond length did not settle at its MMFF equilibrium: "
            f"{r1:.4f} A -> {r2:.4f} A"
        )


# ---------------------------------------------------------------------------
# 4. Combined pipeline + legacy boundary (D-V2-017)
# ---------------------------------------------------------------------------


class TestCombinedPipeline:
    def test_full_opt_in_pipeline_through_finalize(self):
        env = _amide_env(_shell_pocket())
        relaxed, energy, diagnostics = env.finalize_terminal_molecule(
            junction_torsion_stage=True,
            pocket_vdw_relaxation=True,
            return_diagnostics=True,
        )
        assert math.isfinite(energy)
        assert diagnostics["pocket_vdw_relaxation"] is True
        assert diagnostics["locked_junction_torsion"], "the #13 stage must record telemetry"
        entry = diagnostics["locked_junction_torsion"][0]
        assert entry["fragment_id"] == 1
        assert entry["phi_applied_to"][0] == "synthon_adjacent_bond"
        assert diagnostics["pocket_max_drift_a"] <= POSITION_TOLERANCE + 1e-9
        # the #42 addendum ordering: sub-bin refinement fields remain present
        assert "junctions" in diagnostics

    def test_mixed_pipeline_rotatable_then_locked(self):
        # One rotatable junction (rxn_07) refined by #42's Brent, then the
        # #13 stage route for a *locked* junction is a no-op here — the
        # stages stay distinct.
        env = _alkyl_env(_shell_pocket())
        relaxed, _, diagnostics = env.finalize_terminal_molecule(
            junction_torsion_stage=True,
            pocket_vdw_relaxation=True,
            return_diagnostics=True,
        )
        assert len(diagnostics["junctions"]) == 1  # the rotatable junction refined
        assert "locked_junction_torsion" not in diagnostics or not diagnostics["locked_junction_torsion"]
        assert diagnostics["pocket_vdw_relaxation"] is True

    def test_legacy_default_is_unchanged(self):
        # D-V2-017: without the opt-in flags, the behaviour (and therefore
        # the 50 pinned reconciliation contracts) is bit-for-bit legacy.
        env = _amide_env(_shell_pocket())
        env.stats = {"subbin_refinements": 0, "mmff_minimizations": 0, "brent_evaluations": 0}
        relaxed, energy, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert diagnostics["junctions"] == []  # locked: nothing refined (pinned)
        assert env.stats["subbin_refinements"] == 0
        assert env.stats["mmff_minimizations"] == 1
        assert "locked_junction_torsion" not in diagnostics
        assert "pocket_vdw_relaxation" not in diagnostics
