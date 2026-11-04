"""#26 TRAIN-01 (data side) — supervised training loader tests.

Covers: complex-context reconstruction (fail-closed record/artifact
cross-checks), the crystal-intermediate states, the molecule-side handle
pool (D-V2-031), pocket cap trimming, and the #20 augmentation wiring
end-to-end through the loader (issue #20 acceptance criterion 2: the
training loader emits jittered coordinates when ``augment=True``; the
validation loader stays bit-identical pristine).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.reactions import POLICY_REACTIONS
from syntree.data.supervised import (
    MAX_POCKET_ATOMS_PROFILE_A,
    ComplexContext,
    SupervisedTrajectoryDataset,
    SupervisedDataError,
    build_complex_context,
    grow_handle_pool,
    molecule_handle_instances,
    pocket_from_pdb_text,
    pocket_tensors,
    supervised_collate,
)
from syntree.data.trajectories import (
    CatalogView,
    SynthonIndex,
    extract_trajectory,
    trajectory_to_dict,
)
from syntree.engine.environment import PocketDefinition

# ---------------------------------------------------------------------------
# Fixtures: the amide trajectory of the #18 test battery, plus a pocket
# ---------------------------------------------------------------------------

LIGAND_SMILES = "CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1"

AMIDE_CATALOG = [
    ("SEED_PPD", "Nc1ccc(N)cc1", "", "amine"),
    ("BLK_PROPIONYL", "CCC(=O)Cl", "acid_chloride", ""),
    ("BLK_BENZOYL", "O=C(Cl)c1ccccc1", "acid_chloride", ""),
]


def _embed(smiles: str, seed: int = 2026) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=False) == 0
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _build_index(rows):
    return SynthonIndex.build(
        block_ids=[r[0] for r in rows],
        smiles=[r[1] for r in rows],
        synthon_classes=[r[2] for r in rows],
        seed_classes=[r[3] for r in rows],
        catalog_sha256="0" * 64,
        catalog_schema_version="catalog-v2.0",
    )


def _record(split: str = "train") -> dict:
    ligand = _embed(LIGAND_SMILES)
    record = extract_trajectory(
        ligand,
        complex_id="cx_amide",
        split=split,
        index=_build_index(AMIDE_CATALOG),
        catalog=CatalogView(
            block_ids=tuple(r[0] for r in AMIDE_CATALOG),
            smiles=tuple(r[1] for r in AMIDE_CATALOG),
        ),
    )
    assert isinstance(record, object) and hasattr(record, "steps"), getattr(
        record, "detail", "extraction failed"
    )
    return trajectory_to_dict(record)


def _pocket(ligand: Chem.Mol, num_atoms: int = 24) -> PocketDefinition:
    """A synthetic pocket surrounding the ligand (deterministic)."""
    positions = ligand.GetConformer().GetPositions()
    centroid = positions.mean(axis=0)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(num_atoms, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = 4.0 + 2.0 * rng.random(num_atoms)
    pocket_pos = centroid + dirs * radii[:, None]
    elements = [6] * (num_atoms - 4) + [7, 8, 8, 16]
    charges = np.full(num_atoms, 0.0)
    return PocketDefinition(
        positions=pocket_pos,
        atomic_numbers=np.asarray(elements, dtype=np.int64),
        charges=charges,
    )


def _context(split: str = "train") -> ComplexContext:
    record = _record(split)
    ligand = _embed(LIGAND_SMILES)
    return build_complex_context(record, ligand, _pocket(ligand))


# ---------------------------------------------------------------------------
# Pocket parsing and cap trimming
# ---------------------------------------------------------------------------


class TestPocket:
    def test_pdb_roundtrip(self):
        pocket = _pocket(_embed(LIGAND_SMILES))
        lines = []
        for i, (pos, z) in enumerate(zip(pocket.positions, pocket.atomic_numbers)):
            element = {6: "C", 7: "N", 8: "O", 16: "S"}[int(z)]
            lines.append(
                f"ATOM  {i + 1:5d}  CA  ALA A{i % 9 + 1:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
        parsed = pocket_from_pdb_text("\n".join(lines))
        assert parsed.num_atoms == pocket.num_atoms
        assert np.allclose(parsed.positions, pocket.positions, atol=5e-4)  # PDB is 3-decimal
        assert list(parsed.atomic_numbers) == list(pocket.atomic_numbers)

    def test_cap_trims_to_nearest_atoms(self):
        ligand = _embed(LIGAND_SMILES)
        pocket = _pocket(ligand, num_atoms=8)
        centroid = ligand.GetConformer().GetPositions().mean(axis=0)
        pos, z, q = pocket_tensors(pocket, centroid, max_atoms=5)
        assert pos.shape[0] == 5 and z.shape[0] == 5
        # the kept atoms are exactly the 5 nearest the centroid: every
        # kept row matches an original pocket row (centroid-shifted) and
        # the kept distances are the 5 smallest
        distances = np.linalg.norm(pocket.positions - centroid, axis=1)
        nearest5 = set(np.sort(distances)[:5].round(9).tolist())
        kept_distances = np.linalg.norm(pos, axis=1)
        assert set(np.sort(kept_distances).round(9).tolist()) == nearest5
        original = pocket.positions - centroid
        matched = sum(
            1
            for row in pos
            if np.any(np.all(np.isclose(original, row, atol=1e-12), axis=1))
        )
        assert matched == 5  # each kept atom is an original pocket atom
        # no cap: the full pocket is returned (centred)
        pos_all, _, _ = pocket_tensors(pocket, centroid, max_atoms=700)
        assert pos_all.shape[0] == pocket.num_atoms

    def test_profile_a_cap_is_700(self):
        assert MAX_POCKET_ATOMS_PROFILE_A == 700

    def test_empty_pocket_text_fails_closed(self):
        with pytest.raises(SupervisedDataError):
            pocket_from_pdb_text("REMARK nothing here")


# ---------------------------------------------------------------------------
# The molecule-side handle pool (D-V2-031)
# ---------------------------------------------------------------------------


class TestHandlePool:
    def test_all_27_reactions_have_a_molecule_side_attachment(self):
        mol = _embed(LIGAND_SMILES)
        for reaction in POLICY_REACTIONS.values():
            # resolves (and caches) without raising for every reaction
            molecule_handle_instances(mol, reaction)

    def test_pool_contains_the_free_amine_not_the_amide(self):
        seed = _embed("Nc1ccc(N)cc1")
        pool = grow_handle_pool(seed)
        amine_rows = [atom for atom, cls in pool if cls == "amine"]
        assert len(amine_rows) == 2  # both free NH2 positions
        # after one acylation the amide N is a junction atom -> excluded
        acylated = _embed("CCC(=O)Nc1ccc(N)cc1")
        amide = acylated.GetSubstructMatches(
            Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        )
        assert len(amide) == 1
        junction_n = [amide[0][0]]
        pool2 = grow_handle_pool(acylated, junction_atoms=junction_n)
        amine_rows2 = [atom for atom, cls in pool2 if cls == "amine"]
        assert len(amine_rows2) == 1
        assert junction_n and junction_n[0] not in amine_rows2

    def test_pool_is_deterministic_and_deduplicated(self):
        mol = _embed(LIGAND_SMILES)
        assert grow_handle_pool(mol) == grow_handle_pool(mol)
        pairs = grow_handle_pool(mol)
        assert len(pairs) == len(set(pairs))

    def test_reactant_form_graft_handles_are_visible_d_v2_043(self):
        # The authorized-T4 first observation (S-V2-19): pdbbind_4itp step 2
        # couples an incoming aryl bromide onto a product-form methyl via
        # the Negishi alkylzinc class.  The methyl is a legal handle ONLY in
        # the reactant form (CH2-ZnCl): the leaving group is by construction
        # absent from the crystal product graph, so direct template matching
        # against the product graph can never see it.
        acetylated = _embed("CC(=O)NCc1ccc(S(N)(=O)=O)cc1")
        methyl = acetylated.GetSubstructMatches(Chem.MolFromSmarts("[CX4;H3]-[CX3](=O)"))
        assert len(methyl) == 1
        methyl_idx = int(methyl[0][0])
        rxn15 = POLICY_REACTIONS["rxn_15"]
        # direct product-graph matching (the pre-D-V2-043 semantics) is
        # structurally blind: the certified reactant template contains the
        # ZnCl leaving group, which no crystal product graph carries
        from syntree.data.supervised import _molecule_side_pattern
        pattern, attach_pos = _molecule_side_pattern(rxn15)
        direct = {m[attach_pos] for m in acetylated.GetSubstructMatches(pattern, uniquify=False)}
        assert not direct
        # reactant-form graft matching: the methyl is an alkylzinc handle
        assert methyl_idx in molecule_handle_instances(acetylated, rxn15)
        pool = grow_handle_pool(acetylated)
        assert (methyl_idx, "alkylzinc") in pool
        # an atom without a replaceable H can never take the graft: the
        # quaternary acetyl-chloride junction carbon of the same molecule
        quaternary = _embed("CC(C)(C)NCc1ccc(S(N)(=O)=O)cc1")
        central = quaternary.GetSubstructMatches(Chem.MolFromSmarts("[CX4;H0](-[C])(-[C])(-[C])-[N]"))
        assert len(central) == 1
        assert int(central[0][0]) not in molecule_handle_instances(quaternary, rxn15)

    def test_aryl_ch_graft_handles_and_amines_coexist(self):
        # aryl bromide / aryl chloride classes (rxn_09..12) graft onto aryl
        # C-H positions — invisible to direct product-graph matching too.
        arene = _embed("c1ccc(N)cc1")
        rxn09 = POLICY_REACTIONS["rxn_09"]
        instances = molecule_handle_instances(arene, rxn09)
        assert len(instances) >= 2  # the aryl C-H positions (not the N-attached one)
        pool = grow_handle_pool(arene)
        classes = {cls for _, cls in pool}
        assert "aryl_bromide" in classes and "amine" in classes
        # determinism of the (deliberately uncached) derivation
        assert molecule_handle_instances(arene, rxn09) == instances


# ---------------------------------------------------------------------------
# Complex contexts
# ---------------------------------------------------------------------------


class TestComplexContext:
    def test_views_and_labels(self):
        context = _context()
        record_steps = context.record["steps"]
        assert len(record_steps) == 2
        # one pre-step view per step + the terminal view
        assert len(context.views) == len(record_steps) + 1
        terminal = context.views[-1]
        assert terminal.next_junction_atom is None
        assert terminal.product_smiles == Chem.MolToSmiles(_embed(LIGAND_SMILES))
        for view in context.views[:-1]:
            assert view.next_junction_atom is not None
            assert view.next_handle_class == "amine"
        # the junction bond count grows with each placed step
        assert len(context.views[0].junction_bonds) == 0
        assert len(context.views[1].junction_bonds) == 1
        assert len(terminal.junction_bonds) == 2

    def test_corrupted_record_fails_closed(self):
        record = _record()
        ligand = _embed(LIGAND_SMILES)
        broken = dict(record)
        broken["steps"] = record["steps"][:1]  # step-count mismatch
        with pytest.raises(SupervisedDataError, match="mismatch"):
            build_complex_context(broken, ligand, _pocket(ligand))
        wrong_product = dict(record)
        wrong_product["steps"] = [
            {**record["steps"][0], "product_smiles": "CCO"},
            *record["steps"][1:],
        ]
        with pytest.raises(SupervisedDataError, match="disagrees"):
            build_complex_context(wrong_product, ligand, _pocket(ligand))


# ---------------------------------------------------------------------------
# The dataset + augmentation wiring (#20 acceptance criterion 2)
# ---------------------------------------------------------------------------


class TestSupervisedDataset:
    def test_item_table_and_stages(self):
        context = _context()
        dataset = SupervisedTrajectoryDataset([context], split="train")
        assert len(dataset) == 1 + len(context.record["steps"]) + 1  # seed + steps + stop
        stages = [dataset[i]["stage"] for i in range(len(dataset))]
        assert stages[0] == "seed"
        assert stages[-1] == "stop"
        assert all(s == "step" for s in stages[1:-1])

    def test_states_pass_the_frozen_tensor_contract(self):
        from syntree.core.schema import collate_state_batch

        context = _context()
        dataset = SupervisedTrajectoryDataset([context], split="train")
        states = [dataset[i]["state"] for i in range(len(dataset))]
        batch = collate_state_batch(states)
        assert batch.num_graphs == len(states)
        # the seed state is pocket-only (empty ligand sub-graph)
        assert dataset[0]["state"].ligand_pos.shape[0] == 0

    def test_augment_true_emits_jittered_coordinates(self):
        """Issue #20 acceptance criterion 2 (training loader path)."""
        context = _context()
        augmented = SupervisedTrajectoryDataset(
            [context], split="train", augment=True, seed=123
        )
        pristine = SupervisedTrajectoryDataset(
            [context], split="train", augment=False, seed=123
        )
        for index in range(len(augmented)):
            item_aug = augmented[index]
            item_pri = pristine[index]
            assert item_aug["stage"] == item_pri["stage"]
            state_aug, state_pri = item_aug["state"], item_pri["state"]
            if state_aug.ligand_pos.shape[0] == 0:
                continue  # the pocket-only seed state has no ligand coords
            assert not torch.equal(state_aug.ligand_pos, state_pri.ligand_pos), (
                f"item {index}: augmented ligand coordinates are identical "
                "to pristine (augment=True must emit jittered coordinates)"
            )
            # the pocket (fixed conditioning context) stays bit-identical
            assert torch.equal(state_aug.pocket_pos, state_pri.pocket_pos)
            # invariant chemical channels are untouched
            assert torch.equal(state_aug.ligand_z, state_pri.ligand_z)

    def test_validation_loader_is_bit_identical_pristine(self):
        context = _context("val")
        dataset = SupervisedTrajectoryDataset([context], split="val", augment=False)
        for index in range(len(dataset)):
            a = dataset[index]["state"]
            b = dataset[index]["state"]
            assert torch.equal(a.ligand_pos, b.ligand_pos)
        # augmentation on a non-train split is refused at construction
        with pytest.raises(SupervisedDataError, match="train-split only"):
            SupervisedTrajectoryDataset([context], split="val", augment=True)

    def test_augmentation_is_deterministic_and_epoch_fresh(self):
        context = _context()
        dataset = SupervisedTrajectoryDataset(
            [context], split="train", augment=True, seed=9
        )
        first = dataset[1]["state"].ligand_pos.clone()
        again = dataset[1]["state"].ligand_pos
        assert torch.equal(first, again)  # same epoch + seed -> identical
        dataset.set_epoch(1)
        later = dataset[1]["state"].ligand_pos
        assert not torch.equal(first, later)  # fresh draws every epoch

    def test_primary_handle_is_in_the_pool(self):
        context = _context()
        dataset = SupervisedTrajectoryDataset([context], split="train")
        for index in range(1, len(dataset) - 1):
            item = dataset[index]
            assert item["primary_handle"] is not None
            assert 0 <= item["primary_handle"] < len(item["pool"])

    def test_collate_groups_by_stage_with_flat_pool(self):
        context = _context()
        dataset = SupervisedTrajectoryDataset([context], split="train")
        batch = supervised_collate([dataset[i] for i in range(len(dataset))])
        assert batch["seed"] is not None and batch["seed"]["state"].num_graphs == 1
        assert batch["step"]["state"].num_graphs == len(context.record["steps"])
        assert batch["stop"]["state"].num_graphs == 1
        step = batch["step"]
        # flat pool rows carry batched-ligand atom indices (offset applied)
        total_ligand = step["state"].ligand_pos.shape[0]
        assert int(step["handle_atom"].max()) < total_ligand
        assert step["handle_graph"].max() < step["state"].num_graphs
        assert (step["torsion_bin"] >= 0).all() and (step["torsion_bin"] <= 36).all()
        assert step["stop_target"].sum() == 0
        assert batch["stop"]["stop_target"].sum() == 1

    def test_split_mismatch_fails_closed(self):
        context = _context("train")
        with pytest.raises(SupervisedDataError, match="does not match"):
            SupervisedTrajectoryDataset([context], split="val")


# ---------------------------------------------------------------------------
# S-V2-17 regression: hydrogen-laden artifacts (the PDBbind layout)
# ---------------------------------------------------------------------------

class TestHydrogenLadenArtifacts:
    """PDBbind ligand artifacts carry explicit hydrogens; CrossDocked and
    BindingMOAD ones do not.  The records are defined on the heavy-atom
    graph (the #18 extractor's ``removeHs=True`` supplier convention), so
    the supervised loader must parse artifacts the same way — loading
    with ``removeHs=False`` re-derives a different seed/step-count for
    H-laden artifacts and trips the fail-closed record/artifact
    cross-check (first observed on the authorized Tesla T4, S-V2-17:
    ``pdbbind_1px4: re-derived seed graph disagrees with the record``).
    """

    @staticmethod
    def _h_laden_artifact(tmp_path, record):
        """Write the fixture ligand WITH explicit Hs, gzipped (PDBbind layout)."""
        import gzip

        h_laden = Chem.AddHs(_embed(LIGAND_SMILES))
        path = os.path.join(str(tmp_path), f"{record['complex_id']}_ligand.sdf.gz")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            writer = Chem.SDWriter(handle)
            writer.write(h_laden)
            writer.close()
        return path, h_laden

    def test_load_complex_strips_hydrogens_like_the_extractor(self, tmp_path):
        import importlib.util

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "train_stage1", os.path.join(repo_root, "scripts", "train_stage1.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        record = _record()
        path, _ = self._h_laden_artifact(tmp_path, record)
        # the pocket artifact (uncompressed layout, as the driver writes it)
        pocket = _pocket(_embed(LIGAND_SMILES))
        symbols = {6: "C", 7: "N", 8: "O", 16: "S"}
        lines = ["HEADER    S-V2-17 REGRESSION POCKET"]
        for i, (pos, z) in enumerate(zip(pocket.positions, pocket.atomic_numbers), start=1):
            x, y, zz = (float(v) for v in pos)
            sym = symbols[int(z)]
            lines.append(
                f"ATOM  {i:5d} {sym:>4s} {sym:>3s} A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{zz:8.3f}  1.00  0.00          {sym:>2s}"
            )
        with open(os.path.join(str(tmp_path), f"{record['complex_id']}_pocket.pdb"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        ligand, _pocket_def = module._load_complex(str(tmp_path), record)
        # the extractor convention: the heavy-atom graph, no explicit H
        assert not any(a.GetSymbol() == "H" for a in ligand.GetAtoms())
        assert ligand.GetNumAtoms() == _embed(LIGAND_SMILES).GetNumAtoms()
        # ... and therefore the fail-closed cross-check passes
        context = build_complex_context(record, ligand, _pocket(ligand))
        assert context.complex_id == record["complex_id"]
        assert len(context.record["steps"]) == len(record["steps"])

    def test_h_laden_ligand_reproduces_the_t4_failure(self):
        """The pre-fix behaviour (removeHs=False) fails closed — pinned here
        so the convention can never silently drift again."""
        record = _record()
        h_laden = Chem.AddHs(_embed(LIGAND_SMILES))
        with pytest.raises(SupervisedDataError, match="re-derived seed graph"):
            build_complex_context(record, h_laden, _pocket(h_laden))
