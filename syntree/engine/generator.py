"""Autoregressive 3D SBDD inference engine.

Given a target protein pocket, the generator runs the policy step by step:

1. Detect reactive handles on the current ligand.
2. Let the policy choose a reaction-compatible synthon and a dihedral angle.
3. Execute the reaction (RDKit), embed the product, lock the scaffold,
   and apply the predicted torsion.
4. Repeat until no handles remain or ``max_steps`` is reached.

Every generated ligand comes with a step-by-step synthesis recipe built
from catalog IDs and SMARTS reaction names.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.hotspots import PocketHotspotFeaturizer
from syntree.chemistry.conformer import ConformerEngine, vdw_radius
from syntree.chemistry.reactions import (
    REACTION_FAMILY_MEMBERS,
    REACTION_FAMILY_NAMES,
    REACTION_SIDES,
    ReactionEngine,
)
from syntree.chemistry.validator import ChemicalValidator
from syntree.data.featurizer import MolecularFeaturizer

logger = logging.getLogger(__name__)


class SBDDGenerator:
    """Generates pocket-conditioned, synthesis-guaranteed 3D ligands."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        output_dir: str = "./outputs",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.rxn_engine = ReactionEngine()
        self.catalog = SynthonCatalog(
            config["data"]["synthon_catalog_path"],
            embedding_dim=config["model"].get("synthon_embedding_dim", 128),
            min_fsp3=float(config.get("catalog", {}).get("min_fsp3", 0.42)),
            max_mw=float(config.get("catalog", {}).get("max_mw", 220.0)),
        )
        self.conformer_engine = ConformerEngine()
        self.validator = ChemicalValidator()
        self.max_steps = int(config.get("data", {}).get("max_steps_per_molecule", 3))
        self.seed_rng = np.random.default_rng(
            int(config.get("system", {}).get("seed", 42))
        )

    # ------------------------------------------------------------------
    # Single ligand generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_ligand(
        self,
        pocket_pdb_path: str,
        max_steps: Optional[int] = None,
        seed_synthon_idx: Optional[int] = None,
        sample: bool = False,
        temperature: float = 1.0,
        optimize_torsion_grid: bool = False,
        return_trace: bool = False,
    ) -> Dict:
        """Generate one ligand inside the pocket at ``pocket_pdb_path``.

        Args:
            max_steps: override the configured maximum growth steps.
            seed_synthon_idx: catalog index of the seed fragment (random
                valid handle-bearing synthon when None).
            sample: sample the synthon choice instead of argmax.
            temperature: softmax temperature when sampling.
            optimize_torsion_grid: optional diagnostic refinement. Disabled by
                default so inference does not overwrite the neural torsion prediction.

        Returns:
            Dict with ``rdkit_mol`` (3D, explicit H), ``recipe`` (list of
            step records), ``passes_posebusters``, ``smiles``,
            ``clash_score`` and ``descriptors``.
        """
        try:
            pocket_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False)
        except OSError as exc:
            raise ValueError(
                f"Cannot read pocket file {pocket_pdb_path!r}: {exc}"
            ) from exc
        if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
            raise ValueError(f"Cannot read pocket from {pocket_pdb_path}")
        feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
        pocket_pos = feats["pocket_pos"]
        pocket_z = feats["pocket_z"]
        pocket_coords_np = feats["pocket_pos"].numpy()
        pocket_vdw = np.array(
            [vdw_radius(int(z)) for z in pocket_z.tolist()], dtype=np.float64
        )
        pocket_volume = MolecularFeaturizer.vdw_sphere_volume(pocket_mol)
        # Reference frame for ligand/handle coordinates: the pocket centroid
        # (featurize_pocket already centered the cloud on it).
        pocket_centroid = feats["centroid"].view(-1)
        hotspots = PocketHotspotFeaturizer().featurize(
            pocket_mol, center=feats["centroid"].view(-1)
        )

        steps = self.max_steps if max_steps is None else int(max_steps)

        # 1. Seed fragment (implicit H for chemistry; 3D embedded after).
        if seed_synthon_idx is None:
            seed_synthon_idx = self._pick_seed_synthon()
        current_mol = self.catalog.get_mol(seed_synthon_idx, explicit_hs=False)
        current_mol = self.conformer_engine.embed_product(current_mol)
        if current_mol is None:
            raise RuntimeError("Failed to embed the seed synthon in 3D.")
        seed_handle_infos = self.rxn_engine.detect_handles(current_mol)
        seed_handle_type = seed_handle_infos[0].handle_type if seed_handle_infos else None
        # Place the seed at a contact-rich position near the pocket interface
        # rather than at the point farthest from the protein.
        self._place_seed_in_pocket(
            current_mol,
            pocket_coords_np,
            pocket_vdw=pocket_vdw,
            hotspots=hotspots,
            seed_handle_type=seed_handle_type,
            seed_handle_atom_index=(
                seed_handle_infos[0].primary_atom if seed_handle_infos else None
            ),
        )
        current_mol = Chem.AddHs(current_mol, addCoords=True)

        recipe: List[Dict] = [
            {
                "step": 0,
                "action": "seed",
                "synthon_id": self.catalog.get_id(seed_synthon_idx),
                "smiles": self.catalog.get_smiles(seed_synthon_idx),
                "handle": None,
            }
        ]

        total_clash = 0.0
        policy_trace: List[Dict] = []
        # 2. Autoregressive growth.
        for step in range(1, steps + 1):
            # Chemoselectivity: grow through the most reactive handle, not
            # whichever one a SMARTS iteration happens to find first.
            handles = self.rxn_engine.rank_handles(current_mol)
            if not handles:
                break

            target_handle = handles[0]
            if not target_handle.allowed_reactions:
                break

            current_mw = float(Descriptors.MolWt(Chem.RemoveHs(Chem.Mol(current_mol))))
            cap_min_mw = float(
                self.config.get("data", {}).get("terminal_cap_min_mw", 250.0)
            )
            # The policy may only terminate once the ligand is drug-like
            # enough (MW >= cap); the horizon ends the loop instead.
            below_cap = current_mw < cap_min_mw
            allow_stop = not below_cap
            require_remaining_handle = step < steps and below_cap

            reaction_mask = self.catalog.get_reaction_family_compatibility_mask(
                device=self.device,
                core_handle=target_handle.handle_type,
                require_remaining_handle=require_remaining_handle,
                allow_terminal=allow_stop,
            ).unsqueeze(0)
            if not bool((reaction_mask > -1e8).any().item()):
                # No legal chemistry remains (e.g. every compatible partner
                # was filtered out): stop instead of feeding an all-masked
                # distribution to the policy.
                logger.debug(
                    "No legal reaction family at step %d; stopping growth.", step
                )
                break
            synthon_masks = self.catalog.get_reaction_family_masks(
                device=self.device,
                core_handle=target_handle.handle_type,
                require_remaining_handle=require_remaining_handle,
                allow_terminal=allow_stop,
            ).unsqueeze(0)

            # 3. Policy decision: synthon + dihedral.
            handle_feat = MolecularFeaturizer.featurize_handle(
                current_mol,
                target_handle.atom_indices,
                target_handle.handle_type,
            ).unsqueeze(0).to(self.device)
            # Ghost-ligand fix: the policy state carries the full intermediate
            # ligand point cloud (same pocket-centered frame), the reacting
            # handle node index, its position, and global size features.
            lig_feats = MolecularFeaturizer.featurize_ligand(
                current_mol, center=pocket_centroid
            )
            handle_node_idx = lig_feats["heavy_atom_map"].get(
                int(target_handle.primary_atom), -1
            )
            global_feats = MolecularFeaturizer.ligand_global_features(
                current_mol, pocket_volume=pocket_volume
            ).unsqueeze(0).to(self.device)
            handle_pos_t = MolecularFeaturizer.featurize_handle_position(
                current_mol,
                target_handle.atom_indices,
                reference_center=pocket_centroid,
            ).unsqueeze(0).to(self.device)

            batch_data = Data(
                pocket_pos=pocket_pos.to(self.device),
                pocket_z=pocket_z.to(self.device),
                pocket_charge=feats["pocket_charge"].to(self.device),
                pocket_batch=torch.zeros(
                    pocket_pos.size(0), dtype=torch.long, device=self.device
                ),
                ligand_pos=lig_feats["ligand_pos"].to(self.device),
                ligand_z=lig_feats["ligand_z"].to(self.device),
                ligand_charge=lig_feats["ligand_charge"].to(self.device),
                ligand_batch=torch.zeros(
                    lig_feats["ligand_pos"].size(0), dtype=torch.long,
                    device=self.device,
                ),
                handle_features=handle_feat,
                handle_pos=handle_pos_t,
                handle_nodes=torch.tensor(
                    [handle_node_idx], dtype=torch.long, device=self.device
                ),
                global_features=global_feats,
                # Mask STOP out of the action space while the ligand is still
                # below the terminal molecular-weight cap.
                stop_mask=torch.tensor(
                    0.0 if allow_stop else -1e9,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

            if return_trace:
                with torch.enable_grad():
                    decision = self.model.act(
                        batch_data,
                        self.catalog.embeddings.to(self.device),
                        reaction_compatibility_mask=reaction_mask,
                        synthon_masks_by_reaction=synthon_masks,
                        sample=sample,
                        temperature=temperature,
                    )
            else:
                decision = self.model.act(
                    batch_data,
                    self.catalog.embeddings.to(self.device),
                    reaction_compatibility_mask=reaction_mask,
                    synthon_masks_by_reaction=synthon_masks,
                    sample=sample,
                    temperature=temperature,
                )
            selected_family_idx = int(decision["reaction_family_idx"][0].item())
            reaction_family = REACTION_FAMILY_NAMES[selected_family_idx]
            chosen_rxn = self._preferred_reaction(
                reaction_family, target_handle.handle_type
            )
            if chosen_rxn is None:
                logger.debug(
                    "Family %s has no executable backend for handle %s; "
                    "stopping growth.",
                    reaction_family, target_handle.handle_type,
                )
                break
            if bool(decision["stop"][0].item()):
                if return_trace:
                    policy_trace.append({
                        "state": batch_data.clone(),
                        "reaction_mask": reaction_mask.detach().clone(),
                        "synthon_masks": synthon_masks.detach().clone(),
                        "family_idx": int(decision["reaction_family_idx"][0].item()),
                        "action_idx": int(decision["action_idx"][0].item()),
                        "old_log_prob": decision["joint_log_prob"][0],
                        "old_value": decision["state_value"][0],
                    })
                recipe.append({
                    "step": step,
                    "action": "stop",
                    "reason": "policy_stop",
                })
                break

            selected_synthon_idx = int(decision["synthon_idx"][0].item())
            dihedral_pred = float(decision["dihedral"][0].item())

            # The family mask is the final chemistry guard. It is deterministic
            # and comes from the same reaction grammar used by execution.
            selected_mask = self.catalog.get_reaction_family_mask(
                reaction_family,
                device=self.device,
                core_handle=target_handle.handle_type,
                require_remaining_handle=require_remaining_handle,
                allow_terminal=allow_stop,
            )
            if selected_mask[selected_synthon_idx].item() < -1e8:
                logger.warning(
                    "Policy emitted an action that violates the deterministic chemistry mask; stopping this rollout."
                )
                break

            if return_trace:
                policy_trace.append({
                    "state": batch_data.clone(),
                    "reaction_mask": reaction_mask.detach().clone(),
                    "synthon_masks": synthon_masks.detach().clone(),
                    "family_idx": selected_family_idx,
                    "action_idx": int(decision["action_idx"][0].item()),
                    "old_log_prob": decision["joint_log_prob"][0],
                    "old_value": decision["state_value"][0],
                })

            synthon_mol = self.catalog.get_mol(selected_synthon_idx, explicit_hs=False)

            # 4. Chemical execution.
            #    Reactions run on implicit-H copies: RDKit reaction templates
            #    manage H counts implicitly; running them on explicit-H
            #    molecules leaves stray hydrogens on the product (e.g. an
            #    amine N retaining both H atoms through amide coupling,
            #    producing an illegal valence-4 nitrogen). RemoveHs preserves
            #    heavy-atom indices, so the scaffold conformer stays valid.
            core_noH = Chem.RemoveHs(Chem.Mol(current_mol))
            result = self.rxn_engine.apply_reaction(
                core_noH, synthon_mol, chosen_rxn
            )
            if result is None:
                logger.debug(
                    "Reaction %s failed at step %d; stopping growth.",
                    chosen_rxn, step,
                )
                break

            # 5. 3D assembly: embed product (adds H), lock scaffold, rotate.
            product = self.conformer_engine.embed_product(
                result.product,
                core_atom_map=result.core_atom_map,
                core_reference=core_noH,
            )
            if product is None:
                break

            core_atom, synthon_atom = result.junction_bond
            # Resonance-aware torsion application: amide/ester junctions snap
            # to planar {trans, cis} (choosing the lower-LJ-energy option)
            # and the policy angle propagates to the adjacent true single
            # bond, exactly like protein phi/psi angles.
            _, junction_angle, rotated_bond = self.conformer_engine.apply_junction_torsion(
                product,
                (core_atom, synthon_atom),
                dihedral_pred,
                pocket_coords=pocket_coords_np,
                pocket_vdw=pocket_vdw,
            )

            if optimize_torsion_grid:
                _, _, clash = self.conformer_engine.optimize_dihedral(
                    product,
                    (core_atom, synthon_atom),
                    pocket_coords=pocket_coords_np,
                    pocket_vdw=pocket_vdw,
                )
                total_clash = clash

            current_mol = product
            recipe.append(
                {
                    "step": step,
                    "action": "react",
                    "reaction": chosen_rxn,
                    "reaction_family": reaction_family,
                    "handle": target_handle.handle_type,
                    "synthon_id": self.catalog.get_id(selected_synthon_idx),
                    "synthon_smiles": self.catalog.get_smiles(selected_synthon_idx),
                    "dihedral_applied_rad": dihedral_pred,
                    "junction_angle_rad": junction_angle,
                    "torsion_mode": (
                        "planar_snap_adjacent"
                        if rotated_bond != (core_atom, synthon_atom)
                        else "direct"
                    ),
                    "rotated_bond": list(rotated_bond) if rotated_bond else None,
                    "new_atoms": len(result.synthon_atom_map) + len(result.new_atoms),
                }
            )

        # 6. Validation.
        checks = self.validator.validate(
            current_mol,
            pocket_coords=pocket_coords_np,
            pocket_vdw=pocket_vdw,
        )
        smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(current_mol)))

        # Lennard-Jones contact energy (soft-core 6-12 with an attractive
        # dispersion well): unlike the clash score, this penalises both
        # steric overlap AND drifting into open solvent, so it is the
        # physically meaningful pocket-occupation signal for the RL reward.
        try:
            ligand_noH = Chem.RemoveHs(Chem.Mol(current_mol))
            ligand_coords = np.array(
                ligand_noH.GetConformer().GetPositions(), dtype=np.float64
            )
            ligand_vdw = np.array(
                [vdw_radius(int(a.GetAtomicNum())) for a in ligand_noH.GetAtoms()],
                dtype=np.float64,
            )
            contact_energy = float(
                self.conformer_engine.compute_lennard_jones_np(
                    ligand_coords, pocket_coords_np, ligand_vdw, pocket_vdw
                )
            )
        except Exception:
            contact_energy = 0.0

        return {
            "rdkit_mol": current_mol,
            "recipe": recipe,
            "passes_posebusters": all(checks.values()),
            "checks": checks,
            "smiles": smiles,
            "clash_score": float(total_clash),
            "contact_energy": contact_energy,
            "descriptors": self.validator.descriptors(current_mol),
            "policy_trace": policy_trace if return_trace else None,
        }

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        pocket_pdb_path: Optional[str] = None,
        num_ligands: int = 1,
        out_dir: Optional[str] = None,
    ) -> List[Dict]:
        """Generate ``num_ligands`` ligands and write SDF + recipes.

        Results are written to ``out_dir`` (defaults to the generator's
        output directory): one ``ligand_XXX.sdf``, one ``recipe_XXX.json``
        per molecule, plus a consolidated ``batch_summary.json``.
        """
        out = out_dir or self.output_dir
        os.makedirs(out, exist_ok=True)
        if pocket_pdb_path is None:
            pocket_pdb_path = os.path.join(
                self.config["data"]["data_dir"], "sample_pocket.pdb"
            )
        if not os.path.exists(pocket_pdb_path):
            raise FileNotFoundError(
                f"Pocket file not found: {pocket_pdb_path}. Run "
                "scripts/download_assets.py first."
            )

        from rdkit.Chem import SDWriter

        summary: List[Dict] = []
        for i in range(num_ligands):
            t0 = time.time()
            seed = self._pick_seed_synthon()
            res = self.generate_ligand(pocket_pdb_path, seed_synthon_idx=seed)

            sdf_path = os.path.join(out, f"ligand_{i:03d}.sdf")
            writer = SDWriter(sdf_path)
            writer.write(res["rdkit_mol"])
            writer.close()

            recipe_path = os.path.join(out, f"recipe_{i:03d}.json")
            with open(recipe_path, "w") as f:
                json.dump(res["recipe"], f, indent=2)

            record = {
                "index": i,
                "smiles": res["smiles"],
                "num_steps": len(res["recipe"]) - 1,
                "passes_posebusters": res["passes_posebusters"],
                "clash_score": res["clash_score"],
                "descriptors": res["descriptors"],
                "seed_synthon_id": res["recipe"][0]["synthon_id"],
                "sdf": sdf_path,
                "recipe": recipe_path,
                "generation_seconds": round(time.time() - t0, 2),
            }
            summary.append(record)
            print(
                f"[generator] ligand {i:03d}: {res['smiles']} | steps="
                f"{record['num_steps']} | valid={res['passes_posebusters']} | "
                f"{record['generation_seconds']}s"
            )

        with open(os.path.join(out, "batch_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[generator] wrote {num_ligands} ligands + recipes to {out}")
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _place_seed_in_pocket(
        mol: Chem.Mol,
        pocket_coords: np.ndarray,
        pocket_vdw: Optional[np.ndarray] = None,
        hotspots=None,
        seed_handle_type: Optional[str] = None,
        seed_handle_atom_index: Optional[int] = None,
        search_radius: float = 4.0,
        grid: int = 9,
        contact_distance: float = 3.2,
        contact_width: float = 0.8,
    ) -> None:
        """Translate a seed to a contact-rich, clash-avoiding pocket position.

        Candidate translations are scored by a soft contact-shell objective
        and a strong steric-overlap penalty. A 9-point grid plus local
        refinement replaces the old distance-maximising placement.
        """
        if pocket_coords is None or len(pocket_coords) == 0 or mol.GetNumConformers() == 0:
            return
        from rdkit.Geometry import Point3D

        conf = mol.GetConformer()
        pos = np.asarray(conf.GetPositions(), dtype=np.float64)
        pos = pos - pos.mean(axis=0)
        protein_vdw = (
            np.asarray(pocket_vdw, dtype=np.float64)
            if pocket_vdw is not None
            else np.full(len(pocket_coords), 1.70, dtype=np.float64)
        )
        ligand_vdw = np.asarray(
            [vdw_radius(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
            dtype=np.float64,
        )

        def score(offset: np.ndarray) -> float:
            shifted = pos + offset
            dists = np.linalg.norm(
                shifted[:, None, :] - pocket_coords[None, :, :], axis=-1
            )
            contacts = np.exp(
                -0.5 * ((dists - contact_distance) / max(contact_width, 1e-3)) ** 2
            ).sum()
            vdw_sum = ligand_vdw[:, None] + protein_vdw[None, :]
            overlap = np.maximum(0.0, vdw_sum * 0.90 - dists)
            clash_penalty = float((overlap * overlap).sum())
            return float(contacts - 25.0 * clash_penalty)

        def search(center: np.ndarray, radius: float, points: int) -> np.ndarray:
            offsets = np.linspace(-radius, radius, points)
            best = np.zeros(3, dtype=np.float64)
            best_score = -float("inf")
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        candidate = center + np.array([dx, dy, dz])
                        candidate_score = score(candidate)
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best = candidate
            return best

        best_offset = search(np.zeros(3, dtype=np.float64), search_radius, grid)

        if hotspots and seed_handle_type and seed_handle_atom_index is not None:
            ranked = PocketHotspotFeaturizer().rank_for_handle(
                hotspots, seed_handle_type
            )
            if ranked:
                hotspot = ranked[0]
                hp = np.asarray(hotspot.position, dtype=np.float64)
                direction = -hp
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = direction / norm
                target_handle_position = hp + direction * 2.8
                target_offset = target_handle_position - pos[seed_handle_atom_index]
                best_offset = search(target_offset, 1.0, 5)
            else:
                best_offset = search(best_offset, 1.0, 5)
        else:
            best_offset = search(best_offset, 1.0, 5)

        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*(pos[i] + best_offset)))

    def _pick_seed_synthon(self) -> int:
        """Choose a random catalog synthon bearing any reactive handle.

        Prefers sp3-rich synthons (the project's 3D-diversity philosophy):
        aryl coupling partners are only picked when no sp3-rich candidate
        exists.
        """
        sp3_rich_handles = [
            h for h in self.catalog.available_handles
            if h not in ("aryl_halide", "boronic_acid")
        ]
        candidates = []
        for handle in sp3_rich_handles:
            candidates.extend(
                self.catalog.synthon_indices_for_handles([handle]).tolist()
            )
        if not candidates:
            for handle in self.catalog.available_handles:
                candidates.extend(
                    self.catalog.synthon_indices_for_handles([handle]).tolist()
                )
        if not candidates:
            return 0
        multifunctional = [
            idx for idx in candidates if self.catalog.get_handle_count(idx) >= 2
        ]
        if multifunctional:
            candidates = multifunctional
        return int(self.seed_rng.choice(candidates))

    @staticmethod
    def _preferred_reaction(reaction_family: str, core_handle: str) -> Optional[str]:
        """Choose a concrete RDKit backend for a predicted reaction family.

        Returns ``None`` when no member reaction is compatible with the
        handle (callers stop the rollout instead of crashing)."""
        members = REACTION_FAMILY_MEMBERS.get(reaction_family, ())
        for reaction in members:
            if core_handle in REACTION_SIDES[reaction]:
                return reaction
        return None


__all__ = ["SBDDGenerator"]
