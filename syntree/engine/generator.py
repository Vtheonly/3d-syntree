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
from syntree.engine.environment import AssemblyAction, MolecularAssemblyEnv
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
        # The MolecularAssemblyEnv is the single source of environment-side
        # truth (chemistry grammar, reaction execution, 3D assembly, physics
        # scoring); this method only adds the policy loop and packaging.
        env = MolecularAssemblyEnv(
            rxn_engine=self.rxn_engine,
            conformer_engine=self.conformer_engine,
            catalog=self.catalog,
            validator=self.validator,
            config=self.config,
            device=self.device,
        )
        observation = env.reset(
            pocket_pdb_path=pocket_pdb_path,
            seed_synthon_idx=seed_synthon_idx,
            max_steps=max_steps,
            optimize_torsion_grid=optimize_torsion_grid,
        )

        policy_trace: List[Dict] = []
        while not env.done:
            reaction_mask, synthon_masks, has_handle = env.legal_action_masks()
            if not has_handle:
                break
            if not bool((reaction_mask > -1e8).any().item()):
                # No legal chemistry remains (e.g. every compatible partner
                # was filtered out): stop instead of feeding an all-masked
                # distribution to the policy.
                logger.debug(
                    "No legal reaction family at step %d; stopping growth.",
                    env.step_count + 1,
                )
                break

            if return_trace:
                with torch.enable_grad():
                    decision = self.model.act(
                        observation,
                        self.catalog.embeddings.to(self.device),
                        reaction_compatibility_mask=reaction_mask,
                        synthon_masks_by_reaction=synthon_masks,
                        sample=sample,
                        temperature=temperature,
                    )
            else:
                decision = self.model.act(
                    observation,
                    self.catalog.embeddings.to(self.device),
                    reaction_compatibility_mask=reaction_mask,
                    synthon_masks_by_reaction=synthon_masks,
                    sample=sample,
                    temperature=temperature,
                )

            if return_trace:
                policy_trace.append({
                    "state": observation.clone(),
                    "reaction_mask": reaction_mask.detach().clone(),
                    "synthon_masks": synthon_masks.detach().clone(),
                    "family_idx": int(decision["reaction_family_idx"][0].item()),
                    "action_idx": int(decision["action_idx"][0].item()),
                    "old_log_prob": decision["joint_log_prob"][0],
                    "old_value": decision["state_value"][0],
                })

            action = AssemblyAction(
                reaction_family_idx=int(decision["reaction_family_idx"][0].item()),
                synthon_idx=int(decision["synthon_idx"][0].item()),
                dihedral_rad=float(decision["dihedral"][0].item()),
                action_idx=int(decision["action_idx"][0].item()),
            )
            observation, _reward, _done, _info = env.step(action)

        checks = env.validate()
        return {
            "rdkit_mol": env.current_mol,
            "recipe": env.recipe,
            "passes_posebusters": all(checks.values()),
            "checks": checks,
            "smiles": env.smiles(),
            "clash_score": float(env.total_clash),
            "contact_energy": env.contact_energy(),
            "descriptors": self.validator.descriptors(env.current_mol),
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
