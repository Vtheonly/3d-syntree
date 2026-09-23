"""Evaluation pipeline: PoseBusters-style validation, retrosynthetic
feasibility reporting, docking hooks (GNINA / Vina), and property benchmarks.

External engines (GNINA, AiZynthFinder) are optional: when their binaries
or configuration is absent, the pipeline reports those external evaluations
as unavailable rather than replacing them with misleading internal proxies.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from syntree.chemistry.validator import ChemicalValidator

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Runs the full benchmark battery over generated molecules.

    Metrics:
        * ``posebusters_pass_rate`` – fraction passing all sanity checks,
        * ``chemical_validity`` – RDKit sanitizable fraction,
        * ``mean_fsp3`` / ``fsp3_ge_0.42_rate`` – 3D character,
        * ``retrosynthetic_feasibility`` – route solvability from a configured
          AiZynthFinder planner; unavailable is reported explicitly rather than
          substituted with a tautological recipe check,
        * ``mean_vina_score`` – docking affinity (when GNINA/Vina present),
        * ``recipe_completeness`` – fraction with a full synthesis recipe.
    """

    def __init__(self, config: dict, output_dir: str = "./experiments/eval"):
        self.config = config
        eval_cfg = config.get("evaluation", {})
        self.exhaustiveness = int(eval_cfg.get("exhaustiveness", 8))
        self.docking_engine = str(eval_cfg.get("docking_engine", "gnina"))
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.validator = ChemicalValidator()
        self.aizynthfinder_config = eval_cfg.get("aizynthfinder_config")
        self._retro_status = "unavailable"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        mols: List[Chem.Mol],
        recipes: Optional[List[List[Dict]]] = None,
        pocket_pdb_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a batch of generated molecules.

        Args:
            mols: generated molecules (with or without conformers).
            recipes: per-molecule synthesis recipes (when available).
            pocket_pdb_path: pocket used for docking (optional).
        """
        recipes = [None] * len(mols) if recipes is None else list(recipes)
        if len(recipes) != len(mols):
            raise ValueError("recipes length must match mols length")

        n = max(1, len(mols))
        report: Dict[str, object] = {
            "num_molecules": len(mols),
            "tools_available": self._tool_availability(),
        }

        # Chemical validity + PoseBusters-style checks.
        valid_count = 0
        pb_pass = 0
        fsp3_values: List[float] = []
        mw_values: List[float] = []
        per_mol: List[Dict] = []
        for i, (mol, recipe) in enumerate(zip(mols, recipes)):
            checks = self.validator.validate(mol)
            if mol is not None:
                heavy = Chem.RemoveHs(Chem.Mol(mol))
                sanitized = Chem.MolFromSmiles(Chem.MolToSmiles(heavy)) is not None
                smiles = Chem.MolToSmiles(heavy)
            else:
                sanitized = False
                smiles = None
            valid_count += int(sanitized)
            pb_pass += int(all(checks.values()))
            desc = self.validator.descriptors(mol) if mol else {}
            fsp3_values.append(desc.get("fsp3", 0.0))
            mw_values.append(desc.get("molecular_weight", 0.0))
            per_mol.append(
                {
                    "index": i,
                    "smiles": smiles,
                    "checks": checks,
                    "descriptors": desc,
                    "recipe_steps": (len(recipe) - 1) if recipe else 0,
                    "recipe_complete": bool(recipe is not None and len(recipe) >= 1),
                }
            )

        report["chemical_validity"] = valid_count / n
        report["posebusters_pass_rate"] = pb_pass / n
        report["mean_fsp3"] = float(np.mean(fsp3_values)) if fsp3_values else 0.0
        report["fsp3_ge_0.42_rate"] = float(np.mean([f >= 0.42 for f in fsp3_values]))
        report["mean_mw"] = float(np.mean(mw_values)) if mw_values else 0.0
        report["recipe_completeness"] = float(
            np.mean([m["recipe_complete"] for m in per_mol])
        )

        # Retrosynthetic feasibility.
        report["retrosynthetic_feasibility"] = self._retro_feasibility(per_mol)
        report["retrosynthetic_feasibility_status"] = self._retro_status

        # Docking (optional).
        if pocket_pdb_path is not None and self._tool_availability().get(
            self.docking_engine
        ):
            report["docking"] = self._dock(mols, pocket_pdb_path)
            scores = [
                s for s in report["docking"].get("scores", []) if s is not None
            ]
            if scores:
                report["mean_vina_score"] = float(np.mean(scores))

        report["per_molecule"] = per_mol
        report["report_path"] = os.path.join(self.output_dir, "evaluation_report.json")

        out_path = report["report_path"]
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Evaluation report written to %s", out_path)

        return report

    # ------------------------------------------------------------------
    # Retrosynthesis
    # ------------------------------------------------------------------
    def _retro_feasibility(self, per_mol: List[Dict]) -> Optional[float]:
        """Return a genuine AiZynthFinder solve rate when fully configured.

        A forward recipe being non-empty is not evidence that an independent
        retrosynthesis planner can recover that route, so no proxy score is
        returned when the planner is unavailable or unconfigured.
        """
        if not bool(self.config.get("evaluation", {}).get("run_aizynthfinder", True)):
            self._retro_status = "disabled"
            return None
        if shutil.which("aizynthcli") is None:
            self._retro_status = "unavailable_missing_aizynthcli"
            logger.warning("AiZynthFinder CLI is unavailable; retrosynthetic feasibility is not reported.")
            return None
        config_path = self.aizynthfinder_config
        if not config_path or not os.path.isfile(config_path):
            self._retro_status = "unavailable_missing_config"
            logger.warning("AiZynthFinder configuration/model assets are not configured; retrosynthetic feasibility is not reported.")
            return None
        try:
            result = self._aizynth_run(per_mol)
            self._retro_status = "evaluated"
            return result
        except Exception as exc:  # pragma: no cover
            self._retro_status = "failed"
            logger.error("AiZynthFinder evaluation failed: %s", exc)
            return None

    def _aizynth_run(self, per_mol: List[Dict]) -> float:  # pragma: no cover
        """Run aizynthcli over the SMILES list; returns solve rate."""
        smiles_path = os.path.join(self.output_dir, "eval_smiles.txt")
        with open(smiles_path, "w") as f:
            for m in per_mol:
                if m["smiles"]:
                    f.write(m["smiles"] + "\n")
        out_dir = os.path.join(self.output_dir, "aizynth")
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.abspath(str(self.aizynthfinder_config))
        cmd = ["aizynthcli", "--config", config_path, "-i", smiles_path, "-o", out_dir]
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        result_path = os.path.join(out_dir, "results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = json.load(f)
            return float(np.mean([r.get("is_solved", False) for r in results]))
        return 0.0

    # ------------------------------------------------------------------
    # Docking
    # ------------------------------------------------------------------
    def _tool_availability(self) -> Dict[str, bool]:
        return {
            "gnina": shutil.which("gnina") is not None,
            "vina": shutil.which("vina") is not None,
            "smina": shutil.which("smina") is not None,
            "aizynthfinder": shutil.which("aizynthcli") is not None,
            "obabel": shutil.which("obabel") is not None,
        }

    def _dock(self, mols: List[Chem.Mol], pocket_pdb_path: str) -> Dict:  # pragma: no cover
        """Dock every molecule with GNINA (or Vina) and collect scores."""
        engine = self.docking_engine if shutil.which(self.docking_engine) else "vina"
        if not shutil.which(engine):
            return {"scores": []}

        scores: List[Optional[float]] = []
        with tempfile.TemporaryDirectory() as tmp:
            ligand_files = []
            for i, mol in enumerate(mols):
                if mol is None or mol.GetNumConformers() == 0:
                    scores.append(None)
                    continue
                lig_path = os.path.join(tmp, f"lig_{i}.sdf")
                writer = Chem.SDWriter(lig_path)
                writer.write(mol)
                writer.close()
                ligand_files.append((i, lig_path))

            for i, lig_path in ligand_files:
                out_log = os.path.join(tmp, f"dock_{i}.log")
                cmd = [
                    engine,
                    "--receptor", pocket_pdb_path,
                    "--ligand", lig_path,
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--out", os.path.join(tmp, f"out_{i}.sdf"),
                ]
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1800
                    )
                    score = self._parse_docking_score(proc.stdout or "")
                    scores.append(score)
                except Exception:
                    scores.append(None)
                _ = out_log

        return {"engine": engine, "scores": scores}

    @staticmethod
    def _parse_docking_score(stdout: str) -> Optional[float]:
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("Estimated free energy of binding"):
                try:
                    return float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    continue
            if line.startswith("Estimated Affinity"):
                try:
                    return float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    continue
        return None


__all__ = ["EvaluationPipeline"]
