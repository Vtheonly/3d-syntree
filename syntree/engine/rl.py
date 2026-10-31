"""Stage-2 rollout machinery and terminal reward for 3D-SynTree V2.

Stage 2 is GFlowNet Trajectory Balance (#29 FLOW-01); PPO and DPO were
removed from the V2 design as redundant Stage-2 paradigms (#5 ARCH-01).
This module retains the pieces shared by the GFlowNet engine:

* :class:`RolloutTransition` — a single policy transition captured during
  a chemistry-constrained rollout (all tensors detached/CPU-isolated so a
  buffer of transitions never pins the autograd graph).
* :func:`joint_log_prob_and_entropy` — re-evaluates a transition under the
  current policy; the GFlowNet TB loss needs its gradient.
* :func:`collect_episode` — rolls out one assembly-MDP episode.
* :class:`ThreeDReward` — the terminal reward (docking/contact/clash/
  physicochemical components). It reports when docking is unavailable
  instead of substituting a fabricated score.

Known V2 debt (tracked in #30 FLOW-02): the legacy batch Tanimoto
diversity term (``diversity_weight``) makes the terminal reward depend on
rollout-batch siblings, which violates the V2 stationary-reward contract
(#45 §12.4). It must be removed when the reward is re-engineered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, QED, rdFingerprintGenerator

from syntree.core.masking import LOGIT_MASK_LEGAL_ABOVE
from syntree.engine.evaluator import EvaluationPipeline

logger = logging.getLogger(__name__)

# Geometric cutoffs for the pharmacophore key-interaction reward (Angstrom).
SALT_BRIDGE_CUTOFF_A = 4.0
HBOND_CUTOFF_A = 3.5


def _morgan_fp(mol: Chem.Mol):
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    except (AttributeError, NameError):  # pragma: no cover - old RDKit
        from rdkit.Chem import AllChem

        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _positive_residue(residue: str) -> bool:
    return residue in ("LYS", "ARG", "HIP")


def _negative_residue(residue: str) -> bool:
    return residue in ("ASP", "GLU")


def count_key_interactions(
    ligand_mol: Optional[Chem.Mol],
    pocket_mol: Optional[Chem.Mol],
) -> Tuple[int, int]:
    """Count verified pocket-ligand key interactions.

    Returns ``(n_salt_bridges, n_hbonds)``:

    * **salt bridges** - a formally charged ligand atom within
      ``SALT_BRIDGE_CUTOFF_A`` (4.0 A) of a pocket atom carrying an
      opposite-sign physiological charge (Asp/Glu carboxylates, Lys/Arg
      cations, protonated His - the same protonation convention the pocket
      featurizer assigns, so the policy is rewarded for the interactions its
      input features describe);
    * **hydrogen bonds** - ligand donor (N/O/S bearing H) facing a pocket
      acceptor, or ligand acceptor facing a pocket donor (N/O/S of
      non-cationic / non-anionic residues), heavy-atom distance below
      ``HBOND_CUTOFF_A`` (3.5 A).

    Both molecules must carry conformers (the assembly env guarantees this);
    missing inputs simply yield zero interactions.
    """
    if ligand_mol is None or pocket_mol is None:
        return 0, 0
    if ligand_mol.GetNumConformers() == 0 or pocket_mol.GetNumConformers() == 0:
        return 0, 0

    lig = Chem.RemoveHs(Chem.Mol(ligand_mol))
    lig_conf = lig.GetConformer()
    pocket_conf = pocket_mol.GetConformer()

    lig_pos = np.array(
        [list(lig_conf.GetAtomPosition(a.GetIdx())) for a in lig.GetAtoms()],
        dtype=np.float64,
    )
    pocket_pos = np.array(
        [list(pocket_conf.GetAtomPosition(a.GetIdx())) for a in pocket_mol.GetAtoms()],
        dtype=np.float64,
    )
    if lig_pos.size == 0 or pocket_pos.size == 0:
        return 0, 0

    distances = np.linalg.norm(
        lig_pos[:, None, :] - pocket_pos[None, :, :], axis=-1
    )

    from syntree.data.featurizer import assign_pocket_formal_charges

    pocket_charges = assign_pocket_formal_charges(pocket_mol)

    # --- salt bridges ---------------------------------------------------
    lig_charges = np.array(
        [float(a.GetFormalCharge()) for a in lig.GetAtoms()], dtype=np.float64
    )
    salt = 0
    charged_lig = np.nonzero(np.abs(lig_charges) > 0.5)[0]
    if charged_lig.size:
        close = distances[charged_lig] <= SALT_BRIDGE_CUTOFF_A
        opposite = (
            np.sign(lig_charges[charged_lig])[:, None]
            * np.sign(pocket_charges)[None, :]
        ) < -0.5
        salt = int(np.count_nonzero(close & opposite))

    # --- hydrogen bonds ---------------------------------------------------
    lig_donors, lig_acceptors = [], []
    for atom in lig.GetAtoms():
        if atom.GetAtomicNum() not in (7, 8, 16):
            continue
        is_donor = atom.GetTotalNumHs() > 0
        is_acceptor = atom.GetFormalCharge() <= 0
        if is_donor:
            lig_donors.append(atom.GetIdx())
        if is_acceptor:
            lig_acceptors.append(atom.GetIdx())

    pocket_donors, pocket_acceptors = [], []
    for atom in pocket_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        residue = info.GetResidueName().strip().upper() if info else ""
        if atom.GetAtomicNum() not in (7, 8, 16):
            continue
        if _negative_residue(residue) and atom.GetAtomicNum() == 8:
            # carboxylate oxygen: pure acceptor
            pocket_acceptors.append(atom.GetIdx())
        elif _positive_residue(residue) and atom.GetAtomicNum() == 7:
            # cationic nitrogen: pure donor
            pocket_donors.append(atom.GetIdx())
        else:
            pocket_donors.append(atom.GetIdx())
            pocket_acceptors.append(atom.GetIdx())

    hbonds = 0
    for lig_idx in lig_donors:
        for pocket_idx in pocket_acceptors:
            if distances[lig_idx, pocket_idx] <= HBOND_CUTOFF_A:
                hbonds += 1
                break
    for lig_idx in lig_acceptors:
        for pocket_idx in pocket_donors:
            if distances[lig_idx, pocket_idx] <= HBOND_CUTOFF_A:
                hbonds += 1
                break

    return salt, hbonds


@dataclass
class ThreeDRewardConfig:
    docking_weight: float = 1.0
    clash_weight: float = 0.25
    contact_weight: float = 0.25
    fsp3_weight: float = 0.15
    qed_weight: float = 0.15
    validity_weight: float = 0.5
    docking_scale: float = 10.0
    clash_scale: float = 25.0
    contact_scale: float = 5.0
    # tasklist Priority 6: mode-collapse prevention. Defaults keep the
    # legacy reward exactly; production configs opt in explicitly.
    diversity_weight: float = 0.0
    key_interaction_weight: float = 0.0


class ThreeDReward:
    """Bounded terminal reward for a generated ligand."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = ThreeDRewardConfig(
            docking_weight=float(cfg.get("docking_weight", 1.0)),
            clash_weight=float(cfg.get("clash_weight", 0.25)),
            contact_weight=float(cfg.get("contact_weight", 0.25)),
            fsp3_weight=float(cfg.get("fsp3_weight", 0.15)),
            qed_weight=float(cfg.get("qed_weight", 0.15)),
            validity_weight=float(cfg.get("validity_weight", 0.5)),
            docking_scale=float(cfg.get("docking_scale", 10.0)),
            clash_scale=float(cfg.get("clash_scale", 25.0)),
            contact_scale=float(cfg.get("contact_scale", 5.0)),
            diversity_weight=float(cfg.get("diversity_weight", 0.0)),
            key_interaction_weight=float(cfg.get("key_interaction_weight", 0.0)),
        )

    # ------------------------------------------------------------------
    # Batch diversity (tasklist Priority 6)
    # ------------------------------------------------------------------
    @staticmethod
    def batch_diversity(mols: Sequence[Optional[Chem.Mol]]) -> List[float]:
        """Per-molecule batch Tanimoto diversity.

        ``R_div(M_i) = mean_{j != i} (1 - Tanimoto(M_i, M_j))`` over the
        molecules of the same rollout batch. Identical molecules score 0,
        singleton batches (or batches with one valid molecule) score 1.0 by
        convention (nothing to be redundant with).
        """
        valid = [Chem.RemoveHs(Chem.Mol(m)) for m in mols if m is not None]
        n = len(valid)
        if n == 0:
            return [0.0 for _ in mols]
        if n == 1:
            return [1.0 if m is not None else 0.0 for m in mols]

        fps = [_morgan_fp(m) for m in valid]
        diversities = []
        for i, fp in enumerate(fps):
            others = [f for j, f in enumerate(fps) if j != i]
            sims = DataStructs.BulkTanimotoSimilarity(fp, others)
            diversities.append(float(np.mean([1.0 - s for s in sims])))

        out: List[float] = []
        k = 0
        for mol in mols:
            if mol is None:
                out.append(0.0)
            else:
                out.append(diversities[k])
                k += 1
        return out

    def compute(
        self,
        mol: Chem.Mol,
        pocket_pdb_path: Optional[str] = None,
        clash_score: float = 0.0,
        contact_energy: float = 0.0,
        pocket_mol: Optional[Chem.Mol] = None,
        batch_reference_smiles: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        components: Dict[str, float] = {
            "docking": 0.0,
            "clash": 0.0,
            "contact": 0.0,
            "fsp3": 0.0,
            "qed": 0.0,
            "validity": 0.0,
            "key_interaction": 0.0,
            "diversity": 0.0,
        }
        if mol is None:
            return {"reward": 0.0, **components, "docking_available": 0.0}

        try:
            heavy = Chem.RemoveHs(Chem.Mol(mol))
            Chem.SanitizeMol(Chem.Mol(heavy))
            components["validity"] = 1.0
        except Exception:
            return {"reward": 0.0, **components, "docking_available": 0.0}

        try:
            from syntree.chemistry.validator import ChemicalValidator

            validator = ChemicalValidator()
            descriptors = validator.descriptors(mol)
            checks = validator.validate(mol)
            components["validity"] = 1.0 if all(checks.values()) else 0.0
        except Exception:
            descriptors = {}

        fsp3 = float(descriptors.get("fsp3", 0.0))
        components["fsp3"] = float(np.tanh(max(0.0, fsp3 - 0.42) / 0.20))
        try:
            components["qed"] = float(QED.qed(heavy))
        except Exception:
            components["qed"] = 0.0

        clash_score = float(clash_score)
        components["clash"] = float(
            np.exp(-max(0.0, clash_score) / self.cfg.clash_scale)
        )

        # Lennard-Jones contact energy: negative when the ligand occupies the
        # pocket at van der Waals contact (favourable dispersion), positive
        # under steric overlap, and ~0 when the ligand drifts into solvent.
        # tanh maps it to a bounded reward that is maximal for a well-packed
        # pose and *negative* for a clashing one, so the agent can no longer
        # "win" by leaving the pocket.
        contact_energy = float(contact_energy)
        components["contact"] = float(
            np.tanh(-contact_energy / max(self.cfg.contact_scale, 1e-6))
        )

        # tasklist Priority 6: pharmacophore key-interaction bonus. One
        # verified salt bridge earns the full weight (tasklist suggests
        # +1.5 with weight 1.5); hydrogen bonds contribute up to half the
        # bonus so multiple weak contacts cannot fully replace a real
        # ionic anchor.
        if self.cfg.key_interaction_weight > 0.0:
            try:
                n_salt, n_hbond = count_key_interactions(mol, pocket_mol)
                components["n_salt_bridges"] = float(n_salt)
                components["n_hbonds"] = float(n_hbond)
                components["key_interaction"] = float(
                    min(1.0, n_salt + 0.5 * min(n_hbond, 2))
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("key-interaction counting failed: %s", exc)

        # tasklist Priority 6: batch Tanimoto diversity against the other
        # molecules of the same rollout batch.
        if self.cfg.diversity_weight > 0.0 and batch_reference_smiles:
            try:
                reference_mols = [
                    Chem.MolFromSmiles(s) for s in batch_reference_smiles
                ]
                reference_mols = [m for m in reference_mols if m is not None]
                if reference_mols:
                    query_fp = _morgan_fp(Chem.RemoveHs(Chem.Mol(mol)))
                    ref_fps = [_morgan_fp(Chem.RemoveHs(m)) for m in reference_mols]
                    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
                    components["diversity"] = float(
                        np.mean([1.0 - s for s in sims])
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("batch diversity failed: %s", exc)

        docking_available = False
        if pocket_pdb_path:
            try:
                evaluator = EvaluationPipeline(
                    {
                        "evaluation": {
                            "docking_engine": "gnina",
                            "run_aizynthfinder": False,
                        }
                    },
                    output_dir="./experiments/rl_docking",
                )
                available = evaluator._tool_availability()
                engine = "gnina" if available.get("gnina") else (
                    "vina" if available.get("vina") else None
                )
                if engine:
                    docking_available = True
                    result = evaluator._dock([mol], pocket_pdb_path)
                    scores = [s for s in result.get("scores", []) if s is not None]
                    if scores:
                        components["docking"] = float(
                            np.tanh(-float(scores[0]) / self.cfg.docking_scale)
                        )
                    else:
                        docking_available = False
            except Exception:
                docking_available = False

        reward = (
            self.cfg.docking_weight * components["docking"]
            + self.cfg.clash_weight * components["clash"]
            + self.cfg.contact_weight * components["contact"]
            + self.cfg.fsp3_weight * components["fsp3"]
            + self.cfg.qed_weight * components["qed"]
            + self.cfg.validity_weight * components["validity"]
            + self.cfg.key_interaction_weight * components["key_interaction"]
            + self.cfg.diversity_weight * components["diversity"]
        )
        return {
            "reward": float(reward),
            **components,
            "docking_available": float(docking_available),
        }


def _isolate_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Detach a rollout tensor and move it to CPU (CUDA OOM fix, Stage 2).

    Every tensor stored inside a :class:`RolloutTransition` must be isolated
    so that (a) no reference to the forward autograd graph survives between
    rollout steps - the accumulated graph was the memory leak that OOM-ed
    the T4 during Stage-2 minibatch replay - and (b) GPU memory is released
    as soon as the rollout moves on to the next observation.
    """
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return tensor
    return tensor.detach().cpu()


def _isolate_state(state):
    """Detach + CPU every tensor field of a PyG observation (``state``).

    A torch_geometric ``Data`` observation is a store of tensors; the whole
    store is cloned and each tensor attribute is detached and moved to CPU
    so the stored snapshot owns no autograd graph and no CUDA allocation.
    """
    if state is None or not torch.is_tensor(state) and not hasattr(state, "keys"):
        return state
    if torch.is_tensor(state):
        return state.detach().cpu()
    clone = state.clone()
    try:
        for key in list(clone.keys()):
            value = clone[key]
            if torch.is_tensor(value):
                clone[key] = value.detach().cpu()
    except Exception:  # non-PyG state object: keep as-is
        return state
    return clone


@dataclass
class RolloutTransition:
    """A single policy transition plus its environment feedback.

    Invariant (Stage-2 CUDA OOM fix): every tensor field is explicitly
    ``.detach().cpu()``-ed in ``__post_init__`` so transitions stored in a
    rollout buffer never pin the forward autograd graph or VRAM between
    rollout steps, regardless of how the transition was constructed.

    ``reward`` is the reward received *after* taking the action (dense
    physics shaping during growth; the terminal reward is added to the
    final transition of the episode).
    """

    state: object
    reaction_mask: torch.Tensor
    synthon_masks: torch.Tensor
    family_idx: int
    action_idx: int
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.state = _isolate_state(self.state)
        self.reaction_mask = _isolate_tensor(self.reaction_mask)
        self.synthon_masks = _isolate_tensor(self.synthon_masks)
        self.old_log_prob = _isolate_tensor(self.old_log_prob)
        self.old_value = _isolate_tensor(self.old_value)


def joint_log_prob_and_entropy(
    model: torch.nn.Module,
    catalog,
    device: torch.device,
    transition: RolloutTransition,
):
    """Re-evaluate a transition under the *current* policy.

    Used by the GFlowNet Trajectory Balance loss (it needs
    d/dtheta of sum_t log P_F) and evaluation tooling. Returns
    ``(joint_log_prob, state_value, entropy)`` where the joint log-prob is
    ``log P(reaction family) + log P(synthon | family)`` (synthon-only for
    STOP).
    """
    batch = transition.state.to(device)
    reaction_mask = transition.reaction_mask.to(device)
    synthon_masks = transition.synthon_masks.to(device)
    family = int(transition.family_idx)
    selected_mask = synthon_masks[:, family, :]

    out = model(
        batch,
        catalog.embeddings.to(device),
        selected_mask,
        reaction_mask,
    )
    action_log_prob = out["synthon_log_probs"][0, transition.action_idx]
    if transition.action_idx == len(catalog):
        joint_log_prob = action_log_prob
    else:
        reaction_log_prob = out["reaction_log_probs"][0, family]
        joint_log_prob = reaction_log_prob + action_log_prob

    reaction_probs = out["reaction_log_probs"][0].exp()
    synthon_probs = out["synthon_log_probs"][0].exp()
    reaction_entropy = -(reaction_probs * out["reaction_log_probs"][0]).sum()
    synthon_entropy = -(synthon_probs * out["synthon_log_probs"][0]).sum()

    return (
        joint_log_prob,
        out["state_value"][0],
        reaction_entropy + synthon_entropy,
    )


def collect_episode(
    env,
    model: torch.nn.Module,
    catalog,
    pocket_pdb_path: Optional[str] = None,
    terminal_reward_fn=None,
    sample: bool = True,
    temperature: float = 1.0,
    seed_synthon_idx: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[List[RolloutTransition], Dict]:
    """Roll out one episode of the assembly MDP under the current policy."""
    from syntree.engine.environment import AssemblyAction

    observation = env.reset(
        pocket_pdb_path=pocket_pdb_path,
        seed_synthon_idx=seed_synthon_idx,
        max_steps=max_steps,
    )
    device = next(model.parameters()).device
    synthon_embeddings = catalog.embeddings.to(device)

    transitions: List[RolloutTransition] = []
    reward_details: Dict = {}

    with torch.no_grad():  # Auditing F-11: Prevent autograd graph retention during RL rollouts
        while not env.done:
            reaction_mask, synthon_masks, has_handle = env.legal_action_masks()
            if not has_handle or not bool(
                (reaction_mask > LOGIT_MASK_LEGAL_ABOVE).any().item()
            ):
                break
            decision = model.act(
                observation.to(device),
                synthon_embeddings,
                reaction_compatibility_mask=reaction_mask.to(device),
                synthon_masks_by_reaction=synthon_masks.to(device),
                sample=sample,
                temperature=temperature,
            )
            transition = RolloutTransition(
                state=observation.clone().cpu(),
                reaction_mask=reaction_mask.detach().cpu(),
                synthon_masks=synthon_masks.detach().cpu(),
                family_idx=int(decision["reaction_family_idx"][0].item()),
                action_idx=int(decision["action_idx"][0].item()),
                old_log_prob=decision["joint_log_prob"][0].detach().cpu(),
                old_value=decision["state_value"][0].detach().cpu(),
            )
            action = AssemblyAction(
                reaction_family_idx=int(decision["reaction_family_idx"][0].item()),
                synthon_idx=int(decision["synthon_idx"][0].item()),
                dihedral_rad=float(decision["dihedral"][0].item()),
                action_idx=int(decision["action_idx"][0].item()),
            )
            observation, step_reward, done, info = env.step(action)
            transition.reward = float(step_reward)
            transition.done = bool(done)
            transitions.append(transition)

    if terminal_reward_fn is not None:
        try:
            reward_details = terminal_reward_fn(env)
        except Exception as exc:
            logger.warning("terminal reward failed (%s); using zero.", exc)
            reward_details = {"reward": 0.0}
        terminal = float(reward_details.get("reward", 0.0))
        if transitions:
            transitions[-1].reward += terminal
    return transitions, reward_details


def apply_terminal_rewards(
    batch: List[Tuple[Optional[object], Optional[Chem.Mol], Dict[str, float]]],
    reward_fn: ThreeDReward,
) -> List[Dict[str, float]]:
    """Finalize a rollout batch with the diversity-aware terminal rewards.

    Args:
        batch: list of ``(transitions, final_mol, base_reward_details)``
            collected for the *same* rollout batch (the diversity term is
            only meaningful within one batch, exactly as the tasklist's
            R_div formula prescribes).
        reward_fn: the configured :class:`ThreeDReward`.

    Returns:
        Per-episode reward details with the ``diversity`` component added
        and the final ``reward`` updated; the terminal reward has been added
        onto the last transition of every episode (exactly where GAE
        bootstrapping expects it).
    """
    mols = [mol for _, mol, _ in batch]
    if reward_fn.cfg.diversity_weight > 0.0:
        diversities = ThreeDReward.batch_diversity(mols)
    else:
        diversities = [0.0] * len(batch)

    finalized: List[Dict[str, float]] = []
    for (transitions, mol, base), diversity in zip(batch, diversities):
        details = dict(base)
        details["diversity"] = float(diversity)
        details["reward"] = float(base.get("reward", 0.0)) + float(
            reward_fn.cfg.diversity_weight * diversity
        )
        if transitions:
            transitions[-1].reward += float(details["reward"])
        finalized.append(details)
    return finalized


__all__ = [
    "ThreeDRewardConfig",
    "ThreeDReward",
    "RolloutTransition",
    "collect_episode",
    "count_key_interactions",
    "apply_terminal_rewards",
    "joint_log_prob_and_entropy",
    "SALT_BRIDGE_CUTOFF_A",
    "HBOND_CUTOFF_A",
]