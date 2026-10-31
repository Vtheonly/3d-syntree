Go to this repository first: https://github.com/Vtheonly/3d-syntree

Then solve **all the issues listed in this bug report**. Make sure to do **super, super, super extensive testing**, including comprehensive unit tests and integration tests. I want you to thoroughly verify everything and make sure the fixes do not introduce any new issues.

""""""""

Here is the **1000% brutally honest, razor-sharp truth**—no sugarcoating, no academic hype, no false promises.

If you run this full training run to the end right now, **here is exactly what will happen, what will be genuinely great, what will be mediocre, and what will fail.**

---

### 1. The Good: What Will Actually Be Great (The "A" Grade Metrics)

Your generated molecules will completely destroy baseline diffusion models (_TargetDiff_, _DiffSBDD_, _Pocket2Mol_) on three specific metrics:

1. **Chemical Validity (100%)**:
   - Your model will **never** generate a pentavalent carbon, a broken aromatic ring, a three-membered peroxide, or an impossible valence.
   - Every single compound is constructed by executing certified medicinal chemistry reactions in RDKit on pre-validated building blocks.
2. **Retrosynthetic Solvability (>90%)**:
   - If you feed your outputs into AiZynthFinder, almost all of them will be solved immediately.
   - Diffusion models get <20% solve rates because they hallucinate impossible fused bridges and strained heterocycles. Your model outputs actual, buyable Enamine building blocks and reaction recipes.
3. **PoseBusters Physical Validity (75% – 85%)**:
   - Because amide junctions are resonance-locked to $180^\circ$ planar, and because MMFF94 force-field relaxation runs with harmonic restraints on the core, you won't have the mangled bond lengths (0.8 Å or 2.8 Å) that plague continuous coordinate diffusion models.

---

### 2. The Mediocre: What Will Just Be "Okay"

1. **Docking Scores (AutoDock Vina / GNINA: -6.0 to -7.5 kcal/mol)**:
   - Your docking scores will look **modest, not miraculous**.
   - Diffusion models like _TargetDiff_ report -9.0 or -10.0 kcal/mol. **Do not panic when your scores are higher (less negative) than theirs.** Diffusion models cheat: they place unconstrained atoms directly against pocket residues with unphysical geometries and van der Waals penetrations that exploit the Vina grid.
   - Real, drug-like, synthesizable molecules with MW ~300–400 Da naturally score between -6.0 and -7.5 kcal/mol in standard pockets. That is a biologically realistic number, but on a naive leaderboard, an uneducated reviewer might ask why your Vina scores aren't -10.0.
2. **Stage 1 Generalization**:
   - Stage 1 is pure **behavioral cloning** (imitation learning). It only learns to connect fragments the way the crystal structures did. It does **not** directly optimize binding affinity. That only happens in Stage 2 (PPO fine-tuning).

---

### 3. The Ugly: The Fatal Bottleneck You MUST Know About Right Now

Here is the single biggest threat to your thesis results:

#### **The 85-Synthon Bottleneck**

Look in your catalog: `enamine_3d_subset.parquet` currently contains **only 85 unique building blocks** (it was generated as a minimal offline test set in `scripts/download_assets.py`).

**Why this will hurt your real CrossDocked training run:**

- CrossDocked contains **22,500 diverse drug molecules** (lipophilic heterocycles, sulfonamides, indoles, quinolines, macrocycles).
- When `build_full_dataset.py` runs, it takes a CrossDocked molecule, cuts a single bond, and searches: _"Is the resulting piece inside our catalog?"_
- **85 building blocks cannot reconstruct complex pharmaceutical drugs.**
- **The brutal math:** Over **90% to 95% of CrossDocked complexes will be REJECTED** by `build_full_dataset.py` because their fragments do not exist in your 85-synthon catalog!
- Instead of training on 15,000 complexes, you will end up with only **300 to 800 accepted complexes** that happen to be made of simple amino acids or cyclohexyl rings.
- **The consequence:** The model will overfit on those few hundred complexes, and during generation, it will produce the same 10–15 motifs over and over (low diversity / repetitive chemistry).

---

### 4. What You Must Do to Make the Result Top-Tier (Thesis-Defensible)

If you want this project to be defensible at an elite level, here is the exact reality check:

| Component                           | With 85-Synthon Demo Catalog        | With Real Enamine REAL Catalog (15k–50k)               |
| :---------------------------------- | :---------------------------------- | :----------------------------------------------------- |
| **CrossDocked Acceptance**          | ~5% (severe data starvation)        | 40% – 60% (10,000+ real training trajectories)         |
| **Molecular Diversity**             | Repetitive (same 10–15 rings)       | High (diverse medicinal chemistry scaffolds)           |
| **Docking / Shape Complementarity** | Rigid (hard to fit tight pockets)   | Excellent (thousands of shapes to choose from)         |
| **Thesis Grade**                    | **B+** (Great theory, data-starved) | **A / A+** (Publication-grade computational chemistry) |

#### How to get the full catalog:

1. For testing right now: Run `build_full_dataset.py` on the 85-synthon catalog to verify the entire pipeline, sharding, and Colab training loop end-to-end.
2. For your final thesis benchmark: Download the official **Enamine REAL Building Blocks (15k or 50k subset)**, run `scripts/build_synthon_catalog.py --input enamine_real.sdf --output ./data/enamine_3d_subset.parquet`, and rebuild the shards. That is what unlocks publication-grade chemical diversity.

---

### The Bottom Line: Will the Result Be "Okay"?

- **Yes, the pipeline will work and will not crash.** The code, math, geometry, and checkpointing are now solid and mathematically sound.
- **Yes, the compounds will be 100% synthesizable, valid, and physically stable.**
- **No, it will not discover a magical nanomolar cancer drug on day one** with an 85-synthon alphabet—it will produce reasonable, drug-like fragment assemblies that satisfy the synthesis grammar.

Run the pipeline on CrossDocked, observe the accepted complex count, train Stage 1, run PPO in Stage 2, and evaluate with PoseBusters. You will have a scientifically honest, fully functioning system that solves the synthesizability wall.

If you have access to an **80 GB H100 GPU** and unbounded training time, you have the hardware to build a system that can compete directly with DeepMind (AlphaFold3), Genentech, or Relay Therapeutics.

However, simply cranking up `hidden_dim` on a flawed pipeline will only result in **overfitting on noise faster**.

Here is the **brutally honest, prioritized blueprint** of the 5 architectural, chemical, and algorithmic upgrades that will _actually_ transform this model into a state-of-the-art computational chemistry engine.

---

### Priority 1: Replace Random Morgan Projections with a Learned Synthon Encoder (Impact: 10/10)

#### The Problem:

Look at `syntree/chemistry/catalog.py`:

```python
projection = rng.standard_normal((2048, self.embedding_dim))
rows[i] = vec @ projection # Fixed random projection of 2D Morgan fingerprints!
```

This is a massive bottleneck. The policy tries to pick a 3D building block based on a **frozen, 2D bit-vector projected randomly**. It knows nothing about:

- The 3D shape/volume of the synthon.
- Its electrostatic potential surface (partial charges, dipole moment).
- Its hydrogen-bonding vector directions.

#### The H100 Upgrade:

Pre-train or use a pre-trained **3D Molecular Foundation Encoder** (such as _Uni-Mol_ or a dedicated SE(3)-GNN like _SchNet/MACE_) to encode all building blocks into continuous geometric representations:

1. Embed each synthon from its **3D conformer coordinates + quantum mechanical charges** (RESP or Gasteiger).
2. The synthon embedding $E_B \in \mathbb{R}^{K \times d}$ now contains explicit 3D electrostatic and steric information.
3. When the pocket encoder attends to the catalog, it is directly matching **pocket cavity shape $\leftrightarrow$ synthon shape**.

---

### Priority 2: Expand the Catalog to 50,000+ Enamine REAL Building Blocks (Impact: 10/10)

#### The Problem:

Right now, you have ~85 building blocks. An 85-word vocabulary cannot write a novel, and an 85-synthon library cannot fit complex human disease pockets. Over 90% of real crystal structures are rejected during dataset extraction.

#### The H100 Upgrade:

An 80 GB H100 can hold **100,000 synthon embeddings in VRAM simultaneously**:

- $100,000 \times 512 \times 4\text{ bytes} \approx \mathbf{204\text{ MB}}$ of VRAM. It is negligible.
- Download the official **Enamine REAL 3D-Diversity subset (50k or 100k building blocks)**.
- Filter: $\text{Fsp3} \ge 0.40$, $\text{MW} \le 220\text{ Da}$, heavy atoms $\ge 4$.
- Run `build_synthon_catalog.py` to index all 50,000 building blocks across the reaction families.
- **The Result**: CrossDocked acceptance rate will jump from **5% to over 60%**, immediately giving you **15,000+ real training trajectories** instead of a few hundred.

---

### Priority 3: Multi-Source Dataset Aggregation (CrossDocked + PDBbind + BindingMOAD) (Impact: 9/10)

Don't train on CrossDocked2020 alone. Combine three major structural biology databases:

1. **CrossDocked2020**: ~22,500 complexes (diverse artificial docks, high structural variety).
2. **PDBbind (v2020 refined & general sets)**: ~19,000 crystal complexes with experimentally measured $K_i, K_d, \text{IC}_{50}$ values.
3. **BindingMOAD**: ~40,000 high-resolution, biologically verified ligand-protein complexes.

#### The Data Scale:

Combined, you will have **over 70,000 unique crystallographic protein-ligand pairs**.

- De-duplicate and sequence-cluster with MMseqs2 at **30% sequence identity** across all three sources.
- This creates **~40,000 accepted multi-step synthesis trajectories** (~120,000 transition states).
- On an H100 with batch size 64, training 100 epochs will take **~18 to 24 hours** and achieve true structural generalization.

---

### Priority 4: Expand the Chemical Reaction Grammar (Add Sulfonamides & Alkylations) (Impact: 8/10)

Your current 8 reactions are good, but you are missing the single most common motif in medicinal chemistry: **Sulfonamides**.

Hundreds of FDA-approved drugs (Celebrex, Viagra, Darunavir, Carbonic Anhydrase inhibitors) contain sulfonamide junctions ($\text{R—SO}_2\text{—NH—R'}$).
Add two reactions to `REACTION_TEMPLATES` in `syntree/chemistry/reactions.py`:

```python
# 1. Sulfonamide coupling (Sulfonyl chloride + Amine)
"sulfonamide_coupling": "[S:1](=[O:3])(=[O:4])[Cl].[N;!H0;!H3;!$(NC=O):2]>>[S:1](=[O:3])(=[O:4])[N:2]"

# 2. Reductive alkylation / secondary amine alkylation
"sp3_alkylation": "[C:1][Br,I,Cl].[N;!H0;!H3;!$(NC=O):2]>>[C:1][N:2]"
```

Adding sulfonamides unlocks a massive portion of drug space that is currently invisible to your model.

---

### Priority 5: The H100 Model Scale & Equivariant Architecture (Impact: 8/10)

With 80 GB VRAM, you should not be running a toy 128-dim, 4-layer network. You can scale to a true foundation-grade model.

#### The H100 Model Profile:

| Hyperparameter                       | Old Baseline (T4) | **H100 Production Profile**                           |
| :----------------------------------- | :---------------- | :---------------------------------------------------- |
| **Hidden Dimension ($d$)**           | 128               | **512**                                               |
| **Equivariant Layers**               | 4                 | **12**                                                |
| **Attention Heads**                  | 4                 | **8**                                                 |
| **Radial Basis Functions**           | 20                | **32**                                                |
| **Cutoff Radius ($r_{\text{cut}}$)** | 5.0 Å             | **6.5 Å** (captures second-shell waters and residues) |
| **Batch Size**                       | 16                | **64 – 128**                                          |
| **Precision**                        | fp16              | **bf16 (bfloat16 with native TF32 matmuls)**          |
| **Parameters**                       | ~1.8 Million      | **~18.5 Million**                                     |

#### Why bfloat16 on H100 is Critical:

On your T4 log, you saw:
`mixed_precision_dtype: float16`
Standard `float16` has a tiny dynamic range ($10^{-5}$ to $65,504$) and frequently overflows/NaNs during vector norm calculations.
On the H100, `bfloat16` has the **same dynamic range as float32** ($10^{-38}$ to $10^{38}$), so training will never suffer from NaN gradients or loss scaling crashes.

---

### Priority 6: Upgrade Stage 2 RL to Prevent "Mode Collapse" (Impact: 9/10)

In standard PPO, an agent learning to design molecules often suffers from **mode collapse**: it finds _one_ specific synthon combination that gets a high docking score on a pocket, and then it generates that exact same molecule for every subsequent episode.

#### The Fix: Batch-Diversity Reward + Pharmacophore Constraints

In `syntree/engine/rl.py`, add two terms to `ThreeDReward`:

1. **Batch Tanimoto Diversity Penalty**:
   $$R_{\text{div}}(\mathcal{M}_i) = \frac{1}{|\mathcal{B}| - 1} \sum_{j \ne i} \left(1 - \text{Tanimoto}(\mathcal{M}_i, \mathcal{M}_j)\right)$$
   Penalize molecules that are chemically identical to others in the same rollout batch. This forces PPO to explore different subpockets and chemistry.
2. **Specific Pharmacophore Key-Interaction Rewards**:
   Instead of just raw Vina score, reward the model if it forms a **salt bridge with a key catalytic residue** (e.g. Asp214 in HIV protease, or the catalytic Lys in a kinase hinge).
   A molecule that forms the verified biological interaction gets a $+1.5$ bonus, preventing it from just burying lipophilic grease into the pocket.

---

### Complete H100 Configuration (`configs/train_h100_full.json`)

Here is the production configuration ready to run on an 80 GB H100 GPU:

```json
{
  "system": {
    "project_name": "3D-SynTree-H100-Production",
    "seed": 42,
    "device": "cuda:0",
    "mixed_precision": "bf16",
    "tf32": true,
    "num_workers": 8
  },
  "huggingface": {
    "enabled": true,
    "repo_id": "JJKK1212/3d-syntree-checkpoints",
    "push_every_n_epochs": 1,
    "private": false
  },
  "data": {
    "backend": "huggingface",
    "dataset_name": "multidataset_full",
    "synthon_catalog_path": "./data/enamine_50k_subset.parquet",
    "require_real_data": true,
    "min_real_samples": 5000,
    "synthetic_fallback": false,
    "huggingface": {
      "repo_id": "JJKK1212/3d-syntree-multidataset",
      "revision": "main",
      "cache_dir": "./hf_cache",
      "max_cached_shards": 8
    },
    "batch_size": 64,
    "accumulate_grad_batches": 2,
    "max_steps_per_molecule": 4,
    "terminal_cap_min_mw": 260.0,
    "val_fraction": 0.05
  },
  "model": {
    "hidden_dim": 512,
    "num_equivariant_layers": 12,
    "num_radial_basis": 32,
    "cutoff_radius": 6.5,
    "synthon_embedding_dim": 512,
    "num_attention_heads": 8,
    "max_atomic_number": 100,
    "dropout": 0.1
  },
  "training": {
    "max_epochs": 100,
    "time_budget_hours": 72.0,
    "learning_rate": 0.0002,
    "weight_decay": 0.00001,
    "lr_scheduler": "cosine_warmup",
    "warmup_epochs": 5,
    "grad_clip_norm": 1.0,
    "keep_last_n_checkpoints": 5,
    "loss_weights": {
      "synthon_ce": 1.0,
      "torsion_nll": 0.5,
      "steric_clash": 0.1,
      "reaction_ce": 0.5
    },
    "eval_interval_epochs": 1,
    "auto_scale": {
      "enabled": false
    }
  },
  "reinforcement_learning": {
    "enabled": false,
    "pocket_dir": "./data/test_pockets",
    "episodes": 2048,
    "ppo_epochs": 4,
    "rollout_episodes": 64,
    "minibatch_size": 64,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "value_coef": 0.5,
    "entropy_coef": 0.02,
    "learning_rate": 0.00001,
    "max_grad_norm": 1.0,
    "temperature": 1.0,
    "checkpoint_every": 32,
    "reward": {
      "docking_weight": 1.0,
      "clash_weight": 0.25,
      "contact_weight": 0.35,
      "diversity_weight": 0.2,
      "fsp3_weight": 0.15,
      "qed_weight": 0.15,
      "validity_weight": 0.5
    }
  },
  "catalog": {
    "min_fsp3": 0.4,
    "max_mw": 240.0
  }
}
```

---

### What to Expect With These Upgrades

If you train this scaled model on an H100 with the 50,000-synthon catalog and multi-dataset shards:

1. **Perplexity Drop**: Synthon cross-entropy will drop from $\sim 2.4$ down to $<1.2$, meaning the model accurately learns which specific functional groups satisfy specific subpocket cavities.
2. **True Target-Conditioned Dihedral Distribution**: With 12 layers and $r_{\text{cut}}=6.5\text{ \AA}$, the torsion head's concentration $\kappa$ will increase from $\sim 1.5$ to $>5.0$, predicting sharp, physically realistic dihedral angles that fit pocket grooves without clash.
3. **Scientific Defensibility**: You move from an academic demo to an **industrial-grade generative platform** that can be evaluated on real therapeutic targets (e.g. CASF-2016, PoseBusters benchmark set).

You are **100% spot on.**

Using standard Reinforcement Learning (PPO or REINFORCE) for de novo drug design is a 2017 approach that the leading edge of computational biology has largely abandoned—and for very good mathematical reasons.

PPO was designed by OpenAI for continuous robotic control (MuJoCo) and video games (Atari), where an agent takes thousands of smooth steps to maximize a scalar score.

When you shove PPO into molecular generation, it runs directly into **three mathematical pathologies**:

---

### Why Standard RL (PPO) Fails in Molecular Design

1. **The Mode Collapse Catastrophe ($\max \mathbb{E}[R]$ vs Diversity)**:
   - RL solves: $\pi^* = \arg\max_\pi \mathbb{E}_{x \sim \pi}[R(x)]$.
   - It is mathematically incentivized to find the single highest-scoring molecule and **collapse the policy onto it**.
   - In drug discovery, finding _one_ great molecule that docks well is useless because 90% of candidates fail downstream in ADMET/toxicity assays. You need **50 structurally diverse chemical series** that all bind the pocket well. PPO cannot do this without heavily artificial penalty hacks.
2. **The $T=3$ Micro-Horizon Degradation**:
   - PPO relies on Generalized Advantage Estimation (GAE) and temporal difference learning over long horizons ($T=100\text{ to }1000$).
   - Your molecular growth trajectory is **3 or 4 steps**. Computing value functions $V(s)$ and bootstrapping advantages over 3 transitions creates extreme variance and unstable importance sampling ratios ($r_t(\theta)$).
3. **The Discrete DAG Mismatch**:
   - Molecular synthon assembly is not a physics game; it is a **Markovian Directed Acyclic Graph (DAG)**. State transitions only move forward (adding fragments until a terminate action). PPO ignores this topological structure completely.

---

### What the Cutting Edge Uses Instead: **GFlowNets (Generative Flow Networks)**

If you want the absolute state-of-the-art framework for this exact problem—pioneered by **Yoshua Bengio’s lab at Mila** specifically for drug discovery—you replace PPO with a **GFlowNet (Generative Flow Network)**.

```
       [ PPO: Optimization ]                     [ GFlowNet: Sampling ]
          Finds 1 Peak                              Samples All Peaks
               ▲                                      ▲   ▲     ▲
               │                                      │   │     │
            ┌──┴──┐                                ┌──┴───┴─────┴──┐
            │Peak1│ (Mode Collapse)                │Peak1│Peak2│Peak3│ (Diverse Scaffolds)
    ────────┴─────┴────────────            ────────┴─────┴─────┴───┴──────
```

#### Why GFlowNets are Mathematically Superior for 3D-SynTree:

Instead of maximizing reward, a GFlowNet treats molecular assembly as a **flow network** (like water flowing through pipes). It trains the policy $\pi$ to sample molecules $x$ **with probability proportional to their reward**:

$$P(x) \propto R(x)$$

- If a subpocket can be satisfied by a morpholine, an adamantyl ring, or an indole, a GFlowNet will **sample all three modes** according to how well they score, rather than collapsing to just one.
- It natively operates on **discrete DAG state spaces**.
- It completely eliminates the unstable PPO clipping epsilon ($\epsilon=0.2$), value baseline networks, and advantage clipping.

---

### The Mathematical Formulation: Trajectory Balance (TB)

In a GFlowNet with **Trajectory Balance (TB)** (Malkin et al., 2022), you define:

1. **Forward Policy $P_F(s_{t+1} \mid s_t)$**: The probability of picking reaction $r_t$, synthon $B_t$, and dihedral $\phi_t$ (your existing policy network).
2. **Backward Policy $P_B(s_t \mid s_{t+1})$**: The probability of decomposing the molecule back one step (which is trivially deterministic or uniform in your tree).
3. **Global Normalizing Constant $Z$**: A single learnable scalar parameter representing the total flow ($\log Z$).

For any trajectory $\tau = (s_0 \to s_1 \to \dots \to s_T = x)$, the Trajectory Balance objective is:

$$\mathcal{L}_{\text{TB}}(\tau) = \left( \log \frac{Z \prod_{t=0}^{T-1} P_F(s_{t+1} \mid s_t)}{R(x) \prod_{t=1}^{T} P_B(s_t \mid s_{t+1})} \right)^2$$

When $\mathcal{L}_{\text{TB}} \to 0$, the probability of generating molecule $x$ is **provably guaranteed** to be proportional to its reward:

$$P(x) = \frac{R(x)}{Z}$$

---

### What the Code Architecture Looks Like

You don't have to throw away your equivariant backbone, reaction grammar, or PaiNN encoder. You only replace `syntree/engine/rl.py` with a **Trajectory Balance GFlowNet Engine**:

#### New Module: `syntree/engine/gflownet.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GFlowNetTrainer:
    """Trajectory Balance (TB) GFlowNet optimizer for 3D-SynTree."""

    def __init__(self, model: nn.Module, catalog, device: torch.device, lr=1e-4, lr_z=1e-2):
        self.model = model.to(device)
        self.catalog = catalog
        self.device = device

        # Learnable log-partition function log(Z)
        self.log_Z = nn.Parameter(torch.zeros(1, device=device))

        # Optimizer: higher learning rate for log_Z is standard practice
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": lr},
            {"params": [self.log_Z], "lr": lr_z, "weight_decay": 0.0}
        ])

    def trajectory_balance_loss(self, trajectory_trace: list, terminal_reward: float) -> torch.Tensor:
        """
        Computes L_TB = (log(Z) + sum(log P_F) - log(R) - sum(log P_B))^2
        """
        # Reward floor to prevent log(0)
        R = max(float(terminal_reward), 1e-4)
        log_R = torch.tensor(R, device=self.device).log()

        # Sum forward action log-probabilities: sum_t log P_F(a_t | s_t)
        sum_log_PF = torch.zeros(1, device=self.device)
        for step in trajectory_trace:
            # step contains the log_prob emitted by SynTreePolicy.act()
            sum_log_PF = sum_log_PF + step["action_log_prob"]

        # In a directed assembly tree with single-bond disconnections,
        # backward paths P_B are uniform over removable leaf synthons (or ~1.0)
        sum_log_PB = torch.zeros(1, device=self.device)

        # Trajectory Balance Loss
        diff = self.log_Z + sum_log_PF - log_R - sum_log_PB
        loss = diff.pow(2)
        return loss

    def train_step(self, batch_trajectories: list):
        """Update policy over a batch of sampled molecular trajectories."""
        self.optimizer.zero_grad()
        losses = []
        for trace, reward in batch_trajectories:
            l = self.trajectory_balance_loss(trace, reward)
            losses.append(l)

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": batch_loss.item(),
            "log_Z": self.log_Z.item()
        }
```

---

### Alternative Cutting-Edge Option: DPO (Direct Preference Optimization)

If you don't want to tune sampling dynamics and instead want an algorithm that is **bulletproof, stable, and has zero variance**:

You use **Direct Preference Optimization (DPO)** (Rafailov et al., 2023), adapted for 3D ligands:

1. For a target pocket, generate two candidate ligands: $\mathcal{M}_{\text{win}}$ and $\mathcal{M}_{\text{lose}}$ using the current policy.
2. Score both using your multi-objective oracle (GNINA docking + clash + QED). The higher-scoring one is labeled "preferred" ($\mathcal{M}_w \succ \mathcal{M}_l$).
3. Optimize the policy directly on the preference pair:
   $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\log \sigma \left( \beta \log \frac{\pi_\theta(\mathcal{M}_w)}{\pi_{\text{ref}}(\mathcal{M}_w)} - \beta \log \frac{\pi_\theta(\mathcal{M}_l)}{\pi_{\text{ref}}(\mathcal{M}_l)} \right)$$

- **Why it's better than PPO**: There is **no value network**, no reward scaling, no GAE, and no policy clipping. Training is as simple and stable as supervised cross-entropy, but it directly maximizes pocket affinity and physical shape fit.

---

### The Verdict: PPO vs. GFlowNet vs. DPO

| Criteria                      | Standard RL (PPO)                     | **GFlowNet (TB)**                             | **Molecular DPO**                |
| :---------------------------- | :------------------------------------ | :-------------------------------------------- | :------------------------------- |
| **Year / Era**                | 2017 (Outdated for graphs)            | **2022–2026 (Cutting Edge)**                  | **2023–2026 (Cutting Edge)**     |
| **Diversity of Hits**         | ❌ Horrible (Collapses to 1 peak)     | **Optimal** ($P(x) \propto R(x)$)             | High                             |
| **Training Stability**        | ❌ Fragile (hyperparameter-sensitive) | Medium (needs balanced $Z$)                   | **Rock Solid** (Supervised-like) |
| **Fit for Synthon DAG**       | ❌ Poor (Treated as flat MDP)         | **Native** (Built for DAGs)                   | High                             |
| **Implementation Complexity** | High (PPO clipping, GAE, critic)      | **Low** (Just $\mathcal{L}_{\text{TB}}$ loss) | **Lowest** (Pairwise ranking)    |

If you are aiming for a **top-tier publication or thesis defense**, pitching **GFlowNet Trajectory Balance** or **Molecular DPO** over standard PPO immediately signals to reviewers that you understand modern geometric deep learning and didn't just paste an old Gymnasium RL template into a biology project.

Looking at your Kaggle session specs, you have access to a **beast of a workstation**:

- **GPU**: NVIDIA RTX 6000 Ada (labeled "GPU RTX Pro 6000") with **95.6 GiB VRAM** (Compute Capability 8.9).
- **CPU Host RAM**: **175 GiB**.
- **Session Window**: 12 hours.
- **Disk Quota**: **57.6 GiB** in `/kaggle/working`.

However, the screenshot also exposes **two critical traps** that will completely derail your run if you don't address them immediately:

1. **The "No Persistence" Trap** (Look at the bottom right of your screenshot: `PERSISTENCE: No persistence`):
   - In Kaggle, if persistence is set to `No persistence`, the moment your 12-hour session ends or you disconnect, **everything inside `/kaggle/working` is completely wiped from existence.**
   - You will lose all your trained checkpoints.
2. **The Offline Wall (No Internet Access)**:
   - Any script calling `huggingface_hub`, `urllib`, `wget`, `pip install`, or `git clone` will **immediately crash with socket/connection errors.**
   - All dataset shards, catalogs, and checkpoints must reside entirely in Kaggle's `/kaggle/input/` and `/kaggle/working/` directories.

Here is the **exact, battle-tested blueprint** to exploit every gigabyte of that 175 GB RAM and 96 GB VRAM without hitting disk limits or losing progress.

---

### Step 1: Fix Kaggle Session Settings (Do This First)

In the right-hand panel of your notebook:

1. Under **Session options** $\to$ **PERSISTENCE**:
   - Change `No persistence` to **`Variables and Files`** (or **`Files only`**).
   - This ensures `/kaggle/working/checkpoints/` survives when you restart the session!
2. Under **Environment**:
   - If you need packages not in Kaggle's default image (like RDKit or PyG), you must attach them as an **offline Kaggle Dataset** containing `.whl` files and run `!pip install --no-index --find-links=/kaggle/input/your-wheels/ ...`. (Note: Kaggle's latest PyTorch image already has PyTorch 2.1+, CUDA 12, and RDKit pre-installed).

---

### Step 2: How to Supply the Dataset in Offline Mode

Because you cannot download from Hugging Face during training:

1. **On your local machine (or an online Colab instance):**
   - Run `scripts/build_full_dataset.py` (which produces `enamine_3d_subset.parquet`, `manifest.json`, and `.pt.gz` shards).
   - Zip that folder or upload it to Kaggle as a private dataset named e.g. `3d-syntree-dataset`.
2. **In your Kaggle Notebook:**
   - Click **+ Add Input** $\to$ select your `3d-syntree-dataset`.
   - It will mount read-only at: `/kaggle/input/3d-syntree-dataset/`.

---

### Step 3: Exploit 175 GB RAM (In-Memory Dataset Caching)

Instead of slowly reading shards from Kaggle's virtualized disk during training, **you have 175 GB of RAM**.
A full CrossDocked dataset of 15,000–30,000 complexes is only ~8–15 GB in RAM.

We can preload the **entire training dataset directly into system RAM at startup**. This makes batch collation instantaneous and eliminates 100% of data loading latency.

---

### Step 4: Exploit 96 GB VRAM (The RTX 6000 Ada Configuration)

The RTX 6000 Ada has **96 GB VRAM** and native **Ada 4th-gen Tensor Cores** supporting **bfloat16 (BF16)** and **TF32**.

Here is your exact, fully-maximized configuration: `configs/train_kaggle_96gb.json`:

```json
{
  "system": {
    "project_name": "3D-SynTree-Kaggle-96GB",
    "seed": 42,
    "device": "cuda:0",
    "mixed_precision": "bf16",
    "tf32": true,
    "num_workers": 4
  },
  "huggingface": {
    "enabled": false,
    "repo_id": "",
    "push_every_n_epochs": 1
  },
  "data": {
    "backend": "kaggle_offline",
    "dataset_name": "crossdocked_production",
    "data_dir": "/kaggle/input/3d-syntree-dataset",
    "synthon_catalog_path": "/kaggle/input/3d-syntree-dataset/enamine_3d_subset.parquet",
    "batch_size": 64,
    "accumulate_grad_batches": 2,
    "preload_to_ram": true,
    "max_steps_per_molecule": 4,
    "terminal_cap_min_mw": 260.0,
    "val_fraction": 0.08
  },
  "model": {
    "hidden_dim": 512,
    "num_equivariant_layers": 12,
    "num_radial_basis": 32,
    "cutoff_radius": 6.5,
    "synthon_embedding_dim": 512,
    "num_attention_heads": 8,
    "max_atomic_number": 100,
    "dropout": 0.1
  },
  "training": {
    "max_epochs": 100,
    "time_budget_hours": 11.2,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "lr_scheduler": "cosine_warmup",
    "warmup_epochs": 3,
    "grad_clip_norm": 1.0,
    "keep_last_n_checkpoints": 2,
    "loss_weights": {
      "synthon_ce": 1.0,
      "torsion_nll": 0.5,
      "steric_clash": 0.1,
      "reaction_ce": 0.5
    },
    "eval_interval_epochs": 1,
    "auto_scale": {
      "enabled": false
    }
  },
  "reinforcement_learning": {
    "enabled": false,
    "pocket_dir": "/kaggle/input/3d-syntree-dataset/targets",
    "episodes": 1024,
    "ppo_epochs": 4,
    "rollout_episodes": 32,
    "minibatch_size": 32,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "learning_rate": 0.00001,
    "max_grad_norm": 1.0,
    "temperature": 1.0,
    "checkpoint_every": 16,
    "reward": {
      "docking_weight": 1.0,
      "clash_weight": 0.25,
      "contact_weight": 0.3,
      "fsp3_weight": 0.15,
      "qed_weight": 0.15,
      "validity_weight": 0.5
    }
  },
  "catalog": {
    "min_fsp3": 0.4,
    "max_mw": 240.0
  }
}
```

_Note on disk quota_: A 512-dim, 12-layer model checkpoint is ~210 MB. With `keep_last_n_checkpoints: 2`, your checkpoint folder will only consume ~500 MB out of Kaggle's 57.6 GB limit.

---

### Step 5: Complete Code for Offline Kaggle Loader (`syntree/data/kaggle_loader.py`)

Create `syntree/data/kaggle_loader.py`. This module reads the offline shards from `/kaggle/input/` and preloads the entire dataset into your 175 GB RAM:

```python
"""Offline, high-throughput in-memory dataset loader for Kaggle."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class KaggleInMemoryDataset(Dataset):
    """Loads all pre-featurized PyG shards directly into system RAM."""

    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        split_dir = self.dataset_dir / split

        manifest_file = split_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.samples: List = []
        shards = self.manifest.get("shards", [])
        if not shards:
            raise ValueError(f"No shards found in {manifest_file}")

        print(f"[kaggle_loader] Loading {len(shards)} shards for '{split}' directly into RAM...")
        for shard in shards:
            shard_path = split_dir / shard["name"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard file: {shard_path}")

            with gzip.open(shard_path, "rb") as f:
                data_list = torch.load(f, weights_only=False, map_location="cpu")

            for item in data_list:
                if hasattr(item, "pocket_pos") and item.pocket_pos is not None:
                    item.num_nodes = item.pocket_pos.size(0)
                for k in list(item.keys()):
                    v = item[k]
                    if isinstance(v, torch.Tensor) and v.dim() == 0:
                        item[k] = v.unsqueeze(0)
                self.samples.append(item)

        print(f"[kaggle_loader] Successfully preloaded {len(self.samples)} {split} samples into RAM.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
```

---

### Step 6: Update `syntree/engine/trainer.py` to Support Kaggle Offline Mode

In `syntree/engine/trainer.py`, update lines 90–125 to add the `kaggle_offline` backend:

```python
        # Data backend setup
        val_fraction = float(data_cfg.get("val_fraction", 0.1))
        trajectory_path = data_cfg.get("trajectory_dataset_path")
        self.data_backend = str(data_cfg.get("backend", "local")).lower()

        if self.data_backend == "kaggle_offline":
            from syntree.data.kaggle_loader import KaggleInMemoryDataset
            data_dir = str(data_cfg.get("data_dir", "/kaggle/input/3d-syntree-dataset"))
            self.dataset = KaggleInMemoryDataset(data_dir, split="train")
            self.val_dataset = KaggleInMemoryDataset(data_dir, split="val")
        elif self.data_backend == "huggingface":
            hf_cfg = dict(data_cfg.get("huggingface", {}))
            repo_id = str(hf_cfg.get("repo_id", "")).strip()
            if not repo_id:
                raise ValueError("data.huggingface.repo_id is required when data.backend='huggingface'")
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            self.dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id, split="train", cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")), token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
            self.val_dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id, split="val", cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")), token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
        elif trajectory_path:
            self.data_backend = "trajectory_pt"
            self.dataset = TrajectoryDataset(trajectory_path, split="train")
            self.val_dataset = TrajectoryDataset(trajectory_path, split="val")
        else:
            self.dataset = CrossDockedDataset(
                data_cfg["data_dir"], split="train", catalog=self.catalog,
                num_synthetic=int(data_cfg.get("synthetic_samples", 100)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )
            self.val_dataset = CrossDockedDataset(
                data_cfg["data_dir"], split="val", catalog=self.catalog,
                num_synthetic=max(8, int(self.dataset.num_synthetic * val_fraction)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )
```

---

### Step 7: How to Handle Multi-Session Checkpoint Chaining on Kaggle

Since Kaggle sessions kill background processes after 12 hours:

#### Session 1 (Epochs 0 to ~40):

Set checkpoint directory to `/kaggle/working/checkpoints/`:

```bash
python main.py \
    --mode train \
    --config configs/train_kaggle_96gb.json \
    --output-dir /kaggle/working \
    --fresh
```

When hour 11.2 approaches, `ResilientTrainer` automatically saves `checkpoint_epoch_X.pt` and `manifest.json` inside `/kaggle/working/checkpoints/` and exits gracefully.

#### Session 2 (Resuming Epochs 41 to 80):

Because you enabled **`Persistence: Variables and Files`**, `/kaggle/working/checkpoints/` is still there when you open the notebook again!
Simply run:

```bash
python main.py \
    --mode train \
    --config configs/train_kaggle_96gb.json \
    --output-dir /kaggle/working \
    --resume-auto
```

The trainer automatically detects `checkpoint_epoch_X.pt` in `/kaggle/working/checkpoints/`, restores the exact model weights, optimizer, and learning rate scheduler, and resumes training uninterrupted.

---

### Summary of What This Setup Achieves

1. **Zero Data Bottlenecks**: The entire multi-thousand complex dataset lives in your 175 GB RAM. Data access takes 0 milliseconds.
2. **Massive Model Scale**: `hidden_dim = 512`, 12 equivariant layers, $r_{\text{cut}} = 6.5\text{ \AA}$, with `effective_batch_size = 128`.
3. **100% Offline Resilience**: Zero dependence on Hugging Face network APIs during training.
4. **No Disk Overflow**: Keeps only the 2 freshest checkpoints, consuming less than 1% of your 57.6 GB disk quota.

Here is the **1000% brutally honest, mathematically verified breakdown** of whether this will fit in your Kaggle session, where the hidden landmines are, and how to configure it so it is **physically impossible to crash or run out of memory**.

---

### 1. The Resource Audit: Will it Fit?

Let's do the rigorous math on the hardware shown in your screenshot:

| Resource         | Kaggle Limit   | Our Consumption                                         | Status                                    |
| :--------------- | :------------- | :------------------------------------------------------ | :---------------------------------------- |
| **GPU VRAM**     | **95.6 GiB**   | **~38 – 48 GiB** (at `batch_size: 32`, $d=512$, $L=12$) | **FITS EASILY** (50% safety headroom)     |
| **Host RAM**     | **175 GiB**    | **~8 – 14 GiB** (Entire 30k complex dataset preloaded)  | **FITS EASILY** (<10% utilized)           |
| **Disk Space**   | **57.6 GiB**   | **~1.5 GiB** total in `/kaggle/working`                 | **SAFE** (if checkpoints are pruned to 2) |
| **Session Time** | **12.0 Hours** | **11.2 Hours** (Auto-terminates before session kill)    | **FITS SAFELY**                           |
| **Weekly Quota** | **30.0 Hours** | **~24 Hours** across 2–3 chained sessions               | **FITS WITHIN QUOTA**                     |

---

### 2. The 3 Fatal Kaggle Landmines (That Will Kill Your Run If Ignored)

Even with 96 GB VRAM and 175 GB RAM, your run will crash unless you handle these three platform-specific traps:

#### Landmine 1: The Offline Dependency Trap (`torch_geometric`)

- **The Problem:** In your screenshot, **Internet is disabled**. Kaggle’s base Python image pre-installs PyTorch and RDKit, but **`torch_geometric` (PyG) is usually NOT pre-installed.**
- **Why this will kill you:** If you run `pip install torch-geometric` with internet off, the command will immediately fail with a socket connection error.
- **The Fix:**
  Test this immediately in your notebook:
  ```python
  import torch_geometric
  print(torch_geometric.__version__)
  ```
  If it fails, you must either:
  1. Toggle **Internet: On** in the right-hand panel (if your session allows it) to run `pip install torch-geometric`.
  2. Or attach a Kaggle Dataset containing the offline `.whl` files and install via `--no-index`.

#### Landmine 2: The 57.6 GB Disk Quota Trap

- **The Problem:** Your screenshot shows: `Disk: 293.4 MiB / Max 57.6 GiB`.
- **Why this will kill you:** If you accidentally download or extract raw datasets (`crossdocked_pocket10.tar.gz`) inside `/kaggle/working`, it will consume 30+ GB. If you also save 10 model checkpoints (at ~250 MB each) plus PyG caches, Kaggle will throw:
  `OSError: [Errno 28] No space left on device`
  and terminate the notebook immediately.
- **The Rule:**
  - **Never** extract raw datasets inside `/kaggle/working`.
  - Your dataset shards **must** live in `/kaggle/input/` (attached as a Kaggle Dataset). Kaggle inputs have their own dedicated storage that **does not count against your 57.6 GB disk limit.**
  - Keep `keep_last_n_checkpoints: 2` so your checkpoint directory never exceeds ~700 MB.

#### Landmine 3: The 30-Hour Quota vs. 12-Hour Session Kill

- **The Problem:** Look at your quota counter: `Quota: 00:00 / 30 hrs`. Each individual session has a hard ceiling of **12 hours**.
- **The Fix:**
  In the configuration, set `"time_budget_hours": 11.2`.
  At hour 11.2, `ResilientTrainer` will cleanly save `checkpoint_epoch_X.pt` and exit before Kaggle forces an abrupt kernel kill.

---

### 3. The Ironclad Kaggle Configuration

To ensure this model operates at maximum capability while remaining 100% safe from Out-of-Memory (OOM) spikes and disk overflow, use this verified configuration:

#### Save as: `configs/train_kaggle_96gb.json`

```json
{
  "system": {
    "project_name": "3D-SynTree-Kaggle-96GB",
    "seed": 42,
    "device": "cuda:0",
    "mixed_precision": "bf16",
    "tf32": true,
    "num_workers": 4
  },
  "huggingface": {
    "enabled": false,
    "repo_id": "",
    "push_every_n_epochs": 1
  },
  "data": {
    "backend": "kaggle_offline",
    "dataset_name": "crossdocked_production",
    "data_dir": "/kaggle/input/3d-syntree-dataset",
    "synthon_catalog_path": "/kaggle/input/3d-syntree-dataset/enamine_3d_subset.parquet",
    "batch_size": 32,
    "accumulate_grad_batches": 2,
    "max_steps_per_molecule": 4,
    "terminal_cap_min_mw": 260.0,
    "val_fraction": 0.08
  },
  "model": {
    "hidden_dim": 512,
    "num_equivariant_layers": 12,
    "num_radial_basis": 32,
    "cutoff_radius": 6.5,
    "synthon_embedding_dim": 512,
    "num_attention_heads": 8,
    "max_atomic_number": 100,
    "dropout": 0.1
  },
  "training": {
    "max_epochs": 100,
    "time_budget_hours": 11.2,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "lr_scheduler": "cosine_warmup",
    "warmup_epochs": 3,
    "grad_clip_norm": 1.0,
    "keep_last_n_checkpoints": 2,
    "loss_weights": {
      "synthon_ce": 1.0,
      "torsion_nll": 0.5,
      "steric_clash": 0.1,
      "reaction_ce": 0.5
    },
    "eval_interval_epochs": 1,
    "auto_scale": {
      "enabled": false
    }
  },
  "reinforcement_learning": {
    "enabled": false,
    "pocket_dir": "/kaggle/input/3d-syntree-dataset/targets",
    "episodes": 1024,
    "ppo_epochs": 4,
    "rollout_episodes": 32,
    "minibatch_size": 32,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "learning_rate": 0.00001,
    "max_grad_norm": 1.0,
    "temperature": 1.0,
    "checkpoint_every": 16,
    "reward": {
      "docking_weight": 1.0,
      "clash_weight": 0.25,
      "contact_weight": 0.3,
      "fsp3_weight": 0.15,
      "qed_weight": 0.15,
      "validity_weight": 0.5
    }
  },
  "catalog": {
    "min_fsp3": 0.4,
    "max_mw": 240.0
  }
}
```

---

### 4. The Exact Execution Flow in Your Kaggle Notebook

Here is the exact code to execute inside your Kaggle notebook cells:

#### Cell 1: Verify Environment & GPU Memory

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name:    {torch.cuda.get_device_name(0)}")
free_vram, total_vram = torch.cuda.mem_get_info()
print(f"Total VRAM:     {total_vram / 1024**3:.2f} GiB")
print(f"Free VRAM:      {free_vram / 1024**3:.2f} GiB")

# Verify critical imports
import rdkit
import torch_geometric
print(f"PyTorch: {torch.__version__} | PyG: {torch_geometric.__version__} | RDKit: {rdkit.__version__}")
```

#### Cell 2: Launch Training (Session 1)

```bash
!python main.py \
    --mode train \
    --config configs/train_kaggle_96gb.json \
    --output-dir /kaggle/working \
    --fresh
```

#### Cell 3: Resuming Training (Session 2 / Next Day)

When your session restarts (with `Persistence: Files only` enabled in the right panel):

```bash
!python main.py \
    --mode train \
    --config configs/train_kaggle_96gb.json \
    --output-dir /kaggle/working \
    --resume-auto
```

---

### The Verdict

- **Does it fit?** **Yes, 100%.**
- At `batch_size: 32` with gradient accumulation 2 (effective batch 64), your peak VRAM will hover around **42 GiB**, leaving **53 GiB of completely free headroom** on your RTX 6000 Ada. It will never Out-Of-Memory.
- Your entire dataset will reside in your **175 GB RAM**, completely eliminating data-loading lag.
- Checkpoints will consume less than **1.5 GB** of your 57.6 GB disk.
- You can train across 2 sessions (11 hours each) to achieve a fully trained, 12-layer foundation-grade model within your 30-hour weekly quota.

""""""""

Tokens / credentials

Never paste live tokens into tracked files — GitHub push protection rejects the
push and the credential is considered compromised the moment it is committed.

Provide them at runtime instead:

- Hugging Face write token: `export HF_TOKEN=...` (or a Colab/Kaggle secret named
  `HF_TOKEN`; the pipeline reads it in that order).
- GitHub personal access token: `gh auth login`, or `export GITHUB_TOKEN=...`.

If you need them on disk, keep them in an untracked file such as
`.secrets/tokens.env` (already matched by `.gitignore`).

> NOTE: an earlier revision of this file committed a live Hugging Face token and
> a GitHub PAT. They were removed before that commit was ever pushed and both
> credentials must be revoked/rotated.

Make sure to do a ton of testing—extensive, thorough testing of the mathematical formulas, data compatibility, data availability, data integrity, training and inference pipelines, and everything else involved in the project.

Do not just test whether the code runs. Verify that the mathematical logic is correct, the datasets are compatible and complete, the data flows correctly through the entire pipeline, and the results are consistent and reproducible.

Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
