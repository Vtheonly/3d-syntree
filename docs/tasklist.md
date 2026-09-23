Go to this repository first: https://github.com/Vtheonly/3d-syntree

Then solve **all the issues listed in this bug report**. Make sure to do **super, super, super extensive testing**, including comprehensive unit tests and integration tests. I want you to thoroughly verify everything and make sure the fixes do not introduce any new issues.

""""""""

Here is the **brutally honest, razor-sharp assessment** of your project’s current state.

---

### The Verdict: Where You Stand

* **Conceptual & Scientific Formulation: Grade A- (Dramatically Improved)**
  You fixed the critical scientific flaws:
  1. The spatial blindness is resolved: `MolecularFeaturizer` now encodes the reacting handle's 3D coordinates, and `SynTreePolicy` computes distance RBFs between the handle and pocket atoms to modulate attention.
  2. The random target scandal is eliminated: `ReactionConstrainedFragmenter` and `RetrosyntheticTrajectoryBuilder` only accept pairs that replay cleanly via RDKit.
  3. The one-step death trap is solved: the catalog tracks multifunctional linkers vs. caps, and `SynthonHead` now features an explicit learned `STOP` token.
  4. Hotspot-conditioned seeding replaced the "dump in the empty void" algorithm.
  5. The fake `len(recipe) >= 1` metric in `evaluator.py` was replaced with strict handling of AiZynthFinder.

* **Execution Readiness: Grade D (You CANNOT run Colab yet)**
  If you open `run_3d_syntree.ipynb` on Google Colab right this second and hit "Run All", **it will fail with fatal crashes.**

Here are the exact landmines currently in the code, why they will blow up, and what is still missing before you can legitimately start testing.

---

### Landmine 1: The Hugging Face Chicken-and-Egg Crash (Cell 5)

In `run_3d_syntree.ipynb`, Cell 5 executes:
```bash
!python scripts/download_assets.py \
    --dataset-repo JJKK1212/3d-syntree-multidataset \
    --dataset-revision main \
    --output-dir ./data
```

Now look at `scripts/download_assets.py` (lines 296–308):
```python
files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))
required_manifests = {split: f"data/{split}/manifest.json" for split in ("train", "val", "test")}
missing_manifests = [path for path in required_manifests.values() if path not in files]
if missing_manifests:
    raise RuntimeError(f"HF dataset {repo_id}@{revision} is missing required shard manifests: {missing_manifests}")
```

#### The Problem:
Your Hugging Face dataset repository `JJKK1212/3d-syntree-multidataset` is **currently empty** (created less than an hour ago). 
* It contains no `data/train/manifest.json`.
* It contains no `.pt.gz` shards.
* It contains no `enamine_3d_subset.parquet`.

**Cell 5 will throw a fatal `RuntimeError` immediately.** 

#### The Fix:
You cannot test Stage 1 training on Colab until you **build and upload the dataset shards first**. You must run the preprocessing and upload pipeline locally (or in a dedicated data-prep Colab session) using:
```bash
# 1. Preprocess & extract 10A pockets
python scripts/preprocess_multidataset.py ...

# 2. De-leak with MMseqs2
bash scripts/run_mmseqs_split.sh ...

# 3. Build trajectory states
python scripts/build_trajectories.py ...

# 4. Shard and upload to JJKK1212/3d-syntree-multidataset
python scripts/shard_and_upload.py --repo-id JJKK1212/3d-syntree-multidataset --catalog ./data/enamine_3d_subset.parquet ...
```
Only after `shard_and_upload.py` uploads the manifests and shards can your training notebook run.

---

### Landmine 2: Your Test Suite is Out of Sync with Your Code (Immediate Pytest Failures)

You updated the core architecture, but **the test files were left testing the old legacy architecture.** If your thesis committee or an automated CI pipeline runs `pytest`, multiple tests will fail immediately:

1. **`HANDLE_FEATURE_DIM` mismatch:**
   * In `syntree/data/featurizer.py`: `HANDLE_FEATURE_DIM = 67` (64 chemical + 3 xyz).
   * In `tests/test_data_featurizer.py` (line 2753): `assert HANDLE_FEATURE_DIM == 64` $\rightarrow$ **FAILS.**
   * In `tests/test_data_crossdocked.py` (line 2686): `assert sample.handle_features.shape == (64,)` $\rightarrow$ **FAILS.**
2. **`STOP` token dimension mismatch in `SynthonHead`:**
   * In `syntree/models/synthon_head.py`, you added the `STOP` token:
     `logits = torch.cat([logits, stop_logit.unsqueeze(-1)], dim=-1)`
     Output shape is now **$[B, K + 1]$**.
   * In `tests/test_models_heads.py`:
     ```python
     logits, log_probs = head(q, emb)
     assert logits.shape == (3, 50)  # FAILS! Actual shape is (3, 51)
     ```
   * In `tests/test_models_policy.py`:
     ```python
     assert out["synthon_logits"].shape == (4, len(catalog))  # FAILS! Actual shape is (4, len(catalog) + 1)
     ```

#### The Fix:
Update those test assertions to reflect your new, superior architecture (`assert shape == (4, len(catalog) + 1)` and `assert HANDLE_FEATURE_DIM == 67`).

---

### Landmine 3: Fatal Bug in `--mode rl` with the Hugging Face Backend

Look at `main.py` lines 167–173:
```python
if args.mode == "rl":
    ...
    pocket_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith("_pocket.pdb")
    )
    if not pocket_paths:
        print(f"[main] no pocket PDB files found under {data_dir}", file=sys.stderr)
        return 2
```

#### The Problem:
When `data.backend` is set to `"huggingface"`, `ShardedHuggingFaceDataset` streams **pre-featurized `.pt.gz` PyG tensors into `./hf_cache`**. It does **not** download loose `*_pocket.pdb` files into `data_dir` (`./data/crossdocked`).
* `os.listdir(data_dir)` will find zero `.pdb` files.
* `--mode rl` (Cell 9 in your notebook) will immediately crash and exit with code `2`.

#### The Fix:
For Stage 2 (RL), the generator needs real pocket PDB files to calculate docking rewards and conformer placements. You must either:
1. Include a dedicated `targets/` folder of evaluation pocket PDBs in your dataset repo that gets downloaded to `./data/test_pockets/`.
2. Or point the RL loop explicitly to a directory of test pocket PDBs (`--pocket-dir`).

---

### Landmine 4: The Docking Oracle in Colab is Missing

In `configs/rl_colab_12h.json` and `evaluator.py`, docking is configured with:
```json
"docking_engine": "gnina",
"exhaustiveness": 8
```
* Neither `gnina` nor `vina` is installed via `pip install -r requirements.txt`. (They are compiled binary executables, not pure Python packages).
* In `syntree/engine/rl.py`, when `gnina` and `vina` are missing:
  ```python
  docking_available = False
  components["docking"] = 0.0
  ```
* **The consequence:** If you run PPO in Google Colab without downloading the standalone `gnina` binary in a notebook setup cell, **the docking reward will be 0.0 for every single molecule.** Your RL agent will optimize clash avoidance, QED, and Fsp3, but will be completely blind to protein-ligand binding affinity!

#### The Fix:
Add a 2-line setup command in Cell 4 of your Colab notebook to fetch the static binary:
```bash
!wget -q https://github.com/gnina/gnina/releases/download/v1.1/gnina -O /usr/local/bin/gnina && chmod +x /usr/local/bin/gnina
```

---

### What Still Needs to be Done for a Defensible Thesis?

To guarantee this passes peer review and thesis defense:

1. **Ablation Studies Must Be Run:**
   You cannot just report one final set of numbers. A thesis requires ablation tables showing:
   * Policy **without** handle distance RBF vs. **with** handle distance RBF.
   * Supervised Imitation Learning alone (Stage 1) vs. RL-finetuned (Stage 2).
   * Monofunctional catalog vs. Stratified Linker/Cap catalog (showing how the linkers prevent premature 1-step termination).
2. **Export Baseline Outputs for the Same Target Set:**
   In Cell 10, you call `run_comparative_benchmark.py`. This script expects exported SDF files from **TargetDiff**, **DiffSBDD**, and **SyntheMol** under `./benchmarks/outputs/<method>/<target_id>/`. You need to generate those baseline structures on your test targets so your comparative plots are genuine.

---

### Summary: Your Action Plan Before Hitting "Run"

| Step | Action Required | Status |
| :--- | :--- | :--- |
| **1** | Update outdated tests (`test_data_featurizer.py`, `test_models_policy.py`, `test_models_heads.py`) to match 67-dim handles and $K+1$ actions. | ❌ Must do before CI |
| **2** | Run the preprocessing, MMseqs2 split, and trajectory extraction scripts locally or on an HPC cluster. | ❌ Required before Colab |
| **3** | Run `scripts/shard_and_upload.py` to push the shards and catalog to `JJKK1212/3d-syntree-multidataset`. | ❌ Required before Colab |
| **4** | Add `gnina` binary download to Cell 4 in `run_3d_syntree.ipynb` so the RL docking reward is active. | ❌ Must add to notebook |
| **5** | Stage a folder of validation pocket PDBs for the RL engine in `main.py`. | ❌ Must fix in codebase |

Once steps 1–5 are completed, your pipeline is **rock-solid, mathematically sound, and ready for multi-day production training.**

"""""""""



bug report #2 
"""""
Here is the **brutally honest, mathematically and chemically rigorous critique** of the **thesis logic and the AI model architecture itself**.

You fixed the surface-level bugs (67-dim vector, synthetic targets, dead-end reactions). But underneath, the **core AI model still has severe theoretical and architectural flaws** that a sharp thesis examiner will rip apart.

Here are the 4 fundamental logical fallacies currently embedded in your AI model.

---

### Flaw 1: The "Ghost Ligand" Paradox (The State is Not Markovian)

A thesis that claims to formulate molecular design as a **Markov Decision Process (MDP)** must satisfy the Markov Property: **the state $S_t$ must contain all information necessary to choose the next action $A_t$.**

Now, look at what your model actually takes as input in `syntree/models/policy.py`:
```python
# batch_data contains:
pocket_pos  # The protein pocket atoms [N_p, 3]
pocket_z    # The protein pocket atomic numbers [N_p]
handle_features # The SINGLE reacting atom's chemical vector + 3D position [67]
```

#### Where are the other atoms of the ligand you already grew?!
* At Step 1, you attach a 15-atom bicycle into Subpocket A.
* At Step 2, the model prepares to grow from handle $u$.
* **Your GNN only feeds the pocket atoms and the single handle atom $u$ to the network.**
* **The other 14 atoms of your intermediate ligand $M_t$ DO NOT EXIST in the model’s input graph.**

```
WHAT ACTUALLY HAPPENS:

        [ Protein Pocket Wall ]
             ▲
             │ (Clash!)
      [Synthon 1] ◄─── (INVISIBLE TO THE AI!)
           │
     (Handle u) ───► AI says: "Let's attach a bulky adamantyl ring here!"
                     (Because it cannot see Synthon 1 occupying the space behind it!)
```

#### The Logical Failure:
1. **Self-Collision Blindness:** How can the policy choose a synthon that doesn’t crash into the *existing* ligand body if it cannot see the existing ligand body?
2. **Physicochemical Blindness:** How can the policy know whether the ligand has already satisfied its hydrogen bond quota, exceeded drug-like molecular weight, or become too hydrophobic?
3. **Broken MDP:** Your state $S_t$ is **incomplete**. You claim $P(a_t \mid \mathcal{P}, \mathcal{M}_t)$, but your code actually computes $P(a_t \mid \mathcal{P}, u_t)$. The intermediate ligand $\mathcal{M}_t$ is a ghost.

**The Fix:**
The input to the GNN cannot just be pocket atoms. It must be a **Heterogeneous Bipartite Graph** containing:
* Pocket atoms: $(X_P, Z_P)$
* Current intermediate ligand atoms: $(X_L, Z_L)$
* The reacting handle node $u$ explicitly marked.
The PaiNN encoder must pass messages between the pocket atoms AND the existing ligand atoms.

---

### Flaw 2: The Broken Factorization (Torsion is Blind to the Synthon)

In your README and thesis formulation, you write:
$$
\pi_\theta(a_t \mid \mathcal{P}, \mathcal{M}_t) = P(r_t \mid \mathcal{P}, \mathcal{M}_t) \times P(B_t \mid r_t, \mathcal{P}, \mathcal{M}_t) \times \mathbf{p(\phi_t \mid B_t, r_t, \mathcal{P}, \mathcal{M}_t)}
$$
Notice the third term: the dihedral angle $\phi_t$ is conditioned on the **selected synthon $B_t$** and the **reaction $r_t$**.

Now look at your actual PyTorch implementation in `syntree/models/policy.py`:
```python
# 1. Synthon selection
logits, log_probs = self.synthon_head(context, synthon_embeddings, ...)

# 2. Torsion prediction
mu, kappa = self.torsion_head(context, pocket_v, pocket_batch)
```
Where is the chosen synthon embedding $E_B$ in `torsion_head`? 
**It is completely absent.**

#### The Logical Failure:
`torsion_head` only takes `context` (which comes from the handle query and the pocket).
* If the policy selects a tiny **methyl group** ($\text{—CH}_3$), `torsion_head` predicts angle $\mu$.
* If the policy selects a massive, rigid, chiral **spiro-adamantyl core**, `torsion_head` predicts the **exact same angle $\mu$**.

This is chemically nonsensical. The optimal dihedral angle around a single bond is dictated almost entirely by the **steric clashes between the incoming synthon's specific side-chains and the pocket walls**. Predicting the dihedral angle *before* (or independently of) knowing what molecule is attached to that bond violates basic stereochemistry and breaks your own mathematical factorization.

**The Fix:**
The torsion head must take the chosen synthon's embedding:
```python
# Pass the selected synthon's embedding into the torsion head:
selected_synthon_emb = synthon_embeddings[selected_synthon_idx] # [B, d]
torsion_input = torch.cat([context, selected_synthon_emb], dim=-1)
mu, kappa = self.torsion_head(torsion_input, pocket_v, pocket_batch)
```

---

### Flaw 3: The Blind `STOP` Token (The Cavity Completion Paradox)

In `SynthonHead`, you added a learned `STOP` token (index $K$). 
The policy learns to emit `STOP` to terminate molecular growth.

#### But ask yourself: How does the network know when the pocket is full?
* To know if a cavity is full, a model must compare the **volume of the pocket** with the **volume of the ligand**.
* As proven in Flaw 1, your network does not know the volume, atom count, or shape of the ligand grown so far.
* It only sees the pocket and the local handle $u$.
* Therefore, the `STOP` logit:
  $$\text{stop\_logit} = \frac{\mathbf{out} \cdot \mathbf{w}_{\text{stop}}}{\sqrt{d}} + b_{\text{stop}}$$
  is attempting to make a global termination decision **without knowing how much molecule has already been built.**

In practice, the model will either:
1. Learn to predict `STOP` based purely on step count (acting as a noisy constant).
2. Or trigger prematurely on small subpockets because the local context looks crowded, terminating the ligand at 180 Da.

**The Fix:**
You must feed explicit global scalar features into the policy core:
* Current ligand molecular weight ($\text{MW}_t$).
* Current ligand heavy atom count.
* Ratio of estimated ligand volume to pocket cavity volume.

---

### Flaw 4: Mathematical Collapse in Stage 2 PPO (Micro-Batch Degradation)

Look at how PPO fine-tuning is executed in `main.py` (lines 185–205):
```python
for episode in range(rl_start_episode, episodes):
    pocket = pocket_paths[episode % len(pocket_paths)]
    result = generator.generate_ligand(...) # Generates ONE ligand (horizon T = 2 or 3)
    reward_details = reward_fn.compute(...)
    
    # Immediately updates policy on this single ligand:
    stats = finetuner.update_episode(result["policy_trace"], reward_details["reward"])
```

Now look inside `update_episode` in `syntree/engine/rl.py`:
```python
for _ in range(self.ppo_epochs): # e.g., 4 epochs
    # Computes policy loss over `transitions`
    # len(transitions) == 2 or 3!
    loss.backward()
    self.optimizer.step()
```

#### The Mathematical Disaster:
1. **PPO is a batch-sampling algorithm.** It requires an expectation over a distribution of trajectories:
   $$\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A(s, a) \right]$$
2. In your code, the batch size is **literally 2 or 3 transitions** from a single molecule.
3. You are taking **4 epochs of gradient descent on a batch of size 2** using the AdamW optimizer to update a multi-million parameter PaiNN + Attention model.
4. **The result:** The importance sampling ratio $r_t(\theta)$ immediately clips, the policy undergoes **catastrophic forgetting**, and the network’s weights will diverge within 50 episodes. This is not PPO; it is stochastic gradient destabilization.

**The Fix:**
You must accumulate a rollout buffer across multiple pockets before taking a PPO step:
* Roll out 32 to 64 molecules across 32 different pockets ($T \approx 100\text{--}200$ transitions total).
* Normalize advantages across the entire batch of 64 episodes.
* Only then execute your 4 PPO update epochs.

---

### Summary Checklist for Model Architecture

| Component | Current State in Code | Why It Fails Chemically / Mathematically | The Defensible Thesis Fix |
| :--- | :--- | :--- | :--- |
| **State Input ($S_t$)** | Pocket + Single Handle point | Model cannot see the existing ligand body; blind to self-clashes. | Convert GNN to a **Bipartite Heterogeneous Graph** (Pocket + Intermediate Ligand). |
| **Torsion Factorization** | $\phi_t \perp B_t$ (Independent) | Model predicts the same angle for a methyl group and an adamantyl ring. | Condition `ContinuousTorsionHead` on the **selected synthon embedding $E_{B_t}$**. |
| **Termination (`STOP`)** | Blind local logit | Network doesn't know how big the current ligand is. | Feed global scalar features ($\text{MW}_t$, heavy atom count) into policy readout. |
| **Stage 2 RL (PPO)** | Batch size $T = 2$, updated every episode | Destabilizes gradient descent; causes catastrophic forgetting. | Collect a **rollout buffer of 32–64 episodes** before running PPO updates. |

If you address these 4 flaws, your thesis moves from a "naive neural assembly hack" to a **theoretically rigorous, mathematically sound reinforcement learning framework for structural biology.**
"""""


bug report 3 
"""""
Here is the **unfiltered, brutal reality** of where this codebase sits on the spectrum between *"amateur glue code"* and *"elite research-grade software."*

---

### The Unvarnished Verdict

Right now, the codebase lives in the **"Uncanny Valley" of Machine Learning Engineering:**

* **From the outside (The Cosmetics): 9/10.** 
  The packaging, CLI dispatch in `main.py`, structured JSON logging, atomic checkpointing, and RDKit isotope tracking look polished, clean, and professional. It looks like code written by a skilled software engineer or a well-funded AI biotech startup.
* **Under the hood (The Core ML/Chemistry Logic): 4/10.** 
  An expert reviewer at NeurIPS or a senior computational chemist at Novartis will spot **three glaring "amateur tells"** within 10 minutes of reading `syntree/models/policy.py` and `syntree/engine/rl.py`. 

If you leave those tells in, reviewers will immediately dismiss it as:
> *"A junior engineer stitched together PyG, RDKit, and an RL template without understanding how geometric graphs or policy gradients actually work."*

However, if you replace those specific hacky patches with **rigorous architectural abstractions**, the codebase instantly transforms into **top-5% research-grade open-source science**.

Here are the exact amateur tells giving you away right now, and what the code must look like to be indisputably research-grade.

---

### The 3 "Amateur Tells" in Your Current Code

#### Tell 1: The "Franken-Tensor" Hack (`HANDLE_FEATURE_DIM = 67`)
In `featurizer.py` and `policy.py`, you represent the attachment handle by taking a 64-dimensional one-hot chemical vector and **gluing the 3D $(x, y, z)$ coordinates onto the end:**
```python
feats[0:64] = chemical_properties
feats[64:67] = handle_xyz  # <--- AMATEUR SMELTER
```
* **Why it looks amateur:** In modern Geometric Deep Learning (PyG, e3nn, TorchMD-NET), **positions and features are never mixed in the same tensor.** Positions belong in `pos` (subject to $\mathrm{SE}(3)$ transformation laws); invariant chemical identities belong in `x`. 
* Tacking coordinates onto the end of a feature vector is a classic hack used by developers trying to patch a missing feature without updating their data schema.

#### Tell 2: The "Ghost Scaffold" (Missing Heterogeneous Graph)
In `policy.py`, you try to cross-attend from a single handle vector to a protein pocket point cloud.
* **Why it looks amateur:** Real structure-based molecular design operates on **Heterogeneous Bipartite Graphs** (`torch_geometric.data.HeteroData`):
  * Node Type A: `protein_atoms` (with positions `pos_p` and elements `z_p`)
  * Node Type B: `ligand_atoms` (the intermediate molecule grown so far, with positions `pos_l` and elements `z_l`)
  * Edge Type A-A: `protein-protein` radius edges
  * Edge Type B-B: `ligand-ligand` covalent + radius edges
  * Edge Type A-B: `protein-ligand` non-covalent interaction edges
* By ignoring the existing ligand atoms and only passing a single handle point, the code reveals that the developer didn't want to build a dynamic graph collation pipeline for intermediate states.

#### Tell 3: The Micro-Batch RL Loop
In `main.py`, the RL loop rolls out **one single molecule** (2 steps) and immediately runs 4 epochs of AdamW on that 2-step trajectory.
* **Why it looks amateur:** Anyone who has trained PPO in PyTorch knows that a batch size of $N=2$ will destabilize the policy within minutes. It shows that the RL module was written as an untested conceptual demo rather than an actual working optimization algorithm.

---

### What the Code Looks Like When It Is TRULY "Research-Grade"

To remove all traces of "stitched-together glue code," your project needs **three architectural upgrades**:

#### 1. Unified MDP Environment (`syntree/engine/environment.py`)
Amateur code scatters step logic across the generator, conformer engine, and trainer. 
Research-grade code wraps the chemistry inside a clean, Gym-like **Vectorized Molecular MDP Environment**:

```python
class MolecularAssemblyEnv:
    """Rigorous MDP environment wrapping RDKit and Enamine actions."""
    
    def reset(self, pocket_pdb: str) -> HeteroData:
        """Anchors seed using hotspot prior; returns full bipartite graph."""
        ...
        
    def step(self, action: AssemblyAction) -> Tuple[HeteroData, float, bool, Dict]:
        """
        Executes:
        1. Reaction validation & execution via RDKit SMARTS
        2. Scaffold-locked 3D conformer embedding
        3. Synthon-conditioned dihedral rotation
        4. Returns updated Bipartite Graph S_{t+1}, reward r_t, done flag, info
        """
        ...
```
*When a reviewer sees this, they immediately recognize professional, modular, reproducible reinforcement learning engineering.*

#### 2. Native Heterogeneous Equivariant Backbone
Instead of PaiNN on protein atoms alone plus a tacked-on cross-attention trick, a research-grade model processes the pocket and ligand jointly:

```python
class BipartitePaiNN(nn.Module):
    """Processes joint Protein-Ligand heterogeneous graph with SE(3) equivariance."""
    
    def forward(self, hetero_data: HeteroData):
        # 1. Message passing within protein atoms
        # 2. Message passing within growing ligand atoms
        # 3. Inter-molecular message passing across protein-ligand contact edges
        # 4. Readout query conditioned on handle node u AND global ligand mass
        ...
```
*This completely eliminates the "Ghost Ligand" problem. The model naturally sees both the pocket walls AND the molecule it has built so far.*

#### 3. Parallelized Rollout Worker for PPO
Instead of stepping through one molecule at a time in a Python loop:
* The trainer runs an environment vector (e.g., 16 parallel targets).
* It collects an actual **Rollout Buffer** of 64–128 transitions across different pockets.
* It computes Generalized Advantage Estimation (GAE) across the buffer.
* It performs mini-batch PPO updates on chunks of 32 transitions.

---

### The Reality Check: How Academic Code is Judged

Let's be 1000% realistic about how academic research code actually looks in the wild:

1. **Most published academic code is absolute garbage:** If you look at the raw GitHub repos of papers published at NeurIPS, ICML, or nature journals (*TargetDiff*, *DiffSBDD*, *Pocket2Mol*), the code is filled with hardcoded paths, commented-out hacks, zero unit tests, zero docstrings, and brittle shell scripts that fail if you run them on a different machine.
2. **Your codebase already has higher software hygiene than 90% of academic papers:** 
   * You have clean JSON configurations.
   * You have structured JSONL logging.
   * You have an extensive `pytest` test battery.
   * You have an automated Hugging Face shard synchronization layer.
   * You have atomic checkpoint saving.

### The Bottom Line

* **Will it look amateurish right now?** 
  To a senior ML/computational biology researcher: **Yes, in the model architecture and RL loop.** They will see the Franken-tensor coordinates, the missing ligand scaffold graph, and the micro-batch PPO loop as shortcuts.
* **Can you fix it?** 
  **Yes.** If you upgrade the state to a true heterogeneous graph (`HeteroData`), condition the torsion head on the synthon embedding, and collect real rollout buffers for PPO, this project stops looking like glue code entirely. 

It becomes an **industrial-grade, scientifically unassailable framework** that looks cleaner and more principled than 95% of the machine learning repositories coming out of top-tier university labs today.




"""""


bug report 4



""""""
Here is the **strictly chemical and computational chemistry autopsy** of this codebase. 

Right now, from the perspective of a medicinal chemist or structural biophysicist, the underlying chemistry logic is a **4/10**. It treats 3D molecular geometry like a computer-graphics problem (moving shapes around in space) rather than a **quantum mechanical and thermodynamic reality**.

Here are the **6 fatal chemical and biophysical mistakes** currently in the code, why they completely break real-world drug design, and the **exact code and testing protocols** required to make this scientifically unassailable.

---

### Mistake 1: The "Twisted Amide" Catastrophe (Violating Molecular Orbital Theory)

Look at how dihedrals are applied in `generator.py` and `conformer.py`:
```python
# generator.py: Reaction is executed (e.g., amide coupling)
core_atom, synthon_atom = result.junction_bond # This is the C(=O) — N bond!
self.conformer_engine.set_dihedral(product, (core_atom, synthon_atom), dihedral_pred)
```
And look at `conformer.py`:
```python
# _dihedral_definition checks:
if bond.IsInRing():
    return None # Rotatable single bonds only
# Treats C(=O)—N as a rotatable single bond!
```

#### The Chemical Disaster:
* In an amide bond ($\text{R—C(=O)—NR'R''}$), the nitrogen lone pair delocalizes into the carbonyl $\pi^*$ orbital. **The bond order is roughly 1.4, not 1.0.**
* The barrier to rotation around an amide bond is **$\sim 18\text{--}22\text{ kcal/mol}$**. In room-temperature biology, **an amide bond is strictly planar ($180^\circ$ for *trans*, rarely $0^\circ$ for *cis*).**
* Your policy predicts a continuous angle $\phi \in [-\pi, \pi)$ from a von Mises distribution. **It is literally setting amide bonds to $45^\circ$, $90^\circ$, and $-120^\circ$.**
* A $90^\circ$-twisted amide breaks resonance, forces the nitrogen into an unphysical $sp^3$ pyramid, and would cost $\sim 20\text{ kcal/mol}$ of energy penalty. No real protein will ever bind a twisted amide.

#### The Fix:
You must restrict your dihedral action space to **true single bonds with low rotational barriers** ($sp^3\text{--}sp^3$ or $sp^3\text{--}sp^2$).
If the junction is an amide or ester:
1. Lock the junction dihedral strictly to **$180^\circ$ (*trans*)** or allow a discrete binary choice $\{0^\circ, 180^\circ\}$.
2. Rotate the **next adjacent true single bond** (e.g., the $\text{C}_\alpha\text{--C(=O)}$ or the $\text{N--C}_\alpha$ bond), exactly like the $\phi/\psi$ backbone angles in proteins.

---

### Mistake 2: The "Scaffold Lock" Tears Chemical Bonds (`conformer.py`)

Look at how you embed the newly attached synthon in `ConformerEngine.embed_product`:
```python
# 1. RDKit embeds the product using ETKDGv3 in a random local coordinate frame:
status = AllChem.EmbedMolecule(mol, params)

# 2. Hard-lock the scaffold atoms:
for (core_idx, prod_idx) in pairs:
    prod_pos[prod_idx] = ref_pos[core_idx]

for i in range(mol.GetNumAtoms()):
    conf.SetAtomPosition(i, Point3D(*prod_pos[i]))

# 3. Relax with frozen scaffold:
ff.AddFixedPoint(prod_idx) # Freezes ALL core atoms!
ff.Minimize(maxIts=300)
```

#### The Chemical Disaster:
* ETKDG embeds the product with its *own* idea of bond angles.
* When you hard-overwrite the core atoms with `ref_pos`, if the junction bond vector in the reference core does not match ETKDG's random orientation, **the newly formed bond between the core and synthon gets stretched or compressed to unphysical extremes (e.g., 0.9 Å or 2.8 Å).**
* Then you call MMFF with `ff.AddFixedPoint(prod_idx)`. You froze **every single core atom**, including the core atom at the junction!
* Because the core junction atom is completely immovable, the force field is forced to **distort the incoming synthon's bond angles** (bending a $109.5^\circ$ carbon to an unphysical $75^\circ$ or $140^\circ$) just to bridge the gap.
* Furthermore, if MMFF fails on nitrogen/boron-rich heterocycles, your code skips relaxation completely, outputting a molecule with a physically broken bond.

#### The Fix:
Use **Constrained Embed with Harmonic Restraints**, not hard-freezing:
```python
# The proper computational chemistry protocol:
# 1. Align using Match3D or ConstrainedEmbed
# 2. Add harmonic distance restraints to the core (spring constant k = 50.0 kcal/mol/A^2), 
#    DO NOT freeze them rigidly.
# 3. Allow the junction atoms (both core and synthon) to relax their bond length and angle!
```

---

### Mistake 3: Stereochemical Amnesia (Random Chiral Inversion)

Look at what happens during reactions in `reactions.py` and `conformer.py`:
```python
# 1. Reactions run on implicit-H molecules:
core_noH = Chem.RemoveHs(Chem.Mol(current_mol))
result = self.rxn_engine.apply_reaction(core_noH, synthon_mol, chosen_rxn)

# 2. Embeds product:
mol = Chem.AddHs(Chem.Mol(product_mol))
status = AllChem.EmbedMolecule(mol, params)
```

#### The Chemical Disaster:
* The core selling point of your project is using **sp3-rich, 3D chiral synthons** from Enamine.
* When RDKit runs `RunReactants`, stereochemical labels ($R/S$) near the reaction handles are frequently stripped or set to `ChiralType.CHI_UNSPECIFIED`.
* When `EmbedMolecule` embeds a molecule with unspecified chiral centers using random 3D coordinates, **it randomly chooses whether that carbon is $(R)$ or $(S)$!**
* **The consequence:** You pick an $(R)$-enantiomer from Enamine, react it, embed it, and RDKit inverts it into the inactive or toxic $(S)$-enantiomer. The stereochemical identity of your library is completely lost during generation.

#### The Fix:
You must explicitly preserve chiral tags before reaction execution, and enforce chiral constraints during embedding:
```python
# Assign and preserve explicit stereochemistry
Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
params = AllChem.ETKDGv3()
params.enforceChirality = True # CRITICAL: Rejects conformers with inverted stereocenters!
```

---

### Mistake 4: Regiochemistry & Chemoselectivity Blindness (`reactions.py`)

Look at your handle detection logic:
```python
# In reactions.py:
def order_reactants(self, core_mol, synthon_mol, reaction_name):
    side_a, side_b = REACTION_SIDES[reaction_name]
    core_types = self.handle_types(core_mol)
    if side_a in core_types:
        return core_mol, synthon_mol
    ...
```
And in `generator.py`:
```python
handles = self.rxn_engine.detect_handles(current_mol)
target_handle = handles[0] # Takes the first match from SMARTS!
```

#### The Chemical Disaster:
Molecules in medicinal chemistry have multiple competing nucleophiles.
* Suppose your growing ligand has an **aliphatic piperidine amine** and an **aromatic aniline amine**.
* An aliphatic amine ($pK_a \sim 10.5$) is **$10^6$ times more reactive** in an amide coupling than an aromatic aniline ($pK_a \sim 4.5$).
* Your code treats all amines identically using the regex `[N;!H0;!H3;!$(NC=O)]` and picks `handles[0]` (which is just whichever atom happens to have a lower index in the PDB/SDF file!).
* It will cheerfully couple an acid to an unreactive aniline while leaving a super-reactive aliphatic secondary amine untouched. In a real lab, this reaction yields **0% of your predicted product**.

#### The Fix:
You must rank competing handles by **chemical reactivity / nucleophilicity tiers**:
* Tier 1: Aliphatic primary/secondary amines
* Tier 2: Aromatic amines (anilines)
* Tier 3: Sulfonamides / weakly nucleophilic heteroatoms

---

### Mistake 5: The "Clash Loss" Thermodynamic Fallacy (`conformer.py`)

Look at how you define steric clash:
```python
# compute_steric_clash_loss:
thresholds = (ligand_vdw + pocket_vdw) - clash_tolerance
clashes = torch.clamp(thresholds - dists, min=0.0)
return torch.sum(clashes ** 2)
```

#### The Chemical Disaster:
* This is a **repulsion-only hard wall**.
* In statistical thermodynamics, free energy of binding is:
  $$\Delta G_{\text{bind}} = \Delta H_{\text{contacts}} + \Delta H_{\text{desolvation}} - T\Delta S_{\text{conformational}}$$
* Because your function only penalizes $d < \text{threshold}$, the mathematical global minimum of your function is **$\text{distance} \to \infty$**.
* If a synthon swings completely outside the binding pocket into open solvent, your clash loss reports a **perfect score of 0.0**.
* You have **zero attractive potential**: no van der Waals dispersion ($-\frac{C}{r^6}$), no directional hydrogen-bonding potential ($\cos(\theta)$ angle penalty), and no desolvation penalty for burying charged residues.

#### The Fix:
You must use a **Lennard-Jones 6-12 or soft Lennard-Jones 4-8 potential** with an attractive well, not just a repulsive clamp:
$$V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$
The model must learn that an atom placed at $3.2\text{ \AA}$ (the van der Waals contact minimum) gives a **negative (favorable) energy**, whereas placing it at $10.0\text{ \AA}$ gives **zero energy**.

---

### Mistake 6: The Uncharged PDB Delusion (Ignoring Protonation & Tautomers)

In `featurizer.py`:
```python
pocket_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False)
feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
# Featurizes pocket_z (atomic number 6, 7, 8)
```

#### The Chemical Disaster:
* X-ray crystal structures in PDBbind or CrossDocked **do not contain hydrogen atoms** (X-rays scatter from electrons, and hydrogen has only 1 electron; it is invisible at $> 1.0\text{ \AA}$ resolution).
* When RDKit reads a raw PDB file without hydrogens, **all residues are parsed as neutral or undefined.**
* A Aspartate or Glutamate side-chain carboxylate is negatively charged ($-1$).
* A Lysine or Arginine side-chain is positively charged ($+1$).
* A Histidine can exist in three distinct protonation states ($\text{HID, HIE, HIP}$) depending on the local hydrogen-bonding network.
* **Your PaiNN encoder only sees the atomic number $Z=7$ or $Z=8$.** It cannot distinguish a neutral carboxylic acid from a carboxylate ion, or a neutral amine from an ammonium ion. It is physically impossible for the neural network to learn salt-bridge formation.

#### The Fix:
You must preprocess all pockets through **PDB2PQR** or **RDKit MolStandardize / Protonate** at $\text{pH } 7.4$ before generating graph tensors:
* Assign explicit formal charges to residues (+1 for ARG/LYS, -1 for ASP/GLU).
* Include formal charge as a node feature in `pocket_z` or an auxiliary feature tensor.

---

### The Scientific Testing Battery (How to Prove it Actually Works)

To prove to your thesis committee that this is genuine computational chemistry and not an amateur demo, you must implement these **5 automated biophysical tests**:

#### Test 1: The PoseBusters CLI Physical Validation Test
Run the official, standalone **PoseBusters CLI** (which is what pharmaceutical companies use) over your generated SDF files:
```bash
bust outputs/ligands.sdf -p data/crossdocked/target_pocket.pdb --full
```
Assert that your model achieves $>85\%$ pass rates on:
* `bond_lengths_within_bounds`
* `bond_angles_within_bounds`
* `planarity_aromatic_rings`
* `planarity_amide_bonds` (This will catch Mistake #1 immediately!)

#### Test 2: The Amide & Ester Planarity Unit Test
Add this unit test to `tests/test_chemistry_conformer.py`:
```python
def test_amide_junctions_are_strictly_planar(generator, pocket_path):
    """Assert that every newly formed amide bond satisfies resonance planarity."""
    result = generator.generate_ligand(pocket_path, max_steps=2)
    mol = result["rdkit_mol"]
    conf = mol.GetConformer()
    
    # Find all amide bonds: C(=O) - N
    amide_pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[N:3;!H0,H1]")
    matches = mol.GetSubstructMatches(amide_pattern)
    for c_idx, o_idx, n_idx in matches:
        # Get neighbors to define dihedral
        c_nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(c_idx).GetNeighbors() if a.GetIdx() not in (o_idx, n_idx)]
        n_nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(n_idx).GetNeighbors() if a.GetIdx() != c_idx]
        if c_nbrs and n_nbrs:
            deg = abs(rdMolTransforms.GetDihedralDeg(conf, c_nbrs[0], c_idx, n_idx, n_nbrs[0]))
            # Dihedral must be near 180 (trans) or 0 (cis)
            diff_trans = abs(deg - 180.0)
            diff_cis = abs(deg - 0.0)
            assert min(diff_trans, diff_cis) < 15.0, f"Amide bond is twisted by {deg} degrees! Violates orbital planarity."
```

#### Test 3: The Stereochemical Integrity Test
Add a test verifying that $(R)$-synthons do not invert into $(S)$-products:
```python
def test_stereocenter_preservation(engine, catalog):
    """Verify that chiral centers on synthons are not inverted by ETKDG embedding."""
    chiral_synthon = Chem.MolFromSmiles("N[C@@H](C)C(=O)O") # (S)-alanine
    core = Chem.MolFromSmiles("c1ccccc1N")
    result = engine.apply_reaction(core, chiral_synthon, "amide_coupling")
    product_3d = ConformerEngine.embed_product(result.product)
    
    Chem.AssignStereochemistry(product_3d, force=True)
    chiral_centers = Chem.FindMolChiralCenters(product_3d)
    assert len(chiral_centers) == 1
    assert chiral_centers[0][1] == "S", "Chiral center was randomly inverted during 3D conformer generation!"
```

#### Test 4: Conformer Strain Energy Test (MMFF94 $\Delta E$)
A generated 3D molecule must not be trapped in an unphysical, high-energy strain state to fit the pocket:
```python
def test_conformer_strain_energy(generator, pocket_path):
    """Assert that generated ligands do not have severe internal torsional/angle strain."""
    result = generator.generate_ligand(pocket_path)
    mol = result["rdkit_mol"]
    
    # 1. Energy of the as-generated conformer in the pocket
    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    bound_energy = ff.CalcEnergy()
    
    # 2. Relax the conformer in vacuum to find its local minimum
    mol_free = Chem.Mol(mol)
    AllChem.MMFFOptimizeMolecule(mol_free)
    ff_free = AllChem.MMFFGetMoleculeForceField(mol_free, props)
    free_energy = ff_free.CalcEnergy()
    
    # Strain energy = E_bound - E_free
    strain = bound_energy - free_energy
    assert strain < 15.0, f"Ligand is in an unphysical high-strain conformation (Strain: {strain:.2f} kcal/mol)!"
```

---

### Summary Table: From "Amateur Graphics Hack" to "Rigorous Computational Chemistry"

| Component | What Your Code Does (4/10) | What Real Chemistry Requires (9/10) | How to Verify It |
| :--- | :--- | :--- | :--- |
| **Amide/Ester Junctions** | Rotated freely as continuous SO(2) angles ($\phi \in [-\pi, \pi)$). | Locked to planar $180^\circ$ (*trans*); rotate adjacent $sp^3$ single bonds instead. | Automated dihedral angle unit test ($\Delta < 15^\circ$). |
| **Scaffold Assembly** | Hard-overwrites coordinates and freezes all core atoms rigidly. | Harmonic spring restraints ($k=50\text{ kcal/mol/\AA}^2$) allowing junction bond relaxation. | Bond length bounds test ($1.2\text{ \AA} < d < 1.6\text{ \AA}$). |
| **Stereochemistry** | Lost in SMARTS; ETKDG randomly inverts chiral centers. | Enforce `enforceChirality=True` and track CIP parity ($R/S$) tags explicitly. | Stereocenter preservation unit test on chiral amino synthons. |
| **Clash Penalty** | Clamp-based repulsion wall; favors drifting into empty space. | Lennard-Jones 6-12 with attractive dispersion well at $3.0\text{--}3.5\text{ \AA}$. | Distance distribution analysis (peaks at $3.2\text{ \AA}$). |
| **Pocket Charge State** | Parses raw neutral atomic numbers $Z$; blind to ions. | Preprocess pockets with PDB2PQR to assign formal charges at pH 7.4. | Salt bridge presence test with Asp/Glu/Arg residues. |

If you implement these 6 fixes and add the 4 biophysical unit tests to your test suite, your thesis shifts from an *"AI that draws impossible molecular cartoons"* to a **scientifically rigorous, biophysically grounded generative drug discovery platform.**
""""""




Tokens / credentials

Never paste live tokens into tracked files — GitHub push protection rejects the
push and the credential is considered compromised the moment it is committed.

Provide them at runtime instead:

* Hugging Face write token: `export HF_TOKEN=...` (or a Colab/Kaggle secret named
  `HF_TOKEN`; the pipeline reads it in that order).
* GitHub personal access token: `gh auth login`, or `export GITHUB_TOKEN=...`.

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
