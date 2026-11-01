# 3D-SynTree V2 — Formal MDP State-Space, Construction DAG, and Backward-Policy Specification

> **Status:** VERIFIED (issue #38 FLOW-04; owner reconciliation task "Reconcile
> GFlowNet State Space, Torsion Actions, and Reward Oracle"; decision D-V2-009
> in `docs/agent/DECISIONS.md`; constitution #45 §10.2, §12.3, §12.4).
> Implementation: `src/syntree/engine/environment.py`,
> `src/syntree/engine/gflownet.py`, `src/syntree/engine/rl.py`.
> Verification: `tests/test_gflownet_reconciliation.py` (50 tests, including
> the enumerable toy-DAG flow accounting).

This document closes the state-space debate of issue #38: it defines the MDP,
the induced construction DAG, the backward policy, and the exact Trajectory
Balance equation, and proves the properties the GFlowNet correctness claim
rests on.  The MDP and its induced DAG are **complementary objects**
(#45 §2.3): the MDP is the sequential decision process the policy acts in;
the DAG is the state graph GFlowNet flow conservation is defined over.

---

## 1. The two-level architecture

The reconciled design separates two concerns that the pre-reconciliation
backlog had entangled:

```
LEVEL 1 — the discrete GFlowNet MDP (pure probability space)
  state s_t      Discrete Assembly Object (seed + junction assignments; §2)
  action a_t     categorical, dimensionless (STOP / handle / reaction /
                 block / chirality / 36-bin torsion)
  geometry       deterministic bin-centre conformers (exact, no force field)
  backward P_B   Uniform Leaf Deletion Policy (§4)
  loss           Trajectory Balance over discrete log-probabilities (§5)

                 │  emitted only by the STOP action
                 ▼
LEVEL 2 — the physical reward oracle (continuous energy evaluation)
  1. materialise the full 3D ligand from the discrete bin centres
  2. bounded sub-bin refinement (±5°) — coarse-scan-seeded Brent (#42)
  3. scaffold-constrained MMFF94 relaxation (#13, 50 kcal/mol/Å² restraints)
  4. stationary physical reward R(x) (#30)
  5. the scalar R(x) feeds Trajectory Balance as the terminal evaluation
```

**Why the split.**  In GFlowNet theory the transition
`s_t --a_t--> s_{t+1}` must be governed strictly by the policy's transition
model.  If an unmodelled deterministic energy minimiser mutates the 3D
coordinates *after* the action is sampled, the physical `s_{t+1}` is not the
state the policy transitioned to: a continuous coordinate shift has zero
probability density under a discrete categorical distribution, flow
conservation fails, and `P(x) ∝ R(x)` is destroyed.  Moving **all** continuous
physics to the terminal oracle restores the contract: the refinement is a
*deterministic function of the discrete terminal state*, so `R(x)` is a valid,
stationary, positive scalar on discrete states — Trajectory Balance stays
exact while the physical ligand still receives continuous geometric
refinement before scoring.

---

## 2. The MDP (Level 1)

### 2.1 State

The GFlowNet state is the **Discrete Assembly Object**

```
s = ( B_0, { (h_i, r_i, B_i, c_i, k_i) }_{i=1..t} )
```

— the seed building block `B_0` together with the *set* of junction
assignments, where each junction contributes: the attachment handle `h_i`,
the reaction `r_i`, the added block `B_i`, the junction chirality choice
`c_i ∈ {R, S}`, and the torsion bin `k_i ∈ {1..36}`.

Two assembly **orders** of the same set of junction assignments denote the
**same state**.  This is the *canonical-derivation* construction of issue #38
item 3: the state is the quotient of ordered derivations by order, not the
ordered derivation itself.  Section 3 proves why this quotient — and not the
raw derivation tree — is the correct state space for molecule-level flow
accounting.  The environment bookkeeps derivations (add order, fragment ids)
as *implementation labels*; leaf counts, rewards, and TB terms are invariant
under relabelling (tested: atom renumbering, symmetric convergence).

For capacity-1 handles the deterministic geometry is itself
order-independent: each junction's rigid placement depends only on the local
frame at its attachment atom (never moved by other placements), so the same
set of junction assignments yields bit-for-bit identical coordinates
regardless of assembly order (tested).

### 2.2 Action

```
a_t ∈ { STOP } ∪ { (h, r, B, c, k) : h active handle, r registered reaction,
                    B catalog block, c ∈ {R,S,None}, k ∈ {1..36} }
```

All components are categorical and dimensionless (issue #24): every action
log-probability is a clean `log P ∈ (-∞, 0]` with no units and no continuous
density anywhere in the sum.  `STOP` is the explicit absorbing termination
action, evaluated *before* continuation-specific choices (#45 §8.4).

### 2.3 Transition (deterministic geometry, issue #24)

Applying `(h, r, B, c, k)`:

1. the junction bond forms between the handle atom and the block-side
   attachment atom (the leaving groups of both handles are deleted);
2. the block is placed by a **deterministic rigid transform** of its cached
   template conformer realising
   - bond length = sum of covalent radii,
   - junction bond angles from atom hybridisation (sp2 → 120°, sp3 → 109.47°),
   - junction dihedral `φ(a0, a1, b1, b0) = -π + (k - ½)·2π/36` — the exact
     bin centre;
   - the chirality choice `c` selects the enantiotopic placement branch
     (sign of the deterministic perpendicular of the placement frame);
3. resonance-locked junctions (amides/esters, declared by the reaction) snap
   strictly to planar trans 180° (or cis 0°);
4. surviving atom coordinates are never touched; **no continuous
   optimisation, no force field, and no stochastic noise runs** — repeating
   an action sequence reproduces intermediate coordinates bit-for-bit.

### 2.4 Termination and reward (Level 2)

`STOP` ends the trajectory.  Only then does the environment execute the
reward oracle (sub-bin refinement → MMFF94 → stationary `R(x)`, §1) on a deep
copy — the discrete state is never mutated.  `R(x) > 0` is a stationary,
reproducible function of the terminal object and the fixed pocket context
only (issue #30): no batch-relative term exists anywhere in its signature,
and diversity is maintained by GFlowNet sampling itself plus evaluation
telemetry, never by the per-trajectory loss.

---

## 3. The induced construction DAG and multi-path convergence

The forward actions induce a directed acyclic construction graph on the
states of §2.1: `s → s'` when `s' = s ∪ {one junction assignment}`.  Because
states are order-quotiented, the DAG has **multi-parent states**:

* **Symmetric molecules.**  Adding fragment A at handle h₁ then B at h₂
  yields the same state as B at h₂ then A at h₁ — one state, two forward
  paths.
* **Alternative reaction orderings.**  Any two assembly orders of the same
  junction set converge to the same state.
* **Atom renumbering / equivalent SMILES.**  Irrelevant to state identity:
  the state is the labelled graph, not an atom ordering (tested).

`STOP` edges make every state terminal-eligible: each state `s` carries a
stop edge into the absorbing sink with flow `R(s)`.

**Why derivation-tree states would be wrong.**  If the state were the *ordered
derivation*, every state would have a unique tree-parent (drop the last
action), forcing `P_B = 1`.  The flow into a terminal molecule `m` would then
be `F(m) = Σ_{derivations d of m} R(d)` — a molecule reachable in `N!`
orders receives `N!`·`R(m)` mass relative to an equally fit molecule with one
derivation.  The order-quotient repairs exactly this: the multiple orders are
multiple *paths* into one *state*, and the leaf-count backward policy
distributes the state's incoming flow across those paths (§4), so

```
F(x) = R(x)   for every terminal state x        (no multiplicity inflation)
```

**The worked example** (verified numerically in
`TestToyDagFlowBalance::test_multipath_convergence_splits_mass_exactly`).
Seed with two handles, blocks A and B, terminal state `x = {A@h₁, B@h₂}`
with two parents `p₁ = {A@h₁}`, `p₂ = {B@h₂}` and `|leaves(x)| = 2`:

```
F(x) = R(x)
backward edge flows:  F(p₁ ← x) = R(x)/2 ,  F(p₂ ← x) = R(x)/2
F(p₁) = R(p₁) + R(x)/2 ,  F(p₂) = R(p₂) + R(x)/2
optimal forward:  P_F(x | p₁) = R(x) / (2·F(p₁))
trajectory mass:  P(τ_p₁→x) = [F(p₁)/Z]·[R(x)/(2F(p₁))]·P(STOP|x) = R(x)/(2Z)
                  P(τ_p₂→x) = R(x)/(2Z)
molecule mass:    P(x) = P(τ_p₁→x) + P(τ_p₂→x) = R(x)/Z   ∎
```

The `1/|leaves|` factor splits the mass across the two derivations exactly;
with `P_B = 1` each path would have carried `R(x)/Z` and `x` would have
received `2·R(x)/Z` — the over-sampling failure the pre-reconciliation
specification would have caused.

---

## 4. The backward policy: Uniform Leaf Deletion (issue #38)

**Definition.**  For a non-initial state `s` whose molecule has removable
leaf synthons (non-seed fragments with exactly one junction bond in the
fragment junction graph — `get_removable_leaves`),

```
P_B(s' | s) = 1 / |RemovableLeafSynthons(s)|
```

uniform over the parents `s' = s \ {leaf junction}`.  Leaves are identified
on the RDKit graph in < 0.2 ms for `T ≤ 4`; trajectories are short by
construction (#29), so `|leaves| ∈ {1, 2, 3}` in practice.

**Property 1 (proper distribution over forward-parents).**  Every parent
obtained by deleting a removable leaf is reachable from by one forward
action (re-adding that leaf's junction), and every forward-parent of `s`
arises this way (its distinguishing junction is a removable leaf of `s`).
Hence `P_B(·|s)` is supported exactly on the forward-parents of `s` and sums
to 1.

**Property 2 (backward path measure identity).**  For every terminal state
`x`,

```
Σ_{backward paths x → s_0}  ∏ P_B(along the path)  =  1 .
```

*Proof (induction on the number of junctions).*  For the initial state the
empty path has measure 1.  For a state `s` with parents `p₁..p_n` (each
`P_B(p_i|s) = 1/n`), every backward path from `s` passes through exactly one
parent, so the total measure is `Σ_i P_B(p_i|s) · M(p_i) = (1/n)·Σ_i M(p_i)`
where `M(p_i)` is the path measure of `p_i`; by the induction hypothesis
each `M(p_i) = 1`, giving `(1/n)·n = 1`.  ∎

This identity is exactly the condition under which Trajectory Balance
admits flows `F(x) = R(x)` on terminal states (§5); it is verified on the
symmetric two-leaf fixture and, exhaustively, on the enumerable toy DAG.

**Property 3 (STOP backward term).**  The absorbing terminal reached by
`STOP` has exactly one parent — the pre-STOP state — so
`log P_B(s_n | terminal) = log 1 = 0` **by proof**.  Per issue #38's
acceptance rule ("`P_B = 1` only when uniqueness is established for the
actual state representation"), the STOP step contributes *no* backward term;
its recorded leaf count is telemetry only.  All *assembly* steps contribute
their true `-log |leaves(product state)|` term.

**What `P_B = 1` would have broken.**  On the quotient DAG, multi-parent
states would violate flow conservation at every convergence point (the
mass-splitting of §3 would not happen), over-sampling symmetric molecules
and every multi-order-reachable product — the exact failure issue #38 was
opened to prevent.

---

## 5. The exact Trajectory Balance equation (issue #29)

For a complete trajectory `τ = (s_0, a_0, s_1, …, a_{n-1}, s_n)` with
`a_{n-1} = STOP` and terminal object `x`:

```
L_TB(τ) = ( log Z  +  Σ_{t=0}^{n-1} log P_F(a_t | s_t)
                  −  Σ_{t=1}^{n}   log P_B(s_{t-1} | s_t)
                  −  log R(x) )²
```

with the forward term fully expanded over the dimensionless categorical
components (constitution §12.3):

```
log P_F(a_t|s_t) = log P(CONT|s_t) + log P(h_t|s_t) + log P(r_t|h_t)
                 + log P(B_t|r_t) + log P(c_t) + log P(φ_t = k_t | B_t)
```

(including, once #39 lands, the seed/anchor selection action's
log-probability), and the backward term instantiated by §4:

```
Σ log P_B =  Σ_{assembly steps t}  − log |RemovableLeafSynthons(s_t)|
          +  0                     (STOP step; uniqueness proved)
```

**Theorem (TB-zero ⇒ reward-proportional terminal distribution).**  If the
loss is zero for all trajectories, then with `Z = F(s_0)` the total flow,
`F(x) = R(x)` for every terminal state and the terminal distribution is
`P(x) = R(x)/Z`.  *Proof.*  TB-zero gives `Z·∏P_F(τ) = R(x_τ)·∏P_B(τ)`.
Summing over all trajectories ending in `x`:
`Z·P(x) = R(x)·Σ_{τ→x} ∏P_B(τ) = R(x)·1` (Property 2).  ∎

**Verification (the #38 acceptance criterion).**  On the enumerable toy DAG
(seed = 1,2,4-trichlorobenzene; catalog = {methylamine, methanol};
`max_blocks = 2`; 19 quotient states, 31 trajectories, real engine-measured
leaf counts, real stationary rewards): the analytically optimal flows drive
**every** TB residual through the real `TrajectoryBalanceEngine` to zero
(|residual| < 1e-9 in float64), the terminal probability of every outcome
equals `R(x)/Z` to 1e-9 relative, `Σ_x R(x) = Z`, and the two assembly
orders of every two-block state split its mass into exact halves
(`TestToyDagFlowBalance`).

---

## 6. Interface contracts and scope boundaries

| Concern | Owner | This specification's boundary |
|---|---|---|
| 36-bin categorical torsion head | #24 (MODEL-04) | the *environment* realises bin-centre geometry and the resonance snap; the head's logits remain #24's scope |
| Seed/anchor selection action | #39 (MODEL-06) | `s_0` is given; the TB sum reserves a slot for the seed action's log-probability |
| Certified 32-reaction taxonomy | #11 (CHEM-01) | the generic junction model (bond order, handle class, resonance lock) exposes the registry interface the taxonomy fills |
| Sub-bin refinement objective (terminal) | #42 (CHEM-07) | implemented as the terminal oracle stage (coarse-scan-seeded Brent, ±5°); production chemoselectivity and T4 benchmarking remain open |
| Constrained MMFF94 relaxation | #13 (CHEM-03) | terminal oracle stage with 50 kcal/mol/Å² seed restraints; the production chemistry module supersedes the internals |
| Production soft-core pocket potential | #14 (CHEM-04) | documented oracle-internal surrogate (soft-core LJ + softened Coulomb); injection point via `PocketDefinition` |
| Real docking engine | evaluation stack | `S_dock` surrogate documented in `rl.py`; clamp/units/sign are the #30-specified ones |
| `log Z_θ(P, s_0)` conditioning + trainer | #29/#26 | scalar `log_z` parameter behind the same attribute; the loss mathematics is final |
| Reward pharmacophore validation | #14/#35 | documented surrogate rules (salt bridges, H-bonds) with caps |

**Determinism contract.**  Bit-for-bit reproducibility holds per RDKit
build (same library version, same process class): block templates are
embedded once with fixed seeds and cached; every placement is a rigid
transform; the oracle's coarse-scan-seeded Brent and MMFF94 are deterministic
algorithms.  Cross-version coordinate equality is explicitly **not** claimed
(decision D-V2-009).

**Fail-closed guarantees.**  Malformed actions (bins, handles, capacities,
reactions, blocks), chemically invalid products, non-finite physics, missing
MMFF parameters, sub-floored rewards, and non-terminal oracle invocations all
raise precise errors (#45 §8); no NaN masking exists anywhere in the engine.
