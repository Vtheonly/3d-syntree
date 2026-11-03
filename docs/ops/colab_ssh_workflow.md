# OPS-01 — Google Colab SSH daemon & interactive remote workflow (#36)

> The operational bridge for authorized Tesla T4 execution. Target branch
> `3d-synV2`; permanent constitution: issue #45 (§6, §28, §29). This
> document is the **operator-facing workflow**; the runtime-side
> initialization lives in `notebooks/run_3d_syntree.ipynb`.
>
> **Status: PREPARED — awaiting operator authentication/authorization.**
> No SSH connection has been established (per #45 §6.5 the agent never
> claims a connection before it is real; no link is fabricated).

## 1. Responsibilities (constitution #45 §28.1)

| Environment | Responsibility |
|---|---|
| Local dev sandbox (this agent) | inspect/edit source, CPU-only unit/static tests, diff review, commit + push to `3d-synV2` |
| Authorized Colab/Kaggle T4 | pull the exact commit, install deps, verify data/checksums, GPU smoke, launch the canonical training CLI, monitor host RAM + GPU VRAM separately, persist logs/checkpoints |

The GPU training job is **never** launched in the local sandbox, and a
local CPU run is **never** reported as T4 validation.

## 2. Runtime initialization (the operator, in Colab)

1. Open <https://colab.research.google.com>, select
   **Runtime → Change runtime type → T4 GPU**.
2. Upload or open `notebooks/run_3d_syntree.ipynb` and run the cells in
   order:
   - **Cell 1** fails closed unless the allocated GPU is a **Tesla T4**
     (#45 §29.2: any other device must be reported, never trained on);
   - **Cell 2** installs the system packages (`aria2`, `libxrender1`,
     `openssh-server`);
   - **Cell 3** starts `sshd` (the runtime side of the bridge);
   - **Cells 4–6** check out the exact branch commit, install the Python
     environment, and record the mandatory hardware preflight (GPU VRAM
     and host RAM measured **separately**, plus disk);
   - **Cell 7** runs the GPU forward/backward smoke and records peak
     allocated/reserved VRAM;
   - **Cell 8** is the canonical Stage-1 training launch.

## 3. The SSH connection (the operator's local terminal)

The owner-documented Colab CLI patterns (#45 §29) — run these **in your
local terminal**, not in the notebook:

```bash
# direct interactive shell
colab ssh

# connect to a named session
colab ssh -s SESSION

# OpenSSH ProxyCommand integration
ssh -o ProxyCommand="colab ssh --proxy-mode -s %h" SESSION
```

Optional `~/.ssh/config` entry:

```sshconfig
Host colab
  User root
  ProxyCommand colab ssh --proxy-mode -s colab
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  RequestTTY yes
  RemoteCommand cd /content 2>/dev/null; exec bash -l
```

**Authentication handoff (exact, #45 §29.1):**

1. `colab ssh` prints the **actual authentication link** from Colab;
2. open the link and authenticate with **your own Google account**;
3. explicitly confirm to the agent that authorization is complete and
   the session is available;
4. only then does the agent connect through the established session.

The agent never invents or fabricates the link, never authenticates on
the operator's behalf, and never asks for passwords/OAuth tokens/private
keys in chat. If the installed `colab` CLI rejects the flags above,
report its `--help` output — do not substitute a materially different
authentication flow without asking (#45 §29).

## 4. Post-connection verification (the agent, over SSH)

Executed and recorded on the authorized session (#45 §29.2–§29.3):

```bash
git -C /content/3d-syntree rev-parse HEAD        # exact commit
git -C /content/3d-syntree status --short        # clean tree
nvidia-smi                                       # T4 + free/used VRAM
free -h                                          # host RAM (separate pool)
df -h /content                                   # disk
python --version
```

then from the active project environment:

```python
import torch
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | VRAM {p.total_memory/1024**3:.2f} GiB | cc {p.major}.{p.minor}")
```

Hardware policy reconciliation: if the device is **not** a Tesla T4, GPU
training stops and the actual device is reported (#45 §29.2).

## 5. Remote training procedure (#45 §28.5)

1. Confirm the Profile A configuration (T4, FP32, B=12 × accum 6, pocket
   cap 700) and that no FP16/BF16 autocast or GradScaler is active
   (asserted by `syntree/utils/hardware.py`, the #10 gate).
2. Run the GPU forward/backward smoke; record peak allocated/reserved
   VRAM and host RSS (`telemetry.jsonl` carries both pools separately).
3. Verify the dataset/trajectory/catalog manifests and checksums (never
   trust filenames alone).
4. Launch the canonical CLI **inside the SSH session**:

   ```bash
   python scripts/train_stage1.py \
       --config configs/train_colab_t4.json \
       --trajectories /content/work/data/trajectories \
       --processed /content/work/data/processed \
       --catalog "$SYNTREE_CATALOG_H5" \
       --run-dir /content/runs/stage1_t4 \
       --device cuda \
       --time-budget-seconds 8400
   ```

5. Monitor `telemetry.jsonl` (host RAM / GPU VRAM / throughput / losses
   / checkpoint state) and persist run outputs to durable storage (HF
   sync, #37).
6. On session limits, `--time-budget-seconds` triggers a controlled
   checkpoint + graceful stop; the next authorized session resumes with
   `--resume` (#43 contract: exact continuation, no skipped/repeated
   samples).

## 6. Completion criteria (#36 acceptance; honest status)

| Evidence item | Status |
|---|---|
| Notebook + setup scripts + operator commands prepared | **done** (this commit) |
| Interactive SSH connection to a live Colab T4, authorized | **pending operator authentication** |
| T4 verified via `nvidia-smi` **and** PyTorch (~15 GB VRAM) | **pending the authorized session** |
| Host RAM / GPU VRAM captured separately before + during a GPU test | **pending the authorized session** |
| Remote forward/backward test with measured peak memory | **pending the authorized session** |
| Canonical training CLI launched remotely with persisted telemetry | **pending the authorized session** |

An SSH shell alone or a bare `nvidia-smi` does **not** close #36 (#45
§28.7). The issue stays open until the authorized-session evidence
exists.
