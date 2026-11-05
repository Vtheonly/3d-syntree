# OPS-01 — Google Colab SSH daemon & interactive remote workflow (#36)

> The operational bridge for authorized Tesla T4 execution. Target branch
> `3d-synV2`; permanent constitution: issue #45 (§6, §28, §29).
>
> **Status: OPERATIONAL and VERIFIED (S-V2-21, closed as #36).** The
> agent-sandbox bridge runs through the official `google-colab-cli`
> (oauth2 user-consent, refresh-token cache) with real Tesla T4
> sessions (`syntree-t4`): **root shell verified over the SSH-proxy
> channel** (`SSH_BRIDGE_OK`), §29.2 hardware battery PASS with
> transcripts (`ssh_evidence/verify_syntree-t4_*.json`,
> `verify_s-v2-21_*.json`), the HF workspace restore bit-exact, and the
> full T1–T4 training chain executed remotely with persisted telemetry
> (evidence: `ssh_evidence/session1_chain/` + HF
> `runs/stage1_t4_session1/run.tar`; the S-V2-21 work-log entry).
> **Operational caveat (§29.2):** provisioning can silently return a
> CPU-only runtime when the assign endpoint answers `Service
> Unavailable` — `scripts/bootstrap_t4.sh` gates fail-closed on
> nvidia-smi AND torch-CUDA concurring on Tesla T4 before any work
> launches. The notebook tunnel cells below remain the documented
> fallback path.

## 0. Primary bridge — official Colab CLI from the agent sandbox (S-V2-21)

Google shipped the official Colab CLI (`pip install google-colab-cli`,
repo `googlecolab/google-colab-cli`); its documented command forms match
>#45 §29 (`colab ssh`, `-s SESSION`, `--proxy-mode`). The sandbox flow:

1. **Authenticate (user-consent boundary, #45 §29.1).** The CLI's oauth2
   flow prints a Google consent URL; the OPERATOR opens it, authenticates
   with their own Google account, and pastes back the displayed
   one-time authorization code. Because sandbox processes cannot span
   tool calls, `scripts/colab_auth_flow.py` (sandbox-side, untracked)
   persists the PKCE `code_verifier` between the URL step and the code
   step, then writes the token to the CLI's own cache
   (`~/.config/colab-cli/token.json`, refresh-token enabled). The agent
   never handles the user's password; only the one-time code Google's
   landing page shows them.
2. **Provision.** `colab new -s syntree-t4 --gpu T4` (T4 is a supported
   value; a 400 means no GPU entitlement on the account — report, never
   fall back silently). **Verified failure mode (S-V2-21):** under
   capacity pressure the assign endpoint answers `Service Unavailable`
   and the CLI can hand back a **CPU-only runtime without failing** —
   always re-gate on the §29.2 battery before launching work
   (`scripts/bootstrap_t4.sh` does this: nvidia-smi AND torch-CUDA must
   concur on Tesla T4; non-T4 runtimes are disposed and retried, never
   trained on).
3. **Connect without a local `ssh` binary.** `colab ssh` shells out to
   OpenSSH; sandbox builds lack it. `colab ssh --proxy-mode` is an
   OpenSSH-ProxyCommand-compatible WebSocket-stdio bridge, so the repo's
   `scripts/colab_ssh_client.py --proxy 'colab ssh --proxy-mode -s
   syntree-t4'` rides it with paramiko. The CLI reads the sandbox
   PUBLIC key from `~/.ssh/id_ed25519.pub` (WebSocket header) while
   paramiko holds the private key (never committed). Requirements:
   `pip install paramiko google-colab-cli`; pub key at `~/.ssh/`.
   **Do NOT pass `--identity/-i` in this sandbox:** that path shells out
   to `ssh-keygen` (absent here) to derive the public key — the default
   `~/.ssh/id_ed25519.pub` scan reads the file directly (verified
   S-V2-21: `-i` → `failed to derive public key`; default → root shell).
4. **Verify (§29.2).** `colab_ssh_client.py verify --proxy ... --tag
   <session-tag>` runs the full battery (nvidia-smi, free -h, df, torch
   CUDA, T4 policy gate) and saves a JSON transcript.
5. **Operate.** `exec`, `tail`, `put`, `get` subcommands drive the
   runtime; long jobs run with `nohup` on the VM (VM processes survive
   between agent tool calls; sandbox processes do not) and are polled
   through the bridge. `colab stop -s syntree-t4` releases the VM when
   done (idle VMs burn compute units).

## 1. Responsibilities (constitution #45 §28.1)

| Environment | Responsibility |
|---|---|
| Local dev sandbox (this agent) | inspect/edit source, CPU-only unit/static tests, diff review, commit + push to `3d-synV2` |
| Authorized Colab/Kaggle T4 | pull the exact commit, install deps, verify data/checksums, GPU smoke, launch the canonical training CLI, monitor host RAM + GPU VRAM separately, persist logs/checkpoints |

The GPU training job is **never** launched in the local sandbox, and a
local CPU run is **never** reported as T4 validation.

## 2. Runtime initialization (the operator, in Colab) — FALLBACK path

> The primary path is §0 (official Colab CLI). The notebook cells below
> are the fallback when the CLI cannot be used (e.g. no PyPI access from
> the sandbox, or a CLI regression).

1. Open <https://colab.research.google.com>, select
   **Runtime → Change runtime type → T4 GPU**.
2. Upload or open `notebooks/run_3d_syntree.ipynb` and run the cells in
   order:
   - **Cell 1** fails closed unless the allocated GPU is a **Tesla T4**
     (#45 §29.2: any other device must be reported, never trained on);
   - **Cell 2** installs the system packages (`aria2`, `libxrender1`,
     `openssh-server`);
   - **Cell 3** starts `sshd` (public-key only), installs the agent
     sandbox public key, and opens a public TCP tunnel (bore.pub, no
     account needed), printing `ENDPOINT_FOR_AGENT = host:port`;
   - **Cell 3b** is the tunnel fallback/status/reconnect cell (ngrok with
     the operator's authtoken, or a bore restart);
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
| Notebook + setup scripts + operator commands prepared | **done** (fallback path retained) |
| Interactive SSH connection to a live Colab T4, authorized | **done** (S-V2-21: `SSH_BRIDGE_OK` over `colab ssh --proxy-mode` + paramiko, root shell) |
| T4 verified via `nvidia-smi` **and** PyTorch (~15 GB VRAM) | **done** (battery transcripts: Tesla T4 15360 MiB / 14.56 GiB torch-visible, cc 7.5, gate PASS) |
| Host RAM / GPU VRAM captured separately before + during a GPU test | **done** (preflight + per-step telemetry: host 12.67 GiB total pool; GPU peak reserved 2292 MiB / allocated 1501.8 MiB) |
| Remote forward/backward test with measured peak memory | **done** (the T-chain training runs are full fwd/bwd on device; peaks recorded in `telemetry.jsonl`) |
| Canonical training CLI launched remotely with persisted telemetry | **done** (T-chain T4: `TRAIN_CHAIN_DONE`, telemetry + checkpoint persisted to HF `runs/stage1_t4_session1/run.tar`) |

#36 closed 2026-11-06 (S-V2-21) with the authorized-session evidence in
`ssh_evidence/` and on HF.
