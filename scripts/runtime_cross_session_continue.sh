#!/bin/bash
# runtime_cross_session_continue.sh — the #43 cross-session continuation
# evidence driver. Downloads a persisted Stage-1 run from HF (produced by
# runtime_train_chain.sh on a PREVIOUS session), verifies its checkpoint,
# resumes training with max_epochs extended on THIS runtime (a separate
# Colab VM), and asserts the exact-continuation contract:
#   - the resumed history's first N records are identical to the source
#     (no skipped, no repeated, no rewritten samples),
#   - telemetry continues on the new runtime with a fresh hardware
#     baseline (#43 addendum: the resumed process records the new
#     runtime's hardware/memory baseline),
#   - the checkpoint integrity sidecar is verified before resume.
# Usage: runtime_cross_session_continue.sh <source_tag> <target_tag> <extra_epochs>
# Env: HF_TOKEN (required). Requires the workspace restored by
# runtime_restore_from_hf.sh on THIS VM.
set -e
SRC_TAG="${1:-session1}"
DST_TAG="${2:-session2}"
EXTRA_EPOCHS="${3:-1}"
cd /content/3d-syntree
L=/content/logs/cross_session_${SRC_TAG}_to_${DST_TAG}.log
mkdir -p /content/logs /content/runs
touch $L
log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a $L; }

TR=/content/work/data/trajectories
PR=/content/work/data/processed
CT=/content/work/catalog/enamine_real_50k_pharm3d_d256.h5
RD=/content/runs/stage1_t4

log "X1: source run tag: $SRC_TAG -> target: $DST_TAG (+$EXTRA_EPOCHS epochs)"

# ---------- X1: fetch + verify the persisted source run ----------
# D-V2-049: the driver's Python heredocs are QUOTED ('DL'/'ME'/'TB'/'PU'), so
# bash never interpolates ${SRC_TAG}/${DST_TAG}/${EXTRA_EPOCHS}/${N_TEL_BEFORE}
# into them — the f-strings reached Python with the literal `${...}` text and
# died with NameError on the first live execution. Pass the tags through the
# environment and read them with os.environ (no interpolation needed).
SRC_TAG=$SRC_TAG DST_TAG=$DST_TAG python3 - <<'DL' >> $L 2>&1
import os, tarfile, hashlib
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported"
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="JJKK1212/3d-syntree-v2-runs",
                    filename=f"runs/stage1_t4_{os.environ['SRC_TAG']}/run.tar",
                    repo_type="dataset", local_dir="/content/work/bundle")
print("source run.tar:", p, os.path.getsize(p), "bytes")
with tarfile.open(p) as tf:
    tf.extractall("/content/work/source_run")
print("extracted to /content/work/source_run")
DL
SRC=/content/work/source_run/stage1_t4
[ -f "$SRC/trainer_state.pt" ] || { log "X1: source checkpoint missing"; exit 1; }
# D-V2-049 (defect 2): runtime_train_chain.sh tars hardware_preflight.json at
# the ARCHIVE ROOT (arcname="hardware_preflight.json", sibling of stage1_t4/),
# not inside stage1_t4/ — the guard must check the root, exactly like the X2
# copy below already does.
[ -f /content/work/source_run/hardware_preflight.json ] || { log "X1: source preflight missing"; exit 1; }
python3 - <<'IV' >> $L 2>&1
import hashlib
p = "/content/work/source_run/stage1_t4/trainer_state.pt"
digest = hashlib.sha256()
with open(p, "rb") as h:
    for chunk in iter(lambda: h.read(1 << 20), b""):
        digest.update(chunk)
expected = open(p + ".sha256").read().strip()
print("SOURCE_CHECKSUM:", "PASS" if digest.hexdigest() == expected else "FAIL")
assert digest.hexdigest() == expected
IV
log "X1: source checkpoint integrity verified"

# source state + history fingerprint (the continuation baseline)
python3 - <<'SB' >> $L 2>&1
import torch, json, hashlib
p = torch.load("/content/work/source_run/stage1_t4/trainer_state.pt",
               weights_only=False, map_location="cpu")
hist = p["history"]
blob = json.dumps(hist, sort_keys=True).encode()
print("SOURCE_STATE:", json.dumps({
    "epoch": p["progress"]["epoch"],
    "batch_in_epoch": p["progress"]["batch_in_epoch"],
    "global_step": p["progress"]["global_step"],
    "history_len": len(hist),
    "history_sha256": hashlib.sha256(blob).hexdigest(),
    "stop_reason": (p.get("extra") or {}).get("stop_reason"),
}))
SB
log "X1: source state recorded"

# ---------- X2: stage the run dir on THIS runtime ----------
rm -rf "$RD"
mkdir -p "$RD"
cp "$SRC"/trainer_state.pt "$SRC"/trainer_state.pt.sha256 "$RD"/
cp "$SRC"/telemetry.jsonl "$RD"/ 2>/dev/null || log "X2: no source telemetry (fresh file)"
cp /content/work/source_run/hardware_preflight.json /content/logs/source_hardware_preflight.json
N_TEL_BEFORE=$(wc -l < "$RD/telemetry.jsonl" 2>/dev/null || echo 0)
log "X2: run dir staged (telemetry lines: $N_TEL_BEFORE); this runtime's preflight: $(python3 -c 'import json; print(json.load(open("/content/hardware_preflight.json"))["gpu"]["name"])')"

# ---------- X3: resume with extended epochs on the NEW runtime ----------
MAX_EPOCHS=$(EXTRA_EPOCHS=$EXTRA_EPOCHS python3 - <<'ME'
import os, torch
p = torch.load("/content/work/source_run/stage1_t4/trainer_state.pt",
               weights_only=False, map_location="cpu")
profile = p.get("profile") or {}
print(int(profile.get("max_epochs", 2)) + int(os.environ["EXTRA_EPOCHS"]))
ME
)
log "X3: resuming with max_epochs=$MAX_EPOCHS on this runtime"
OUT=$(python3 scripts/train_stage1.py --config configs/train_colab_t4.json \
  --trajectories "$TR" --processed "$PR" --catalog "$CT" \
  --run-dir "$RD" --device cuda --resume --max-epochs "$MAX_EPOCHS" 2>&1 | tail -20)
echo "$OUT" >> $L
log "X3: resume run finished"

# ---------- X4: the exact-continuation assertions ----------
python3 - <<'VF' >> $L 2>&1
import torch, json, hashlib
src = torch.load("/content/work/source_run/stage1_t4/trainer_state.pt",
                 weights_only=False, map_location="cpu")
dst = torch.load("/content/runs/stage1_t4/trainer_state.pt",
                 weights_only=False, map_location="cpu")
s_hist, d_hist = src["history"], dst["history"]
n = len(s_hist)
prefix_identical = d_hist[:n] == s_hist
new_records = len(d_hist) - n
src_blob = hashlib.sha256(json.dumps(s_hist, sort_keys=True).encode()).hexdigest()
print("CROSS_SESSION_CONTINUATION:", "PASS" if prefix_identical and new_records > 0 else "FAIL")
print("history: source", n, "-> resumed", len(d_hist), f"(+{new_records} new records)")
print("source history sha256:", src_blob)
print("resumed prefix identical (no repeats/skips/rewrites):", prefix_identical)
print("resumed state:", json.dumps({
    "epoch": dst["progress"]["epoch"],
    "global_step": dst["progress"]["global_step"],
    "stop_reason": (dst.get("extra") or {}).get("stop_reason"),
}))
assert prefix_identical, "the resumed run rewrote or skipped source history"
assert new_records > 0, "the resumed run trained nothing new"
assert (dst.get("extra") or {}).get("stop_reason") == "max_epochs_reached"
VF
log "X4: continuation assertions passed"

# the new runtime's hardware baseline must appear in the resumed telemetry
N_TEL_BEFORE=$N_TEL_BEFORE python3 - <<'TB' >> $L 2>&1
import json, os
lines = [json.loads(l) for l in open("/content/runs/stage1_t4/telemetry.jsonl") if l.strip()]
n_before = int(os.environ["N_TEL_BEFORE"])
new_lines = lines[n_before:] if n_before else lines
print("telemetry lines: total", len(lines), "| new on this runtime:", len(new_lines))
assert new_lines, "no telemetry recorded on the resumed runtime"
sample = new_lines[0]
print("new-runtime telemetry sample keys:", sorted(sample.keys())[:8])
print("NEW_RUNTIME_TELEMETRY_BASELINE:", "PASS" if "host_memory" in sample and "gpu_memory" in sample else "FAIL")
assert "host_memory" in sample and "gpu_memory" in sample
TB
log "X4: new-runtime telemetry baseline verified"

# ---------- X5: persist the cross-session run ----------
DST_TAG=$DST_TAG python3 - <<'PU' >> $L 2>&1
import os, tarfile
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported"
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset",
                private=True, exist_ok=True)
with tarfile.open("/content/runs/cross_session_run.tar", "w") as tf:
    tf.add("/content/runs/stage1_t4", arcname="stage1_t4")
    tf.add("/content/hardware_preflight.json", arcname="hardware_preflight.json")
    tf.add("/content/logs/source_hardware_preflight.json", arcname="source_hardware_preflight.json")
api.upload_file(path_or_fileobj="/content/runs/cross_session_run.tar",
                path_in_repo=f"runs/stage1_t4_{os.environ['DST_TAG']}/run.tar",
                repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset")
print("HF_CROSS_SESSION_UPLOAD_DONE:", os.environ["DST_TAG"])
PU
log "X5: cross-session run persisted (runs/stage1_t4_${DST_TAG}/run.tar)"
log "CROSS_SESSION_DONE"
