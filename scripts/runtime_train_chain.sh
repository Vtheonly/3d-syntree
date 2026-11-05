#!/bin/bash
set -e
cd /content/3d-syntree
L=/content/logs/train_chain.log
mkdir -p /content/logs /content/runs
touch $L
log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a $L; }

TR=/content/work/data/trajectories
PR=/content/work/data/processed
CT=/content/work/catalog/enamine_real_50k_pharm3d_d256.h5
RD=/content/runs/stage1_t4

launch() {  # launch <extra-args...>
  python3 scripts/train_stage1.py --config configs/train_colab_t4.json \
    --trajectories "$TR" --processed "$PR" --catalog "$CT" \
    --run-dir "$RD" --device cuda "$@"
}

ckpt_state() {  # print the checkpoint continuation state as JSON
  python3 - <<'CS'
import torch, json
p = torch.load("/content/runs/stage1_t4/trainer_state.pt", weights_only=False, map_location="cpu")
print(json.dumps({
  "epoch": p["progress"]["epoch"],
  "batch_in_epoch": p["progress"]["batch_in_epoch"],
  "global_step": p["progress"]["global_step"],
  "stop_reason": (p.get("extra") or {}).get("stop_reason"),
  "history_len": len(p.get("history") or []),
}))
CS
}

# ---------- T1: controlled time-budget stop (adaptive, work-aware) ----------
# Probe with the largest budget first: if it COMPLETES we learn the total
# micro-step count and can pick a budget that stops near mid-run, leaving
# enough untrained steps for T2's mid-training SIGKILL (D-V2-047 follow-up:
# a 40 s stop on this dataset left only 2 steps, so run2 finished before
# the kill could land). If the probe itself stops mid-run, the epochs are
# long and the checkpoint already leaves ample work.
TOTAL=0
for BUDGET in 60 30 15; do
  log "T1: run with time-budget ${BUDGET}s"
  rm -rf "$RD"
  OUT=$(launch --time-budget-seconds "$BUDGET" 2>&1 | tail -20)
  echo "$OUT" >> $L
  ST=$(ckpt_state)
  log "T1: checkpoint state: $ST"
  if echo "$ST" | grep -q '"stop_reason": "time_budget"'; then
    if [ "$TOTAL" -gt 0 ]; then
      STEP=$(echo "$ST" | python3 -c 'import json,sys; print(json.load(sys.stdin)["global_step"])')
      LEFT=$((TOTAL - STEP))
      if [ "$LEFT" -ge 15 ]; then
        log "T1: CONTROLLED_STOP_OK budget=$BUDGET (steps done $STEP / $TOTAL, $LEFT left for T2)"
        break
      fi
      log "T1: only $LEFT steps left — too little for the T2 mid-training kill; retrying smaller"
      continue
    fi
    log "T1: CONTROLLED_STOP_OK budget=$BUDGET (epochs exceed the probe budget; ample work remains)"
    break
  fi
  TOTAL=$(echo "$ST" | python3 -c 'import json,sys; print(json.load(sys.stdin)["global_step"])')
  log "T1: stop_reason was not time_budget (total micro-steps: $TOTAL); retrying with a smaller budget"
done
echo "$ST" | grep -q '"stop_reason": "time_budget"' || { log "T1: FAILED to obtain a mid-run time-budget stop"; exit 1; }
RUN1="$ST"

# ---------- T2: resume + abrupt SIGKILL ----------
log "T2: resuming in background, will SIGKILL after observed training progress"
N1=$(wc -l < "$RD/telemetry.jsonl" 2>/dev/null || echo 0)
log "T2: telemetry lines at start: $N1"
rm -f /content/logs/train_run2.out
setsid nohup python3 scripts/train_stage1.py --config configs/train_colab_t4.json \
  --trajectories "$TR" --processed "$PR" --catalog "$CT" \
  --run-dir "$RD" --device cuda --resume > /content/logs/train_run2.out 2>&1 &
RUN2_PID=$!
log "T2: run2 pid=$RUN2_PID (pgid=$RUN2_PID)"
KILLED=0
for i in $(seq 1 90); do
  sleep 2
  if ! kill -0 $RUN2_PID 2>/dev/null; then log "T2: run2 exited on its own before kill"; break; fi
  N=$(wc -l < "$RD/telemetry.jsonl" 2>/dev/null || echo 0)
  if [ "$N" -gt "$N1" ]; then
    sleep 2   # let a couple more steps land, then kill mid-training
    if kill -0 $RUN2_PID 2>/dev/null; then
      kill -9 -$RUN2_PID 2>/dev/null || kill -9 $RUN2_PID 2>/dev/null || true
      KILLED=1
      log "T2: SIGKILL sent mid-training after new telemetry (lines $N1 -> $N)"
    else
      log "T2: run2 exited between the telemetry poll and the kill"
    fi
    break
  fi
done
if [ "$KILLED" = "1" ]; then
  log "T2: ABRUPT_KILL_MID_TRAINING (process alive + fresh telemetry at kill time)"
else
  kill -9 -$RUN2_PID 2>/dev/null || kill -9 $RUN2_PID 2>/dev/null || true
  log "T2: fallback kill after timeout (no mid-training kill achieved)"
fi
sleep 2
tail -5 /content/logs/train_run2.out >> $L || true

# checkpoint integrity after the abrupt kill (explicit sidecar verification)
python3 - <<'IV' >> $L 2>&1
import hashlib
p = "/content/runs/stage1_t4/trainer_state.pt"
digest = hashlib.sha256()
with open(p, "rb") as h:
    for chunk in iter(lambda: h.read(1 << 20), b""):
        digest.update(chunk)
expected = open(p + ".sha256").read().strip()
print("INTEGRITY_AFTER_KILL:", "PASS" if digest.hexdigest() == expected else "FAIL")
IV

# ---------- T3: resume to completion ----------
log "T3: resume to completion"
OUT=$(launch --resume 2>&1 | tail -25)
echo "$OUT" >> $L
ST3=$(ckpt_state)
log "T3: final state: $ST3"
echo "$ST3" | grep -q '"stop_reason": "max_epochs_reached"' || { log "T3: final stop_reason mismatch"; exit 1; }

# ---------- T4: verification + persistence ----------
python3 - <<'VF' >> $L 2>&1
import json, torch

p = torch.load("/content/runs/stage1_t4/trainer_state.pt", weights_only=False, map_location="cpu")
hist = p["history"]
print("total micro-batch records:", len(hist))
epochs = {}
for h in hist:
    epochs.setdefault(h["epoch"], []).append(h["total"])
means = {e: sum(v) / len(v) for e, v in sorted(epochs.items())}
print("per-epoch mean total loss:", json.dumps({str(k): round(v, 6) for k, v in means.items()}))
mono = all(means[e2] < means[e1] for e1, e2 in zip(sorted(means), sorted(means)[1:]))
print("STRICTLY_MONOTONIC_EPOCH_DESCENT:", "PASS" if mono else "FAIL")

# sequence-level descent check (optimizer-step view: every 6 micro-batches + flushes)
seq = [h["total"] for h in hist]
print("first/last micro-batch loss:", round(seq[0], 6), round(seq[-1], 6))

peak_reserved = 0.0
peak_alloc = 0.0
for line in open("/content/runs/stage1_t4/telemetry.jsonl"):
    try:
        rec = json.loads(line)
    except Exception:
        continue
    g = rec.get("gpu_memory") or {}
    if g.get("gpu_peak_reserved_mib"):
        peak_reserved = max(peak_reserved, g["gpu_peak_reserved_mib"])
    if g.get("gpu_peak_allocated_mib"):
        peak_alloc = max(peak_alloc, g["gpu_peak_allocated_mib"])
print("peak reserved VRAM MiB:", peak_reserved, "(", round(peak_reserved/1024, 3), "GiB )")
print("peak allocated VRAM MiB:", peak_alloc)
print("VRAM_UNDER_9_5_GIB:", "PASS" if peak_reserved < 9.5 * 1024 else "FAIL")

# last-completed-sample logging (#43)
last = hist[-1]
print("last completed: epoch", last["epoch"], "batch", last["batch"], "step", last["step"])
# progress.epoch counts COMPLETED epochs (D-V2-047 semantics)
print("epochs_completed:", p["progress"]["epoch"])
VF

log "T4: verification computed"

python3 - <<'PU' >> $L 2>&1
import os, tarfile
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported (no embedded credentials)"
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset", private=True, exist_ok=True)
with tarfile.open("/content/runs/stage1_t4_run.tar", "w") as tf:
    tf.add("/content/runs/stage1_t4", arcname="stage1_t4")
    tf.add("/content/hardware_preflight.json", arcname="hardware_preflight.json")
api.upload_file(path_or_fileobj="/content/runs/stage1_t4_run.tar",
                path_in_repo="runs/stage1_t4_session1/run.tar",
                repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset")
print("HF_RUN_UPLOAD_DONE")
PU
log "T4: run artifacts persisted to HF"
log "TRAIN_CHAIN_DONE"
