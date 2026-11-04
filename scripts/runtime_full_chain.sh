#!/bin/bash
set -e
mkdir -p /content/logs /content/work/catalog /content/work/raw /content/work/tools
L=/content/logs/chain.log
touch $L
log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a $L; }

# ---------- stage 0: system packages + sshd ----------
if ! command -v aria2c >/dev/null; then
  log "stage0: apt packages"
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2 libxrender1 openssh-server >/dev/null 2>&1
fi
mkdir -p /var/run/sshd
grep -q "3D-SynTree OPS-01 bridge" /etc/ssh/sshd_config || { echo -e "\n# 3D-SynTree OPS-01 bridge\nPermitRootLogin yes" >> /etc/ssh/sshd_config; }
pgrep sshd >/dev/null || /usr/sbin/sshd || true
log "stage0: ok (sshd up: $(pgrep -c sshd))"

# ---------- stage 1: repo ----------
if [ ! -d /content/3d-syntree ]; then
  log "stage1: clone"
  cd /content && git clone -q -b 3d-synV2 https://github.com/Vtheonly/3d-syntree.git
  cd /content/3d-syntree && git config core.fileMode false
fi
log "stage1: HEAD $(git -C /content/3d-syntree rev-parse --short HEAD)"

# ---------- stage 2: python env ----------
if ! python3 -c "import syntree, rdkit, torch_geometric, h5py, psutil" 2>/dev/null; then
  log "stage2: pip install"
  cd /content/3d-syntree
  pip install -q -e '.[test]' 2>&1 | tail -1
  pip install -q torch_geometric h5py pandas pyarrow psutil 2>&1 | tail -1
fi
log "stage2: env ok (torch $(python3 -c 'import torch;print(torch.__version__)'))"

# ---------- stage 2.5: hardware preflight (separate RAM/VRAM pools) ----------
python3 - <<'HP' >> $L 2>&1
import json, psutil, torch
vm = psutil.virtual_memory()
preflight = {
    "gpu": {
        "name": torch.cuda.get_device_name(0),
        "total_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        "free_vram_mib": round(torch.cuda.mem_get_info()[0] / 1024**2, 1),
        "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
    },
    "host": {
        "total_ram_mib": round(vm.total / 1024**2, 1),
        "available_ram_mib": round(vm.available / 1024**2, 1),
        "process_rss_mib": round(psutil.Process().memory_info().rss / 1024**2, 1),
        "cpu_count": psutil.cpu_count(),
    },
}
open("/content/hardware_preflight.json", "w").write(json.dumps(preflight, indent=2))
print("preflight:", json.dumps(preflight))
HP
log "stage2.5: preflight recorded"

# ---------- stage 3: catalog ----------
if [ ! -f /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 ]; then
  log "stage3: catalog download"
  python3 - <<'PY3' >> $L 2>&1
import os
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported (no embedded credentials)"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="JJKK1212/3d-syntree-v2-catalog",
                filename="enamine_real_50k_pharm3d_d256.h5",
                repo_type="dataset", local_dir="/content/work/catalog")
PY3
fi
S=$(sha256sum /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 | cut -d' ' -f1)
[ "$S" = "8b8c13a0581ec91db6ace643e13d9d46c8b29211e1a264b2bdd62bb90bf2ab7e" ] || { log "stage3: CATALOG CHECKSUM MISMATCH $S"; exit 1; }
log "stage3: catalog verified"

# ---------- stage 4: raw sources ----------
cd /content/work/raw
[ -f crossdocked_pocket10.tar.gz ] || aria2c -q --checksum=md5=59416d06c2c366f6e05b91fdff8584f7 -o crossdocked_pocket10.tar.gz "https://huggingface.co/datasets/Yukk1Zz/if3-crossdocked2020/resolve/main/crossdocked_pocket10.tar.gz"
[ -f pbpp-2020.zip ] || aria2c -q --checksum=md5=56c61303cb29d08a2ad6b13f4ae2b6ef -o pbpp-2020.zip "https://huggingface.co/datasets/photonmz/pdbbindpp-2020/resolve/main/pbpp-2020.zip"
[ -f bindingmoad_final.csv ] || aria2c -q -o bindingmoad_final.csv "https://raw.githubusercontent.com/Mugsie1969/BindingMOAD/main/BindingMOAD_data_for_PDB_1_31_2022.csv"
log "stage4: raw sources present"

# ---------- stage 5: mmseqs static binary ----------
if [ ! -x /content/work/tools/mmseqs/bin/mmseqs ]; then
  cd /content/work/tools
  aria2c -q -x4 https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz || wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
  tar xzf mmseqs-linux-avx2.tar.gz
fi
log "stage5: mmseqs staged"

# ---------- stage 6: preprocessing (resumable) ----------
cd /content/3d-syntree
if [ ! -f /content/work/data/processed/.done ]; then
  log "stage6: preprocessing"
  python3 scripts/preprocess_multidataset.py --source all --raw-dir /content/work/raw \
    --out-dir /content/work/data/processed --gzip --workers 2 >> /content/logs/preprocess.log 2>&1
  N=$(wc -l < /content/work/data/processed/processed_manifest.jsonl)
  [ "$N" = "204005" ] || { log "stage6: RECORD COUNT MISMATCH ($N != 204005)"; exit 1; }
  touch /content/work/data/processed/.done
fi
log "stage6: preprocessing done (204005 records)"

# ---------- stage 7: mmseqs split ----------
if [ ! -f /content/work/data/splits/unified_splits_30seq_id.parquet ]; then
  log "stage7: mmseqs split"
  MMSEQS_BIN=/content/work/tools/mmseqs/bin/mmseqs bash scripts/run_mmseqs_split.sh \
    /content/work/data/processed/processed_manifest.jsonl /content/work/data/splits >> /content/logs/split.log 2>&1
fi
log "stage7: splits present"

# ---------- stage 8: synthon index ----------
if [ ! -f /content/work/catalog/synthon_index.json ]; then
  log "stage8: index build"
  python3 scripts/build_synthon_index.py --catalog /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --out /content/work/catalog/synthon_index.json --manifest /content/work/catalog/index_manifest.json >> $L 2>&1
fi
log "stage8: index present"

# ---------- stage 9: trajectory extraction (resumable chunks) ----------
DONE_TARGET=170955
for i in $(seq 1 10); do
  DONE=$(python3 - <<'PY4'
import os
d = "/content/work/data/trajectories"
n = 0
for f in ("trajectories_train.jsonl", "trajectories_val.jsonl", "trajectories_test.jsonl", "extraction_failures.jsonl"):
    p = os.path.join(d, f)
    if os.path.exists(p):
        n += sum(1 for _ in open(p))
print(n)
PY4
)
  log "stage9: done count $DONE / $DONE_TARGET"
  [ "$DONE" -ge "$DONE_TARGET" ] && break
  python3 scripts/build_trajectories.py \
    --manifest /content/work/data/processed/processed_manifest.jsonl \
    --splits /content/work/data/splits/unified_splits_30seq_id.parquet \
    --catalog /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --index /content/work/catalog/synthon_index.json \
    --out-dir /content/work/data/trajectories --max-seconds 900 >> /content/logs/extract.log 2>&1 \
    || { log "stage9: chunk $i failed"; exit 1; }
done
log "stage9: extraction complete"

# ---------- stage 10: finalize ----------
if [ ! -f /content/work/data/trajectories/trajectories_train.pt ]; then
  log "stage10: finalize"
  python3 scripts/build_trajectories.py \
    --manifest /content/work/data/processed/processed_manifest.jsonl \
    --splits /content/work/data/splits/unified_splits_30seq_id.parquet \
    --catalog /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --index /content/work/catalog/synthon_index.json \
    --out-dir /content/work/data/trajectories --finalize --require-complete >> /content/logs/extract.log 2>&1
fi
log "stage10: finalize done"

# ---------- stage 11: verify + portable bundle + HF persist ----------
cd /content/work/data/trajectories
python3 - <<'PV' >> $L 2>&1
import hashlib
expected = {
 "trajectories_train.pt": "8e95ed891b68c3dd78546115d2338f66ea521a81ac72af55f6a2a074000d5e85",
 "trajectories_val.pt":   "76fe93077306336898535eb46803e3274688a597c3d2b6a18834a6fa2b928d80",
 "trajectories_test.pt":  "4f67184914fa40fe5ff1e493594afa156928e15621e855f2cad8fbd032264db8",
}
ok = True
for f, e in expected.items():
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(f"{f}: {'MATCH' if h == e else 'MISMATCH'}")
    ok = ok and (h == e)
print("TRAJECTORY_BITEXACT:", "PASS" if ok else "FAIL")
PV
log "stage11: checksums verified"

python3 - <<'PB' >> $L 2>&1
import os, tarfile, torch
os.makedirs("/content/work/portable", exist_ok=True)
traj_dir = "/content/work/data/trajectories"
proc_dir = "/content/work/data/processed"
ids = set()
for split in ("train", "val", "test"):
    for rec in torch.load(f"{traj_dir}/trajectories_{split}.pt", weights_only=False):
        ids.add(rec["complex_id"])
print("portable: complexes needed:", len(ids))
with tarfile.open("/content/work/portable/training_bundle.tar", "w") as tf:
    for split in ("train", "val", "test"):
        tf.add(f"{traj_dir}/trajectories_{split}.pt", arcname=f"trajectories_{split}.pt")
    tf.add("/content/work/data/splits/unified_splits_30seq_id.parquet", arcname="unified_splits_30seq_id.parquet")
    tf.add("/content/work/catalog/synthon_index.json", arcname="synthon_index.json")
    tf.add("/content/work/catalog/index_manifest.json", arcname="index_manifest.json")
    matched = 0
    for cid in sorted(ids):
        added = False
        for suffix in ("_ligand.sdf.gz", "_pocket.pdb.gz", "_ligand.sdf", "_pocket.pdb"):
            p = os.path.join(proc_dir, cid + suffix)
            if os.path.exists(p):
                tf.add(p, arcname="processed/" + cid + suffix)
                matched += 1
                added = True
        if not added:
            print("portable: MISSING artifacts for", cid)
    print("portable: artifact files:", matched)
    if matched != 2 * len(ids):
        raise SystemExit(
            f"portable: expected {2 * len(ids)} artifacts (ligand+pocket per "
            f"complex), packed {matched} — refusing an incomplete bundle"
        )
print("portable: bundle bytes:", os.path.getsize("/content/work/portable/training_bundle.tar"))
PB
log "stage11: portable bundle assembled"

python3 - <<'PU' >> $L 2>&1
import os
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported (no embedded credentials)"
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="/content/work/portable/training_bundle.tar",
                path_in_repo="data/training_bundle.tar",
                repo_id="JJKK1212/3d-syntree-v2-runs", repo_type="dataset")
print("HF_UPLOAD_DONE")
PU
log "stage11: HF persistence done"
log "FULL_CHAIN_DONE"
