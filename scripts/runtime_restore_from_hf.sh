#!/bin/bash
# restore_from_hf.sh — S-V2-19 runtime recovery #2: restore the training-ready
# workspace from the HF-persisted bundle (no full pipeline re-run needed).
set -e
mkdir -p /content/logs /content/work/catalog /content/work/data/trajectories /content/work/data/splits /content/work/data/processed /content/work/bundle /content/runs
L=/content/logs/restore.log
touch $L
log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a $L; }

# ---------- repo + env ----------
if [ ! -d /content/3d-syntree ]; then
  log "clone"
  cd /content && git clone -q -b 3d-synV2 https://github.com/Vtheonly/3d-syntree.git
fi
log "HEAD $(git -C /content/3d-syntree rev-parse --short HEAD)"
if ! python3 -c "import syntree, rdkit, torch_geometric, h5py, psutil" 2>/dev/null; then
  cd /content/3d-syntree
  pip install -q -e '.[test]' 2>&1 | tail -1
  pip install -q torch_geometric h5py pandas pyarrow psutil 2>&1 | tail -1
fi
log "env ok (torch $(python3 -c 'import torch;print(torch.__version__)'))"

# ---------- hardware preflight of the new runtime ----------
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
log "preflight recorded"

# ---------- catalog ----------
if [ ! -f /content/work/catalog/enamine_real_50k_pharm3d_d256.h5 ]; then
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
[ "$S" = "8b8c13a0581ec91db6ace643e13d9d46c8b29211e1a264b2bdd62bb90bf2ab7e" ] || { log "CATALOG CHECKSUM MISMATCH $S"; exit 1; }
log "catalog verified"

# ---------- training bundle from HF ----------
if [ ! -f /content/work/data/trajectories/trajectories_train.pt ]; then
  python3 - <<'PB' >> $L 2>&1
import os, tarfile, hashlib, shutil
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be exported (no embedded credentials)"
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="JJKK1212/3d-syntree-v2-runs",
                    filename="data/training_bundle.tar",
                    repo_type="dataset", local_dir="/content/work/bundle")
print("bundle path:", p, "bytes:", os.path.getsize(p))
R = "/content/work/restore"
os.makedirs(R, exist_ok=True)
with tarfile.open(p) as tf:
    tf.extractall(R)
for f in ("trajectories_train.pt", "trajectories_val.pt", "trajectories_test.pt"):
    shutil.move(f"{R}/{f}", f"/content/work/data/trajectories/{f}")
shutil.move(f"{R}/unified_splits_30seq_id.parquet", "/content/work/data/splits/unified_splits_30seq_id.parquet")
shutil.move(f"{R}/synthon_index.json", "/content/work/catalog/synthon_index.json")
shutil.move(f"{R}/index_manifest.json", "/content/work/catalog/index_manifest.json")
src, dst = f"{R}/processed", "/content/work/data/processed"
n = 0
for name in os.listdir(src):
    shutil.move(os.path.join(src, name), os.path.join(dst, name))
    n += 1
print("processed artifacts restored:", n)
expected = {
 "trajectories_train.pt": "8e95ed891b68c3dd78546115d2338f66ea521a81ac72af55f6a2a074000d5e85",
 "trajectories_val.pt":   "76fe93077306336898535eb46803e3274688a597c3d2b6a18834a6fa2b928d80",
 "trajectories_test.pt":  "4f67184914fa40fe5ff1e493594afa156928e15621e855f2cad8fbd032264db8",
}
ok = True
for f, e in expected.items():
    h = hashlib.sha256(open(f"/content/work/data/trajectories/{f}", "rb").read()).hexdigest()
    print(f"{f}: {'MATCH' if h == e else 'MISMATCH'} (from the HF-persisted copy)")
    ok = ok and (h == e)
print("HF_RESTORE_BITEXACT:", "PASS" if ok else "FAIL")
PB
fi
log "RESTORE_DONE"
