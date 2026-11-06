#!/usr/bin/env bash
# MMseqs2 30%-identity sequence-clustered split generation (issue #17 DATA-02).
#
# Orchestrates the three stages documented in scripts/split_clusters.py:
#   1. extract  — processed_manifest.jsonl -> all_proteins.fasta (+ seq_map)
#   2. cluster  — mmseqs easy-cluster at the pinned contract
#   3. allocate — cluster TSVs -> unified_splits_30seq_id.parquet + manifest
#
# Usage:
#   scripts/run_mmseqs_split.sh <processed_manifest.jsonl> <workdir> \
#       [--mmseqs /path/to/mmseqs]
#
# The mmseqs binary may be supplied via --mmseqs or the MMSEQS_BIN env var;
# the script fails closed if it cannot be executed (constitution #45 §11:
# the clustering step is a hard dependency of leakage-free splits, never
# silently skipped or replaced with a stand-in).
set -euo pipefail

MANIFEST="${1:?usage: run_mmseqs_split.sh <processed_manifest.jsonl> <workdir> [--mmseqs <binary>]}"
WORKDIR="${2:?usage: run_mmseqs_split.sh <processed_manifest.jsonl> <workdir> [--mmseqs <binary>]}"
MMSEQS="${MMSEQS_BIN:-mmseqs}"
if [ "${3:-}" = "--mmseqs" ]; then
  MMSEQS="${4:?--mmseqs requires a path}"
fi

echo "[mmseqs-split] stage 1/3: extracting unique chain sequences"
python3 scripts/split_clusters.py extract --manifest "${MANIFEST}" --workdir "${WORKDIR}"

echo "[mmseqs-split] stage 2/3: mmseqs easy-cluster (min-seq-id 0.30, cov 0.80, cov-mode 0, cluster-mode 2)"
if ! command -v "${MMSEQS}" >/dev/null 2>&1; then
  echo "[mmseqs-split] FATAL: mmseqs binary not found or not executable: ${MMSEQS}" >&2
  echo "[mmseqs-split]        install mmseqs2 or pass --mmseqs /path/to/mmseqs (fail-closed)" >&2
  exit 1
fi
mkdir -p "${WORKDIR}/tmp"
"${MMSEQS}" easy-cluster "${WORKDIR}/all_proteins.fasta" \
  "${WORKDIR}/mmseqs_clusters" "${WORKDIR}/tmp" \
  --min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode 2

echo "[mmseqs-split] stage 3/3: union-find components, deterministic allocation, leakage check"
python3 scripts/split_clusters.py allocate \
  --clusters "${WORKDIR}/mmseqs_clusters_cluster.tsv" --workdir "${WORKDIR}"

echo "[mmseqs-split] done: ${WORKDIR}/unified_splits_30seq_id.parquet"
