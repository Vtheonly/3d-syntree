#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: $0 <processed_manifest.jsonl> [output_dir] [threads]}"
OUTPUT_DIR="${2:-./data/multidataset}"
THREADS="${3:-8}"

mkdir -p "$OUTPUT_DIR/mmseqs_tmp"
python scripts/dump_all_fastas.py   --manifest "$MANIFEST"   --output "$OUTPUT_DIR/all_proteins.fasta"

command -v mmseqs >/dev/null 2>&1 || {
  echo "mmseqs executable not found; install MMseqs2 before running this stage." >&2
  exit 1
}

mmseqs easy-cluster   "$OUTPUT_DIR/all_proteins.fasta"   "$OUTPUT_DIR/mmseqs_clusters"   "$OUTPUT_DIR/mmseqs_tmp"   --min-seq-id 0.30   -c 0.80   --cov-mode 0   --threads "$THREADS"

python scripts/split_clusters.py   --fasta "$OUTPUT_DIR/all_proteins.fasta"   --cluster-tsv "$OUTPUT_DIR/mmseqs_clusters_cluster.tsv"   --output "$OUTPUT_DIR/unified_splits_30seq_id.parquet"

echo "Leakage-safe sequence-cluster split written to $OUTPUT_DIR/unified_splits_30seq_id.parquet"
