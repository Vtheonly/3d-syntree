#!/usr/bin/env python3
"""Extract all protein-chain FASTA records from a unified raw-source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from syntree.data.multidataset import extract_chain_sequences, iter_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(args.manifest):
            sample_id = row["sample_id"]
            for chain, sequence in sorted(extract_chain_sequences(row["protein_path"]).items()):
                if len(sequence) < 20:
                    continue
                header = f"{sample_id}|chain={chain}"
                f.write(f">{header}\n")
                for i in range(0, len(sequence), 80):
                    f.write(sequence[i:i + 80] + "\n")
                count += 1

    print(f"Wrote {count} protein-chain records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
