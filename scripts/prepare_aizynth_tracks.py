#!/usr/bin/env python
"""#44 EVAL-03 — prepare the two AiZynthFinder evaluation track configs.

The two tracks of #44 are **scientifically distinct questions** and are
configured separately here:

* **Track A — generator-stock-aligned**: the planner runs with the
  exported Enamine catalog stock (``scripts/export_aizynth_stock.py``
  output).  Route success under this stock is *stock-aligned route
  verification*, never independent evidence.

* **Track B — independent external-stock**: the planner runs with a
  separately sourced external vendor stock — the official AiZynthFinder
  public ZINC stock snapshot (figshare deposit 23086469, downloaded via
  the documented ``download_public_data`` entry point of the
  ``aizynthfinder`` package).  Solve rates under this stock are the
  independent external-stock track.

Both tracks share the public USPTO-derived expansion policy (ONNX models
+ template libraries from the same documented download) — the planner's
own knowledge base, external to the generator's certified 32-reaction
taxonomy.  This is the anti-circularity requirement of #34: retrosolved
means "an external planner found a route", not "our forward reactions
replayed".

Usage::

    python scripts/prepare_aizynth_tracks.py \
        --models-dir work/eval/aizynth_public \
        --stock-a work/eval/stocks/enamine_stock.txt \
        --stock-b work/eval/aizynth_public/zinc_inchikeys.txt \
        --output-dir work/eval

Writes ``track_a_stock_aligned.yml`` and
``track_b_external_zinc.yml`` plus ``tracks_manifest.json`` (URLs,
file SHA-256s, versions — the reproducibility record of #44).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

#: The documented public-model deposit (aizynthfinder's own
#: ``download_public_data`` entry point fetches exactly these files).
PUBLIC_FILES = {
    "uspto_model.onnx": (
        "https://zenodo.org/record/7797465/files/uspto_model.onnx"
    ),
    "uspto_templates.csv.gz": (
        "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz"
    ),
    "uspto_ringbreaker_model.onnx": (
        "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx"
    ),
    "uspto_ringbreaker_templates.csv.gz": (
        "https://zenodo.org/record/7341155/files/"
        "uspto_ringbreaker_unique_templates.csv.gz"
    ),
    "uspto_filter_model.onnx": (
        "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx"
    ),
}


class TrackConfigError(ValueError):
    """Raised when a track's inputs are missing (fail-closed)."""


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_yaml(models: dict[str, str], stock_name: str, stock_path: str) -> str:
    return (
        "expansion:\n"
        "  uspto:\n"
        f"    - {models['uspto_model.onnx']}\n"
        f"    - {models['uspto_templates.csv.gz']}\n"
        "  ringbreaker:\n"
        f"    - {models['uspto_ringbreaker_model.onnx']}\n"
        f"    - {models['uspto_ringbreaker_templates.csv.gz']}\n"
        "filter:\n"
        f"  uspto: {models['uspto_filter_model.onnx']}\n"
        "stock:\n"
        f"  {stock_name}: {stock_path}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#44: write the two AiZynthFinder track configs"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="directory with the public USPTO ONNX models + templates",
    )
    parser.add_argument(
        "--stock-a",
        required=True,
        help="Track A stock file (the exported Enamine stock text file)",
    )
    parser.add_argument(
        "--stock-b",
        required=True,
        help="Track B stock file (the external ZINC stock text file)",
    )
    parser.add_argument(
        "--stock-b-source",
        default="zinc_public_figshare_23086469",
        help="provenance label for the Track B external stock",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for the track config YMLs + manifest",
    )
    args = parser.parse_args()

    models = {
        name: os.path.join(args.models_dir, name)
        for name in PUBLIC_FILES
    }
    missing = [p for p in models.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "error: public model files missing (run the documented "
            "`download_public_data` from the aizynthfinder package): "
            f"{missing}"
        )
    for label, stock in (("A", args.stock_a), ("B", args.stock_b)):
        if not os.path.exists(stock):
            raise SystemExit(f"error: Track {label} stock not found: {stock}")

    os.makedirs(args.output_dir, exist_ok=True)
    track_a_path = os.path.join(args.output_dir, "track_a_stock_aligned.yml")
    track_b_path = os.path.join(args.output_dir, "track_b_external_zinc.yml")
    with open(track_a_path, "w") as handle:
        handle.write(
            _config_yaml(
                {k: os.path.abspath(v) for k, v in models.items()},
                "enamine_stock",
                os.path.abspath(args.stock_a),
            )
        )
    with open(track_b_path, "w") as handle:
        handle.write(
            _config_yaml(
                {k: os.path.abspath(v) for k, v in models.items()},
                "zinc_stock",
                os.path.abspath(args.stock_b),
            )
        )

    manifest = {
        "track_a": {
            "config": os.path.abspath(track_a_path),
            "label": (
                "generator-stock-aligned route verification — NEVER "
                "report as independent of the generator's stock"
            ),
            "stock": os.path.abspath(args.stock_a),
            "stock_sha256": _sha256_file(args.stock_a),
            "stock_format": "aizynthfinder InMemoryInchiKeyQuery text",
        },
        "track_b": {
            "config": os.path.abspath(track_b_path),
            "label": "independent external-stock retrosynthesis",
            "stock": os.path.abspath(args.stock_b),
            "stock_sha256": _sha256_file(args.stock_b),
            "stock_source": args.stock_b_source,
        },
        "shared_policy": {
            "models": {
                name: os.path.abspath(path)
                for name, path in models.items()
            },
            "model_sha256": {
                name: _sha256_file(path) for name, path in models.items()
            },
            "source_urls": dict(PUBLIC_FILES),
            "note": (
                "the public USPTO-derived policy of the aizynthfinder "
                "package — external to the generator's reaction taxonomy"
            ),
        },
        "never_merge": (
            "the two tracks' solve rates are separate metrics (#44); no "
            "headline merge exists anywhere in this pipeline"
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = os.path.join(args.output_dir, "tracks_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[tracks] Track A config: {track_a_path}")
    print(f"[tracks] Track B config: {track_b_path}")
    print(f"[tracks] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
