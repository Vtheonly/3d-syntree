"""CLI-level smoke tests: real subprocess invocations of main.py (the
interface the Colab notebook calls). The legacy build_full_dataset.py CLI
tests were removed with the script in #5 ARCH-01; the V2 multi-source
builder arrives with #16 DATA-01)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(args, cwd=REPO, env=None, timeout=300):
    merged_env = dict(os.environ)
    merged_env.pop("HF_TOKEN", None)
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable] + args,
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestMainCLI:
    def test_info_mode(self):
        result = _run(["main.py", "--mode", "info"])
        assert result.returncode == 0, result.stderr[-2000:]
        assert "reactions registered" in result.stdout
        assert "[main] device" in result.stdout

    def test_train_missing_config_fails(self, tmp_path):
        result = _run([
            "main.py", "--mode", "train", "--config", str(tmp_path / "nope.json"),
        ])
        assert result.returncode == 2
        assert "config not found" in result.stderr

    def test_missing_catalog_fails_closed(self, tmp_path):
        """train mode without a staged catalog must fail, not fall back."""
        cfg = {
            "system": {"seed": 42, "device": "cpu", "num_workers": 0},
            "data": {
                "backend": "huggingface",
                "huggingface": {"repo_id": "test/repo"},
                "data_dir": str(tmp_path),
                "synthon_catalog_path": str(tmp_path / "missing.parquet"),
                "require_real_data": True,
            },
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        result = _run(["main.py", "--mode", "train", "--config", str(cfg_path)])
        assert result.returncode == 2
        assert "catalog missing" in result.stderr
