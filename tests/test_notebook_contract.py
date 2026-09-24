"""Notebook contract tests for notebooks/run_3d_syntree.ipynb.

The notebook is the production entrypoint (Google Colab), so it must satisfy
a strict contract:
* valid nbformat v4 JSON with the documented cell sequence,
* CELL 4.5 build gate (BUILD_REAL_DATASET) invoking build_full_dataset.py,
* CELL 5 syncing assets and asserting train_samples >= 50,
* CELL 1 RUNTIME_CONFIG merging (via main.deep_update) with
  configs/train_colab_12h.json into a fail-closed production config,
* no embedded credentials anywhere in any cell.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "run_3d_syntree.ipynb")

_SECRET_PATTERNS = re.compile(
    r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _cells(nb: dict) -> list:
    return nb["cells"]


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _find_cell(nb: dict, marker: str) -> tuple:
    for i, cell in enumerate(_cells(nb)):
        if marker in _src(cell):
            return i, cell
    raise AssertionError(f"no cell contains {marker!r}")


class TestNotebookStructure:
    def test_valid_nbformat(self, notebook):
        assert notebook["nbformat"] == 4
        assert isinstance(notebook["cells"], list)
        for cell in notebook["cells"]:
            assert cell["cell_type"] in ("code", "markdown", "raw")
            assert isinstance(cell.get("source", []), list)

    def test_cell_sequence(self, notebook):
        markers = [
            "CELL 1: Environment",
            "CELL 2: Clone Repository",
            "CELL 3: Dedicated Force-Update",
            "CELL 4: Install Dependencies",
            "CELL 4.5",
            "CELL 5: Sync Clean Sharded",
            "CELL 6: Hardware Verification",
            "CELL 6.5: Training Mode",
            "CELL 7: Launch",
            "CELL 8: Status",
            "CELL 9: Stage 2 PPO",
            "CELL 10: Comparative Benchmark",
        ]
        positions = []
        for marker in markers:
            idx, _ = _find_cell(notebook, marker)
            positions.append(idx)
        assert positions == sorted(positions), (
            f"notebook cells out of order: {list(zip(markers, positions))}"
        )

    def test_cell45_precedes_cell5(self, notebook):
        i45, _ = _find_cell(notebook, "CELL 4.5")
        i5, _ = _find_cell(notebook, "CELL 5: Sync Clean Sharded")
        assert i45 < i5


class TestBuildGate:
    def test_build_gate_defaults_off(self, notebook):
        _, cell = _find_cell(notebook, "CELL 4.5")
        src = _src(cell)
        assert "BUILD_REAL_DATASET = False" in src

    def test_build_gate_invokes_builder(self, notebook):
        _, cell = _find_cell(notebook, "CELL 4.5")
        src = _src(cell)
        assert "build_full_dataset.py" in src
        assert "--upload" in src
        assert "--repo-id JJKK1212/3d-syntree-multidataset" in src


class TestSyncGate:
    def test_cell5_reads_assets_manifest(self, notebook):
        _, cell = _find_cell(notebook, "CELL 5: Sync Clean Sharded")
        src = _src(cell)
        assert "assets_manifest.json" in src
        assert 'meta["splits"]["train"]["total_samples"]' in src

    def test_cell5_asserts_minimum_samples(self, notebook):
        _, cell = _find_cell(notebook, "CELL 5: Sync Clean Sharded")
        src = _src(cell)
        assert "assert train_samples >= 50" in src

    def test_cell5_mentions_build_command(self, notebook):
        _, cell = _find_cell(notebook, "CELL 5: Sync Clean Sharded")
        assert "build_full_dataset.py" in _src(cell)

    def test_cell5_uses_download_assets(self, notebook):
        _, cell = _find_cell(notebook, "CELL 5: Sync Clean Sharded")
        src = _src(cell)
        assert "scripts/download_assets.py" in src
        assert "--dataset-repo" in src


class TestRuntimeConfigMerge:
    """Simulate the exact notebook flow: CELL 1 writes runtime_config.json;
    main.load_config merges config_override into train_colab_12h.json."""

    @pytest.fixture()
    def merged_config(self, notebook, repo_root, tmp_path):
        _, cell = _find_cell(notebook, "CELL 1: Environment")
        src = _src(cell)
        # Extract the RUNTIME_CONFIG literal by executing the cell with a
        # stubbed environment (no HF token prompt: provide a fake value).
        from getpass import getpass  # noqa: F401  (used inside exec)

        import builtins

        real_input = builtins.input
        builtins.input = lambda *_: "hf_fake_notebook_token"
        saved_env = os.environ.get("HF_TOKEN")
        os.environ.pop("HF_TOKEN", None)
        try:
            namespace = {}
            body = "\n".join(
                line for line in src.split("\n")
                if not line.strip().startswith("!") and "getpass(" not in line
            )
            body = body.replace(
                'from getpass import getpass', 'pass'
            ).replace('HF_TOKEN = getpass', 'HF_TOKEN = "hf_fake_notebook_token"')
            exec(compile(body, "<cell1>", "exec"), namespace)  # noqa: S102
        finally:
            builtins.input = real_input
            if saved_env is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = saved_env
        runtime_config = namespace["RUNTIME_CONFIG"]
        runtime_path = tmp_path / "runtime_config.json"
        runtime_path.write_text(json.dumps(runtime_config))

        from main import load_config

        return load_config(
            os.path.join(repo_root, "configs", "train_colab_12h.json"),
            str(runtime_path),
        )

    def test_merged_config_is_fail_closed(self, merged_config):
        data = merged_config["data"]
        assert data["backend"] == "huggingface"
        assert data["require_real_data"] is True
        assert data["min_real_samples"] == 50
        assert data["synthetic_fallback"] is False
        assert data["huggingface"]["repo_id"] == "JJKK1212/3d-syntree-multidataset"
        assert data["huggingface"]["max_cached_shards"] == 4

    def test_merged_config_keeps_production_model(self, merged_config):
        model = merged_config["model"]
        assert model["hidden_dim"] == 256
        assert model["num_equivariant_layers"] == 8

    def test_merged_config_checkpoints_repo(self, merged_config):
        hf = merged_config["huggingface"]
        assert hf["repo_id"] == "JJKK1212/3d-syntree-checkpoints"

    def test_runtime_writes_config_file(self, notebook, tmp_path):
        _, cell = _find_cell(notebook, "CELL 1: Environment")
        assert 'with open("runtime_config.json", "w") as f' in _src(cell)


class TestCell7Launch:
    def test_launch_uses_production_config_and_runtime(self, notebook):
        _, cell = _find_cell(notebook, "CELL 7: Launch")
        src = _src(cell)
        assert "configs/train_colab_12h.json" in src
        assert "--runtime-config ../runtime_config.json" in src.replace("\\\n", " ")
        assert "--mode train" in src

    def test_fresh_flag_available(self, notebook):
        _, cell = _find_cell(notebook, "CELL 6.5: Training Mode")
        src = _src(cell)
        assert "FRESH_TRAINING" in src
        assert '"--fresh"' in src
        assert '"--resume-auto"' in src


class TestCell6HalfRemotePurge:
    def test_purges_remote_checkpoints_on_fresh(self, notebook):
        _, cell = _find_cell(notebook, "CELL 6.5: Training Mode")
        src = _src(cell)
        assert "delete_file" in src
        assert "checkpoints/" in src


class TestNoSecrets:
    def test_no_credentials_in_any_cell(self, notebook):
        for i, cell in enumerate(notebook["cells"]):
            src = _src(cell)
            match = _SECRET_PATTERNS.search(src)
            assert match is None, (
                f"cell {i} contains an embedded credential: {match.group(0)[:12]}..."
            )

    def test_token_comes_from_environment(self, notebook):
        _, cell = _find_cell(notebook, "CELL 1: Environment")
        src = _src(cell)
        assert 'os.environ["HF_TOKEN"]' in src
        assert "userdata.get" in src or "getpass" in src

    def test_no_secrets_in_repo_text_files(self, repo_root):
        """Scan tracked source/config/docs for credential patterns."""
        suspicious = []
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [
                d for d in dirnames
                if d not in (".git", "__pycache__", ".pytest_cache", "node_modules")
            ]
            for name in filenames:
                if not name.endswith((".py", ".json", ".md", ".ipynb", ".yml", ".yaml", ".toml", ".cfg")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                match = _SECRET_PATTERNS.search(text)
                if match:
                    suspicious.append((path, match.group(0)[:12]))
        assert not suspicious, f"credentials found in tracked files: {suspicious}"

    def test_gitignore_covers_secrets(self, repo_root):
        gi = open(os.path.join(repo_root, ".gitignore")).read()
        assert ".secrets/" in gi
        assert "*.env" in gi
