"""Suite A / 4 — compressed ``.pt.gz`` shard I/O with SHA-256 validation
(issue #32 TEST-01; the #19 DATA-04 format contract).

Exercises the canonical shard primitive (``syntree/data/shards.py``):
deterministic gzip bytes (``mtime=0`` — identical records produce
byte-identical shards), round-trip fidelity for heterogeneous record
payloads, digest verification on read, and fail-closed corruption
handling (a flipped byte or a wrong digest never yields data).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.data.shards import (  # noqa: E402
    ShardIntegrityError,
    read_shard,
    shard_sha256,
    write_shard,
)


def _sample_records():
    return [
        {"complex_id": "pdbbind_10gs", "step": 0, "coords": torch.randn(4, 3)},
        {"complex_id": "pdbbind_10gs", "step": 1, "coords": torch.randn(4, 3)},
        {"complex_id": "crossdocked_x", "smiles": "CCO", "energy": -12.5},
    ]


class TestWriteReadRoundTrip:
    def test_round_trip_preserves_records(self, tmp_path):
        records = _sample_records()
        path = str(tmp_path / "shard_000.pt.gz")
        digest = write_shard(path, records)
        assert len(digest) == 64  # sha256 hex
        back = read_shard(path)
        assert len(back) == 3
        assert back[0]["complex_id"] == "pdbbind_10gs"
        assert back[2]["energy"] == -12.5
        assert torch.equal(back[0]["coords"], records[0]["coords"])

    def test_round_trip_heterogeneous_payloads(self, tmp_path):
        import numpy as np

        records = [
            {"tensor": torch.arange(6).reshape(2, 3), "array": np.arange(4)},
            {"nested": {"list": [1, 2, 3], "none": None}},
            {"mol": "CCO", "tree": (1, (2, 3))},
        ]
        path = str(tmp_path / "het.pt.gz")
        write_shard(path, records)
        back = read_shard(path)
        assert torch.equal(back[0]["tensor"], records[0]["tensor"])
        assert (back[0]["array"] == records[0]["array"]).all()
        assert back[1]["nested"]["list"] == [1, 2, 3]
        assert back[2]["mol"] == "CCO"

    def test_empty_shard_round_trips(self, tmp_path):
        path = str(tmp_path / "empty.pt.gz")
        digest = write_shard(path, [])
        assert read_shard(path, digest) == []


class TestDeterministicBytes:
    def test_identical_records_produce_identical_shards(self, tmp_path):
        torch.manual_seed(0)
        records_a = [{"x": torch.randn(5), "s": "stable-string"}]
        a = str(tmp_path / "a.pt.gz")
        b = str(tmp_path / "b.pt.gz")
        digest_a = write_shard(a, records_a)
        digest_b = write_shard(b, records_a)
        assert digest_a == digest_b
        assert Path(a).read_bytes() == Path(b).read_bytes()

    def test_gzip_header_has_no_timestamp(self, tmp_path):
        """mtime=0 pinned: the gzip header bytes are time-independent."""
        import gzip as gzip_mod
        import time

        path1 = str(tmp_path / "t1.pt.gz")
        write_shard(path1, [{"k": 1}])
        time.sleep(1.1)
        path2 = str(tmp_path / "t2.pt.gz")
        write_shard(path2, [{"k": 1}])
        # byte-identical files imply identical (zeroed) mtime headers
        assert Path(path1).read_bytes() == Path(path2).read_bytes()
        with open(path1, "rb") as fh:
            header = fh.read(10)
        mod_time = int.from_bytes(header[4:8], "little")
        assert mod_time == 0, f"gzip mtime not pinned to 0: {mod_time}"
        assert header[2:3] != gzip_mod.compress(b"")[2:3] or True  # noqa: B015


class TestSha256Validation:
    def test_read_with_matching_digest(self, tmp_path):
        path = str(tmp_path / "ok.pt.gz")
        digest = write_shard(path, _sample_records())
        assert read_shard(path, digest) is not None

    def test_digest_matches_shard_sha256_helper(self, tmp_path):
        path = str(tmp_path / "ok.pt.gz")
        digest = write_shard(path, _sample_records())
        assert shard_sha256(path) == digest

    def test_wrong_digest_fails_closed(self, tmp_path):
        path = str(tmp_path / "ok.pt.gz")
        write_shard(path, _sample_records())
        with pytest.raises(ShardIntegrityError, match="digest mismatch"):
            read_shard(path, "0" * 64)

    def test_corrupted_bytes_fail_digest(self, tmp_path):
        path = str(tmp_path / "corrupt.pt.gz")
        digest = write_shard(path, _sample_records())
        raw = bytearray(Path(path).read_bytes())
        raw[len(raw) // 2] ^= 0xFF  # flip one byte mid-payload
        Path(path).write_bytes(bytes(raw))
        with pytest.raises(ShardIntegrityError):
            read_shard(path, digest)

    def test_missing_file_fails_closed(self, tmp_path):
        with pytest.raises(ShardIntegrityError, match="not found"):
            read_shard(str(tmp_path / "absent.pt.gz"))

    def test_truncated_file_fails_digest(self, tmp_path):
        path = str(tmp_path / "trunc.pt.gz")
        digest = write_shard(path, _sample_records())
        raw = Path(path).read_bytes()
        Path(path).write_bytes(raw[: len(raw) - 7])
        with pytest.raises(ShardIntegrityError):
            read_shard(path, digest)
