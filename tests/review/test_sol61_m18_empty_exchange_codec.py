"""Exact native empty-wire protocol; no synthetic positive scientific states."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "33dee906850d56eec542dd0a7d0b21954aa244d3"
spec = importlib.util.spec_from_file_location("m18_empty_codec_review", Path(__file__).with_name(
    "sol61_m18_entropy_offline_oracle.py"))
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)
EMPTY01 = b"POPSEX01" + bytes(8)
EMPTY02 = b"POPSEX02" + bytes(24)


def images(*rows):
    offsets = np.cumsum([0, *map(len, rows)], dtype=np.int64)
    raw = np.frombuffer(b"".join(rows), dtype=np.uint8)
    return oracle.empty_exchange_images(raw, offsets, len(rows))


def test_empty_wire_lengths_follow_actual_frozen_writer_and_trailing_guard():
    source = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                      SOURCE + ":include/pops/runtime/program/accepted_exchange.hpp"], text=True)
    writer = source.split("checkpoint(bool force_extended=false) const {", 1)[1].split(
        "static AcceptedExchangeLedger from_checkpoint", 1)[0]
    assert "const bool extended = force_extended || !integrals_.empty();" in writer
    assert "static_cast<std::uint8_t>(extended ? '2' : '1')" in writer
    assert "word(records_.size());" in writer
    assert "if (extended) {\n      word(integrals_.size());" in writer
    assert "word(consumed_.size());" in writer
    assert "if (cursor != bytes.size())" in source
    runtime = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                       SOURCE + ":python/pops/runtime/_checkpoint_exchanges.py"], text=True)
    assert "payload[_OFFSETS] = np.asarray(offsets, dtype=np.int64)" in runtime
    assert "offsets.dtype != np.dtype(np.int64)" in runtime
    assert len(EMPTY01) == 8 + 8 and len(EMPTY02) == 8 + 3 * 8


@pytest.mark.parametrize("rows", ((EMPTY01,), (EMPTY02,), (EMPTY01, EMPTY01),
                                  (EMPTY01, EMPTY02), (EMPTY02, EMPTY01, EMPTY02)))
def test_protocol_only_exact_empty_01_and_02_at_each_rank_boundary(rows):
    assert images(*rows) == list(rows)


@pytest.mark.parametrize("version,offset", ((1, 8), (2, 8), (2, 16), (2, 24)))
@pytest.mark.parametrize("count", (1, 1 << 63, (1 << 64) - 1))
def test_any_nonzero_record_quantity_or_consumption_count_is_refused(version, offset, count):
    raw = bytearray(EMPTY01 if version == 1 else EMPTY02)
    raw[offset:offset + 8] = count.to_bytes(8, "little")
    with pytest.raises(ValueError, match="fabricated physical exchanges"):
        images(bytes(raw))


@pytest.mark.parametrize("raw", (b"", EMPTY01[:-1], EMPTY02[:-1], EMPTY01 + bytes(16),
                                 EMPTY02[:16], EMPTY01 + b"\0", EMPTY02 + b"\0",
                                 b"POPSEX03" + bytes(24), b"POPSEX04" + bytes(24),
                                 b"FOREIGN1" + bytes(8)))
def test_truncation_trailing_old_wrong_01_extent_and_foreign_versions_refuse(raw):
    with pytest.raises(ValueError):
        images(raw)


@pytest.mark.parametrize("offsets,ranks", (([1, 16], 1), ([0, 15], 1), ([0, 17], 1),
                                         ([0, 0, 16], 2), ([0, -1, 16], 2),
                                         ([0, 8, 16], 2), ([0, 16], 2), ([0, 16], True)))
def test_offsets_cannot_omit_alias_truncate_or_reassign_a_rank_image(offsets, ranks):
    with pytest.raises(ValueError):
        oracle.empty_exchange_images(np.frombuffer(EMPTY01, dtype=np.uint8),
                                     np.asarray(offsets, dtype=np.int64), ranks)


@pytest.mark.parametrize("dtype", (np.uint64, np.int32, np.float64, np.uint8))
def test_offsets_use_exact_native_int64_type(dtype):
    with pytest.raises(ValueError, match="rank exchange offsets mismatch"):
        oracle.empty_exchange_images(np.frombuffer(EMPTY01, dtype=np.uint8),
                                     np.asarray([0, 16], dtype=dtype), 1)


@pytest.mark.parametrize("rank", (0, 1, 2))
def test_one_bad_peer_cannot_be_hidden_by_other_empty_ranks(rank):
    rows = [EMPTY01] * 3
    bad = bytearray(EMPTY01)
    bad[8] = 1
    rows[rank] = bytes(bad)
    with pytest.raises(ValueError, match="fabricated physical exchanges"):
        images(*rows)


def test_reader_version_is_explicit_and_original_math_functions_are_byte_exact():
    assert oracle.contract()["exchange_reader_contract"] == "sol61.m18-empty-exchange-reader@2"
    assert oracle.contract()["empty_exchange_image_lengths"] == {"POPSEX01": 16, "POPSEX02": 32}
    old = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                   SOURCE + ":tests/review/sol61_m18_entropy_offline_oracle.py"], text=True)
    current = Path(oracle.__file__).read_text()
    # Scientific equations/classifiers live before snapshot; only the new wire
    # helper is inserted at that boundary. No residual/entropy/cone change.
    assert current.split("EMPTY_EXCHANGE_READER =", 1)[0] == old.split("def snapshot(", 1)[0]


def observe_owner(path, owner_sha):
    """Read authentic ROOT-pinned byte images only; this is not full M18 reception."""
    raw = Path(path).read_bytes()
    oracle.require(hashlib.sha256(raw).hexdigest() == owner_sha, "external ROOT native owner seal differs")
    owner = oracle.strict_json(raw)
    oracle.require(owner["schema"] == "pops.root.m18-native-owner-auth@1"
                   and owner["source_commit"] == SOURCE and owner["scientific_reception"] is False,
                   "native owner scope differs")
    rows = owner["closed_phase_files"]
    oracle.require(len(rows) == 31 and len({row["path"] for row in rows}) == 31,
                   "native owner closed phase inventory differs")
    observations = {}
    for row in rows:
        file = Path(row["path"])
        oracle.require(file.is_absolute() and file == file.resolve() and not file.is_symlink(),
                       "native owner path is aliased")
        data = file.read_bytes()
        oracle.require(hashlib.sha256(data).hexdigest() == row["sha256"], "native owner file changed")
        if file.suffix != ".npz" or file.name.endswith("-state.npz"):
            continue
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            offsets = archive["program_exchange_offsets"]
            accepted = oracle.empty_exchange_images(archive["program_exchange_state"], offsets, owner["ranks"])
            observations[file.stem] = dict(checkpoint_sha256=row["sha256"], offsets=offsets.tolist(),
                                           image_hex=[image.hex() for image in accepted])
    oracle.require(set(observations) == set(oracle.PHASES), "native owner phase images differ")
    return dict(schema="sol61.m18-empty-exchange-wire-observation@2", status="received",
                scientific_reception=False, native_execution_here=False, owner_sha256=owner_sha,
                source_commit=SOURCE, ranks=owner["ranks"], phases=observations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = observe_owner(args.owner, args.owner_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "phases"}))
