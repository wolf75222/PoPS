"""Python archive boundary tests; native codec/capabilities have separate C++ tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._checkpoint_finite_continuation import (
    FINITE_CONTINUATION_KEYS, capture_finite_continuation, prepare_finite_continuation,
)


class Native:
    def __init__(self, schema="schema/exact", capacity=128):
        self.schema, self.capacity, self.validated = schema, capacity, []

    def _checkpoint_continuation_schema(self):
        return self.schema

    def _checkpoint_continuation_capacity(self):
        return self.capacity

    def _checkpoint_continuation_image(self):
        return b"immutable-native-bytes"

    def _validate_checkpoint_continuation_image(self, image, time, step):
        self.validated.append((image, time, step))


@pytest.fixture
def fixture(monkeypatch):
    from pops.output import _checkpoint_collective
    topology = SimpleNamespace(distributed=False, size=1, rank=0, communicator=None)
    monkeypatch.setattr(_checkpoint_collective, "checkpoint_topology", lambda owner: topology)
    owner = SimpleNamespace(_s=Native())
    return owner, topology


def payload():
    return {"finite_continuation_schema": np.asarray("schema/exact"),
            "finite_continuation_bytes": np.frombuffer(b"firstsecond", dtype=np.uint8).copy(),
            "finite_continuation_offsets": np.asarray([0, 11], dtype=np.int64),
            "t": np.asarray(.25), "macro_step": np.asarray(1, dtype=np.int64)}


def test_capture_and_prepare_forward_exact_native_bytes(fixture):
    owner, _ = fixture
    out = {"t": np.asarray(.25), "macro_step": np.asarray(1, dtype=np.int64)}
    capture_finite_continuation(owner, out)
    assert FINITE_CONTINUATION_KEYS <= out.keys()
    image = prepare_finite_continuation(owner, out)
    assert image == b"immutable-native-bytes"
    assert owner._s.validated == [(image, .25, 1)]


def test_rank_selects_own_shard_and_never_substitutes_rank_zero(fixture):
    owner, topology = fixture
    topology.size, topology.rank = 2, 1
    out = payload(); out["finite_continuation_offsets"] = np.asarray([0, 5, 11], dtype=np.int64)
    assert prepare_finite_continuation(owner, out) == b"second"
    topology.size = 3
    with pytest.raises(ValueError, match="rank mapping"):
        prepare_finite_continuation(owner, out)


@pytest.mark.parametrize("mutation", ["missing", "wrong_schema", "too_large", "negative_offset", "float_offsets", "duplicate_offset"])
def test_malformed_archive_refused_before_native_decode(fixture, mutation):
    owner, _ = fixture
    out = payload()
    if mutation == "missing": out.pop("finite_continuation_offsets")
    elif mutation == "wrong_schema": out["finite_continuation_schema"] = np.asarray("other")
    elif mutation == "too_large": owner._s.capacity = 4
    elif mutation == "negative_offset": out["finite_continuation_offsets"] = np.asarray([-1, 11], dtype=np.int64)
    elif mutation == "float_offsets": out["finite_continuation_offsets"] = np.asarray([0., 11.])
    elif mutation == "duplicate_offset": out["finite_continuation_offsets"] = np.asarray([0, 0], dtype=np.int64)
    with pytest.raises(ValueError): prepare_finite_continuation(owner, out)
    assert not owner._s.validated


def test_unconfigured_legacy_runtime_has_no_fabricated_image(fixture):
    owner, _ = fixture
    owner._s = Native("", 0)
    out = {}; capture_finite_continuation(owner, out)
    assert out == {} and prepare_finite_continuation(owner, {}) is None
    with pytest.raises(ValueError, match="no matching installed"):
        prepare_finite_continuation(owner, payload())
