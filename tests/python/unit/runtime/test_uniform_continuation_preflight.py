"""The strict Uniform archive reader admits only the mandatory continuation wire members."""
from io import BytesIO

import numpy as np
import pytest

from pops.runtime._checkpoint_exchanges import (
    CONTINUATION_CHECKPOINT_KEYS, capture_checkpoint_continuation,
    prepare_checkpoint_continuation,
)
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
from tests.python.unit.runtime.test_continuation_transitions import _owner


@pytest.fixture(scope="module")
def continuation_payload():
    owner = _owner()
    payload = {}
    capture_checkpoint_continuation(owner, payload)
    return payload


def _payload(continuation):
    payload = {
        "t": np.array(0.0, dtype=np.float64),
        "macro_step": np.array(0, dtype=np.int64),
        "pops_spatial_contract": np.array("{}"),
        "pops_embedded_boundary_contract": np.array("{}"),
        "program_hash": np.array("ab" * 32),
        "history_names": np.array([], dtype="U1"),
        "cache_nodes": np.array([], dtype=np.int64),
        "cache_names": np.array([], dtype="U1"),
        "temporal_restart_state": np.array("{}"),
        "program_cadence_substeps": np.array(1, dtype=np.int64),
        "program_cadence_stride": np.array(1, dtype=np.int64),
        "program_cadence_window_steps": np.array(0, dtype=np.int64),
        "program_cadence_window_dt": np.array(0.0, dtype=np.float64),
        "program_cadence_window_start_time": np.array(0.0, dtype=np.float64),
        "program_last_dt": np.array(0.0, dtype=np.float64),
    }
    payload.update({key: value.copy() for key, value in continuation.items()})
    return payload


def test_uniform_reader_accepts_current_producer_members_in_npz(continuation_payload):
    assert set(continuation_payload) == CONTINUATION_CHECKPOINT_KEYS
    stream = BytesIO()
    np.savez_compressed(stream, **_payload(continuation_payload))
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as payload:
        preflight_uniform_restart(payload)


@pytest.mark.parametrize("missing", sorted(CONTINUATION_CHECKPOINT_KEYS))
def test_uniform_reader_refuses_missing_continuation_member(continuation_payload, missing):
    payload = _payload(continuation_payload)
    del payload[missing]
    with pytest.raises(ValueError, match="missing " + missing):
        preflight_uniform_restart(payload)


@pytest.mark.parametrize(("key", "value", "reason"), [
    ("program_exchange_state", np.zeros(16, dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_state", np.zeros((1, 16), dtype=np.uint8), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0, 16], dtype=np.int32), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0, 16], dtype=np.uint64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([[0, 16]], dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0], dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([1, 16], dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0, 15], dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0, 0, 16], dtype=np.int64), "ranked byte offsets"),
    ("program_exchange_offsets", np.array([0, 12, 8, 16], dtype=np.int64), "ranked byte offsets"),
    ("continuation_transition_plan", np.array(["{}"]), "Unicode scalar"),
    ("continuation_transition_plan", np.array(b"{}"), "Unicode scalar"),
    ("continuation_transition_plan", np.array(""), "Unicode scalar"),
])
def test_uniform_reader_refuses_malformed_continuation_arrays(
    continuation_payload, key, value, reason
):
    payload = _payload(continuation_payload)
    payload[key] = value
    with pytest.raises(ValueError, match=reason):
        preflight_uniform_restart(payload)


def test_uniform_reader_keeps_exact_unknown_member_refusal(continuation_payload):
    payload = _payload(continuation_payload)
    payload["program_exchange_extra"] = np.array(0)
    with pytest.raises(ValueError, match="unknown archive members program_exchange_extra"):
        preflight_uniform_restart(payload)


def test_owner_validation_still_consumes_policy_and_native_bytes_after_schema_preflight():
    owner = _owner()
    continuation = {}
    capture_checkpoint_continuation(owner, continuation)
    payload = _payload(continuation)
    preflight_uniform_restart(payload)
    assert prepare_checkpoint_continuation(owner, payload) == owner._s.image
    before = owner._s.image
    payload["program_exchange_state"][-1] = 1
    preflight_uniform_restart(payload)  # Structurally valid bytes are still native-owned records.
    with pytest.raises(ValueError, match="invalid native exchange record"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.validated[-1] == payload["program_exchange_state"].tobytes()
    assert owner._s.image == before
    payload["program_exchange_state"][-1] = 0
    payload["continuation_transition_plan"] = np.array("{}")
    preflight_uniform_restart(payload)
    with pytest.raises(ValueError, match="policies differ"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.image == before
