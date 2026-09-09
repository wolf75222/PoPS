from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._checkpoint_history_flux_snapshots import (
    SNAPSHOT_OFFSETS_KEY,
    SNAPSHOT_STATE_KEY,
    capture_history_flux_snapshots,
    prepare_history_flux_snapshots,
)


def test_ranked_history_flux_snapshot_archive_roundtrips_and_preserves_empty_image(
    monkeypatch,
) -> None:
    from pops import _native_collectives
    from pops.output import _checkpoint_collective

    topology = SimpleNamespace(distributed=True, communicator=object(), size=2)
    rows = (b"POPSHFS1/rank/0", b"POPSHFS1/rank/1")
    monkeypatch.setattr(_native_collectives, "allgather_bytes", lambda communicator, local: rows)
    monkeypatch.setattr(
        _checkpoint_collective,
        "consensus",
        lambda topology, phase, *, error=None: (_ for _ in ()).throw(error) if error else None,
    )

    payload = {}
    capture_history_flux_snapshots(topology, rows[0], 64, payload)
    assert payload[SNAPSHOT_OFFSETS_KEY].tolist() == [0, len(rows[0]), len(rows[0]) + len(rows[1])]
    assert prepare_history_flux_snapshots(payload, checkpoint_ranks=2, shard_capacity=64) == rows

    empty = {}
    capture_history_flux_snapshots(topology, b"", 0, empty)
    assert empty[SNAPSHOT_STATE_KEY].tolist() == []
    assert empty[SNAPSHOT_OFFSETS_KEY].tolist() == [0, 0, 0]
    assert prepare_history_flux_snapshots(empty, checkpoint_ranks=2, shard_capacity=0) == (b"", b"")


@pytest.mark.parametrize(
    ("state", "offsets", "message"),
    (
        (np.array([1], dtype=np.int16), np.array([0, 1], dtype=np.int64), "uint8"),
        (np.array([1], dtype=np.uint8), np.array([0, 1], dtype=np.int32), "int64"),
        (np.array([1], dtype=np.uint8), np.array([0, 0], dtype=np.int64), "cover"),
        (
            np.array([1, 2], dtype=np.uint8),
            np.array([0, 2, 2], dtype=np.int64),
            "non-empty",
        ),
    ),
)
def test_history_flux_snapshot_preflight_refuses_malformed_rank_archive(
    state, offsets, message
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_history_flux_snapshots(
            {
                SNAPSHOT_STATE_KEY: state,
                SNAPSHOT_OFFSETS_KEY: offsets,
            },
            checkpoint_ranks=len(offsets) - 1,
            shard_capacity=64,
        )


def test_history_flux_snapshot_preflight_preserves_legacy_absence_but_refuses_partial_pair() -> (
    None
):
    assert prepare_history_flux_snapshots({}, checkpoint_ranks=2, shard_capacity=64) is None
    with pytest.raises(ValueError, match="partial"):
        prepare_history_flux_snapshots(
            {SNAPSHOT_STATE_KEY: np.array([], dtype=np.uint8)},
            checkpoint_ranks=2,
            shard_capacity=64,
        )
