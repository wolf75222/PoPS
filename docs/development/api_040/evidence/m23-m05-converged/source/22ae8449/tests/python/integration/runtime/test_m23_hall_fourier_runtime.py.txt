"""Installed native M23 Hall classification: five saved Fourier trajectories."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def test_hall_fourier_signed_phase_norm_permutations_and_zero_limit(tmp_path, monkeypatch):
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(1)
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    monkeypatch.syspath_prepend(str(examples))
    witness = importlib.import_module("api040_m23_hall_fourier")
    records = witness.run_and_archive(tmp_path)
    assert len(records) == 5
    assert {(row["cells"], tuple(row["order"]), row["eta_h"]) for row in records} == {
        (32, (0, 1), .3), (64, (0, 1), .3),
        (32, (1, 0), .3), (64, (1, 0), .3),
        (32, (0, 1), 0.),
    }
    assert all(row["continuous_phase"] == -.24 for row in records if row["eta_h"] == .3)
    assert all(row["continuous_phase"] == 0 for row in records if row["eta_h"] == 0)
    world = native.mpi_world()
    failure = b""
    if world.rank == 0:
        try:
            assert (tmp_path / "receipt.json").is_file()
            assert len(list(tmp_path.glob("state_*.npz"))) == 5
        except Exception as exception:
            failure = (type(exception).__name__+": "+str(exception)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise AssertionError(failure.decode())
