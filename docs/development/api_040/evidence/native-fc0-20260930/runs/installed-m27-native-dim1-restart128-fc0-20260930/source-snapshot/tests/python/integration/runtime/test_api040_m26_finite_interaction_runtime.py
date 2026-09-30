"""Installed finite M26 reception from three saved native batches."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.python.support.collective_checks import collective_check


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def test_finite_interaction_two_densities_rebind_permutation_and_adjoint(tmp_path, monkeypatch):
    from pops._native_selector import select_native_dimension

    native = select_native_dimension(1)
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    monkeypatch.syspath_prepend(str(examples))
    witness = importlib.import_module("api040_m26_finite_interaction")
    records = witness.run_and_archive(tmp_path)
    world = native.mpi_world()
    with collective_check(world):
        assert [(row["permuted"], row["variation"]) for row in records] == [
            (False, False), (False, True), (True, False)]
        assert records[0]["artifact_identity"] == records[1]["artifact_identity"]
        assert all(row["mpi_ranks"] == world.size for row in records)
        assert all(row["adjoint_defect"] <= witness.CRITERIA["adjoint_defect"]
                   for row in records)
    failure = b""
    if world.rank == 0:
        try:
            assert (tmp_path / "receipt.json").is_file()
            assert len(tuple(tmp_path.glob("state_*.npz"))) == 3
        except Exception as exception:
            failure = (type(exception).__name__ + ": " + str(exception)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise AssertionError(failure.decode())
