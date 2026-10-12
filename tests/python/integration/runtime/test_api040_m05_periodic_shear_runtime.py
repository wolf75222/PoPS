"""Installed Dim1 M05 shear reception from saved states and accepted face ledger."""
from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path

import pytest

from tests.python.support.collective_checks import collective_check


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def test_periodic_shear_three_grids_dissipation_and_accepted_ledger(tmp_path, monkeypatch):
    from tests.python.support.public_native_profile import native_profile, rank_and_size

    native, world = native_profile(dimension=1, scope="saved-state Dim1 reception")
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    monkeypatch.syspath_prepend(str(examples))
    witness = importlib.import_module("api040_m05_periodic_shear")
    records = witness.run_and_archive(tmp_path)
    with collective_check(world):
        assert [row["cells"] for row in records] == [32, 64, 128]
        assert all(row["accepted_steps"] == row["cells"] for row in records)
        assert all(row["accepted_exchange_count"] == 2 * row["cells"] for row in records)
        assert all(row["final_energy"] < row["initial_energy"] for row in records)
    failure = b""
    if rank_and_size(world)[0] == 0:
        try:
            assert (tmp_path / "receipt.json").is_file()
            assert len(list(tmp_path.glob("state_*.npz"))) == 3
            assert len(list(tmp_path.glob("ledger_*.json"))) == 3
            for row in records:
                ledger = tmp_path / row["saved_ledger"]
                assert hashlib.sha256(ledger.read_bytes()).hexdigest() == row["saved_ledger_sha256"]
                assert len(json.loads(ledger.read_text())) == 2 * row["cells"]
        except Exception as exception:
            failure = (type(exception).__name__+": "+str(exception)).encode()
    failure = failure if world is None else world.broadcast_bytes(failure, root=0)
    if failure:
        raise AssertionError(failure.decode())
