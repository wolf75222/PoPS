from __future__ import annotations

import json

import pytest

from pops.runtime._checkpoint_embedded_boundary import (
    CheckpointEmbeddedBoundaryContract,
    EMBEDDED_BOUNDARY_CONTRACT_KEY,
    add_checkpoint_embedded_boundary_contract,
    authenticate_checkpoint_embedded_boundary_contract,
    inspect_checkpoint_embedded_boundary_contract,
)


def _digest(kind: str, value: str) -> str:
    return "pops.prepared-eb-%s.v1:sha256:%s" % (kind, value * 64)


def _report(*, enabled: bool, semantic: str = "a") -> dict[str, object]:
    return {
        "topology": {"dimension": 3, "periodicity": [True, False, True]},
        "eb": {
            "enabled": enabled,
            "geometry_mode": "staircase" if enabled else "none",
            "kappa_min": 0.02,
            "face_open_eps": 1.0e-6,
            "cut_theta_min": 1.0e-3,
            "semantic_digest": _digest("semantic", semantic) if enabled else "",
            "materialization_digest": _digest("geometry", "b") if enabled else "",
            "generation": 4 if enabled else 0,
        },
    }


class _Native:
    def __init__(self, report: dict[str, object]) -> None:
        self._report = report

    def effective_options_report(self) -> dict[str, object]:
        return self._report


class _Owner:
    def __init__(self, report: dict[str, object]) -> None:
        self._s = _Native(report)


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_embedded_boundary_roundtrips_exact_schema(enabled: bool) -> None:
    contract = CheckpointEmbeddedBoundaryContract.from_effective_report(_report(enabled=enabled))
    assert CheckpointEmbeddedBoundaryContract.from_data(contract.to_data()) == contract

    payload: dict[str, object] = {}
    add_checkpoint_embedded_boundary_contract(payload, contract)
    assert inspect_checkpoint_embedded_boundary_contract(payload) == contract
    assert json.loads(str(payload[EMBEDDED_BOUNDARY_CONTRACT_KEY])) == contract.to_data()


def test_checkpoint_embedded_boundary_authenticates_semantics_not_rank_materialization() -> None:
    contract = CheckpointEmbeddedBoundaryContract.from_effective_report(_report(enabled=True))
    payload: dict[str, object] = {}
    add_checkpoint_embedded_boundary_contract(payload, contract)

    rematerialized = _report(enabled=True)
    rematerialized["eb"]["materialization_digest"] = _digest("geometry", "c")  # type: ignore[index]
    rematerialized["eb"]["generation"] = 12  # type: ignore[index]
    assert (
        authenticate_checkpoint_embedded_boundary_contract(_Owner(rematerialized), payload)
        == contract
    )


def test_checkpoint_embedded_boundary_refuses_changed_geometry_before_restart() -> None:
    contract = CheckpointEmbeddedBoundaryContract.from_effective_report(_report(enabled=True))
    payload: dict[str, object] = {}
    add_checkpoint_embedded_boundary_contract(payload, contract)

    with pytest.raises(ValueError, match="geometry differs"):
        authenticate_checkpoint_embedded_boundary_contract(
            _Owner(_report(enabled=True, semantic="d")), payload
        )


def test_checkpoint_embedded_boundary_refuses_missing_or_forged_authority() -> None:
    with pytest.raises(ValueError, match="lacks"):
        inspect_checkpoint_embedded_boundary_contract({})

    contract = CheckpointEmbeddedBoundaryContract.from_effective_report(_report(enabled=True))
    data = contract.to_data()
    data["mode"] = "cutcell"
    payload = {EMBEDDED_BOUNDARY_CONTRACT_KEY: json.dumps(data)}
    with pytest.raises(ValueError, match="authenticate"):
        inspect_checkpoint_embedded_boundary_contract(payload)
