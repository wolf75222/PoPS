"""Contracts for convergence values derived from sealed sine-wave fields."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
PLOTTER = REPOSITORY / "benchmarks/verification/advection/sine_wave/plot_results.py"


def _case(identifier: str, dimension: int, resolution: int) -> dict[str, object]:
    return {
        "id": identifier,
        "dimension": dimension,
        "resolution": str(resolution),
        "mode": "x",
        "layout": "uniform",
        "subcycling": "synchronous",
        "block_size": 16 if dimension < 3 else 8,
        "mpi": False,
        "mpi_ranks": 1,
        "cycles": 1,
        "time_snapshots": 17,
    }


def _sealed_fixture() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, np.ndarray]],
]:
    """Build the three exact controlled series without executing PoPS."""
    cases: list[dict[str, object]] = []
    metadata: dict[str, dict[str, object]] = {}
    data: dict[str, dict[str, np.ndarray]] = {}
    series: dict[str, dict[str, object]] = {}
    for dimension, resolutions in ((1, (32, 64, 128)), (2, (16, 32, 64)), (3, (16, 32, 64))):
        identifiers = ["conv-d%d-n%d" % (dimension, resolution) for resolution in resolutions]
        series["dim%d" % dimension] = {
            "case_ids": identifiers,
            "qualified_norm": "l1",
            "reported_norms": ["l1", "l2", "linf"],
            "minimum_order": 1.75,
        }
        for identifier, resolution in zip(identifiers, resolutions, strict=True):
            cases.append(_case(identifier, dimension, resolution))
            error = float(resolution**-2)
            shape = (resolution,) * dimension
            numerical = np.full(shape, error, dtype=np.float64)
            exact = np.zeros(shape, dtype=np.float64)
            mask = np.ones(shape, dtype=bool)
            data[identifier] = {"numeric": numerical, "exact": exact, "mask": mask}
            metadata[identifier] = {
                "dimension": dimension,
                "resolution": [resolution] * dimension,
                "layout": "uniform",
                "source_fingerprint": "f" * 64,
                "metrics": {
                    "method": {"time": "SSPRK2", "reconstruction": "MUSCL(VanLeer)"},
                    "errors": {"l1": error, "l2": error, "linf": error},
                },
                "provenance": {"execution": {"runtime": {"kokkos_backend": "OpenMP"}}},
            }
    matrix = {"convergence_series": series}
    by_id = {case["id"]: case for case in cases}
    return matrix, by_id, metadata, data


def test_complete_convergence_is_recomputed_from_npz_fields_before_receipt_use() -> None:
    scope = runpy.run_path(str(PLOTTER))
    matrix, by_id, metadata, data = _sealed_fixture()
    recomputed = scope["_recomputed_convergence_receipt"](matrix, by_id, metadata, data)

    assert set(recomputed) == {"dim1", "dim2", "dim3"}
    for receipt in recomputed.values():
        assert receipt["orders"]["l1"] == pytest.approx([2.0, 2.0])
        assert receipt["minimum_order"] == 1.75

    scope["_validate_convergence_receipt"](copy.deepcopy(recomputed), recomputed)
    forged = copy.deepcopy(recomputed)
    forged["dim2"]["orders"]["l1"][0] = 1.999
    with pytest.raises(ValueError, match="disagrees with recomputation"):
        scope["_validate_convergence_receipt"](forged, recomputed)


def test_recomputed_convergence_rejects_npz_metadata_and_threshold_falsification() -> None:
    scope = runpy.run_path(str(PLOTTER))
    matrix, by_id, metadata, data = _sealed_fixture()

    data["conv-d1-n128"]["numeric"] *= 1.5
    with pytest.raises(ValueError, match="metadata"):
        scope["_recomputed_convergence_receipt"](matrix, by_id, metadata, data)

    matrix, by_id, metadata, data = _sealed_fixture()
    identifier = "conv-d1-n128"
    degraded_error = 1.0 / 64.0**2
    data[identifier]["numeric"].fill(degraded_error)
    metadata[identifier]["metrics"]["errors"] = {
        "l1": degraded_error,
        "l2": degraded_error,
        "linf": degraded_error,
    }
    with pytest.raises(ValueError, match="below its declared final-order threshold"):
        scope["_recomputed_convergence_receipt"](matrix, by_id, metadata, data)


def test_sealed_native_compatibility_requires_common_abi_and_mpi_np1_proof() -> None:
    scope = runpy.run_path(str(PLOTTER))
    non_mpi = _case("serial", 1, 32)
    mpi = _case("mpi-np1", 2, 32)
    mpi["mpi"] = True
    mpi["mpi_ranks"] = 1
    mpi["mpi_topology"] = [1, 1]
    by_id = {"serial": non_mpi, "mpi-np1": mpi}

    def metadata_for(case: dict[str, object]) -> dict[str, object]:
        enabled = bool(case["mpi"])
        topology = [1, 1] if enabled else None
        return {
            "mpi": enabled,
            "coverage": {
                "mpi_topology": (
                    {
                        "requested_ranks": 1,
                        "observed_ranks": 1,
                        "expected_spatial_decomposition": topology,
                        "ownership_active": True,
                            "rank_ownership": [
                                {"rank": 0, "local_boxes": [[[0, 0], [32, 32]]]}
                            ],
                        "rank_coordinates": [{"rank": 0, "coordinate": [0, 0]}],
                    }
                    if enabled
                    else None
                )
            },
            "provenance": {
                "execution": {
                    "runtime": {
                        "mpi_compiled": enabled,
                        "mpi_active": enabled,
                        "mpi_ranks": case["mpi_ranks"],
                    }
                },
                "source": {
                    "native": {
                        "dimension": case["dimension"],
                        "version": "test-version",
                        "build_fingerprint": "a" * 64,
                        "abi_key": (
                            "compiler=clang;std=c++20;headers=abc;kokkos=1;stdlib=libcxx;"
                            "dim=%d;mpi=%d;mpi_abi=%s"
                            % (case["dimension"], int(enabled), "openmpi" if enabled else "none")
                        ),
                        "has_mpi": enabled,
                        "has_kokkos": True,
                    }
                },
            },
        }

    metadata = {case_id: metadata_for(case) for case_id, case in by_id.items()}
    receipt = scope["_recomputed_native_compatibility"](by_id, metadata)
    assert receipt["abi_common"] == [
        ["compiler", "clang"],
        ["headers", "abc"],
        ["kokkos", "1"],
        ["std", "c++20"],
        ["stdlib", "libcxx"],
    ]
    scope["_validate_native_compatibility"](receipt, by_id, metadata)
    forged = copy.deepcopy(receipt)
    forged["build_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="disagrees with authenticated pairs"):
        scope["_validate_native_compatibility"](forged, by_id, metadata)

    metadata["mpi-np1"]["provenance"]["execution"]["runtime"]["mpi_active"] = False
    with pytest.raises(ValueError, match="MPI receipt"):
        scope["_recomputed_native_compatibility"](by_id, metadata)
