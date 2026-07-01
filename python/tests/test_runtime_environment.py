"""ADC-609 runtime environment report and explicit unsupported-policy validators."""

import sys

import pytest

pops = pytest.importorskip("pops")

from pops.runtime_environment import (  # noqa: E402
    NATIVE_AMR_REFINEMENT_RATIO,
    NATIVE_DIMENSION,
    NATIVE_PRECISION,
    runtime_environment_report,
    validate_runtime_environment,
)


def test_runtime_environment_report_shape():
    report = pops.runtime_environment_report()
    assert report["dimension"] == NATIVE_DIMENSION == 2
    assert report["amr_refinement_ratio"] == NATIVE_AMR_REFINEMENT_RATIO == 2
    assert report["precision"] == NATIVE_PRECISION == "double"
    assert report["real_bytes"] == 8
    assert report["supports_single_precision"] is False
    assert report["supports_mixed_precision"] is False
    assert report["supports_custom_communicator"] is False
    assert report["communicator"] in ("serial", "MPI_COMM_WORLD", "unknown")
    assert "allocator_lifetime" in report
    assert "kokkos_lifecycle" in report


def test_runtime_environment_validators_accept_native_facts():
    accepted = validate_runtime_environment(
        dimension=2, amr_refinement_ratio=2, precision="double", communicator="serial")
    assert accepted == {
        "dimension": 2,
        "amr_refinement_ratio": 2,
        "precision": "double",
        "communicator": "serial",
    }


def test_runtime_environment_validators_reject_unsupported_requests():
    with pytest.raises(ValueError, match="dimension=3"):
        validate_runtime_environment(dimension=3)
    with pytest.raises(ValueError, match="ratio 3"):
        validate_runtime_environment(amr_refinement_ratio=3)
    with pytest.raises(ValueError, match="precision"):
        validate_runtime_environment(precision="single")
    with pytest.raises(ValueError, match="communicator"):
        validate_runtime_environment(communicator="MPI_COMM_SELF")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
