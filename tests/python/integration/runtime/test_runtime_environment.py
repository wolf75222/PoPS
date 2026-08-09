"""ADC-609 runtime environment report and explicit unsupported-policy validators."""


from types import SimpleNamespace

import pytest

pops = pytest.importorskip("pops")

from pops.runtime_environment import (  # noqa: E402
    NATIVE_AMR_REFINEMENT_RATIO,
    NATIVE_PRECISION,
    NATIVE_SUPPORTED_DIMENSIONS,
    RuntimeCapabilityError,
    native_dimension,
    runtime_environment_report,
    validate_runtime_environment,
)


def test_runtime_environment_report_shape():
    report = runtime_environment_report()
    assert NATIVE_SUPPORTED_DIMENSIONS == (1, 2, 3)
    assert report["dimension"] == native_dimension()
    assert report["amr_refinement_ratio"] == NATIVE_AMR_REFINEMENT_RATIO == 2
    assert report["precision"] == NATIVE_PRECISION == "double"
    assert report["real_bytes"] == 8
    assert report["supports_single_precision"] is False
    assert report["supports_mixed_precision"] is False
    assert report["supports_custom_communicator"] is False
    assert report["communicator"] in ("serial", "MPI_COMM_WORLD", "unknown")
    assert "allocator_lifetime" in report
    assert "kokkos_lifecycle" in report
    assert isinstance(report["kokkos_concurrency"], int)
    if report["kokkos_initialized"]:
        assert report["kokkos_concurrency"] > 0
    else:
        assert report["kokkos_concurrency"] == 0


def test_native_execution_resource_reaches_the_component_bridge_exactly():
    from pops._native_selector import selected_native_module

    _pops = selected_native_module(required=False)
    if _pops is None:
        pytest.skip("requires a dimension-selected native module")
    from pops._platform_contracts import ExecutionContext, ExecutionResource
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.runtime._component_execution_context import component_execution_data
    from pops.runtime._platform_manifest import (
        native_device_resource,
        native_runtime_backend_for_route,
    )

    communicator_name = native_mpi_communicator(_pops)
    backend = native_runtime_backend_for_route(
        "production", "system", communicator_name)
    if communicator_name == "MPI_COMM_WORLD":
        communicator = _pops.mpi_world()
        communicator_resource = ExecutionResource(
            "communicator", communicator_name, handle=communicator)
        datatype_resource = ExecutionResource(
            "datatype", "float64", handle=communicator.datatype_float64)
    else:
        communicator_resource = ExecutionResource("communicator", "serial")
        datatype_resource = ExecutionResource("datatype", "float64")
    context = ExecutionContext(
        backend=backend,
        communicator=communicator_resource,
        datatype=datatype_resource,
        device=native_device_resource(backend),
    )
    projected = component_execution_data(context)
    resource = context.device.handle
    assert projected["device_identity"] == resource.device_identity
    assert projected["memory_space"] == {
        "host": 1, "device": 2, "managed": 3,
    }[resource.memory_space_identity]
    assert projected["stream_handle"] == resource.stream_handle
    assert projected["stream_identity"] == resource.stream_identity


def test_static_runtime_report_does_not_fabricate_kokkos_concurrency(monkeypatch):
    import pops.runtime_environment as environment
    import pops._native_selector as selector

    monkeypatch.setattr(selector, "selected_native_module", lambda *, required=False: None)
    report = environment.runtime_environment_report()
    assert report["kokkos_concurrency"] == 0


def test_present_native_runtime_report_failure_is_not_silently_downgraded(monkeypatch):
    import pops.runtime_environment as environment
    import pops._native_selector as selector

    def broken_report():
        raise RuntimeError("native runtime report ABI failure")

    native = SimpleNamespace(runtime_environment_report=broken_report)
    monkeypatch.setattr(
        selector, "selected_native_module", lambda *, required=False: native)
    with pytest.raises(RuntimeError, match="ABI failure"):
        environment.runtime_environment_report()


def test_present_native_fallback_report_failure_is_not_silently_downgraded(monkeypatch):
    import pops.runtime.fallbacks as fallbacks
    import pops._native_selector as selector

    def broken_report():
        raise RuntimeError("native fallback report ABI failure")

    native = SimpleNamespace(fallback_diagnostics_report=broken_report)
    monkeypatch.setattr(
        selector, "selected_native_module", lambda *, required=False: native)
    with pytest.raises(RuntimeError, match="ABI failure"):
        fallbacks.fallback_diagnostics_report()


def test_runtime_environment_validators_accept_native_facts(monkeypatch):
    import pops.runtime_environment as environment

    monkeypatch.setattr(environment, "native_dimension", lambda: 2)
    accepted = validate_runtime_environment(
        dimension=2, amr_refinement_ratio=2, precision="double", communicator="serial")
    assert accepted == {
        "dimension": 2,
        "amr_refinement_ratio": 2,
        "precision": "double",
        "communicator": "serial",
    }


def test_runtime_environment_validators_reject_unsupported_requests(monkeypatch):
    import pops.runtime_environment as environment

    monkeypatch.setattr(environment, "native_dimension", lambda: 2)
    with pytest.raises(RuntimeCapabilityError, match="dimension=3") as excinfo:
        validate_runtime_environment(dimension=3)
    assert excinfo.value.field == "dimension"
    assert excinfo.value.to_dict()["runtime_environment"]["dimension"] == 2
    with pytest.raises(ValueError, match="ratio 3"):
        validate_runtime_environment(amr_refinement_ratio=3)
    with pytest.raises(ValueError, match="precision"):
        validate_runtime_environment(precision="single")
    with pytest.raises(ValueError, match="communicator"):
        validate_runtime_environment(communicator="MPI_COMM_SELF")
    with pytest.raises(RuntimeCapabilityError, match="exactly 1, 2, or 3"):
        validate_runtime_environment(dimension=True)
