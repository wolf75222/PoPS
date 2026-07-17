"""ADC-683 explicit platform/backend/execution/field-view contracts."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pops._platform_contracts import (
    CapabilityProof,
    ExecutionContext,
    ExecutionResource,
    FieldViewDescriptor,
    PlatformContractError,
    launch_checked,
    proven_serial_manifest,
    validate_component_launch,
    validate_launch,
)
from pops.identity import make_identity
from pops.runtime._run_manifest import RunManifest
from pops.runtime._step_strategy import run_control_payload
from pops.time import AdaptiveCFL


def _platform(**overrides):
    base = proven_serial_manifest(
        backend="production", target="system", abi="headers|clang|c++23")
    return replace(base, **overrides)


def _context(**overrides):
    backend = proven_serial_manifest(
        backend="production", target="system", abi="headers|clang|c++23", runtime=True)
    values = {
        "backend": backend,
        "communicator": ExecutionResource("communicator", "serial"),
        "datatype": ExecutionResource("datatype", "float64"),
        "device": ExecutionResource("device", "host"),
    }
    values.update(overrides)
    return ExecutionContext(**values)


def _field(**overrides):
    values = {
        "name": "state", "dimension": 2, "extents": (16, 12), "strides": (12, 1),
        "centering": "cell", "ghosts": ((0, 0), (0, 0)), "scalar": "float64",
        "memory_space": "host", "patch": "patch-0", "layout": "right",
        "ownership": "borrowed",
    }
    values.update(overrides)
    return FieldViewDescriptor(**values)


def _proof(value):
    return CapabilityProof.proven(value, "test-proof")


def test_platform_compatibility_facts_change_artifact_identity():
    baseline = _platform()
    variants = [
        replace(baseline, backend=_proof("aot")),
        replace(baseline, target=_proof("amr_system")),
        replace(baseline, abi=_proof("other|clang|c++23")),
        replace(baseline, precision=replace(baseline.precision, compute=_proof("float32"))),
        replace(baseline, device=_proof("cuda:0")),
        replace(baseline, memory_spaces=_proof(("device",))),
    ]
    identities = {
        make_identity("artifact", {"platform": item.to_data()}).token
        for item in (baseline, *variants)
    }
    assert len(identities) == len(variants) + 1


def test_platform_manifest_strict_data_round_trip():
    manifest = _platform()
    assert type(manifest).from_data(manifest.to_data()) == manifest
    malformed = manifest.to_data()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        type(manifest).from_data(malformed)
    malformed = manifest.to_data()
    malformed["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        type(manifest).from_data(malformed)
    malformed = manifest.to_data()
    malformed["device"] = {"value": "host", "evidence": None}
    with pytest.raises(ValueError, match="without evidence"):
        type(manifest).from_data(malformed)


def test_execution_context_changes_bind_and_run_identity():
    serial = _context()
    other = _context(datatype=ExecutionResource("datatype", "float32"))
    bind_a = make_identity("bind", {"execution_context": serial.to_data()})
    bind_b = make_identity("bind", {"execution_context": other.to_data()})
    assert bind_a != bind_b
    controls = {
        "t_end": 1.0,
        "step_transaction": run_control_payload(AdaptiveCFL(0.4)),
        "max_steps": 8,
        "output_mode": "memory",
    }
    assert RunManifest(bind_identity=bind_a, start_time=0.0, start_macro_step=0,
                       controls=controls).run_identity != RunManifest(
                           bind_identity=bind_b, start_time=0.0, start_macro_step=0,
                           controls=controls).run_identity


def test_unknown_is_missing_proof_and_3d_is_representable_then_refused():
    with pytest.raises(PlatformContractError, match="absence of proof"):
        launch_checked(replace(_platform(), device=CapabilityProof.unknown()),
                       _context(), [_field()], lambda *_: None)
    three_d = _field(
        dimension=3, extents=(8, 8, 8), strides=(64, 8, 1),
        ghosts=((0, 0), (0, 0), (0, 0)))
    assert three_d.dimension == 3
    with pytest.raises(PlatformContractError, match="unsupported dimension=3"):
        launch_checked(_platform(), _context(), [three_d], lambda *_: None)


@pytest.mark.parametrize("changed", [
    {"centering": "node"},
    {"scalar": "float32"},
    {"extents": (15, 12)},
    {"memory_space": "device"},
])
def test_field_mismatch_refuses_before_kernel(changed):
    launched = []
    with pytest.raises(PlatformContractError):
        launch_checked(
            _platform(), _context(), [_field(**changed)],
            lambda *_: launched.append(True), expected_fields=[_field()])
    assert launched == []


def test_generic_2d_double_descriptor_launches_once():
    launched = []
    assert launch_checked(
        _platform(), _context(), [_field()],
        lambda context, fields: launched.append((context, fields)) or fields[0].extents,
        expected_fields=[_field()]) == (16, 12)
    assert len(launched) == 1


def test_aot_component_build_route_is_checked_against_simulation_execution_facts():
    component = proven_serial_manifest(
        backend="aot-component", target="component", abi="headers|clang|c++23")
    context = _context()

    # The generic gate still rejects different execution routes.  Only the typed component gate
    # recognizes aot-component/component as a build-role namespace.
    with pytest.raises(PlatformContractError, match="backend mismatch"):
        validate_launch(component, context, ())
    validate_component_launch(component, context, ())

    with pytest.raises(PlatformContractError, match="ABI mismatch"):
        validate_component_launch(replace(component, abi=_proof("other|clang|c++23")), context, ())
    with pytest.raises(PlatformContractError, match="device mismatch"):
        validate_component_launch(replace(component, device=_proof("cuda:0")), context, ())
    with pytest.raises(PlatformContractError, match="component artifact route"):
        validate_component_launch(_platform(), context, ())


def test_aot_component_rejects_openmpi_mpich_abi_mix_even_with_same_headers_and_standard():
    openmpi = (
        "compiler=clang;std=202002;headers=same;kokkos=1;stdlib=libc++;"
        "mpi=1;mpi_abi=" + "a" * 64
    )
    mpich = openmpi[:-64] + "b" * 64
    component = proven_serial_manifest(
        backend="aot-component", target="component", abi=openmpi)
    runtime = proven_serial_manifest(
        backend="production", target="system", abi=mpich, runtime=True)
    context = _context(backend=runtime)
    with pytest.raises(PlatformContractError, match="exact native ABI mismatch"):
        validate_component_launch(component, context, ())


def test_final_generic_contract_has_no_implicit_device_capture():
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    paths = [
        root / "include/pops/runtime/config/platform_manifest.hpp",
        root / "python/pops/_platform_contracts.py",
        root / "python/pops/runtime/_platform_validation.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("DefaultExecutionSpace", "current_device", "mpi" + "4py"):
        assert forbidden not in text
    assert "def mpi_world(cls, artifact: Any)" in text
    assert "handle=communicator.datatype_float64" in text
