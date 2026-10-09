"""Actual test DSO compilation/control: Source/Host preparation, not Native dispatch proof."""

import ctypes
from pathlib import Path
import subprocess
from pops import interfaces
from tests.python.integration.native_loader.test_external_field_solver_runtime import _manifest
from tests.python.support.external_field_fault_component import fault_source, SharedTableControl


def test_real_component_source_builds_and_typed_arm_restores_table(tmp_path, monkeypatch):
    manifest = _manifest(
        "dispatch-host-probe", interfaces.FieldSolver, ({"name": "answer", "kind": "runtime"},)
    )
    source = tmp_path / "component.cpp"
    source.write_text(fault_source(manifest))
    binary = tmp_path / "component.so"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "clang++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-DPOPS_NATIVE_DIM=2",
            "-I",
            str(root / "include"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    path = tmp_path / "actual-control.bin"
    monkeypatch.setenv("POPS_TEST_FIELD_CONTROL", str(path))
    library = ctypes.CDLL(str(binary))
    library.pops_component_interface_v1.restype = ctypes.c_void_p
    assert library.pops_component_interface_v1()
    library.pops_test_actual_table_address.restype = ctypes.c_void_p
    address = library.pops_test_actual_table_address()
    header = ctypes.c_uint32.from_address(address)
    control = SharedTableControl(path)
    original_size = header.value
    assert original_size == control.layout["table_bytes"]
    for mode in (-1, 3, True):
        import pytest

        with pytest.raises(ValueError):
            control.arm(mode)
    assert header.value == original_size
    control.arm(2)
    assert header.value == control.layout["header_bytes"] < original_size
    assert control.counts() == {"callbacks": 0, "writes": 0}
    control.arm(0)
    assert header.value == original_size
    control.arm(1)
    assert header.value == original_size
    assert library.pops_test_actual_table_address() == address


def test_adapter_preparation_and_actual_provider_fault_scopes():
    root = Path(__file__).resolve().parents[2]
    adapter = (root / "include/pops/runtime/system/prepared_field_solver_component.hpp").read_text()
    body = adapter[adapter.index("  SolveReport execute_bound_solve_") :]
    assert body.index("callback preparation failed collectively") < body.index(
        "component::solve_field"
    )
    assert body.index("completion.reset()") < body.index("all_reduce_max(solve_error")
    source = fault_source(_manifest("ordering", interfaces.FieldSolver))
    assert (
        source.index("++control->callbacks;")
        < source.index("++control->writes;")
        < source.index("throw std::runtime_error")
    )


def test_actual_identity_metadata_is_lossless_json_not_default_str():
    import json
    from pops.identity.digest import Identity
    from tests.python.support.external_field_fault_component import json_evidence

    identity = Identity("binary", 1, "sha256", bytes(range(32)))
    decoded = json.loads(json.dumps(json_evidence(identity.to_data()), allow_nan=False))
    assert decoded["digest"] == {"encoding": "hex", "bytes": bytes(range(32)).hex()}


def test_public_fault_provider_composition_resolves(tmp_path):
    from pops.fields import ExternalFieldSolver
    from tests.python.integration.native_loader.test_external_field_solver_runtime import (
        _component,
        _topology_source,
        _program,
    )
    from tests.python.integration._final_field_program import (
        passive_field_model,
        resolve_periodic_field_program,
    )

    topology = _component(
        tmp_path,
        name="source-topology",
        interface=interfaces.FieldTopology,
        source_factory=_topology_source,
    )
    solver = _component(
        tmp_path,
        name="source-solver",
        interface=interfaces.FieldSolver,
        source_factory=fault_source,
        manifest_parameters=({"name": "answer", "kind": "runtime"},),
        instance_parameters={"answer": 7},
    )
    provider = ExternalFieldSolver(
        topology=topology, solver=solver, relative_tolerance=1e-11, max_iterations=23
    )
    plan = resolve_periodic_field_program(
        passive_field_model("dispatch preparation", coefficient=0.0),
        _program,
        name="dispatch preparation",
        block_name="material",
        target="system",
        n=8,
        field_solver=provider,
        components=(topology, solver),
        compile_options={"model_source_policy": "require"},
    )
    plan.verify()
    assert len(plan.component_inputs) == 2
    assert plan.compile_options["model_source_policy"] == "require"
    (field,) = plan.field_plans.values()
    from pops.fields import MeanValueGauge, ConstantNullspace
    assert type(field.discretization.gauge) is MeanValueGauge
    assert field.discretization.gauge.value == 0.0
    assert type(field.discretization.nullspace) is ConstantNullspace


def test_shared_table_rejects_foreign_layout_and_symlink(tmp_path):
    import json
    import os
    import pytest

    path = tmp_path / "control"
    path.write_bytes(bytes(100))
    layout = {
        "schema": "sol61.test-field-shared-table@1",
        "pid": os.getpid(),
        "bytes": 100,
        "table_bytes": 64,
        "header_bytes": 32,
        "mode": 64,
        "callbacks": 68,
        "writes": 72,
    }
    sidecar = Path(str(path) + ".layout.json")
    sidecar.write_text(json.dumps(layout))
    for key, value in (("pid", os.getpid() + 1), ("mode", True), ("bytes", 99), ("callbacks", 64)):
        bad = dict(layout, **{key: value})
        sidecar.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            SharedTableControl(path)
    sidecar.write_text(json.dumps(layout))
    link = tmp_path / "link"
    link.symlink_to(path)
    Path(str(link) + ".layout.json").write_text(json.dumps(layout))
    with pytest.raises(OSError):
        SharedTableControl(link)
