"""Actual test DSO compilation/control: Source/Host preparation, not Native dispatch proof."""

import ctypes
from pathlib import Path
import subprocess
from pops import interfaces
from tests.python.integration.native_loader.test_external_field_solver_runtime import _manifest
from tests.python.support.external_field_fault_component import fault_source


def test_real_component_source_builds_and_typed_arm_restores_table(tmp_path):
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
    library = ctypes.CDLL(str(binary))
    library.pops_test_field_fault_arm.argtypes = [ctypes.c_int]
    library.pops_test_field_table_size.restype = ctypes.c_size_t
    original_size = library.pops_test_field_table_size()
    assert library.pops_test_field_fault_arm(-1) == 1
    assert library.pops_test_field_fault_arm(3) == 1
    assert library.pops_test_field_table_size() == original_size
    assert library.pops_test_field_fault_arm(2) == 0
    assert 0 < library.pops_test_field_table_size() < original_size
    assert library.pops_test_field_callback_count() == 0
    assert library.pops_test_field_write_count() == 0
    assert library.pops_test_field_fault_arm(0) == 0
    assert library.pops_test_field_table_size() == original_size
    assert library.pops_test_field_fault_arm(1) == 0
    assert library.pops_test_field_callback_count() == 0


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
        source.index("++observed_callbacks;")
        < source.index("++observed_writes;")
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
