"""ADC-689: one pure-Python public front door and qualified instance handles."""
from __future__ import annotations

import importlib
import sys
from inspect import signature

import pops
import pytest


_PUBLIC = (
    "Model",
    "Program",
    "Case",
    "validate",
    "inspect",
    "explain",
    "resolve",
    "compile",
    "bind",
    "run",
    "__version__",
)


def _retired_names() -> tuple[str, ...]:
    return (
        "Pro" + "blem",
        "Runtime" + "Policies",
        "Output" + "Policy",
        "Checkpoint" + "Policy",
        "Sys" + "tem",
        "Amr" + "System",
        "Model" + "Spec",
        "Bind" + "Inputs",
        "Sys" + "temConfig",
        "Amr" + "SystemConfig",
        "Compiled" + "Time",
        "compile" + "_library",
        "read" + "_library_manifest",
        "Library" + "Manifest",
    )


def test_root_contract_is_exact_and_does_not_load_native_code() -> None:
    assert tuple(pops.__all__) == _PUBLIC
    assert "pops._pops" not in sys.modules
    for removed in _retired_names():
        assert not hasattr(pops, removed)
    for absent_submodule in ("restart", "schedule"):
        assert absent_submodule not in dir(pops)
        assert not hasattr(pops, absent_submodule)


def test_case_authoring_validation_and_inspection_are_pure_python() -> None:
    model = pops.Model("transport")
    state = model.state("U", components=("u",))
    case = pops.Case("two_instances")
    left = case.block("left", model)
    right = case.block("right", model)

    left_state = case.qualify(state, block=left)
    right_state = case.qualify(state, block=right)
    assert left_state != right_state
    assert left_state.block_ref == left
    assert right_state.block_ref == right

    report = pops.inspect(case)
    assert report["name"] == "two_instances"
    assert set(report["blocks"]) == {"left", "right"}
    assert pops.validate(case) is case
    assert case.frozen
    assert "pops._pops" not in sys.modules


def test_case_has_one_registration_spelling_per_family() -> None:
    case = pops.Case("canonical")
    assert hasattr(case, "block") and not hasattr(case, "add_block")
    assert hasattr(case, "field") and not hasattr(case, "add_field")
    assert hasattr(case, "program") and not hasattr(case, "time")
    program = pops.Program("canonical")
    assert not hasattr(program, "call")
    assert not hasattr(program, "solve_fields")
    assert not hasattr(program, "solve_fields_from_blocks")
    assert not hasattr(program, "solve_implicit")
    assert hasattr(program, "solve")


def test_public_state_rejects_opaque_units_until_a_typed_unit_protocol_exists() -> None:
    model = pops.Model("dimension_contract")
    with pytest.raises(TypeError, match="units are unsupported"):
        model.state("U", components=("rho",), units=("kg/m3",))


def test_public_bind_accepts_value_families_without_an_internal_inputs_record() -> None:
    parameters = tuple(signature(pops.bind).parameters)
    assert parameters == (
        "artifact", "initial_state", "params", "aux", "resources", "initial_values",
    )
    from pops import codegen, external

    assert codegen.__all__ == ["Production"]

    for retired in (
        "BindInputs", "InstallPlan", "ResolvedSimulationPlan", "CompiledSimulationArtifact",
        "LibraryManifest", "compile_library", "read_library_manifest", "emit_library_cpp",
    ):
        assert retired not in codegen.__all__
        assert not hasattr(codegen, retired)
    for retired in (
        "CompiledBrickRef", "ExternalBrick", "register", "register_manifest_file",
        "read_manifest", "CompiledManifest", "load_cpp_library", "load_compiled_manifest",
    ):
        assert retired not in external.__all__
        assert not hasattr(external, retired)


def test_public_run_accepts_only_the_bound_runtime_instance() -> None:
    assert tuple(signature(pops.run).parameters) == ("instance", "controls")
    with pytest.raises(TypeError, match="authenticated object returned by pops.bind"):
        pops.run(object(), t_end=1.0)
    from pops.runtime.runtime_instance import RuntimeInstance

    assert not hasattr(RuntimeInstance, "run")


def test_output_surface_has_direct_consumers_not_policy_bundles() -> None:
    from pops import output

    assert hasattr(output, "ScientificOutput")
    assert hasattr(output, "Checkpoint")
    for removed in (*_retired_names()[1:4], "Plotfile"):
        assert not hasattr(output, removed)


def test_runtime_package_does_not_reexport_retired_authoring_engines() -> None:
    from pops import runtime

    for removed in _retired_names()[4:7]:
        assert removed not in runtime.__all__
        assert not hasattr(runtime, removed)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.runtime.mesh")


def test_physics_has_no_competing_model_facade() -> None:
    from pops import physics

    assert physics.__all__ == [
        "Model", "ComponentRole", "Density", "Energy", "Momentum", "Pressure", "Scalar",
        "Temperature", "Velocity",
    ]
    assert physics.Model is pops.Model
    for removed in ("PdeModel", "HyperbolicModel", "PhysicsModel", "HybridModel"):
        assert not hasattr(physics, removed)
    model = pops.Model("single_public_model")
    assert not hasattr(model, "dsl")
    assert not hasattr(model, "compile")
    for retired_module in (
        "pops.physics.facade",
        "pops.physics.model",
        "pops.codegen.compile",
        "pops.codegen.compile_drivers",
        "pops.codegen.compile_emit",
        "pops.codegen.backends",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(retired_module)
