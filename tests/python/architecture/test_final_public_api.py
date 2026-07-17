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
    "RunReport",
    "RunStopReason",
    "ExecutionContext",
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


def test_program_and_time_library_expose_only_final_authoring_spelling() -> None:
    import pops.lib.time as libtime

    assert tuple(signature(pops.Program.state).parameters) == ("self", "state", "clock")
    assert tuple(libtime.__all__) == (
        "AdamsBashforth", "BDF", "ButcherTableau", "FORWARD_EULER_TABLEAU",
        "ForwardEuler", "IMEX", "IMEX_EULER_TABLEAU", "Lie", "PredictorCorrector",
        "RK4", "RK4_TABLEAU", "RungeKutta", "SSPRK2", "SSPRK2_TABLEAU", "SSPRK3",
        "SSPRK3_TABLEAU", "Strang",
    )
    for removed in (
        "forward_euler", "ssprk2", "ssprk3", "rk4", "rk", "explicit_rk", "strang",
        "lie", "adams_bashforth", "adams_bashforth2", "bdf", "imex_local",
        "imex_local_linear", "predictor_corrector_local_linear", "CondensedSchur",
    ):
        assert not hasattr(libtime, removed)

    from pops import fields

    for removed in ("FieldProblem", "PoissonProblem", "HoldPrevious"):
        assert not hasattr(fields, removed)


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

    assert codegen.__all__ == ["Production", "CompilerLowerable", "CompilerLowering"]
    assert "compile_component" not in codegen.__all__
    assert not hasattr(codegen, "compile_component")
    assert "compile_component" in external.__all__
    assert callable(external.compile_component)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.codegen.component_packages")
    for retired_module in (
        "pops.codegen.math_options",
        "pops.codegen.optimization",
        "pops.codegen.orchestration",
        "pops.codegen.compiled_artifact",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(retired_module)

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
    with pytest.raises(TypeError, match="exact RuntimeInstance returned by pops.bind"):
        pops.run(object(), t_end=1.0)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.runtime.runtime_instance")

    from pops.runtime._runtime_instance import RuntimeInstance

    assert not hasattr(RuntimeInstance, "run")
    assert "__getattr__" not in RuntimeInstance.__dict__


def test_runtime_instance_has_only_the_explicit_read_and_restart_surface() -> None:
    from pops.runtime._runtime_instance import RuntimeInstance

    public = {
        name
        for name in RuntimeInstance.__dict__
        if not name.startswith("_")
    }
    assert public == {
        "amr",
        "bind_identity",
        "bound_snapshot",
        "block_level_state",
        "block_level_state_global",
        "block_names",
        "checkpoint",
        "cleanup_consumer_recovery",
        "consumer_cursors",
        "consumer_graph",
        "consumer_recoveries",
        "field_potential_global",
        "field_potential_level_global",
        "field_provider_levels",
        "field_provider_slots",
        "get_state",
        "history_depth",
        "history_global",
        "history_names",
        "history_ncomp",
        "inspect",
        "integral",
        "installed_program_hash",
        "last_restart_identity",
        "last_run_identity",
        "layout_identity",
        "local_boxes",
        "local_state",
        "macro_step",
        "n_levels",
        "nx",
        "ny",
        "patch_boxes",
        "patch_rectangles",
        "program_report",
        "restart",
        "restore_consumer_recovery",
        "retry_consumer_finalizers",
        "state_global",
        "time",
    }
    assert public.isdisjoint({
        "assembly",
        "executor_for_block",
        "executor_for_layout",
        "install_plan",
        "native_executor",
        "profile",
        "run",
        "runtime_plan",
        "step",
        "step_cfl",
    })


def test_output_surface_has_direct_consumers_not_policy_bundles() -> None:
    from pops import output, runtime

    assert hasattr(output, "ScientificOutput")
    assert hasattr(output, "Checkpoint")
    assert hasattr(output, "ConsumerGraph")
    assert "ConsumerGraph" in output.__all__
    assert output.ConsumerGraph.__module__ == "pops.output._consumer_contracts"
    assert "ConsumerGraph" not in runtime.__all__
    assert not hasattr(runtime, "ConsumerGraph")
    for retired_module in (
        "pops.runtime.consumer",
        "pops.runtime.output_publisher",
        "pops.runtime._consumer_contracts",
        "pops.runtime._consumer_authoring",
        "pops.runtime.restart_provider",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(retired_module)
    for removed in (*_retired_names()[1:4], "Plotfile"):
        assert not hasattr(output, removed)


def test_runtime_package_does_not_reexport_retired_authoring_engines() -> None:
    from pops import runtime

    assert runtime.__all__ == []
    assert dir(runtime) == []
    for removed in _retired_names()[4:7]:
        assert removed not in runtime.__all__
        assert not hasattr(runtime, removed)
    for retired_module in (
        "pops.runtime.mesh",
        "pops.runtime.system",
        "pops.runtime.amr_system",
        "pops.runtime.profile",
        "pops.runtime.threading",
        "pops.runtime.platform_manifest",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(retired_module)


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
