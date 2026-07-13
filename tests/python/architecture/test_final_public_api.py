"""ADC-689: one pure-Python public front door and qualified instance handles."""
from __future__ import annotations

import sys

import pops


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


def test_physics_has_no_competing_model_facade() -> None:
    from pops import physics

    assert physics.__all__ == ["Model"]
    assert physics.Model is pops.Model
    for removed in ("PdeModel", "HyperbolicModel", "PhysicsModel", "HybridModel"):
        assert not hasattr(physics, removed)
