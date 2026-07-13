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


def test_root_contract_is_exact_and_does_not_load_native_code() -> None:
    assert tuple(pops.__all__) == _PUBLIC
    assert "pops._pops" not in sys.modules
    for removed in (
        "Problem",
        "RuntimePolicies",
        "OutputPolicy",
        "CheckpointPolicy",
        "System",
        "AmrSystem",
        "ModelSpec",
        "BindInputs",
    ):
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
    for removed in ("RuntimePolicies", "OutputPolicy", "CheckpointPolicy"):
        assert not hasattr(output, removed)
