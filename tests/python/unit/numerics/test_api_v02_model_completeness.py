"""A real dispatch and an explicit component review are separate from science."""

from dataclasses import replace
import importlib
import inspect
from hashlib import sha256
from types import SimpleNamespace

import pytest

from examples.api_v02.common import CaseSpec, MissingCapabilityError
from examples.api_v02.coverage import LEGACY_OBLIGATIONS, OBLIGATIONS, scope_report
from examples.api_v02.profile_contracts import MissingComponentsError, selected_profile
from examples.api_v02.profile_dispatch import build_profile, reviewed_manifest


# Enumerated independently of dispatch, including the changed public defaults.
BUILDERS = {
    **{f"M{i:02}": f"models_M01_M14:build_m{i:02}" for i in range(1, 12)},
    "M09": "models_magnetic_midpoint:build_m09_midpoint",
    **{f"M{i:02}": f"models_M{i:02}:build_m{i:02}" for i in range(12, 15)},
    "M15": "models_M15_orders:build_m15_orders",
    "M16": "models_M15_M18:build_m16",
    "M17": "models_M15_M18:build_m17",
    "M18": "models_M18_entropy:build_m18_entropy",
    "M19": "models_kinetic_nd:build_kinetic_nd",
    "M20": "models_kinetic_nd:build_kinetic_nd",
    "M21": "model_gyrokinetic_nd:build_gyrokinetic_nd",
    "M22": "models_M22_M25_M28:build_m22",
    "M23": "model_hall_mhd_ct:build_hall_mhd_ct",
    "M24": "models_M22_M25_M28:build_m24",
    "M25": "models_M22_M25_M28:build_m25",
    "M26": "models_M15_M28:build_m26",
    "M27": "models_M15_M28:build_m27",
    "M28": "model_ale_nd:build_ale_nd",
    "T01": "methods_T01_T12:build_t01",
    "T02": "methods_coupled_nonsymmetric:build_t02_imex_second_order",
    "T03": "method_splitting_boundary:build_t03_dirichlet",
    "T04": "methods_coupled_nonsymmetric:build_t04_nonsymmetric",
    "T05": "methods_T01_T12:build_t05",
    "T06": "methods_coupled_nonsymmetric:build_t06_selected_global",
    "T07": "methods_T01_T12:build_t07",
    "T08": "methods_T01_T12:build_t08",
    "T09": "method_interval_transport:build_interval_transport",
    "T10": "method_relaxed_composite:build_t10_composite",
    "T11": "methods_patankar_network:build_t11_network",
    "T12": "method_flux_correction_amr:build_t12_amr",
}


def _public(identifier, **kwargs):
    if identifier.startswith("T"):
        from examples.api_v02.methods_T01_T12 import build_method

        return build_method(identifier, **kwargs)
    module = "models_M01_M14" if int(identifier[1:]) <= 14 else "models_M15_M28"
    return importlib.import_module("examples.api_v02." + module).build_model(identifier, **kwargs)


def _spy(monkeypatch, symbol):
    module_name, name = symbol.split(":")
    module = importlib.import_module("examples.api_v02." + module_name)
    original = getattr(module, name)
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return CaseSpec(
            object(),
            object(),
            {"q": 0.0},
            {"q": object()},
            1.0,
            0.1,
            lambda runtime: {"passed": False},
            "dispatch-only fixture",
        )

    capture.__signature__ = inspect.signature(original)
    monkeypatch.setattr(module, name, capture)
    return original, calls


@pytest.mark.parametrize("identifier", tuple(BUILDERS))
def test_all_public_defaults_call_selected_builder_and_retain_actual_parameters(
    monkeypatch, identifier
):
    original, calls = _spy(monkeypatch, BUILDERS[identifier])
    contract = selected_profile(identifier)
    spec = _public(identifier, require_complete=True)
    expected = inspect.signature(original).bind(**contract.builder.kwargs())
    expected.apply_defaults()
    assert calls == [expected.arguments]
    assert spec.selection.identifier == identifier
    assert spec.selection.builder == "examples.api_v02." + BUILDERS[identifier]
    assert spec.selection.kwargs() == expected.arguments
    assert spec.selection.implementation_manifest.profile == contract.key
    assert OBLIGATIONS[identifier]  # Scientific questions still exist after component admission.
    assert not spec.invariants(None)["passed"]  # No synthetic native PASS from dispatch.


def test_missing_equation_refuses_before_any_builder_runs(monkeypatch):
    _, calls = _spy(monkeypatch, BUILDERS["M09"])
    manifest = reviewed_manifest("M09")
    manifest = replace(
        manifest,
        bindings=tuple(binding for binding in manifest.bindings if binding.component != "midpoint"),
    )
    with pytest.raises(MissingComponentsError) as failure:
        _public("M09", require_complete=True, implementation_manifest=manifest)
    assert failure.value.missing == ("midpoint",)
    assert calls == []


def test_parameter_cannot_disable_a_required_equation_and_keep_complete_label(monkeypatch):
    _, calls = _spy(monkeypatch, BUILDERS["M09"])
    with pytest.raises(MissingComponentsError) as failure:
        _public("M09", require_complete=True, source_only=True)
    assert failure.value.missing == ("euler",)
    assert calls == []


def test_changed_dependency_refuses_the_old_review(monkeypatch):
    _, calls = _spy(monkeypatch, BUILDERS["M19"])
    manifest = reviewed_manifest("M19")
    assert manifest.dependencies
    module, _ = manifest.dependencies[0]
    changed = replace(manifest, dependencies=((module, "0" * 64), *manifest.dependencies[1:]))
    with pytest.raises(ValueError, match="digest changed"):
        _public("M19", require_complete=True, implementation_manifest=changed)
    assert calls == []


def test_historical_profile_calls_its_own_builder_and_cannot_claim_selected_contract(monkeypatch):
    _, calls = _spy(monkeypatch, "models_M15_M28:build_m19")
    spec = _public("M19", profile="legacy-1x1v", velocity_cells=10)
    assert calls[0]["velocity_cells"] == 10
    assert "configuration_shape" not in calls[0]
    assert spec.selection.name == "legacy-1x1v"
    assert spec.selection.implementation_manifest is None
    with pytest.raises(MissingCapabilityError, match="selected equation contract"):
        _public("M19", profile="legacy-1x1v", require_complete=True)
    assert len(calls) == 1


def test_parameter_override_is_visible_but_model_identity_cannot_switch(monkeypatch):
    _, calls = _spy(monkeypatch, BUILDERS["M20"])
    spec = _public("M20", ranks=3, configuration_shape=(4, 5), velocity_shape=(6, 7))
    assert spec.selection.kwargs()["ranks"] == 3
    assert spec.selection.kwargs()["configuration_shape"] == (4, 5)
    assert spec.selection.parameter_overrides == ("configuration_shape", "ranks", "velocity_shape")
    with pytest.raises(ValueError, match="model parameter"):
        _public("M20", model="M19")
    assert len(calls) == 1


def test_scope_reclassifies_old_gaps_without_erasing_the_old_audit():
    assert "milieu/Schur" in LEGACY_OBLIGATIONS["M09"][0]
    report = scope_report("M09")
    assert "midpoint" in report["required_components"]
    assert report["unestablished_properties"]
    assert "native_pass" not in report
    assert "nonhyperbolic" in scope_report("M16")["model_properties"][0]
    assert not any("hyperbolic" in key for key in scope_report("M16")["required_components"])


def test_continuation_reception_records_the_real_checkpoint_route(tmp_path):
    from examples.api_v02.reception import record_state

    class CheckpointOnly:
        def checkpoint(self, target):
            path = target.with_suffix(".npz")
            path.write_bytes(b"test-fixture-public-checkpoint")
            return str(path)

        def n_levels(self):
            raise AssertionError("a zero-block profile has no mesh snapshot to collect")

    row = record_state(CheckpointOnly(), SimpleNamespace(subjects={}), tmp_path)
    assert (
        row["continuation_checkpoint_sha256"]
        == sha256(b"test-fixture-public-checkpoint").hexdigest()
    )
    assert "states_sha256" not in row


@pytest.mark.parametrize("identifier", ("M22", "M23", "M24", "M25", "M28"))
def test_continuum_compatibility_entry_point_uses_the_same_selected_route(monkeypatch, identifier):
    from examples.api_v02.models_M22_M25_M28 import build_model

    _, calls = _spy(monkeypatch, BUILDERS[identifier])
    spec = build_model(identifier)
    assert calls
    assert spec.selection.builder == "examples.api_v02." + BUILDERS[identifier]


@pytest.mark.parametrize(
    "identifier,parameters",
    (
        ("M09", {"cells": (3, 3), "steps": 1}),
        ("T02", {"count": 3, "steps": 1}),
    ),
)
def test_dispatch_constructs_real_equation_programs_without_compiling(identifier, parameters):
    import pops

    spec = build_profile(identifier, require_complete=True, **parameters)
    assert spec.case._time is not None
    assert spec.selection.kwargs().items() >= parameters.items()
    assert pops.validate(spec.case) is not None
