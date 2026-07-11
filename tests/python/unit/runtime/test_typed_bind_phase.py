"""ADC-660: the runtime bind boundary is typed and fails closed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.runtime import _bind_validation as validation
from pops.runtime._bind_adapters import (
    _AmrRuntimeAdapter,
    _UniformRuntimeAdapter,
    adapter_for,
    install_plan,
)


def test_adapter_selection_is_exhaustive():
    layout = object()
    assert isinstance(adapter_for("system", None), _UniformRuntimeAdapter)
    assert isinstance(adapter_for("amr_system", layout), _AmrRuntimeAdapter)
    with pytest.raises(ValueError, match="exactly 'system' or 'amr_system'"):
        adapter_for("future-or-tampered-route", layout)
    with pytest.raises(ValueError, match="exactly 'system' or 'amr_system'"):
        adapter_for(None, None)


def test_typed_install_rejects_every_wrong_phase_value():
    with pytest.raises(TypeError, match="exact InstallPlan"):
        install_plan(object())
    with pytest.raises(TypeError, match="exact InstallPlan"):
        install_plan(SimpleNamespace(target="system", layout=None))


def test_bind_gates_reject_missing_artifact_metadata_before_runtime_probe(monkeypatch):
    called = False

    def facts():
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(validation, "loaded_runtime_facts", facts)
    with pytest.raises(TypeError, match=r"manifest\(\) and arguments\(\)"):
        validation.run_bind_gates(SimpleNamespace(), None, {}, {}, {})
    assert not called


def test_bind_gates_reject_none_artifact_metadata_before_runtime_probe(monkeypatch):
    artifact = SimpleNamespace(manifest=lambda: None, arguments=lambda: object())
    monkeypatch.setattr(
        validation,
        "loaded_runtime_facts",
        lambda: pytest.fail("runtime must not be probed for incomplete artifact metadata"),
    )
    with pytest.raises(ValueError, match="incomplete manifest/arguments"):
        validation.run_bind_gates(artifact, None, {}, {}, {})


@pytest.mark.parametrize(
    "facts, error",
    [
        ({}, "required identity/capability"),
        ({
            "abi_key": "abi", "precision": "double", "communicator": "serial",
            "supports_mpi": None, "supports_gpu": False,
        }, "supports_mpi"),
        ({
            "abi_key": "abi", "precision": "unknown", "communicator": "serial",
            "supports_mpi": False, "supports_gpu": False,
        }, "exact fact"),
        ({
            "abi_key": "abi", "precision": "double", "communicator": "serial",
            "supports_mpi": 0, "supports_gpu": False,
        }, "must be bool"),
    ],
)
def test_runtime_facts_are_complete_and_typed(facts, error):
    with pytest.raises((TypeError, ValueError), match=error):
        validation._require_runtime_facts(facts)


def test_loaded_runtime_fact_probe_does_not_swallow_provider_failure(monkeypatch):
    def broken_report():
        raise RuntimeError("runtime facts unavailable")

    monkeypatch.setattr("pops._bootstrap.abi_key", lambda: "abi")
    monkeypatch.setattr("pops.runtime_environment.runtime_environment_report", broken_report)
    with pytest.raises(RuntimeError, match="runtime facts unavailable"):
        validation.loaded_runtime_facts()


def test_install_validation_rejects_degraded_or_unreadable_artifact():
    sim = SimpleNamespace(block_names=lambda: ())
    with pytest.raises(TypeError, match=r"callable arguments\(\)"):
        validation.validate_install_arguments(sim, SimpleNamespace(), {}, {}, {}, {})

    def broken_arguments():
        raise RuntimeError("metadata corruption")

    with pytest.raises(RuntimeError, match="metadata corruption"):
        validation.validate_install_arguments(
            sim, SimpleNamespace(arguments=broken_arguments), {}, {}, {}, {})


def test_install_validation_requires_runtime_block_inventory():
    artifact = SimpleNamespace(arguments=lambda: SimpleNamespace(
        instances={}, params={}, aux={}, solvers={}))
    with pytest.raises(TypeError, match=r"block_names\(\)"):
        validation.validate_install_arguments(
            SimpleNamespace(), artifact, {}, {}, {}, {})
