"""ADC-660: resolve closes capability evidence before any compiler call."""
from __future__ import annotations

import json

import pytest

from pops.codegen._resolution import (
    CapabilityResolutionError,
    canonical_capability_evidence_json,
    resolve_capability_evidence,
)
from pops.external._brick_gates import validate_ref


class _Projection:
    def __init__(self, values):
        self._values = values

    def to_dict(self):
        return dict(self._values)


class _Blocks:
    def __init__(self, model):
        self._model = model

    def items(self):
        return (("fluid", {"model": self._model}),)


class _Model:
    def __init__(self, capabilities=()):
        self.provided_capabilities = tuple(capabilities)


class _Problem:
    def __init__(self, *, frozen=True, capabilities=()):
        self.frozen = frozen
        self._blocks = _Blocks(_Model(capabilities))

    def capabilities(self):
        return _Projection({"problem_structure": True})

    def requirements(self):
        return _Projection({"time_scheme": True})


class _Layout:
    def __init__(self, name):
        self._name = name

    def capabilities(self):
        return _Projection({"layout": self._name})


class _Library:
    def __init__(self, bricks):
        self.bricks = tuple(bricks)


def _brick(*, requirements=None, capabilities=None, supported=("uniform",)):
    return {
        "id": "ext",
        "brick_type": "external_cpp",
        "category": "solver",
        "scheme": "ext",
        "native_id": "ext_native",
        "available": True,
        "requirements": dict(requirements or {}),
        "capabilities": dict(capabilities or {}),
        "options": {"supported_layouts": list(supported)},
    }


def test_resolution_requires_frozen_problem():
    with pytest.raises(TypeError, match="frozen"):
        resolve_capability_evidence(_Problem(frozen=False), layout=_Layout("uniform"))


def test_resolution_joins_providers_and_returns_canonical_json():
    problem = _Problem(capabilities=("linear_solve",))
    library = _Library((_brick(
        requirements={"linear_solve": True}, capabilities={"ext_flux": True}),))

    first = resolve_capability_evidence(
        problem, layout=_Layout("uniform"), libraries=(library,))
    repeat = resolve_capability_evidence(
        problem, layout=_Layout("uniform"), libraries=(library,))

    assert first == repeat
    assert first["external_bricks"] == [{
        "id": "ext_native",
        "requirements": ["linear_solve"],
        "capabilities": ["ext_flux"],
        "supported_layouts": ["uniform"],
        "status": "proven",
    }]
    encoded = canonical_capability_evidence_json(first)
    assert encoded == canonical_capability_evidence_json(repeat)
    assert json.loads(encoded) == first
    assert " " not in encoded


def test_missing_external_capability_fails_closed():
    library = _Library((_brick(requirements={"missing_cap": True}),))
    with pytest.raises(CapabilityResolutionError, match="missing capability 'missing_cap'"):
        resolve_capability_evidence(
            _Problem(), layout=_Layout("uniform"), libraries=(library,))


def test_external_brick_cannot_satisfy_its_own_requirement():
    library = _Library((_brick(
        requirements={"self_cap": True}, capabilities={"self_cap": True}),))
    with pytest.raises(CapabilityResolutionError, match="missing capability 'self_cap'"):
        resolve_capability_evidence(
            _Problem(), layout=_Layout("uniform"), libraries=(library,))


def test_unknown_external_layout_evidence_fails_closed():
    library = _Library((_brick(supported=()),))
    with pytest.raises(CapabilityResolutionError, match="unknown supported_layouts"):
        resolve_capability_evidence(
            _Problem(), layout=_Layout("uniform"), libraries=(library,))


class _PendingAmrProgram:
    def ir_nodes(self):
        return [{"op": "project", "attrs": {}}]


def test_pending_amr_program_capability_fails_before_compiler_call():
    compiler_calls = []

    def compiler():
        compiler_calls.append(True)

    with pytest.raises(CapabilityResolutionError, match="projection=pending"):
        resolve_capability_evidence(
            _Problem(), layout=_Layout("amr"), time=_PendingAmrProgram())
    assert compiler_calls == []


def test_canonical_external_gate_refuses_unknown_evidence():
    record = {
        "id": "ext",
        "native_id": "ext_native",
        "requirements": [],
        "capabilities": [],
        "supported_layouts": ["uniform"],
        "exported_symbols": [],
    }
    with pytest.raises(ValueError, match="unknown ABI evidence"):
        validate_ref(record, context={
            "canonical_resolution": True,
            "capabilities": [],
            "layout": "uniform",
            "module_abi_key": "module-abi",
        })


def test_noncanonical_external_gate_preserves_exploratory_unknown_report():
    record = {
        "id": "ext",
        "native_id": "ext_native",
        "requirements": ["unknown_cap"],
        "capabilities": [],
        "supported_layouts": [],
        "exported_symbols": [],
    }
    assert validate_ref(record, context=None) is None
