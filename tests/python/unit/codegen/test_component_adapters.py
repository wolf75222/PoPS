"""ADC-681 manifest-driven small-interface conformance."""
from __future__ import annotations

import pytest

from pops.model import (
    ComponentInterfaceError,
    ComponentManifest,
    ComponentManifestError,
    ComponentRegistry,
    EvaluationOutcome,
)
from pops.runtime.routes import component_registry_snapshot, resolve


class ExternalOperator:
    def __init__(self, manifest):
        self.component_manifest = manifest
        self.layouts = []

    def requirements(self):
        return ("state",)

    def lower(self, context):
        return {"plan": context["layout"]}

    def stencil(self):
        return {"depth": 1}

    def stability(self):
        return {"kind": "spectral-radius"}

    def providers(self):
        return ("state",)

    def effects(self):
        return ("rate-write",)

    def restart(self):
        return {"mode": "stateless"}

    def report(self):
        return {"kind": "external"}

    def evaluate(self, context):
        self.layouts.append(context["layout"])
        return EvaluationOutcome.ok(context["layout"])

    def format(self, value):
        return "external:%s" % value


_METHODS = {
    "requirement": "requirements",
    "lowering": "lower",
    "stencil": "stencil",
    "stability": "stability",
    "provider": "providers",
    "effects": "effects",
    "restart": "restart",
    "report": "report",
    "fallible_evaluation": "evaluate",
    "format": "format",
}


def _manifest(*, facets=tuple(_METHODS), target=None):
    return ComponentManifest(
        uri="pops://external.test/components/all-interfaces",
        component_type="spatial_operator",
        version="1.0.0",
        facets=facets,
        interfaces=tuple({"name": name, "mode": "method", "binding": _METHODS[name]}
                         for name in facets),
        layouts=("uniform", "amr"),
        target=target or {"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
    )


def _cpu2d():
    return {"dimension": 2, "scalar": "float64", "device": "cpu", "features": []}


def test_one_external_component_executes_uniform_and_amr_through_the_same_adapter():
    registry = ComponentRegistry()
    external = ExternalOperator(_manifest())
    registry.register(external, origin="external", platform=_cpu2d())
    adapter = registry.adapter(external.component_manifest.component_id)

    assert adapter.invoke("lowering", {"layout": "uniform"}) == {"plan": "uniform"}
    assert adapter.invoke("fallible_evaluation", {"layout": "uniform"}).value == "uniform"
    assert adapter.invoke("fallible_evaluation", {"layout": "amr"}).value == "amr"
    assert external.layouts == ["uniform", "amr"]
    assert adapter.invoke("format", 3) == "external:3"


def test_registration_rejects_malformed_interface_and_target_before_mutation():
    class MissingLower:
        component_manifest = _manifest(facets=("lowering",))

    registry = ComponentRegistry()
    with pytest.raises(ComponentInterfaceError) as error:
        registry.register(MissingLower(), origin="external", platform=_cpu2d())
    assert error.value.code == "missing_interface_binding"
    assert registry.revision == 0

    external = ExternalOperator(_manifest())
    with pytest.raises(ComponentManifestError) as target_error:
        registry.register(external, origin="external", platform={
            "dimension": 3, "scalar": "float64", "device": "cpu", "features": [],
        })
    assert getattr(target_error.value, "code", None) == "unsupported_target"
    assert registry.revision == 0


def test_fallible_interface_refuses_an_implicit_success_value():
    class Implicit(ExternalOperator):
        def evaluate(self, context):
            return context

    component = Implicit(_manifest(facets=("fallible_evaluation",)))
    registry = ComponentRegistry()
    registry.register(component, origin="external")
    with pytest.raises(ComponentInterfaceError) as error:
        registry.adapter(component.component_manifest.component_id).invoke(
            "fallible_evaluation", {"layout": "uniform"})
    assert error.value.code == "implicit_evaluation_outcome"


def test_builtin_and_external_registration_reports_have_identical_shape():
    builtin_snapshot = component_registry_snapshot(_cpu2d())
    builtin = builtin_snapshot.adapter(
        resolve("riemann", "rusanov").component_manifest().component_id)

    external_registry = ComponentRegistry()
    external = ExternalOperator(_manifest())
    external_registry.register(external, origin="external", platform=_cpu2d())
    extension = external_registry.adapter(external.component_manifest.component_id)

    assert set(builtin.to_data()) == set(extension.to_data())
    assert set(builtin.to_data()["provenance"]) == set(extension.to_data()["provenance"])
    assert builtin.to_data()["provenance"]["origin"] == "builtin"
    assert extension.to_data()["provenance"]["origin"] == "external"


def test_native_interface_is_declared_and_unbound_never_falls_back():
    builtin = component_registry_snapshot().adapter(
        resolve("riemann", "rusanov").component_manifest().component_id)
    with pytest.raises(ComponentInterfaceError) as error:
        builtin.invoke("fallible_evaluation", object())
    assert error.value.code == "unbound_component_entry_point"
