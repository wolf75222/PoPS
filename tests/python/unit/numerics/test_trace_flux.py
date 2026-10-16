"""The two-trace author uses the existing expression and component authorities."""
import re
import pytest

from pops import interfaces
from pops.model.scalar_contract import Var
from pops.numerics import TraceFlux


def _make(speed=.2, *, dimension=2, components=1):
    return TraceFlux(lambda l,r,n,t,h: tuple(speed*l[i]*n[0] + t*h*(r[i]-l[i])
                                            for i in range(components)),
                     dimension=dimension, components=components,
                     stability=lambda l,r,n,t,h: abs(speed*n[0])+t*h)


@pytest.mark.parametrize("dimension,components", ((1,1), (2,4), (3,7)))
def test_trace_dag_has_exact_shapes_and_real_component_authority(dimension, components):
    component = _make(dimension=dimension, components=components)
    interface = component.component_type.interface
    assert interface == interfaces.NumericalFlux
    component.component_type.package.verify()
    manifest = component.component_manifest
    assert manifest.signature["state_components"] == components
    assert manifest.signature["trace_flux"]["dimension"] == dimension
    assert component.component_type.snapshot_data()["manifest"] == manifest.manifest_digest.token
    source = component.component_type.package.payloads[0].content.decode()
    assert "Kokkos::parallel_for" in source and "pops_component_interface_v1" in source
    assert not re.search(r"@[A-Z]+@", source)


def test_changed_equation_changes_authenticated_compiled_payload():
    a, equal, b = _make(), _make(), _make(.3)
    assert a.component_type.package.identity == equal.component_type.package.identity
    assert a.component_type.package.identity != b.component_type.package.identity
    assert a.component_type.package.payloads[0].content != b.component_type.package.payloads[0].content


def test_callbacks_elaborate_once_and_retained_package_has_no_callback():
    calls = []
    def body(l,r,n,t,h):
        calls.append("body")
        return (l[0]+r[0],)
    def bound(l,r,n,t,h):
        calls.append("bound")
        return 1.
    component = TraceFlux(body, stability=bound, components=1, dimension=2)
    component.component_type.snapshot_data()
    component.component_type.package.verify()
    assert calls == ["body", "bound"]


def test_foreign_trace_with_forged_name_cannot_capture_another_expression_scope():
    forged = Var("trace_left_0", "trace_input")
    with pytest.raises(ValueError, match="foreign variable"):
        TraceFlux(lambda l,r,n,t,h: (forged,), stability=lambda *a: 1., components=1, dimension=2)


def test_wrong_component_shape_is_rejected_before_compilation():
    with pytest.raises(TypeError, match="declared component count"):
        TraceFlux(lambda *a: (1.,), stability=lambda *a: 1., components=2, dimension=2)
