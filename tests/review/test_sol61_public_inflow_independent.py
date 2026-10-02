"""Independent Source counterexamples; no Native qualification."""
import pytest
from pops.boundary import interior_trace
from pops.boundary.transport import Inflow,TransportBoundarySet
from pops.math import ValueExpr
from pops.codegen.inferred_boundary_expression import lower_expression
from tests.python.unit.boundary.test_transport_authoring import _authoring

def test_trace_does_not_hide_sibling_own_ghost_dependency():
    from tests.python.support.public_inflow_field_case import build
    case,_=build(ghost=False)
    state=case._resolved_numerics_for("marker").boundaries[0].conditions[0].state
    expression=2*interior_trace(state,"c")+ValueExpr(state)["c"]
    with pytest.raises(ValueError,match="own output ghost"):
        lower_expression(expression,output_state=state)

def test_same_expression_four_boundaries_has_distinct_geometry_authority():
    frame,_,_,_,numerics,case,block,state=_authoring()
    inflow=Inflow(state=state,value=interior_trace(state,"u")+1)
    numerics.boundaries.add(TransportBoundarySet({face:inflow for face in frame.boundaries.all}))
    case.numerics(numerics,block=block)
    authority=case._resolved_numerics_for("tracer").boundaries[0]
    first=authority.inferred_component_bindings();second=authority.inferred_component_bindings()
    ids=lambda rows:[c.component_manifest.component_id for _,c in rows]
    assert len(first)==len(set(ids(first)))==4 and ids(first)==ids(second)
    geometry=[c.component_manifest.signature["inferred_boundary_expression"]["geometry"] for _,c in first]
    assert len({repr(x) for x in geometry})==4
    for _,component in first:
        leaves=component.component_manifest.signature["inferred_boundary_expression"]["leaves"]
        assert all(x.get("support")=="clamped-primary-state-interior-trace@1" for row in leaves for x in row)

def test_trace_component_bool_and_foreign_state_not_reinterpreted():
    *_,state=_authoring()
    with pytest.raises((TypeError,ValueError)):interior_trace(state,True)
    *_,foreign=_authoring()
    with pytest.raises(ValueError,match="exact primary"):
        lower_expression(interior_trace(foreign,"u"),output_state=state)
