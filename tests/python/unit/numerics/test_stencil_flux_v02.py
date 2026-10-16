"""Closed expression/stencil authority and receipt derivation, not native qualification."""
from fractions import Fraction

import pytest
import pops
from pops._ir.expr import Var
from pops.numerics.stencil_flux import StencilFlux
from examples.api_v02.models_M01_M14 import build_m05, navier_stokes_face


def heat(sample, h, axis):
    left = tuple(-int(d == axis) for d in range(2))
    return (-.1*(sample((0, 0), 0)-sample(left, 0))/h[axis],), .1/h[axis]**2


@pytest.mark.parametrize("offset", ((2, 0), (0,), (True, 0), [0, 0]))
def test_stencil_refuses_unbounded_or_inexact_indices(offset):
    with pytest.raises(ValueError, match="offset"):
        StencilFlux(dimension=2, components=1, radius=1,
                    body=lambda sample,h,axis: ((sample(offset, 0),), 1.))


@pytest.mark.parametrize("component", (-1, 1, True, .0))
def test_stencil_refuses_component_outside_exact_state(component):
    with pytest.raises(ValueError, match="component"):
        StencilFlux(dimension=2, components=1, radius=1,
                    body=lambda sample,h,axis: ((sample((0, 0), component),), 1.))


def test_stencil_refuses_unbound_scientific_dependency():
    with pytest.raises(ValueError, match="foreign"):
        StencilFlux(dimension=2, components=1, radius=1,
                    body=lambda sample,h,axis: ((Var("mu", "aux"),), 1.))


def test_normal_and_transverse_extent_are_derived_from_real_reads():
    normal = StencilFlux(dimension=2, components=1, radius=1, body=heat)
    cross = navier_stokes_face()
    assert normal.ghost_extent == (1, 1)
    assert cross.ghost_extent == (2, 2)
    assert any(offset[0] and offset[1] for offset, component, symbol in cross.samples)
    assert not hasattr(cross, "body"), "no callable survives authoring"


def test_stencil_order_is_not_limited_by_a_method_catalogue():
    wide = StencilFlux(dimension=2, components=1, radius=9,
        body=lambda sample,h,axis: ((sample(tuple(9*int(d==axis) for d in range(2)),0),),1.))
    assert wide.ghost_extent == (10,10)
    wide.validate()
    with pytest.raises(ValueError, match="Int32"):
        StencilFlux(dimension=2,components=1,radius=2**31-1,body=heat)


def test_body_and_accepted_weights_survive_real_program_detachment():
    from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
    spec = build_m05(cells=(8, 8), two_dimensional=True)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    # This test establishes authoring/resolution only, never a compiled/native claim.
    assert resolved.resolved_dimension == 2
    program = resolved.time
    rows = accepted_stencil_quadrature(program)
    assert len(rows) == 2
    assert [weight for _, weight in rows] == [{1: Fraction(1, 2)}]*2
    assert all(row.attrs["stencil_flux"].components == 4 for row, _ in rows)
    program.to_graph()
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    source = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "PreparedStencilFlux" in source and "capture_stencil_exchanges" in source
    assert "publish_stencil_exchanges" in source


@pytest.mark.parametrize("field,component", ((-1,0), (1,0), (True,0), (0,1)))
def test_stencil_refuses_foreign_field_slot_or_component(field,component):
    with pytest.raises(ValueError, match="field"):
        StencilFlux(dimension=2,components=1,radius=1,field_components=(1,),
            body=lambda sample,h,axis: ((sample((0,0),component,field=field),), 1.))


def test_typed_poisson_field_reads_keep_source_and_bounded_halo():
    from examples.api_v02.models_M01_M14 import build_m11
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.stencil_flux import _validate_fields
    from types import SimpleNamespace
    spec=build_m11(cells=(8,8))
    resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    nodes=[v for v in resolved.time._values if v.op == "stencil_flux_divergence"]
    assert len(nodes)==2 and all(len(v.inputs)==2 for v in nodes)
    flux=nodes[0].attrs["stencil_flux"]
    assert flux.field_components==(1,) and flux.field_ghost_extents==((1,1),)
    state,field=nodes[0].inputs
    _validate_fields(state,(field,),flux)
    with pytest.raises(ValueError,match="stage"):
        _validate_fields(state,(nodes[1].inputs[1],),flux)
    # Equal physical type and evaluation point cannot authorize a different SSA state.
    changed=SimpleNamespace(id=state.id+10000,block=state.block,state_ref=state.state_ref,
                            space=state.space,point=state.point,attrs=state.attrs,op=state.op)
    with pytest.raises(ValueError,match="exact State"):
        _validate_fields(changed,(field,),flux)
    with pytest.raises(ValueError,match="arity"):
        _validate_fields(state,(),flux)
    cpp=emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert cpp.count("PreparedStencilInput<")==2 and "stencil_field_" in cpp
    assert cpp.count("capture_stencil_exchanges")==2
    assert cpp.count("publish_stencil_exchanges")==1


def test_native_scalar_stencil_arguments_survive_resolution_without_callbacks():
    from examples.api_v02.methods_T01_T12 import _storage
    from pops.numerics import DiscretizationPlan, StateStorage
    from pops.time import FixedDt
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case,layout,model,block,U,p,q,_=_storage("scalar_stencil_port",("u",),(1.,))
    plan=DiscretizationPlan();plan.states.add(U,StateStorage());case.numerics(plan,block=block)
    flux=StencilFlux(dimension=2,components=1,radius=1,parameters=("duration",),
        body=lambda sample,h,axis,duration: ((duration*sample((-1,0),0),),duration/h[axis]))
    step=p.requested_dt()
    with pytest.raises(ValueError,match="arity"):
        p.stencil_flux_divergence("missing",q.n,flux=flux)
    with pytest.raises(TypeError,match="Float64"):
        p.stencil_flux_divergence("wrong",q.n,flux=flux,parameters=(p.int64(1),))
    value=p.stencil_flux_divergence("parameterized",q.n,flux=flux,parameters=(step,))
    p.commit(q.next,p.value("after",q.n+p.dt*value,at=q.next.point))
    p.step_strategy(FixedDt(.01));case.program(p)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    cpp=emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "stencil_parameter_" in cpp and "capture_stencil_exchanges" in cpp
    assert "publish_stencil_exchanges" in cpp
    assert not hasattr(flux,"body") and flux.to_data()["parameters"]==("duration",)


@pytest.mark.parametrize("parameters",(["h"],("h","h"),("bad name",),(True,)))
def test_scalar_parameter_declaration_is_closed_and_unambiguous(parameters):
    with pytest.raises(ValueError,match="distinct"):
        StencilFlux(dimension=2,components=1,radius=1,parameters=parameters,body=heat)
