"""T12 numerical equations, computed SSA authority, and exact accepted face selection."""
from fractions import Fraction

import numpy as np
import pops
import pytest

from pops.fields import CellMidpoint
from pops.model import PhysicalDimension
from pops.numerics.stencil_flux import StencilFlux
from examples.api_v02.method_flux_correction import build_t12_meshface, reference_step, authored_flux
from examples.api_v02.models_M01_M14 import _domain
from pops.codegen.program_emit_stencil_flux import accepted_stencil_quadrature
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def test_t12_replaces_one_native_face_ledger_and_retains_true_candidate_stage():
    spec=build_t12_meshface()
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    nodes=[v for v in plan.time._values if v.op=="stencil_flux_divergence"]
    assert len(nodes)==2
    high,final=nodes
    sample=final.inputs[1]
    assert sample.attrs["cell_field_projection"]=="pops.cell-midpoint-sampling@1"
    assert sample.inputs[0].name=="provisional_high_state"
    assert sample.point==sample.inputs[0].point!=final.inputs[0].point
    assert any(value is high for value in sample.inputs[0].inputs)
    rows=accepted_stencil_quadrature(plan.time)
    assert len(rows)==1 and rows[0][0] is final and rows[0][1]=={1:Fraction(1)}
    source=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "PreparedStencilInput" in source
    assert source.count("ctx.stage_exchange_batch(")==1
    assert "stencil_parameter_" in source


def test_t12_numpy_reference_detects_both_incidents_and_unilateral_mass_loss():
    initial=np.full((1,4,32),.15);initial[...,6:12]=[.15,.06,.005,1.,.65,.3]
    result=reference_step(initial,dt=.4/32,dx=1/32)
    assert result["provisional"].min()<-.09
    assert result["corrected"].min()>0
    assert np.count_nonzero(result["bad"])==4
    assert np.count_nonzero(result["replaced"])==8
    assert abs(np.mean(result["corrected"])-np.mean(initial))<1.e-14
    assert abs(np.mean(result["unilateral"])-np.mean(initial))>1.e-4
    assert np.max(np.abs(result["one_incident"]-result["corrected"]))>1.e-3
    assert build_t12_meshface().invariants({"method":result["corrected"]})["passed"]


def test_computed_stencil_port_rejects_other_block_and_relabelled_time():
    frame,_=_domain("port_guard",(8,4))
    m=pops.Model("guard",frame=frame)
    U=m.state("q",components=("q",),sampling="cell_average",units=(PhysicalDimension(),))
    case=pops.Case("guard")
    left=case.block("left",m);right=case.block("right",m)
    p=pops.Program("guard");q=p.state(left[U]);r=p.state(right[U]);old=q.n
    unrelated=p.cell_field(r.n,quadrature=CellMidpoint())
    with pytest.raises(ValueError,match="authority"):
        p.stencil_flux_divergence("foreign",old,flux=authored_flux(corrected=True),
                                  fields=(unrelated,),parameters=(p.requested_dt(),))
    valid=p.cell_field(p.value("next_candidate",old,at=q.next.point),quadrature=CellMidpoint())
    fake=p._new("scalar_field",valid.op,valid.inputs,dict(valid.attrs),"forged",valid.block,
                space=valid.space,point=old.point,inherit_state_ref=False)
    with pytest.raises(ValueError,match="same exact evaluation point"):
        p.stencil_flux_divergence("relabelled",old,flux=authored_flux(corrected=True),
                                  fields=(fake,),parameters=(p.requested_dt(),))


def test_computed_port_does_not_admit_an_unproven_scalar_buffer():
    frame,_=_domain("scratch_guard",(8,4))
    m=pops.Model("scratch_guard",frame=frame)
    U=m.state("q",components=("q",),sampling="cell_average",units=(PhysicalDimension(),))
    case=pops.Case("scratch_guard");block=case.block("state",m)
    p=pops.Program("scratch_guard");q=p.state(block[U]);old=q.n
    midpoint=p.cell_field(old,quadrature=CellMidpoint())
    scratch=p.scalar_field(like=midpoint)
    with pytest.raises(ValueError,match="authenticated"):
        p.stencil_flux_divergence("scratch",old,flux=authored_flux(corrected=True),
                                  fields=(scratch,),parameters=(p.requested_dt(),))
