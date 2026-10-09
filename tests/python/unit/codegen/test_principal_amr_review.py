"""Independent source checks for complete C11 AMR face/reflux realization.

No native build: the AMR emitter must retain one joint face evaluation and an
exact projected face basis for every physical row, including cache hits.
"""
from __future__ import annotations

import re

import numpy as np
import pops
import pytest

from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
                      AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                      Hysteresis, Tag)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.math import ValueExpr
from pops.params import RuntimeParam
from pops.time import every
from tests.python.integration.runtime.test_principal_group_runtime import _problem


def _matrix(n):
    return np.asarray([[1.+.1*i if i == j else .03*(i+j+1)
                        for j in range(n)] for i in range(n)])


@pytest.mark.parametrize("n", (2, 3, 5))
@pytest.mark.parametrize("reverse", (False, True))
def test_amr_emission_projects_every_row_of_one_joint_face_evaluation(n, reverse):
    permutation = tuple(reversed(range(n))) if reverse else tuple(range(n))
    matrix = _matrix(n)[np.ix_(permutation, permutation)]
    case, layout, _, _, _ = _problem(matrix, vector=False, permutation=permutation)
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(plan.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks),
        target="amr_system")
    calls = re.findall(r"\b\w+_evaluate\(ctx,\d+,\w+_inputs,", source)
    attach = re.findall(r"ctx\.attach_principal_flux\([^;]+;", source)
    assert len(calls) == 1, "one joint face evaluation must serve all physical rows"
    assert len(attach) == n, "every cache-hit row must publish its own face basis"
    assert len({re.search(r"attach_principal_flux\((\d+),", item).group(1)
                for item in attach}) == n
    offsets = sorted(int(match.group(1)) for item in attach
                     if (match := re.search(r"\.component_faces\((\d+),1\)", item)))
    assert offsets == list(range(n)), "face projection must follow exact packed component order"
    assert "prepare_principal_inputs" in source
    assert "pack_prepared" in source and "evaluate_prepared" in source


@pytest.mark.parametrize("n", (2, 3, 5))
def test_public_synchronous_amr_layout_retains_all_principal_blocks(n):
    permutation = tuple(range(n))
    case, uniform, blocks, states, _ = _problem(
        _matrix(n), vector=False, permutation=permutation)
    transfer = AMRTransfer()
    for block, state in zip(blocks, states, strict=True):
        transfer.state(block[state], StateTransfer())
    program = case._time_registry.program  # only to recover the authored schedule clock
    threshold = case.param(RuntimeParam("principal_refine_threshold", default=1.))
    layout = AMR(
        grid=uniform.mesh,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(blocks[0][states[0]]) > case.value(threshold)),
                   Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer, execution=AMRExecution.synchronous())
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert len(plan.blocks) == n
    source = emit_cpp_program(plan.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks),
        target="amr_system")
    assert source.count("ctx.attach_principal_flux(") == n


def test_late_initial_row_projection_restores_its_exact_stage_after_midpoint():
    """The cached initial face must attach at c=0 after a midpoint changed ctx."""
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.numerics.variables import Conservative
    from pops.time import FixedDt

    frame = Rectangle("principal_stage_interleave", (0.,0.), (1.,1.)).frame(Cartesian2D())
    model = pops.Model("principal_stage_model", frame=frame)
    first = model.species("first", state=("a",))
    second = model.species("second", state=("b",))
    a, b = first[0], second[0]
    fluxes = (model.flux("first_flux", state=first, frame=frame,
                components={frame.x:(a+.25*b,),frame.y:(0*a,)},
                waves={frame.x:(2.,),frame.y:(0.,)}),
              model.flux("second_flux", state=second, frame=frame,
                components={frame.x:(.25*a+b,),frame.y:(0*b,)},
                waves={frame.x:(2.,),frame.y:(0.,)}))
    rates = (model.rate("first_rate", equation=ddt(first)==-div(fluxes[0])),
             model.rate("second_rate", equation=ddt(second)==-div(fluxes[1])))
    case = pops.Case("principal_stage_interleave_case")
    blocks = (case.block("first_block", model, states=(first,)),
              case.block("second_block", model, states=(second,)))
    for block,state,flux,rate,other in zip(blocks,(first,second),fluxes,rates,
                                            (second,first),strict=True):
        plan = DiscretizationPlan()
        plan.rates.add(rate,FiniteVolume(flux=flux,variables=Conservative(state),
                       reconstruction=FirstOrder(),riemann=Rusanov(),sampling=(other,)))
        case.numerics(plan,block=block)
    program = pops.Program("interleaved_principal_stages")
    t0,t1 = (program.state(block[state]) for block,state in zip(blocks,(first,second)))
    initial = {first:t0.n,second:t1.n}
    first_row = rates[0](t0.n,bindings=initial)
    midpoint = program.stage("midpoint",c=.5)
    mid0 = program.value("mid_first",t0.n+.5*program.dt*first_row,at=midpoint)
    mid1 = program.value("mid_second",1*t1.n,at=midpoint)
    mid_bindings = {first:mid0,second:mid1}
    mid_rate0 = rates[0](mid0,bindings=mid_bindings)
    mid_rate1 = rates[1](mid1,bindings=mid_bindings)
    late_initial_row = rates[1](t1.n,bindings=initial)
    program.commit(t0.next,program.value("next_first",.5*t0.n+.5*mid0+.5*program.dt*mid_rate0,
                                         at=t0.next.point))
    program.commit(t1.next,program.value("next_second",
                                         .5*t1.n+.5*mid1+.5*program.dt*(mid_rate1+late_initial_row),
                                         at=t1.next.point))
    program.step_strategy(FixedDt(.001))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame,cells=(8,8),
                                   periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case),layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
        target="amr_system")
    assert len(re.findall(r"\b\w+_evaluate\(ctx,\d+,\w+_inputs,",source)) == 2
    attachments = list(re.finditer(r"ctx\.attach_principal_flux\([^;]+;",source))
    assert len(attachments) == 4  # first row c=0, two midpoint rows, late c=0 row
    late = attachments[-1].start()
    assert source.rfind("ctx.set_stage_time(0,1);",0,late) > source.rfind(
        "ctx.set_stage_time(1,2);",0,late), "cache-hit projection retained midpoint stage"
