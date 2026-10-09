"""Public authoring shared by source and installed C11 AMR receipts."""
import math
import numpy as np
import pops
from pops.analytic import cos, x, y
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

CELLS, DT = 16, .0002


def principal_amr_case(widths=(1, 1), *, levels=2, reverse=False, singular=False, stages=False):
    count = sum(widths)
    order = tuple(reversed(range(count))) if reverse else tuple(range(count))
    frame = Rectangle("principal_amr_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("principal_amr_physics", frame=frame)
    states = tuple(model.species("row%d" % i, state=tuple("c%d" % j for j in range(width)))
                   for i, width in enumerate(widths))
    parameters = tuple((model.param(RuntimeParam("slope%d" % i, default=.07 + .04*i)),
                        model.param(RuntimeParam("dissipation%d" % i, default=.6 + .1*i)))
                       for i in range(len(widths)))
    gate = model.param(RuntimeParam("gate", default=1.e6)) if singular else None
    q = tuple(component for state in states for component in state)
    row, col = np.indices((count, count))
    matrices = tuple(a[np.ix_(order, order)] for a in (
        np.diag(1. + .1*np.arange(count)) + .025*(row + col + 1),
        np.diag(-.5 - .05*np.arange(count)) - .015*(row + col + 1)))
    speeds = tuple(float(np.max(np.sum(abs(a), axis=1))) for a in matrices)
    starts = np.cumsum((0, *widths))
    fluxes, rates = [], []
    for i, state in enumerate(states):
        flux = model.flux("flux%d" % i, state=state, frame=frame,
            components={axis: tuple(sum(float(a[k,j])*q[j] for j in range(count))
                         for k in range(starts[i], starts[i+1]))
                         for axis, a in zip(frame.axes, matrices)},
            waves=({axis: (speed,)*widths[0] for axis, speed in zip(frame.axes, speeds)}
                   if i == 0 else None))
        fluxes.append(flux)
        rates.append(model.rate("rate%d" % i, equation=ddt(state) == -div(flux)))
    case = pops.Case("principal_amr_case")
    rows = tuple(reversed(range(len(states)))) if reverse else tuple(range(len(states)))
    blocks = {i: case.block("block%d" % i, model, states=(states[i],)) for i in rows}
    transfer = AMRTransfer()
    for i in rows:
        slope, dissipation = (model.value(p) for p in parameters[i])
        gate_value = None if gate is None else model.value(gate)
        def face(left, right, fl, fr, speed, d=dissipation, threshold=gate_value, width=widths[i]):
            normal = .5*(fl+fr)-d*speed*(right-left)
            if threshold is None:
                return normal
            return tuple(where(left[0] > threshold,
                               lambda j=j: 1/(left[j]-left[j]),
                               lambda j=j: normal[j]) for j in range(width))
        method = FiniteVolume(flux=fluxes[i], variables=variables.Conservative(states[i]),
            reconstruction=reconstruction.User(
                lambda sample, slope=slope: sample(0)+slope*(sample(1)-sample(-1)), formal_order=1),
            riemann=riemann.User(body=face, state=states[i],
                stability=lambda left,right,fl,fr,speed,d=dissipation: 2*d*speed),
            sampling=tuple(states[j] for j in rows if j != i))
        plan = DiscretizationPlan()
        plan.rates.add(rates[i], method)
        case.numerics(plan, block=blocks[i])
        profiles = tuple(2.+order[k]+.13*(1+.05*order[k])*cos(2*math.pi*(x(frame)-.5))
                         + .07*(1+.03*order[k])*cos(2*math.pi*(y(frame)-.5)) for k in range(starts[i],starts[i+1]))
        case.initials.add(InitialCondition(state=blocks[i][states[i]],
            value=Analytic(frame=frame, components=profiles), projection=ConservativeCellAverage()))
        transfer.state(blocks[i][states[i]], StateTransfer())
    program = pops.Program("joint_amr_stage")
    temporal = {i: program.state(blocks[i][states[i]]) for i in rows}
    bindings = {states[i]: temporal[i].n for i in rows}
    first = {i: rates[i](temporal[i].n, bindings=bindings) for i in rows}
    if stages:
        point = program.stage("predictor", c=1)
        sampled = {i: program.value("stage%d" % i, temporal[i].n + program.dt*first[i], at=point)
                   for i in rows}
        bindings = {states[i]: sampled[i] for i in rows}
        evaluations = {i: rates[i](sampled[i], bindings=bindings) for i in rows}
    else:
        evaluations = first
    for i in rows:
        expression = (.5*temporal[i].n + .5*sampled[i] + .5*program.dt*evaluations[i]
                      if stages else temporal[i].n+program.dt*evaluations[i])
        result = program.value("accepted%d" % i, expression,
                               at=temporal[i].next.point)
        program.commit(temporal[i].next, result)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    threshold = case.param(RuntimeParam("refine_threshold",default=2.+order[0]+.09))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(CELLS,CELLS), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,)*(levels-1)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(blocks[0][states[0]])[states[0].components[0]] > case.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    handles = tuple(tuple(tuple(blocks[i][p] for p in pair) for pair in parameters)
                    for i in range(len(widths)))
    gates = () if gate is None else tuple(blocks[i][gate] for i in range(len(widths)))
    return case, layout, handles, gates, matrices
