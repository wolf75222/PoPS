"""Source-only independent storage-owner witnesses; no native execution/JIT.

The physical coupled field is global and independent of the chosen history
storage block. These baseline probes do not qualify the future owner_block API.
"""
from __future__ import annotations


import pops
from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.math import ddt, div, laplacian
from pops.model import Handle, OwnerPath
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.solvers import CG
from pops.time import FailRun


def authored(*, nonlinear=False, endpoint=False):
    frame = Rectangle("storage-domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    case = pops.Case("storage-case")
    program = pops.Program("global-coupled-field")
    blocks, states, inputs = [], [], {}
    for name in ("thermal", "matter", "spectator"):
        model = pops.Model("physical-"+name, frame=frame)
        state = model.state("U", components=("load", "independent-component"))
        flux = model.flux("stationary", frame=frame, state=state,
            components={axis: tuple(0*x for x in state) for axis in frame.axes},
            waves={axis: tuple(0*x for x in state) for axis in frame.axes})
        rate = model.rate("balance", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerics, block=block)
        current = program.state(block[state])
        blocks.append(block)
        states.append(state)
        inputs[block[state]] = current.n
    unknowns = tuple(Handle(name, kind="field", owner=OwnerPath.model("global-equations"))
                     for name in ("T", "z", "w"))
    from pops._ir.elliptic import Reaction
    equations = []
    for i, (q, state) in enumerate(zip(unknowns, states, strict=True)):
        lhs = -laplacian(q)+Reaction(q, 2)
        for j, other in enumerate(unknowns):
            if i != j:
                lhs -= Reaction(other, .25)
        equations.append(lhs == state[0])
    problem = FieldProblem("original-global", unknowns=unknowns, equations=tuple(equations),
        boundaries=tuple(FieldBoundary(q, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for q in unknowns))
    if nonlinear:
        from pops.fields import CellCenteredNonlinearCoupled
        from pops.solvers import Newton
        solver = Newton()
        method = CellCenteredNonlinearCoupled(finite_difference_step=1e-7)
    else:
        solver = CG(max_iter=200)
        method = CellCenteredSecondOrder()
    field = case.field(problem, FieldDiscretization(method=method, boundaries=(), solver=solver))
    point = (program.state(blocks[0][states[0]]).next.point if endpoint
             else program.stage("same-physical-point", c=0))
    if nonlinear:
        request = field.bind_program_inputs(program=program, values=inputs, at=point, solver=solver)
        consumed = program.solve(request, solver=solver).consume(action=FailRun())
    else:
        consumed = program.solve(field, values=inputs, at=point).consume(action=FailRun())
    observation = field.observe(consumed)
    return case, program, blocks, states, problem, observation, field, point
