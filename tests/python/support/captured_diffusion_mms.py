"""Test-only original equations and independent arithmetic-face MMS.

No PoPS field action, emitted kernel or solver is used to manufacture the load.
"""
from __future__ import annotations

import numpy as np
import pops

from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.domain import Rectangle
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import SourceTerm
from pops.solvers import Newton
from pops.time import FailRun, FixedDt

DT = .01
HISTORY_MAX_LAG = 1
FD_STEP = 1e-6
CONTROLS = dict(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                linear_max_iterations=240, restart=60, armijo=1e-4,
                minimum_step=1/1024)
# Declared MMS tolerances: the nonlinear relative stopping criterion is 1e-10.
# The independent double FV evaluation permits its accumulation roundoff and
# an error up to 300 times that tolerance, not a discretization error fit.
SOLUTION_TOL = 3e-8
RESIDUAL_TOL = 3e-8
CANDIDATE_BETA = 3.


def matrices(width):
    if width == 1:
        return np.array([[.012]]), np.array([[1.1]])
    if width == 3:
        # Signed/nonsymmetric and singular D; the complete local reaction closes
        # the third equation. This makes no SPD certificate or general solvability
        # claim for other finite matrices.
        return (np.array([[.012, .002, 0.], [-.001, .014, 0.], [.001, 0., 0.]]),
                np.array([[1.1, .03, -.01], [-.02, 1.3, .02], [.01, -.03, 1.5]]))
    raise ValueError("this manufactured witness declares one or three fields")


def original_action(q, alpha, *, candidate_diffusion=False):
    """Independent periodic Cartesian FV -div(D grad q)+R q+.2 q^3."""
    width, ny, nx = q.shape
    diffusion, reaction = matrices(width)
    cell_d = diffusion[:, :, None, None] * (1 + alpha[None, None])
    if candidate_diffusion:
        cell_d = cell_d * (1 + CANDIDATE_BETA*q[None]**2)
    spatial = np.zeros_like(q)
    for axis, cells in ((1, ny), (2, nx)):
        # q is (component,y,x), D is (row,column,y,x).
        d_axis = axis + 1
        high_d = .5 * cell_d + .5 * np.roll(cell_d, -1, d_axis)
        low_d = .5 * np.roll(cell_d, 1, d_axis) + .5 * cell_d
        high_gradient = np.roll(q, -1, axis) - q
        low_gradient = q - np.roll(q, 1, axis)
        for row in range(width):
            spatial[row] -= cells**2 * sum(
                high_d[row, column] * high_gradient[column] -
                low_d[row, column] * low_gradient[column] for column in range(width))
    local = np.einsum("ij,jyx->iyx", reaction, q) + .2 * q**3
    return spatial + local, spatial


def initial_data(cells, width, *, candidate_diffusion=False):
    coordinate = (np.arange(cells) + .5)/cells
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    alpha = .25*np.sin(2*np.pi*x) + .15*np.cos(2*np.pi*y)
    target = np.stack(tuple(.15 + .025*np.cos(2*np.pi*(index+1)*x)
                           + .02*np.sin(2*np.pi*y) for index in range(width)))
    forcing, spatial = original_action(target, alpha, candidate_diffusion=candidate_diffusion)
    assert np.ptp(alpha) > .5 and all(np.ptp(row) > .07 for row in target)
    assert np.max(np.abs(spatial)) > .01, "MMS must exercise diffusion"
    return ({"response": np.zeros_like(target), "forcing": np.ascontiguousarray(forcing),
             "material": np.ascontiguousarray(alpha[None])}, target, spatial)


def build(cells, width, order, *, omit_material=False, candidate_diffusion=False):
    if tuple(sorted(order)) != tuple(range(width)):
        raise ValueError("MMS order must be an exact field permutation")
    diffusion, reaction = matrices(width)
    frame = Rectangle("captured-diffusion-domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    fluid = pops.Model("field-consumer", frame=frame)
    response = fluid.state("response", components=tuple("u%d" % i for i in range(width)))
    unknowns = tuple(fluid.field("q%d" % i) for i in range(width))
    auxiliaries = tuple(fluid.aux("observed%d" % i) for i in range(width))
    source = fluid.source("actual_field_response", on=response, value=auxiliaries)
    load_model = pops.Model("load-capture", frame=frame)
    load = load_model.state("forcing", components=tuple("f%d" % i for i in range(width)))
    material_model = pops.Model("material-capture", frame=frame)
    material = material_model.state("material", components=("alpha",))

    # Original physical equations precede method/Program selection.
    equations = []
    for row in range(width):
        lhs = Reaction(unknowns[row], .2*ValueExpr(unknowns[row])**2)
        for column in range(width):
            if reaction[row, column] != 0:
                lhs += Reaction(unknowns[column], float(reaction[row, column]))
            if diffusion[row, column] != 0:
                coefficient = float(diffusion[row, column])*(1+material[0])
                if candidate_diffusion:
                    coefficient *= 1+CANDIDATE_BETA*ValueExpr(unknowns[column])**2
                lhs -= DivCoeffGrad(unknowns[column], coefficient)
        equations.append(lhs == load[row])
    problem = FieldProblem("captured-D-original-equations", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order), boundaries=tuple(FieldBoundary(unknowns[i],
            bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for i in order))
    case = pops.Case("captured-diffusion-MMS")
    blocks = []
    for name, model, state in (("response", fluid, response), ("forcing", load_model, load),
                               ("material", material_model, material)):
        flux = model.flux("stationary-flux", frame=frame, state=state,
            components={axis: tuple(0*q for q in state) for axis in frame.axes},
            waves={axis: tuple(0*q for q in state) for axis in frame.axes})
        rate = model.rate("balance", equation=ddt(state) == (-div(flux)+source if model is fluid else -div(flux)))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerics, block=block)
        blocks.append(block)
    solver = Newton(**CONTROLS)
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=FD_STEP, face_policy="Arithmetic@1",
        coefficient_evaluation="PerCandidate@1" if candidate_diffusion else None),
        boundaries=(), solver=solver))
    program = pops.Program("captured-diffusion-accepted-response")
    current, forcing, coefficient = (program.state(block[state]) for block, state in
        zip(blocks, (response, load, material), strict=True))
    captures = {blocks[1][load]: forcing.n}
    if not omit_material:
        captures[blocks[2][material]] = coefficient.n
    request = field.bind_program_inputs(program=program, values=captures,
        at=program.stage("frozen-original-coefficients", c=0), solver=solver)
    outcome = program.solve(request, solver=solver)
    observed = field.observe(outcome.consume(action=FailRun()))
    outputs = tuple(observed[field[unknown]] for unknown in unknowns)
    for index, output in enumerate(outputs):
        program.store_history("q%d" % index, output, depth=HISTORY_MAX_LAG)
    module = fluid.module
    carrier = blocks[0][module.field_handle(module.field_spaces()["fields"])]
    publication = observed.publish({(carrier, "observed%d" % i): output for i, output in enumerate(outputs)},
                                   states={blocks[0][response]: current.n})
    rhs = program.rhs(state=current.n, fields=publication,
        terms=[SourceTerm(blocks[0][module.operator_handle("actual_field_response")])])
    program.commit(current.next, program.value("response-update", current.n+program.dt*rhs, at=current.next.point))
    for time in (forcing, coefficient):
        program.commit(time.next, program.value("preserved-"+time.n.name, 1*time.n, at=time.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, outcome._token
