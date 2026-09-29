"""M18: five-node, three-moment discrete minimum-entropy local solve.

The quadrature and moment basis are data, not a model-specific runtime route.
Only the moderate interior set is expected to have finite dual multipliers.
"""
from __future__ import annotations

import numpy as np
import pops

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments.closures import DiscreteEntropyQuadrature
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual


VELOCITIES = (-1., -.5, 0., .5, 1.)
WEIGHTS = (.1, .2, .4, .2, .1)
BASIS = (tuple(1. for _ in VELOCITIES), VELOCITIES,
         tuple(v*v for v in VELOCITIES))
QUADRATURE = DiscreteEntropyQuadrature(VELOCITIES, WEIGHTS, BASIS)


def moderate_multipliers() -> np.ndarray:
    """Twenty specified interior targets, three multipliers by 4×5 cells."""
    indices = np.arange(20, dtype=float).reshape(4, 5)
    return np.stack((-.18 + .018*indices,
                     .14*np.sin(.23*indices),
                     -.12 + .01*indices))


def target_moments(multipliers: np.ndarray) -> np.ndarray:
    """Input preparation outside the cell loop; run-time equations remain native."""
    if multipliers.shape[0] != QUADRATURE.moment_count:
        raise ValueError("multiplier array does not match the quadrature")
    matrix = np.asarray(BASIS)
    population = np.asarray(WEIGHTS)[:, None, None] * np.exp(
        np.einsum("cn,cxy->nxy", matrix, multipliers))
    return np.einsum("cn,nxy->cxy", matrix, population)


def make_case(*, cells=(4, 5)):
    frame = Rectangle("entropy_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    dual = pops.Model("entropy_dual", frame=frame)
    lam = dual.state("multipliers", components=("lambda_0", "lambda_1", "lambda_2"))
    data = pops.Model("entropy_data", frame=frame)
    moment = data.state("moments", components=("u_0", "u_1", "u_2"))
    case = pops.Case("m18_discrete_entropy")
    dual_block = case.block("dual", dual)
    data_block = case.block("target", data)
    subjects = (dual_block[lam], data_block[moment])
    program = pops.Program("m18_entropy_step")
    dual_state, data_state = (program.state(subject) for subject in subjects)
    seed = program.value("dual_seed", dual_state.n, at=dual_state.next.point)

    def residual(_program, unknowns, *, target):
        return {"dual": QUADRATURE.residual(unknowns["dual"], target)}

    solved = program.solve(LocalResidual(
        residual, {"dual": seed}, captures={"target": data_state.n}),
        solver=LocalNewton(tolerance=2.e-11, max_iterations=12,
                           safeguard="backtracking", max_backtracks=16,
                           minimum_step=2.**-16)).consume(action=FailRun())
    program.commit(dual_state.next, solved[dual_block])
    program.step_strategy(FixedDt(.01))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=cells,
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, subjects
