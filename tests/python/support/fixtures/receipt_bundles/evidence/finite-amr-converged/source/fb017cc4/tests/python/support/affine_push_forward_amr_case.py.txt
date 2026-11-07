"""Public affine-moment body on AMR storage, with a particle data source."""

import itertools
import math

import numpy as np
import pops

from pops.amr import (
    AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer,
    ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
)
from pops.analytic import x
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import affine_push_forward
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual


# A tag band in the interior avoids wrapping the periodic x seam when the
# native hierarchy grows/nests tagged patches. This is a geometry selection;
# the linear particle density and its exact cell-average oracle are unchanged.
CELLS, DT = 16, .01
REFINE_DENSITY_LOWER, REFINE_DENSITY_UPPER = 1.25, 1.35
INDICES = tuple(index for index in itertools.product(range(3), repeat=2)
                if sum(index) <= 2)
PARTICLES = np.asarray(((-.4, .2), (-.1, -.3), (.25, .45), (.6, -.15)), dtype=float)
WEIGHTS = np.asarray((.2, .3, .4, .1), dtype=float)
MATRIX = np.asarray(((math.cos(.2), math.sin(.2)),
                     (-math.sin(.2), math.cos(.2))), dtype=float)
OFFSET = np.asarray((.1, -.07), dtype=float)


def raw_particle_moments():
    return np.asarray([np.dot(WEIGHTS, np.prod(PARTICLES ** index, axis=1))
                       for index in INDICES])


def mapped_particle_moments():
    moved = PARTICLES @ MATRIX.T + OFFSET
    return np.asarray([np.dot(WEIGHTS, np.prod(moved**index, axis=1))
                       for index in INDICES])


def expected_level_cell_averages(moment_vector, level):
    width = CELLS * 2**level
    density = 1.0 + .6 * ((np.arange(width) + .5) / width)
    return np.broadcast_to(moment_vector[:, None, None] * density[None, None, :],
                           (len(moment_vector), width, width))


def affine_amr_case(*, levels=1, case_name="affine_amr_public_body",
                    impossible_residual=False):
    frame = Rectangle("affine_amr_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("affine_amr_measure", frame=frame)
    state = model.state("U", components=tuple("m%d_%d" % index for index in INDICES))
    case = pops.Case(case_name)
    block = case.block("moments", model)
    subject = block[state]
    program = pops.Program("affine_amr_local_solve")
    q = program.state(subject)

    def image(value):
        return affine_push_forward(value, indices=INDICES, matrix=tuple(map(tuple, MATRIX)),
                                   offset=tuple(OFFSET))

    target = program.value("mapped_stage", image(q.n), at=q.next.point)
    seed = program.value("distinct_seed", 1.1*q.n, at=q.next.point)

    def residual(_program, unknown, target):
        mapped = image(unknown)
        rows = tuple(mapped[k]-target[k] for k in range(len(INDICES)))
        if impossible_residual:
            # A separate negative problem has no zero of its original first row.
            return ((rows[0] * 0.0 + 1.0), *rows[1:])
        return rows

    solved = program.solve(LocalResidual(residual, seed, captures={"target": target}),
                           solver=LocalNewton(tolerance=1.e-12, max_iterations=8)).consume(
                               action=FailRun())
    program.commit(q.next, program.value("accepted_image", image(solved), at=q.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)

    density = 1. + .6*x(frame)
    case.initials.add(InitialCondition(
        state=subject,
        value=Analytic(frame=frame, components=tuple(float(raw)*density
                                                      for raw in raw_particle_moments())),
        projection=ConservativeCellAverage(),
    ))
    lower = case.param(RuntimeParam("refine_density_lower", default=REFINE_DENSITY_LOWER))
    upper = case.param(RuntimeParam("refine_density_upper", default=REFINE_DENSITY_UPPER))
    indicator = ValueExpr(subject)[state.components[0]]
    transfer = AMRTransfer()
    transfer.state(subject, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,)*(levels-1)),
        tagging=AMRTagging(
            rules=(Tag((indicator > case.value(lower)) & ~(indicator > case.value(upper))),
                   Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous(),
    )
    return case, layout, subject
