"""Public nondimensional kinetic transport on the Cartesian product x × v.

Physics is declared at module scope. Discretization, time stepping, quadrature,
realization and acceptance are separate choices, not properties of the PDE.
"""
from fractions import Fraction
import math

import pops
from pops.analytic import coordinate, cos
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import NoFlux
from pops.domain import Rectangle
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.math import ddt, div
from pops.mesh import AxisQuadrature, CartesianGrid, PeriodicAxes
from pops.model import PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt


# The existing generic product chart uses native axis 0=v, axis 1=x. The
# physical support retains those names; coordinate aliases are not inferred.
FRAME = Rectangle("kinetic product v times periodic x", (-1., 0.), (1., 1.)).frame(Cartesian2D())
V_AXIS, X_AXIS = FRAME.axes
SUPPORT = PhysicalSupport((("velocity", "[-1,1]"), ("position", "periodic-[0,1]")))
UNIT = PhysicalDimension()  # Explicit nondimensional control.
MODEL = pops.Model("linear kinetic freestreaming", frame=FRAME)
POPULATION = MODEL.state("f", components=("population",), support=SUPPORT,
                         units=(UNIT,), sampling="cell_average")
VELOCITY = MODEL.auxiliary("velocity_coordinate", frame=FRAME.canonical_id, unit="1")
FLUX = MODEL.flux("kinetic_transport", frame=FRAME, state=POPULATION,
                  components={V_AXIS: (0 * POPULATION[0],), X_AXIS: (VELOCITY * POPULATION[0],)},
                  waves={V_AXIS: (0 * POPULATION[0],), X_AXIS: (VELOCITY,)})
MODEL.wave_speeds(FLUX, frame=FRAME, values={V_AXIS: (0 * POPULATION[0], 0 * POPULATION[0]),
                                          X_AXIS: (VELOCITY, VELOCITY)})
RATE = MODEL.rate("freestreaming", equation=ddt(POPULATION) == -div(FLUX))
MODULE = MODEL.module
VELOCITY_HANDLE = MODULE.aux_handle(MODULE.aux()["velocity_coordinate"])
MODULE.aux_provider(AnalyticAux(VELOCITY_HANDLE, coordinate(FRAME, V_AXIS), frame=FRAME,
                               boundary=AuxiliaryBoundary(width=1, kind="foextrap")))

FINAL_TIME = Fraction(1, 8)
CASES = ((32, 8), (64, 12))


def build(nx, nv):
    if type(nx) is not int or type(nv) is not int or nx < 2 or nv < 2:
        raise ValueError("kinetic transport requires positive spatial and velocity cell counts >=2")
    case = pops.Case("M19 kinetic freestreaming %d x %d" % (nx, nv))
    block = case.block("kinetic", MODEL)
    numerics = DiscretizationPlan()
    numerics.rates.add(RATE, FiniteVolume(flux=FLUX, variables=variables.Conservative(POPULATION),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.HLL(waves=riemann.waves.ExplicitPair())))
    numerics.boundaries.add(TransportBoundarySet({
        FRAME.boundaries.x_min: NoFlux(state=block[POPULATION]),
        FRAME.boundaries.x_max: NoFlux(state=block[POPULATION]),
    }, periodic=PeriodicAxes((X_AXIS,))))
    case.numerics(numerics, block=block)
    program = pops.Program("SSPRK2 kinetic transport")
    f = program.state(block[POPULATION])
    stage = program.stage("forward Euler stage", c=Fraction(1))
    predictor = program.value("kinetic predictor", f.n + program.dt * RATE(f.n, program.input_fields(f.n, for_rate=RATE)), at=stage)
    accepted = program.value("kinetic accepted", Fraction(1, 2) * f.n
                             + Fraction(1, 2) * (predictor + program.dt * RATE(predictor, program.input_fields(predictor, for_rate=RATE))), at=f.next.point)
    program.commit(f.next, accepted)
    # max |v| < 1, so this exact binary dt gives Courant <=1/4 and exactly
    # nx/2 steps to FINAL_TIME. No endpoint clipping changes the issued dt.
    dt = Fraction(1, 4 * nx)
    program.step_strategy(FixedDt(float(dt)))
    case.program(program)
    x, v = coordinate(FRAME, X_AXIS), coordinate(FRAME, V_AXIS)
    case.initials.add(InitialCondition(state=block[POPULATION],
        value=Analytic(frame=FRAME, components=((1 + .1 * cos(2 * math.pi * x)) * (1 + .25 * v),)),
        projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=FRAME, cells=(nv, nx), periodic=PeriodicAxes((X_AXIS,))))
    # This is a diagnostic velocity quadrature, not the transport coefficient.
    quadrature = AxisQuadrature(0, -1, 1, nv, UNIT)
    return case, layout, quadrature, dt
