"""2D advection of a disc on an adaptive mesh: SSPRK2 subcycled over the levels, refinement on the front.

Section 17.11 of spec 0.3 gives the program of T08 without a whole case. Written to test the layout,
the subcycling, the coarse-fine values and the regrid decision in the proposed surface.
"""

import pops
from pops.math import *
from pops.lib.numerics import upwind, muscl, vanleer, minmod, linear_prolongation
from pops.lib.time import ssprk2, subcycle, interpolate, finite, composite_balance
from pops.lib.output import hdf5

CELLS, LEVELS, RATIO = 64, 3, 2
AX, AY = 1, Fraction(1, 2)
DT = Fraction(2, 5) / CELLS / max(AX, AY)    # CFL 0.4 on the coarsest level; each finer level takes RATIO substeps
THRESHOLD = Fraction(1, 20)

domain = pops.Box(x=pops.Periodic(0, 1), y=pops.Periodic(0, 1))
x = domain.x
y = domain.y

model = pops.Model("amr_advection")
u = model.quantity("u", pops.Scalar(domain))
advection = model.flux(of=u, value=vector(AX, AY) * u)
model.balance(ddt(u) == -div(advection))

numerics = pops.Numerics(
    represent={u: pops.CellAverage()},                            # restriction derived: mean of children
    schemes={advection: upwind(reconstruction=muscl(limiter=vanleer))},
)

prolong = linear_prolongation(limiter=minmod)                     # a Transfer from a cell to its children (4.6)


def jump_along(cell, direction):
    ahead = cell.shift(direction, 1)
    behind = cell.shift(direction, -1)
    return abs(ahead.value(u) - behind.value(u))


def largest_jump(cell):
    """The largest difference of u between the two neighbours of the cell along one axis."""
    return largest([jump_along(cell, direction) for direction in domain.axes])


def refine(cell):
    return largest_jump(cell) > THRESHOLD


def coarsen(cell):
    return largest_jump(cell) < THRESHOLD / 4


program = pops.Program(
    advance=pops.Advance(subcycle(method=ssprk2(rate=model), coarse_fine=prolong, in_time=interpolate), dt=DT,
                         accept=[finite(), composite_balance(atol=1e-11)]),     # all of them accept
    initial={u: pops.project(where((x - 0.3)**2 + (y - 0.3)**2 < Fraction(3, 20)**2, 1.0, 0.0),
                             quadrature=pops.GaussLegendre(2))},
    regrid=pops.Regrid(refine=refine, coarsen=coarsen, fill=prolong, when=pops.every(steps=2)),
)

outputs = [pops.Output(pops.record(u=u), when=pops.every(time=0.1), write=hdf5("amr_advection.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={domain: pops.AMR(cells=(CELLS, CELLS), max_levels=LEVELS, ratios=RATIO)},
                 outputs=outputs)
artifact = pops.compile(case, target=pops.CUDA())
simulation = pops.bind(artifact)
result = simulation.run(until=1.0)
