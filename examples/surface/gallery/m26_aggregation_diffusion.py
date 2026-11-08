"""M26 (spec section 17.7) in the proposed surface: nonlocal aggregation-diffusion on a periodic interval.

drho/dt = div(rho grad(psi)) + D lap(rho), with psi = W * rho. The potential psi is a definition that
integrates over a bound point (E3), not a balance term, so the schemes cover the two fluxes once each (D5).
Its scheme sums over all cells (sum_cells, D19); the plan may run that sum as an FFT convolution when it
proves translation invariance (IR15, D19b).
"""

import pops
from pops.math import *
from pops.discrete import CellValue, sum_cells
from pops.lib.numerics import upwind, two_point
from pops.lib.time import ssprk2
from pops.lib.output import hdf5

HALF_LENGTH, CELLS = 5, 256  # each a value chosen for the gallery
DT, T_END = 0.01, 5.0        # each a value chosen for the gallery: DT is below
                             # 1 / (max|u| / h + 2 D / h^2) = 0.0123, with max|u| <= mass * max|W'| = 0.61
                             # and D = 0.05

line = pops.Box(x=pops.Periodic(-HALF_LENGTH, HALF_LENGTH))
x = line.x
y = line.point("y")


# The kernel W: value chosen for the gallery.
def kernel(z):
    """W(z) = -exp(-z^2 / 2) / sqrt(2 pi), an attractive Gaussian well (value chosen for the gallery)."""
    return -exp(-z**2 / 2) / sqrt(2 * pi)


model = pops.Model("aggregation_diffusion")
rho = model.quantity("rho", pops.Scalar(line, admissible=pops.Positive()))
D = model.parameter("D", admissible=pops.Positive())
psi = model.define("psi", integral(kernel(x - y) * rho.at(y), over=y))       # psi = W * rho, a definition (E3)
aggregation = model.flux(of=rho, value=-rho * grad(psi))
diffusion = model.diffusive_flux(of=rho, value=-D * grad(rho))
model.balance(ddt(rho) == -div(aggregation + diffusion))


@pops.scheme
def convolution(cell, term):
    """psi on a cell: the sum over all cells of W(x_cell - x_source) times the mass of the source cell."""
    term.expect(integral(kernel(x - y) * rho.at(y), over=y))     # RealizationMismatch for another key

    def contribution(source):
        return kernel(cell.displacement(source, line.x)) * source.value(rho) * source.measure

    return CellValue(sum_cells(contribution, over=line))


def aggregation_speed(face):
    """The normal speed -dpsi/dn at the face, by the two-point difference of the cell values of psi."""
    return -face.normal_gradient(psi)


numerics = pops.Numerics(
    represent={rho: pops.CellAverage()},
    schemes={psi: convolution(), aggregation: upwind(speed=aggregation_speed), diffusion: two_point()},
)

program = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=DT),
    initial={rho: pops.project(exp(-2 * (x + 1)**2) + exp(-2 * (x - 1)**2),     # value chosen for the gallery:
                               quadrature=pops.GaussLegendre(2))},              # two bumps that merge
)
# The output times are a value chosen for the gallery.
outputs = [pops.Output(pops.record(rho=rho, psi=psi), when=pops.every(time=0.5),
                       write=hdf5("aggregation_diffusion.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={line: pops.Uniform(cells=CELLS)}, outputs=outputs)
artifact = pops.compile(case, target=pops.OpenMP())     # value chosen for the gallery
simulation = pops.bind(artifact, params={D: 0.05})       # value chosen for the gallery
result = simulation.run(until=T_END)
