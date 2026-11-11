"""L2 (spec section 17.2) in the proposed surface: advection-Poisson, the M08 guiding centre.

The density is carried by the drift v = J grad(phi) of its own potential, -lap(phi) = rho - nbar, on the
periodic square [0, 2 pi)^2 of the notes [N section 11.2.1]. The background nbar is fixed, not a
time-updated average: the case binds it to the mean of the initial data that it passes. SSPRK2 solves
the field for the state of each stage (T6).
"""

import numpy as np
import pops
from pops.math import *
from pops.lib.math import perp
from pops.lib.numerics import upwind, adjacent_mean, poisson_residual
from pops.lib.time import ssprk2, cfl_limit
from pops.lib.solve import auto
from pops.lib.output import hdf5
from pops.lib.quadrature import GaussLegendre

CELLS = 128                  # value chosen for the gallery
CFL, T_END = 0.4, 10.0       # each a value chosen for the gallery

plane = pops.Box(x=pops.Periodic(0, 2 * pi), y=pops.Periodic(0, 2 * pi))    # 17.2 writes the unit square

model = pops.Model("guiding_centre")
rho = model.quantity("rho", pops.Scalar(plane))
phi = model.quantity("phi", pops.Scalar(plane))
nbar = model.parameter("nbar")               # the fixed background, bound at bind to the mean of the initial data
rho0 = model.data("rho0", pops.Scalar(plane))
poisson = model.constraint(-div(grad(phi)) == rho - nbar, unknowns=phi,
                           normalization=mean(phi, over=plane) == 0, name="poisson")
drift = model.flux(of=rho, value=rho * perp(grad(phi)))
model.balance(ddt(rho) == -div(drift))


def drift_speed(face):
    """The normal drift speed: it needs the tangential derivative of phi, so it reads the full gradient
    from the mean of the two adjacent cell-centred gradients; the two-point gradient gives only the normal one."""
    return dot(face.normal, perp(face.gradient(phi, scheme=adjacent_mean())))


# The load of the constraint is integrated over each cell (cell.load), which needs a quadrature (defaults of
# 2.6.9); the midpoint rule is exact for a load that is affine in the cell average of rho.
numerics = pops.Numerics(
    represent={rho: pops.CellAverage(), phi: pops.CellAverage(quadrature=GaussLegendre(1))},
    schemes={drift: upwind(speed=drift_speed), poisson: poisson_residual()},
)

# cfl_limit reads cell.speed(drift, axis), the largest |speed| that the face fluxes of drift record on the two
# faces of the cell along the axis, here drift_speed. These face speeds sum to zero over each square cell, so
# the step keeps tau sum_f |f| (v_f . n)+ / |K| at most 2 CFL < 1, the condition (152) of the notes.
program = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=cfl_limit(drift, cfl=CFL),
                         tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)),     # value chosen for the gallery
    initial={rho: rho0},
    solvers={poisson: auto()},       # an FFT when the plan proves the symbol of the stencil, else CG (SV11, SV15)
)
# phi at an output time is solved for the state of that time (T8).
# The output times are a value chosen for the gallery.
outputs = [pops.Output(pops.record(rho=rho, phi=phi), when=pops.every(time=1.0), write=hdf5("guiding_centre.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={plane: pops.Uniform(cells=(CELLS, CELLS))}, outputs=outputs)
artifact = pops.compile(case, target=pops.OpenMP())     # value chosen for the gallery

# Exact cell averages of n0 = 1 + 0.2 cos x + 0.1 cos 2y [N(144)] on the cells of the compiled layout, whose
# arrays have the shape of cells= (x first).
geometry = artifact.cells(plane)
x_lower = geometry.lower(plane.x)
x_upper = geometry.upper(plane.x)
y_lower = geometry.lower(plane.y)
y_upper = geometry.upper(plane.y)
rho0_cells = (1 + 0.2 * (np.sin(x_upper) - np.sin(x_lower)) / (x_upper - x_lower)
              + 0.1 * (np.sin(2 * y_upper) - np.sin(2 * y_lower)) / (2 * (y_upper - y_lower)))
background = np.sum(geometry.measure * rho0_cells) / np.sum(geometry.measure)     # the mean of the initial data
simulation = pops.bind(artifact, params={nbar: background}, data={rho0: rho0_cells})
result = simulation.run(until=T_END)
