"""Cahn-Hilliard on a periodic square: a fourth-order equation written as two second-order relations,
advanced by Eyre's convex splitting.

Not in the corpus of spec 0.3. Written to test the physics of a new paper in the proposed surface (G1):
a definition that holds a differential operator, a diffusive flux of a derived variable, and a time
method that reads one part of the right-hand side at the start of the step.
"""

import pops
from pops.math import *
from pops.discrete import CellValue, sum_faces
from pops.claims import Dissipates, Theorem
from pops.lib.numerics import two_point
from pops.lib.solve import newton, gmres
from pops.lib.output import hdf5

square = pops.Box(x=pops.Periodic(0, 1), y=pops.Periodic(0, 1))

# Model: dc/dt = div(M grad(mu)), mu = c^3 - c - kappa lap(c), the variational derivative of the free energy.
model = pops.Model("cahn_hilliard")
c = model.quantity("c", pops.Scalar(square))
kappa = model.parameter("kappa", default=1e-4, admissible=pops.Positive())
mobility = model.constant("mobility", value=1.0)
mu = model.define("mu", c**3 - c - kappa * div(grad(c)))
flux = model.diffusive_flux(of=c, value=-mobility * grad(mu))
model.balance(ddt(c) == -div(flux))
free_energy = model.inventory("free_energy", {c: c**4 / 4 - c**2 / 2 + kappa / 2 * dot(grad(c), grad(c))})

# Numerics: the concave part -c of mu reads c in the role `concave`. A method may bind that role to another
# state; unbound, it reads the same state as the other reads, so the realization is consistent with mu (T18).
concave = pops.role("concave")


def normal_derivative(face):
    return face.normal_gradient(c) * face.measure


@pops.scheme
def split_potential(cell, term):
    """mu on a cell: its convex part at the state read, its concave part at the state bound to `concave`."""
    term.expect(c**3 - c - kappa * div(grad(c)))
    laplacian = sum_faces(cell, normal_derivative) / cell.measure
    return CellValue(cell.value(c)**3 - cell.value(c.role(concave)) - kappa * laplacian)


def interface_energy(face):
    """The cell's half of the gradient energy of a face, kappa/2 |f| d_f ((c_R - c_L) / d_f)^2."""
    return kappa / 4 * face.measure * face.distance * face.normal_gradient(c)**2


@pops.scheme
def energy_density(cell, term):
    """The density of the free energy on a cell, F(c_K) plus half the gradient energy of each of its faces, so
    that the output publishes sum_K |K| F(c_K) + kappa/2 sum_f |f| d_f ((c_R - c_L) / d_f)^2."""
    term.expect({c: c**4 / 4 - c**2 / 2 + kappa / 2 * dot(grad(c), grad(c))})
    bulk = cell.value(c)**4 / 4 - cell.value(c)**2 / 2
    return CellValue(bulk + sum_faces(cell, interface_energy) / cell.measure)


numerics = pops.Numerics(represent={c: pops.CellAverage()},
                         schemes={mu: split_potential(), flux: two_point(), free_energy: energy_density()})


@pops.method
def convex_splitting(step, rate):
    """Eyre: the convex part at the end of the step, the concave part at its start."""
    C = pops.unknown(step.state.space)
    end = pops.solve(C == step.state + step.dt * rate(C, step.time + step.dt, roles={concave: step.state}), C)
    return pops.Candidate(end)


pops.claim(flux, Dissipates(free_energy),
           evidence=Theorem("gradient flow of the free energy, mu its variational derivative",
                            ref="Cahn and Hilliard 1958"))

eyre = convex_splitting(rate=model)
x = square.x
y = square.y
program = pops.Program(
    advance=pops.Advance(eyre, dt=1e-4, tolerance=pops.Tolerance(rtol=1e-8, atol=1e-10)),
    initial={c: pops.project(sin(2 * pi * x) * sin(4 * pi * y) / 20 - cos(6 * pi * y) / 50,
                             quadrature=pops.GaussLegendre(2))},
    solvers={eyre: newton(linear=gmres(restart=50))},
)
outputs = [pops.Output(pops.record(c=c, mu=mu), when=pops.every(time=0.01), write=hdf5("cahn_hilliard.h5")),
           pops.Output(pops.record(free_energy=free_energy), when=pops.every(steps=10),
                       write=hdf5("cahn_hilliard_energy.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={square: pops.Uniform(cells=(256, 256))}, outputs=outputs)
artifact = pops.compile(case, target=pops.CUDA())
simulation = pops.bind(artifact)
result = simulation.run(until=0.5)
