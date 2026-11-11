"""M19 (spec section 17.6) in the proposed surface: Vlasov-Poisson-BGK in 3x3v, Strang splitting."""

import pops
from pops.math import *
from pops.lib.models import characteristic
from pops.lib.models.kinetic import maxwellian
from pops.lib.numerics import upwind, adjacent_mean, pointwise, poisson_residual, ghosts
from pops.lib.numerics.kinetic import discrete_maxwellian
from pops.lib.time import ssprk2, strang

LENGTH, VMAX = 4 * pi, 6.0
DT, T_END = 0.01, 20.0      # each a value chosen for the gallery: at |v_i| = VMAX, the Courant numbers
                            # of the three axes sum to 0.92 over DT, so 0.46 over a transport substep (DT / 2)

space = pops.Box(x=pops.Periodic(0, LENGTH), y=pops.Periodic(0, LENGTH), z=pops.Periodic(0, LENGTH))
velocity = pops.Box(vx=(-VMAX, VMAX), vy=(-VMAX, VMAX), vz=(-VMAX, VMAX))
phase = pops.product(space, velocity)                 # a support of dimension 6 (S6)
x = space.x
v = velocity.position                                 # the velocity vector; read on phase space through the projection

model = pops.Model("vlasov_poisson_bgk")
f = model.quantity("f", pops.Scalar(phase, admissible=pops.Positive()))
phi = model.quantity("phi", pops.Scalar(space))
nu = model.parameter("nu", default=0.1, admissible=pops.Positive())
poisson = model.constraint(-div(grad(phi)) == 1 - integral(f, over=velocity), unknowns=phi,
                           normalization=mean(phi, over=space) == 0, name="poisson")
E = -grad(phi)                                        # a field on space, read on phase space through the projection
vlasov = model.flux(of=f, value={space: v * f, velocity: -E * f})       # one block per factor of the product
equilibrium = model.define("equilibrium", maxwellian(f, over=velocity))
collisions = model.source(of=f, value=nu * (equilibrium - f))
model.balance(ddt(f) == -div(vlasov) + collisions)
model.on(phase.boundary, characteristic(f, incoming=0))       # the faces |v_i| = VMAX; space is periodic


def field_speed(face):
    """The speed of the velocity block on a velocity face, -E.n = grad(phi).n, the axes of the two factors
    paired in order. Both sides lie over one cell of space, so the mean of their centred gradients is the
    centred gradient of phi on that cell."""
    return dot(face.normal, face.gradient(phi, scheme=adjacent_mean()))


# The load of the constraint is integrated over each cell of space (cell.load), which needs a quadrature
# (defaults of 2.6.9); the integral over velocity is the sum over the fibre (sum_fibre), exact for cell averages.
# The discrete Maxwellian has the discrete moments 1, v and |v|^2/2 of f on each fibre. Its solve keeps the
# relative tolerance of 17.6, 1e-13; its absolute tolerance is a value chosen for the gallery.
numerics = pops.Numerics(
    represent={f: pops.CellAverage(), phi: pops.CellAverage(quadrature=pops.GaussLegendre(1))},
    schemes={vlasov: upwind(speed={velocity: field_speed}),     # the space block keeps its derived speed v.n
             equilibrium: discrete_maxwellian(tolerance=pops.Tolerance(rtol=1e-13, atol=1e-15)),
             collisions: pointwise(),
             poisson: poisson_residual()},
    boundary=ghosts(),
)


@pops.method
def bgk_relaxation(step, rate):
    """Exact flow of df/dt = nu (M - f): the moments of f, hence M, do not change during the step, so
    f(t + h) = M + (f - M) exp(-nu h) = f + (1 - exp(-nu h)) / nu * nu (M - f)."""
    k = rate(step.state, step.time)                   # nu (M - f), with the discrete Maxwellian of the state
    return pops.Candidate(step.state + (1 - exp(-nu * step.dt)) / nu * k)


# Each child advance takes its step from strang and the request of the step, so neither states dt=.
transport = pops.Advance(ssprk2(rate=vlasov))
relaxation = pops.Advance(bgk_relaxation(rate=collisions))
# phi is solved for the state of each stage (T6), at the request of the step, which 17.6 does not state.
program = pops.Program(
    advance=pops.Advance(strang(first=transport, second=relaxation), dt=DT,
                         tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)),     # value chosen for the gallery
    initial={f: pops.project((1 + cos(x / 2) / 100) * exp(-dot(v, v) / 2) / sqrt(2 * pi)**3,
                             quadrature=pops.GaussLegendre(2))},
)

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={space: pops.Uniform(cells=(64, 64, 64)), velocity: pops.Uniform(cells=(32, 32, 32))})
artifact = pops.compile(case, target=pops.CUDA(mpi=True))     # value chosen for the gallery
simulation = pops.bind(artifact)
result = simulation.run(until=T_END)
