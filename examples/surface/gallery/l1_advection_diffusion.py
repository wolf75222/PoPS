"""L1 (spec section 17.1) in the proposed surface: advection-diffusion, SSPRK2, then IMEX Euler with CG."""

import pops
from pops.math import *
from pops.claims import Preserves, Courant, Theorem
from pops.lib.numerics import upwind, two_point
from pops.lib.time import ssprk2, imex_euler
from pops.lib.solve import cg

# Model: du/dt + div(a u - nu grad u) = 0 on a periodic interval.
domain = pops.Box(x=pops.Periodic(0, 1))

model = pops.Model("advection_diffusion")
u = model.quantity("u", pops.Scalar(domain))
a = model.constant("a", value=1.0)
nu = model.parameter("nu", default=1e-2, admissible=pops.Positive())
advection = model.flux(of=u, value=a * u)                       # on the interval, the flux is a scalar
diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))     # Fick's flux; its gradient variable u is derived
transport = model.balance(ddt(u) == -div(advection + diffusion))

numerics = pops.Numerics(
    represent={u: pops.CellAverage()},
    schemes={advection: upwind(), diffusion: two_point()},
)


def combined_limit(cell):
    """The combined restriction c_a + 2 r_d <= 1 of the forward Euler step, cell by cell [N(141)]."""
    h = cell.width(domain.x)
    return 1 / (abs(a) / h + 2 * nu / h**2)


def advective_limit(cell):
    """The restriction c_a <= 1 alone: the implicit diffusion adds none."""
    return cell.width(domain.x) / abs(a)


# The subject is the realization of the balance (17.1 claims it on bal_h).
pops.claim(numerics.of(transport), Preserves(pops.Positive()), assumptions=Courant(combined_limit),
           evidence=Theorem("convex combination of Euler steps", ref="[N§10.2]"))

# Exact cell averages: the jumps of the initial state, at 0.25 and 0.75, sit on faces of the 256 cells.
x = domain.x
initial = {u: pops.project(where(abs(x - 0.5) < 0.25, 1.0, 0.0), quadrature=pops.GaussLegendre(2))}

# Program A: SSPRK2 under the combined restriction.
program_a = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=combined_limit),
    initial=initial,
)

# Program B: advection explicit, diffusion implicit, the implicit stage solved by CG.
imex = imex_euler(explicit=advection, implicit=diffusion)
program_b = pops.Program(
    advance=pops.Advance(imex, dt=advective_limit,
                         tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)),    # value chosen for the gallery
    initial=initial,
    solvers={imex: cg()},
)

case_a = pops.Case(model=model, numerics=numerics, program=program_a,
                   layout={domain: pops.Uniform(cells=256)})
artifact = pops.compile(case_a, target=pops.OpenMP())
simulation = pops.bind(artifact, params={nu: 1e-2})
result = simulation.run(until=1.0, max_steps=10_000)
u_final = result.state[u]                         # shape (256,): the accepted cell averages (CC14)
