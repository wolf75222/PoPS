"""L4 (spec sections 3.11 and 17.4) in the proposed surface: two fluids, drag and frictional heating, exact substep."""

import pops
from pops.math import *
from pops.claims import Conserves
from pops.lib.numerics import rusanov, pointwise
from pops.lib.time import ssprk2, strang

GAMMA = Fraction(7, 5)
DT, T_END = 1e-3, 0.5            # each a value chosen for the gallery: a Courant number below 0.3
                                 # for the initial states

domain = pops.Box(x=pops.Periodic(0, 1))


def euler_flux(U, gamma):
    """F(U) = (m, m u + p, (E + p) u), field by field: a Python function that returns expressions (E4)."""
    velocity = U.m / U.rho
    p = (gamma - 1) * (U.E - dot(U.m, U.m) / (2 * U.rho))
    return pops.record(rho=U.m, m=outer(U.m, velocity) + p * identity(domain), E=(U.E + p) * velocity)


model = pops.Model("two_fluid_drag")
fluid = pops.Components(domain, rho=pops.Scalar(admissible=pops.Positive()), m=pops.Vector(), E=pops.Scalar())
Ua = model.quantity("Ua", fluid)
Ub = model.quantity("Ub", fluid)
K = model.constant("K", value=5.0)
theta = model.constant("theta", value=0.5)
flux_a = model.flux(of=Ua, value=euler_flux(Ua, GAMMA))
flux_b = model.flux(of=Ub, value=euler_flux(Ub, GAMMA))

va = Ua.m / Ua.rho
vb = Ub.m / Ub.rho
force = K * (vb - va)                                  # on fluid a; fluid b receives the opposite
heat_a = theta * K * dot(vb - va, vb - va)
heat_b = (1 - theta) * K * dot(vb - va, vb - va)
drag = model.interaction(deliver={Ua: pops.record(m=force, E=dot(va, force) + heat_a),      # no rho: zero
                                  Ub: pops.record(m=-force, E=-dot(vb, force) + heat_b)})
model.balance(ddt(Ua) == -div(flux_a) + drag[Ua])
model.balance(ddt(Ub) == -div(flux_b) + drag[Ub])

momentum = model.inventory("momentum", {Ua: Ua.m, Ub: Ub.m})
energy = model.inventory("energy", {Ua: Ua.E, Ub: Ub.E})

numerics = pops.Numerics(
    represent={Ua: pops.CellAverage(), Ub: pops.CellAverage()},
    schemes={flux_a: rusanov(), flux_b: rusanov(), drag: pointwise()},
)


@pops.method
def drag_substep(step):
    """Exact solution of the drag subproblem with rho_a, rho_b, K and theta constant on the step [N§11.4.3]."""
    a = step.state[Ua]
    b = step.state[Ub]
    mu = a.rho * b.rho / (a.rho + b.rho)
    total = a.m + b.m
    w0 = b.m / b.rho - a.m / a.rho
    w = w0 * exp(-K * step.dt / mu)                    # relative velocity, dw/dt = -K w / mu
    lost = mu / 2 * (dot(w0, w0) - dot(w, w))          # kinetic energy turned into heat
    ua = (total - b.rho * w) / (a.rho + b.rho)
    ub = (total + a.rho * w) / (a.rho + b.rho)
    ea = a.E - dot(a.m, a.m) / (2 * a.rho)
    eb = b.E - dot(b.m, b.m) / (2 * b.rho)
    end_a = pops.record(rho=a.rho, m=a.rho * ua, E=ea + theta * lost + a.rho * dot(ua, ua) / 2)
    end_b = pops.record(rho=b.rho, m=b.rho * ub, E=eb + (1 - theta) * lost + b.rho * dot(ub, ub) / 2)
    return pops.Candidate({Ua: end_a, Ub: end_b})


exact_drag = drag_substep()                            # a claim names a value: the method that the program uses
pops.claim(drag, Conserves(momentum, energy))           # evidence derived: ExactIdentity (R22)
pops.claim(exact_drag, Conserves(momentum, energy))     # evidence derived

# Each child advance takes the step that strang gives it, so neither states dt=.
transport = pops.Advance(ssprk2(rate=[flux_a, flux_b]))
friction = pops.Advance(pops.exact_flow(exact_drag, of=drag))
program = pops.Program(
    advance=pops.Advance(strang(first=transport, second=friction), dt=DT),
    initial={Ua: pops.record(rho=1.0, m=0.1, E=2.5), Ub: pops.record(rho=0.5, m=-0.1, E=1.25)},
)

case = pops.Case(model=model, numerics=numerics, program=program, layout={domain: pops.Uniform(cells=200)})
artifact = pops.compile(case, target=pops.OpenMP())               # value chosen for the gallery
simulation = pops.bind(artifact)
result = simulation.run(until=T_END)
