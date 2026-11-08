"""M10 (spec section 17.5) in the proposed surface: drift-diffusion-Poisson, Scharfetter-Gummel, joint Newton."""

import pops
from pops.math import *
from pops.lib.numerics import scharfetter_gummel, poisson_residual, ghosts
from pops.lib.solve import newton, gmres, cg, bicgstab, block_elimination, multigrid, v_cycle, line_search

L_DEVICE, N_D = 1, 1
DT, T_END = 1e-3, 1.0

device = pops.Box(x=(0, L_DEVICE))
contacts = [device.boundary.x_min, device.boundary.x_max]
x = device.x

model = pops.Model("drift_diffusion")
n = model.quantity("n", pops.Scalar(device, admissible=pops.Positive()))
psi = model.quantity("psi", pops.Scalar(device))
eps = model.constant("eps", value=1.0)
charge = model.constant("charge", value=1.0)
D = model.constant("D", value=1.0)
V_contact = model.parameter("V_contact", default=0.5)
n_contact = model.parameter("n_contact", default=1.0)
doping = model.define("doping", N_D * where(x < Fraction(L_DEVICE, 2), 1, 0))      # an analytic datum is a definition

poisson = model.constraint(-div(eps * grad(psi)) == charge * (n - doping), unknowns=psi, name="poisson")
drift = model.flux(of=n, value=-D * n * grad(psi))           # a positive carrier, thermal-voltage units (D8)
diffusion = model.diffusive_flux(of=n, value=-D * grad(n))   # Fick's flux; grad(psi) of the drift is an input (E2)
model.balance(ddt(n) == -div(drift + diffusion))
model.on(contacts, psi == V_contact)                         # reads psi only: it closes the constraint poisson (R10)
model.on(contacts, n == n_contact)                           # reads n only: it closes the balance of n

# The load of the constraint is integrated over each cell (cell.load), which needs a quadrature (defaults of
# 2.6.9). The midpoint rule is exact here: n is a cell average and the doping jumps on a face.
numerics = pops.Numerics(
    represent={n: pops.CellAverage(), psi: pops.CellAverage(quadrature=pops.GaussLegendre(1))},
    schemes={pops.jointly(drift, diffusion): scharfetter_gummel(),       # D8: one joint flux
             poisson: poisson_residual()},
    boundary=ghosts(),
)


@pops.method
def coupled(step, rate, constraint):
    """Backward Euler for the state, solved jointly with the field constraint at the new state."""
    end = step.time + step.dt
    U = pops.unknown(step.state.space)
    Phi = pops.unknown(constraint.space)                     # the space of psi, the unknown of the constraint
    stage = U == step.state + step.dt * rate(U, end, fields={constraint: Phi})
    solution = pops.solve([stage, constraint(U, end, Phi)], [U, Phi])     # the constraint as its discrete relation
    return pops.Candidate(solution[U])


joint = coupled(rate=model, constraint=poisson)
program = pops.Program(
    advance=pops.Advance(joint, dt=DT,
                         tolerance=pops.Tolerance(rtol=1e-8, atol=1e-10)),    # value chosen for the gallery
    initial={n: pops.project(N_D * (1 + x / L_DEVICE), quadrature=pops.GaussLegendre(1))},   # exact: linear
    solvers={joint: newton(                              # the Jacobian is exact unless stated (SU9)
        linear=gmres(restart=30, preconditioner=block_elimination(
            order=[psi, n], inner={psi: cg(preconditioner=multigrid(cycle=v_cycle())), n: bicgstab()})),
        globalization=line_search(c=1e-4, max_backtracks=20))},
)

case = pops.Case(model=model, numerics=numerics, program=program, layout={device: pops.Uniform(cells=400)})
artifact = pops.compile(case, target=pops.OpenMP())
simulation = pops.bind(artifact)
result = simulation.run(until=T_END)
