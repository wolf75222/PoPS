import pops
from pops.math import *
from pops.lib.numerics import upwind, muscl, vanleer, ghosts
from pops.lib.time import ssprk2, cfl_limit
from pops.lib.output import npz

NX, NY = 64, 64
AX, AY = 1.0, 0.25
FAR_FIELD, AMPLITUDE, BETA, X0, Y0 = 0.05, 0.95, 120.0, 0.30, 0.35
CFL, MAX_DT, T_END, MAX_STEPS = 0.45, 1.0e-2, 0.20, 10_000

domain = pops.Box(x=(0.0, 1.0), y=(0.0, 1.0))
sides = domain.boundary

model = pops.Model("scalar_advection")
u = model.quantity("u", pops.Scalar(domain))
advection = model.flux(of=u, value=vector(AX, AY) * u)
model.balance(ddt(u) == -div(advection))
model.on([sides.x_min, sides.y_min], u == FAR_FIELD)
model.on([sides.x_max, sides.y_max], outflow(u))

numerics = pops.Numerics(
    represent={u: pops.CellAverage()},
    schemes={advection: upwind(reconstruction=muscl(limiter=vanleer))},
    boundary=ghosts(),
)

x = domain.x
y = domain.y
bump = FAR_FIELD + AMPLITUDE * exp(-BETA * ((x - X0)**2 + (y - Y0)**2))
program = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=cfl_limit(advection, cfl=CFL, max_dt=MAX_DT)),
    initial={u: pops.project(bump, quadrature=pops.GaussLegendre(1))},
)
outputs = [pops.Output(pops.record(u=u), when=[0.0, T_END],
                       write=npz("results/02_openmp_explicit_ssprk2.npz"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={domain: pops.Uniform(cells=(NX, NY))}, outputs=outputs)
artifact = pops.compile(case, target=pops.OpenMP())
simulation = pops.bind(artifact, resources=pops.Resources(threads=4))
result = simulation.run(until=T_END, max_steps=MAX_STEPS)
