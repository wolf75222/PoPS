"""2D Euler with gravity in a channel: characteristic sides, slip walls, HLL on MUSCL, embedded RK with PI control.

Not in the corpus of spec 0.3. Written to test a system of conservation laws in the proposed surface.
"""

import pops
from pops.math import *
from pops.lib.models import characteristic
from pops.lib.numerics import hll, muscl, minmod, pointwise, ghosts
from pops.lib.time import explicit_rk, bogacki_shampine, pi_controller
from pops.lib.output import hdf5

GAMMA, GRAVITY = Fraction(7, 5), 9.81
channel = pops.Box(x=(0, 4), y=(0, 1))
sides = channel.boundary
x = channel.x
y = channel.y

model = pops.Model("euler_gravity")
U = model.quantity("U", pops.Components(channel, rho=pops.Scalar(admissible=pops.Positive()),
                                        m=pops.Vector(), E=pops.Scalar()))
gamma = model.constant("gamma", value=GAMMA)
g = vector(0, -GRAVITY)
rho = U.rho
m = U.m
E = U.E
velocity = model.define("velocity", m / rho)
pressure = model.define("pressure", (gamma - 1) * (E - dot(m, m) / (2 * rho)))

euler = model.flux(of=U, value=pops.record(rho=m, m=outer(m, velocity) + pressure * identity(channel),
                                           E=(E + pressure) * velocity))
gravity = model.source(of=U, value=pops.record(rho=0, m=rho * g, E=dot(m, g)))
model.balance(ddt(U) == -div(euler) + gravity)

stratified_rho = exp(-GRAVITY * y)                     # isothermal hydrostatic state, R T = 1
stratified_E = stratified_rho / (gamma - 1)
far_field = pops.record(rho=stratified_rho, m=stratified_rho * vector(0.2, 0), E=stratified_E)
model.on([sides.x_min, sides.x_max], characteristic(U, incoming=far_field))     # data only where waves enter
model.on([sides.y_min, sides.y_max], dot(velocity, normal()) == 0)                # slip walls

# The output reads the definition pressure, so it takes a scheme (CoverageNotExact otherwise).
numerics = pops.Numerics(
    represent={U: pops.CellAverage()},
    schemes={euler: hll(reconstruction=muscl(limiter=minmod)), gravity: pointwise(), pressure: pointwise()},
    boundary=ghosts(),
)

bump = 1 + exp(-50 * ((x - 1)**2 + (y - 0.5)**2)) / 10
program = pops.Program(
    advance=pops.Advance(explicit_rk(tableau=bogacki_shampine, rate=model), dt=pi_controller(), first_dt=1e-4,
                         attempts=8, tolerance=pops.Tolerance(rtol=1e-4, atol=1e-6)),
    initial={U: pops.project(pops.record(rho=stratified_rho * bump, m=stratified_rho * bump * vector(0.2, 0),
                                         E=stratified_E),
                             quadrature=pops.GaussLegendre(2))},
)

outputs = [pops.Output(pops.record(rho=rho, pressure=pressure), when=pops.every(time=0.1),
                       write=hdf5("euler_gravity.h5"))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={channel: pops.Uniform(cells=(400, 100))}, outputs=outputs)
artifact = pops.compile(case, target=pops.CUDA())
simulation = pops.bind(artifact)
result = simulation.run(until=2.0)
