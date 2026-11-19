#!/usr/bin/env python3
"""L1-d (spec 17.1, CF17): L1-a with IMEX Euler in place of SSPRK2.

The advection is explicit and the diffusion implicit, solved by cg(); the step is the restriction
c_a <= 1 of the advection alone, a number until N8 elaborates dt= as a function of cell.
"""
import numpy

import pops.surface as pops
from pops.surface.math import *
from pops.lib.numerics import upwind, two_point
from pops.lib.time import imex_euler
from pops.lib.solve import cg

NX, A, NU, T_END, MAX_STEPS = 256, 1.0, 1e-2, 1.0, 10_000

domain = pops.Box(x=pops.Periodic(0, 1))
model = pops.Model("advection_diffusion")
u = model.quantity("u", pops.Scalar(domain))
a = model.constant("a", value=A)
nu = model.constant("nu", value=NU)
u0 = model.data("u0", pops.Scalar(domain))
advection = model.flux(of=u, value=a * u)
diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
model.balance(ddt(u) == -div(advection + diffusion))

numerics = pops.Numerics(represent={u: pops.CellAverage()},
                         schemes={advection: upwind(), diffusion: two_point()})

DT = 1 / (abs(A) * NX)
imex = imex_euler(explicit=advection, implicit=diffusion)
program = pops.Program(advance=pops.Advance(imex, dt=DT, attempts=2,
                                            tolerance=pops.Tolerance(rtol=1e-10, atol=1e-12)),
                       initial={u: u0}, solvers={imex: cg()})

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={domain: pops.Uniform(cells=NX)})
artifact = pops.compile(case, target=pops.OpenMP())
x_c = (numpy.arange(NX) + 0.5) / NX
u0_avg = numpy.where(abs(x_c - 0.5) < 0.25, 1.0, 0.0)
simulation = pops.bind(artifact, data={u0: u0_avg})
result = simulation.run(until=T_END, max_steps=MAX_STEPS)
