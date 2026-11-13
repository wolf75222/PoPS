#!/usr/bin/env python3
"""L1-b (spec 17.1, CF17): L1-a with rusanov() in place of upwind().

The swap changes the scheme of the advection and its import, nothing else; for this linear flux
both faces compute (F(u_L) + F(u_R)) / 2 - |a| (u_R - u_L) / 2, so L1-b gives L1-a bit for bit.
"""
import numpy

import pops.surface as pops
from pops.surface.math import *
from pops.lib.numerics import rusanov, two_point
from pops.lib.time.methods import ssprk2

NX, A, NU, T_END, MAX_STEPS = 256, 1.0, 1e-2, 1.0, 10_000

domain = pops.Box(x=pops.Periodic(0, 1))
model = pops.Model("advection_diffusion")
u = model.quantity("u", pops.Scalar(domain))
a = model.constant("a", value=A)
nu = model.constant("nu", value=NU)
u0 = model.data("u0", pops.Scalar(domain))
advection = model.flux(of=u, value=a * u)
diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
model.balance(ddt(u) == -div(advection) - div(diffusion))

numerics = pops.Numerics(represent={u: pops.CellAverage()},
                         schemes={advection: rusanov(), diffusion: two_point()})

DT = 1 / (abs(A) * NX + 2 * NU * NX**2)
program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=DT, attempts=2),
                       initial={u: u0})

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={domain: pops.Uniform(cells=NX)})
artifact = pops.compile(case, target=pops.OpenMP())
x_c = (numpy.arange(NX) + 0.5) / NX
u0_avg = numpy.where(abs(x_c - 0.5) < 0.25, 1.0, 0.0)
simulation = pops.bind(artifact, data={u0: u0_avg})
result = simulation.run(until=T_END, max_steps=MAX_STEPS)
