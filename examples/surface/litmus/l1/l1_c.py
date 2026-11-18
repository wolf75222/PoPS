#!/usr/bin/env python3
"""L1-c (spec 17.1, CF17): L1-a with the upwind face written in the case in place of upwind().

The swap adds the lines of the function, written with @pops.scheme, and takes upwind out of the
import of pops.lib; the key of schemes= keeps its name.
"""
import numpy

import pops.surface as pops
from pops.surface.math import *
from pops.lib.numerics import two_point
from pops.surface.discrete import FaceFlux
from pops.lib.time import ssprk2

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


@pops.scheme
def upwind(face, term):
    """The upwind face written in the case: the mean of the two fluxes minus |s| (u_R - u_L) / 2,
    s = F(1).n, which for a linear flux is s u_L where s > 0 and s u_R elsewhere."""
    q = term.quantity
    term.expect(term.law(states={q: 1}) * q)
    s = abs(dot(term.law(states={q: 1}), face.normal))
    left = dot(term.law(states={q: face.left(q)}), face.normal)
    right = dot(term.law(states={q: face.right(q)}), face.normal)
    return FaceFlux((left + right) / 2 - s * (face.right(q) - face.left(q)) / 2, speed=s)


numerics = pops.Numerics(represent={u: pops.CellAverage()},
                         schemes={advection: upwind(), diffusion: two_point()})

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
