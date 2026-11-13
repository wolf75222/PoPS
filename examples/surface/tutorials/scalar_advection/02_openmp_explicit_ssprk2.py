#!/usr/bin/env python3
"""Advection scalaire 2D OpenMP avec SSPRK2 ecrit explicitement dans pops.Program.

Le script reste top-level et lineaire. Les expressions Python construisent le graphe temporel ;
les stages et les operateurs sont ensuite compiles et executes en C++/Kokkos OpenMP sur sept
threads.
"""
import pops.surface as pops
from pops.surface.math import *
from pops.surface.discrete import FaceFlux
from pops.lib.numerics import muscl, vanleer, ghosts
from pops.lib.time import cfl_limit
from pops.lib.quadrature import GaussLegendre
from pops.lib.output import npz

NX, NY = 64, 64
AX, AY = 1.0, 0.25
FAR_FIELD = 0.05
GAUSSIAN_AMPLITUDE, GAUSSIAN_BETA = 0.95, 120.0
GAUSSIAN_CENTER_X, GAUSSIAN_CENTER_Y = 0.30, 0.35
CFL, MAX_DT, T_END, MAX_STEPS = 0.45, 1.0e-2, 0.20, 10_000

domain = pops.Box(x=(0.0, 1.0), y=(0.0, 1.0))
sides = domain.boundary

model = pops.Model("scalar_advection")
u = model.quantity("u", pops.Scalar(domain))
advection = model.flux(of=u, value=vector(AX, AY) * u)
model.balance(ddt(u) == -div(advection))
model.on([sides.x_min, sides.y_min], u == FAR_FIELD)
model.on([sides.x_max, sides.y_max], outflow(u))


@pops.scheme
def upwind(face, term, reconstruction=None):
    """Le decentrement de master (ScalarUpwind), evalue comme master l'evalue : la moyenne des deux
    flux moins |s| (u_R - u_L) / 2, s = F(1).n ; pour un flux lineaire c'est s u_L ou s > 0,
    s u_R ailleurs."""
    q = term.quantity
    unit_flux = term.law(states={q: 1})
    term.expect(unit_flux * q)
    s = abs(dot(unit_flux, face.normal))
    left = dot(term.law(states={q: face.left(q)}), face.normal)
    right = dot(term.law(states={q: face.right(q)}), face.normal)
    return FaceFlux((left + right) / 2 - s * (face.right(q) - face.left(q)) / 2,
                    reconstruction=reconstruction, speed=s)


numerics = pops.Numerics(
    represent={u: pops.CellAverage()},
    schemes={advection: upwind(reconstruction=muscl(limiter=vanleer))},
    boundary=ghosts(),
)


@pops.method
def ssprk2(step, rate):
    """SSPRK2 de master : u1 = u + h k0, puis u + h/2 k0 + h/2 k1."""
    u, t, h = step.state, step.time, step.dt
    k0 = rate(u, t)
    k1 = rate(u + h * k0, t + h)
    return pops.Candidate(u + h / 2 * k0 + h / 2 * k1)


x, y = domain.x, domain.y
bump = FAR_FIELD + GAUSSIAN_AMPLITUDE * exp(
    -GAUSSIAN_BETA * ((x - GAUSSIAN_CENTER_X)**2 + (y - GAUSSIAN_CENTER_Y)**2))
program = pops.Program(
    advance=pops.Advance(ssprk2(rate=model), dt=cfl_limit(advection, cfl=CFL, max_dt=MAX_DT)),
    initial={u: pops.project(bump, quadrature=GaussLegendre(1))},
)
metadata = pops.record(nx=NX, ny=NY, ax=AX, ay=AY, far_field=FAR_FIELD,
                       gaussian_amplitude=GAUSSIAN_AMPLITUDE, gaussian_beta=GAUSSIAN_BETA,
                       gaussian_center_x=GAUSSIAN_CENTER_X, gaussian_center_y=GAUSSIAN_CENTER_Y,
                       cfl=CFL, t_end=T_END)
outputs = [pops.Output(pops.record(u=u), when=[0.0, T_END],
                       write=npz("results/02_openmp_explicit_ssprk2.npz", metadata=metadata))]

case = pops.Case(model=model, numerics=numerics, program=program,
                 layout={domain: pops.Uniform(cells=(NX, NY))}, outputs=outputs)
artifact = pops.compile(case, target=pops.OpenMP())
simulation = pops.bind(artifact)
result = simulation.run(until=T_END, max_steps=MAX_STEPS)
