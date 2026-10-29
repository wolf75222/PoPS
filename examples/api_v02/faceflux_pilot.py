"""N7a pilot: the M05 and M10 faces written as FaceFlux schemes (spec 0.3 IR25).

Same physics and the same arithmetic as ``navier_stokes_face`` and
``scharfetter_gummel_face`` of ``models_M01_M14.py``, which stay unchanged.
The difference is how the reads are written: a scheme reads cells reached
from the face (``face.left_cell``, ``face.right_cell``, ``cell.shift``) and
components by name. It writes no dimension, no radius and no halo depth;
``pops.lib.numerics.realize_face_flux`` derives them (IR14, IR17).
"""
from __future__ import annotations

import numpy as np
from pops import math as pm
from pops.lib.numerics import FaceFlux


def navier_stokes_scheme(U, *, gamma=1.4, viscosity=.01, prandtl=.72):
    """The M05 face: Rusanov inviscid part and centred viscous part, one face transfer."""
    if gamma <= 1 or viscosity < 0 or prandtl <= 0:
        raise ValueError("NS requires gamma>1, viscosity>=0, Pr>0")
    cv, cp = 1/(gamma-1), gamma/(gamma-1)
    conserved = ("rho", "mx", "my", "E")

    def scheme(face):
        axis = face.axis
        sampled_states = {}

        def data(cell):
            if cell in sampled_states:
                return sampled_states[cell]
            rho, mx, my, energy = (cell.value(U[name]) for name in conserved)
            u, v = mx/rho, my/rho
            pressure = (gamma-1)*(energy-(mx*mx+my*my)/(2*rho))
            sampled_states[cell] = rho, u, v, pressure/rho, pressure, energy
            return sampled_states[cell]
        l, r = data(face.left_cell), data(face.right_cell)

        def derivative(component, direction):
            if direction == axis:
                return (r[component]-l[component])/face.h(direction)
            return (data(face.right_cell.shift(direction, 1))[component]
                    - data(face.right_cell.shift(direction, -1))[component]
                    + data(face.left_cell.shift(direction, 1))[component]
                    - data(face.left_cell.shift(direction, -1))[component])/(4*face.h(direction))
        grad_u = tuple(tuple(derivative(i+1, j) for j in range(2)) for i in range(2))
        divergence = grad_u[0][0]+grad_u[1][1]
        mu = viscosity*((l[3]+r[3])/2)**.7
        conductivity = cp*mu/prandtl
        tau = tuple(mu*(grad_u[i][axis]+grad_u[axis][i]
                        - (2/3)*int(i == axis)*divergence) for i in range(2))

        def inviscid(q):
            rho, u, v, temperature, pressure, energy = q
            velocity = (u, v); normal = velocity[axis]
            return (rho*normal, rho*u*normal+int(axis == 0)*pressure,
                    rho*v*normal+int(axis == 1)*pressure, (energy+pressure)*normal)
        fl, fr = inviscid(l), inviscid(r)
        speed = pm.maximum(abs(l[axis+1])+pm.sqrt(gamma*l[3]),
                           abs(r[axis+1])+pm.sqrt(gamma*r[3]))
        euler = tuple((fl[i]+fr[i])/2-speed*(face.right(U[name])-face.left(U[name]))/2
                      for i, name in enumerate(conserved))
        work = sum((l[i+1]+r[i+1])*tau[i]/2 for i in range(2))
        flux = (euler[0], euler[1]-tau[0], euler[2]-tau[1],
                euler[3]-work-conductivity*derivative(3, axis))
        rho_min = pm.minimum(l[0], r[0])
        h = face.h(axis)
        frequency = (speed/(2*h)
                     + 2*((4/3)*mu+conductivity/cv)/(rho_min*h**2)
                     + 2*mu/(rho_min*h*face.h(1-axis)))
        domain_floor = l[0]
        for q in sampled_states.values():
            domain_floor = pm.minimum(domain_floor, pm.minimum(q[0], q[3]))
        return FaceFlux(dict(zip(conserved, flux)),
                        frequency=pm.where(domain_floor > 0, lambda: frequency, lambda: -1.))
    return scheme


def _bernoulli(z):
    # B(z) = z/(exp(z)-1), as in models_M01_M14.scharfetter_gummel_face.
    square = z*z
    return pm.where(abs(z) < 1.e-4,
        lambda: 1-z/2+square*(1/12+square*(-1/720+square/30240)),
        lambda: pm.where(z > 50,
            lambda: z*pm.exp(-z)/(1-pm.exp(-z)),
            lambda: pm.where(z < -50,
                lambda: -z/(1-pm.exp(z)), lambda: z/pm.expm1(z))))


def scharfetter_gummel_scheme(U, phi, *, diffusion=.1, mobility=.1):
    """The M10 face: the fitted Scharfetter-Gummel flux of n in the solved potential phi."""
    if not np.isfinite(diffusion) or diffusion <= 0 or not np.isfinite(mobility) or mobility < 0:
        raise ValueError("fitted face requires finite D>0 and mobility>=0")
    n = U["n"]

    def scheme(face):
        nl, nr = face.left(n), face.right(n)
        jump = mobility/diffusion*(face.right(phi)-face.left(phi))
        outgoing = diffusion*_bernoulli(jump)/face.distance
        incoming = diffusion*_bernoulli(-jump)/face.distance
        frequency = (outgoing+incoming)/face.distance
        return FaceFlux({"n": outgoing*nl-incoming*nr},
                        frequency=pm.where(pm.minimum(nl, nr) >= 0, lambda: frequency, lambda: -1.))
    return scheme


__all__ = ["navier_stokes_scheme", "scharfetter_gummel_scheme"]
