"""Ternary Maxwell--Stefan face equations for the pending typed-field stencil.

No runtime or native qualification is supplied by this file alone. The builder
will bind ``potential_sample`` to the actual same-stage Poisson observation.
S09: https://arxiv.org/html/1809.00413v1, equations (19), (22), (24).
Total mass is normalized to one and F/(RT)=1. We use z dot x from (19)/(22),
not the inconsistent rho dot x printed in equation (2).
"""
from pops import math as pm


def ternary_face(sample, potential_sample, spacing, axis, *,
                 masses=(1., 2., 4.), charges=(1., -1., 0.),
                 binary=((0., 1., .8), (1., 0., 1.2), (.8, 1.2, 0.))):
    """Return J, all three friction residuals, and a declared frequency envelope.

    Exact 2x2 elimination of J2=-J0-J1 enforces the mass constraint in every
    face expression. This is a centered FV realization of the constitutive
    equations, not a reproduction of the article's finite-element algorithm.
    """
    dimension = len(spacing)
    left = tuple(-int(d == axis) for d in range(dimension))
    right = (0,)*dimension
    rho_l = tuple(sample(left, i) for i in range(3))
    rho_r = tuple(sample(right, i) for i in range(3))
    rho = tuple((a+b)/2 for a, b in zip(rho_l, rho_r))
    def fractions(values):
        concentration = tuple(values[i]/masses[i] for i in range(3))
        total = sum(concentration)
        return total, tuple(c/total for c in concentration)
    ct, x = fractions(rho)
    _, xl = fractions(rho_l); _, xr = fractions(rho_r)
    grad_phi = (potential_sample(right)-potential_sample(left))/spacing[axis]
    charge_fraction = sum(charges[i]*x[i] for i in range(3))
    drive = tuple((xr[i]-xl[i])/spacing[axis]
                  +(charges[i]*x[i]-charge_fraction*rho[i])*grad_phi for i in range(3))
    friction = tuple(tuple(0. if i == j else 1/(ct**3*masses[i]*masses[j]*binary[i][j])
                          for j in range(3)) for i in range(3))
    matrix = tuple(tuple(sum(friction[i][j]*rho[j] for j in range(3) if j != i)
                         if i == k else -friction[i][k]*rho[i]
                         for k in range(3)) for i in range(3))
    a, b = matrix[0][0]-matrix[0][2], matrix[0][1]-matrix[0][2]
    c, d = matrix[1][0]-matrix[1][2], matrix[1][1]-matrix[1][2]
    determinant = a*d-b*c
    j0 = (-d*drive[0]+b*drive[1])/determinant
    j1 = (c*drive[0]-a*drive[1])/determinant
    flux = j0, j1, -j0-j1
    residual = tuple(sum(matrix[i][j]*flux[j] for j in range(3))+drive[i] for i in range(3))
    inverse_norm = (abs(a)+abs(b)+abs(c)+abs(d))/abs(determinant)
    minimum_density = pm.minimum(pm.minimum(rho[0],rho[1]),rho[2])
    frequency = 8*inverse_norm/(minimum_density*spacing[axis]**2)*(1+abs(grad_phi)*spacing[axis])
    return flux, residual, frequency


__all__ = ["ternary_face"]
