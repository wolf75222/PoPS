"""Relations of pops.lib.models and pops.lib.models.kinetic, and the function of pops.lib.math, in the surface of
section 2.6 (version 2). The three modules share this gallery file; `maxwellian` belongs to pops.lib.models.kinetic
and `perp` to pops.lib.math.

Each word is a plain Python function that runs once, at authoring time, and returns an expression or a relation
built with the words of pops.math (E4, SU5). None of them names a scheme: the realization of a relation is chosen
by the numerics, `boundary=ghosts()` or a scheme under its handle (SU6, D9). `outflow(q)`, the empty relation, is a
word of pops.math (SD43), not of this module. `normal()` is the outward unit normal of the piece on which its
expression is read, each piece of `model.on` in turn, so that e.n is written `dot(e, normal())` (SD50). A wall,
velocity.n = 0, is one line of the paper, `dot(velocity, normal()) == 0`, and has no word here (SD64).
"""

import pops
from pops.math import *


def characteristic(U, incoming):
    """P_in (U - g) == 0 on the trace: the part of U - g on the incoming characteristics vanishes.
    P_in = incoming_projector(U, normal=normal()), the sum of r_i l_i^T over the eigenvalues lambda_i < 0 of the
    Jacobian of the first-order flux of the balance of U along normal(), the outward unit normal of each piece on
    which model.on places the relation (R9, SD43, SD50). g is a value or a pops.record of U."""
    return dot(incoming_projector(U, normal=normal()), U - incoming) == 0


def maxwellian(f, over):
    """M = n exp(-|v - u|^2 / (2 T)) / (2 pi T)^(d/2), the Maxwellian with the moments 1, v and |v|^2 / 2 of f
    over the velocity support `over` of dimension d: n = int f dv, n u = int f v dv, E = int f |v|^2 / 2 dv and
    T = (2 E / n - |u|^2) / d. The moments are fields of space, pulled back to the phase space where M is read;
    u has the components of the velocity axes, so v - u stays within one factor (SD48)."""
    v = over.position
    d = len(over.axes)
    n = integral(f, over=over)
    u = integral(v * f, over=over) / n
    energy = integral(dot(v, v) / 2 * f, over=over)
    T = (2 * energy / n - dot(u, u)) / d
    return n * exp(-dot(v - u, v - u) / (2 * T)) / sqrt(2 * pi * T) ** d


def perp(g):
    """perp(g) = (-g_y, g_x): g turned by a quarter turn in the oriented 2-D frame of its support (FC5), whose axes
    g.support.axes lists in order, x then y (SD42)."""
    axes = g.support.axes
    return vector(-component(g, axes[1]), component(g, axes[0]))
