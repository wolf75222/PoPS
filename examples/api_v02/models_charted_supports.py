"""Charted supports, written with the surface: the cases of N17 L2a (ROM-110, D0b, DN25 to DN30).

``σ.chart(x=e_x, y=e_y)`` states the chart of a support of rank two. Its keywords name the axes of
the Cartesian frame, which key the components of a flux and of a vector; the coordinates of σ stay
the logical axes, which name the mesh axes (``cells=`` in their order), the boundary pieces and the
node functions of ``pops.Graded``. The layout states the quadrature rule of the measures,
``pops.Uniform(cells=..., geometry=GaussLegendre(2))`` (D0b, SU9). The elaborated layout is the
chart of C1, served by the static geometry arrays: measures and centroids by the rule, chord area
vectors and normals, distances (models_static_geometry.py writes the same chart below the surface).

* ``build_chart_annulus``: a scalar advected at a constant Cartesian velocity by ``rusanov()`` on
  the annulus 1 <= r <= 2, theta periodic, ``outflow`` at both walls, with the SSPRK2 of
  ``pops.lib.time.methods``; ``graded=f`` places the radial nodes at f(s) on the same chart.
* ``build_chart_diffusion``: a scalar diffused by ``two_point()`` on a doubly periodic chart of the
  unit square (N17 L2b-1, DN35, DN36): the square turned by pi / 6, whose faces are orthogonal to
  the line of the centroids, or sheared, x = s + t / 4, whose faces are not (the sine of the angle
  is 1 / sqrt(17)); ``chart=None`` writes the same case on the box.

Each builder returns the case with its evolved quantity and its datum; the initial values and the
oracles live in the tests that run them.
"""
from __future__ import annotations

import pops.surface as pops
from pops.lib.numerics import rusanov, two_point
from pops.lib.numerics.boundaries import ghosts
from pops.lib.quadrature import GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.surface.math import cos, ddt, div, grad, outflow, pi, sin, vector

# The annulus: r in [1, 2], theta periodic, (r, theta) the order of the cells.
ANNULUS_RADII = (1.0, 2.0)
ANNULUS_CELLS = (8, 16)
ANNULUS_VELOCITY = (1.0, 0.5)
ANNULUS_DT = 1.0e-2
ANNULUS_STEPS = 6

# The doubly periodic square, turned or sheared, diffused by two_point().
PLANE_CELLS = (8, 8)
PLANE_TURN = pi / 6
PLANE_SHEAR = 0.25
PLANE_CONDUCTIVITY = 0.01
PLANE_DT = 1.0e-2
PLANE_STEPS = 8


def radial_nodes(s):
    """Radial nodes on [1, 2], denser toward the inner wall: 1 + s (1 + s) / 2."""
    return 1 + s * (1 + s) / 2


def build_chart_annulus(geometry=None, graded=None):
    """A scalar advected by rusanov() on the charted annulus; ``geometry`` replaces the rule
    GaussLegendre(2) (the refusal tests), ``graded`` grades the radial axis on the same chart."""
    ring = pops.Box(r=ANNULUS_RADII, theta=pops.Periodic(0, 2 * pi))
    annulus = ring.chart(x=ring.r * cos(ring.theta), y=ring.r * sin(ring.theta))
    model = pops.Model("chart_annulus")
    u = model.quantity("u", pops.Scalar(annulus))
    u0 = model.data("u0", pops.Scalar(annulus))
    a = model.constant("a", value=ANNULUS_VELOCITY[0])
    b = model.constant("b", value=ANNULUS_VELOCITY[1])
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    model.on([annulus.boundary.r_min, annulus.boundary.r_max], outflow(u))
    numerics = pops.Numerics(represent={u: pops.CellAverage()}, schemes={advection: rusanov()},
                             boundary=ghosts())
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=ANNULUS_DT),
                           initial={u: u0})
    layout = pops.Uniform(cells=ANNULUS_CELLS,
                          geometry=GaussLegendre(2) if geometry is None else geometry)
    if graded is not None:
        layout = pops.Graded(base=layout, nodes={annulus.r: graded})
    case = pops.Case(model=model, numerics=numerics, program=program, layout={annulus: layout})
    return case, u, u0



def build_chart_diffusion(chart="turned"):
    """A scalar diffused by two_point() on the doubly periodic unit square under a chart: the
    square turned by pi / 6 (``"turned"``), sheared along x (``"sheared"``), or the box itself
    (``None``), 8 x 8 cells with the SSPRK2 of ``pops.lib.time.methods``."""
    square = pops.Box(s=pops.Periodic(0, 1), t=pops.Periodic(0, 1))
    if chart is None:
        # A box names the axes of its frame (DN26): the same square, written x and y.
        square = pops.Box(x=pops.Periodic(0, 1), y=pops.Periodic(0, 1))
    if chart == "turned":
        support = square.chart(x=cos(PLANE_TURN) * square.s - sin(PLANE_TURN) * square.t,
                               y=sin(PLANE_TURN) * square.s + cos(PLANE_TURN) * square.t)
    elif chart == "sheared":
        support = square.chart(x=square.s + PLANE_SHEAR * square.t, y=square.t)
    elif chart is None:
        support = square
    else:
        raise ValueError("chart is 'turned', 'sheared' or None")
    model = pops.Model("chart_diffusion")
    u = model.quantity("u", pops.Scalar(support))
    u0 = model.data("u0", pops.Scalar(support))
    k = model.constant("k", value=PLANE_CONDUCTIVITY)
    diffusion = model.diffusive_flux(of=u, value=-k * grad(u))
    model.balance(ddt(u) == -div(diffusion))
    numerics = pops.Numerics(represent={u: pops.CellAverage()}, schemes={diffusion: two_point()})
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=PLANE_DT),
                           initial={u: u0})
    layout = pops.Uniform(cells=PLANE_CELLS,
                          geometry=None if chart is None else GaussLegendre(2))
    case = pops.Case(model=model, numerics=numerics, program=program, layout={support: layout})
    return case, u, u0
