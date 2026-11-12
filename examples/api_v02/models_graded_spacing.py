"""Graded spacing, written with the surface: the cases of N17 L1b (ROM-110, D0b, DN3).

``pops.Graded(base=pops.Uniform(cells=c), nodes={Ω.x: f})`` places node k of an axis at f(s),
s = k / c the logical coordinate of the node, correctly rounded; f is a plain function that PoPS
calls once with s (SU5, as section 13 of the design note N17 states it). The elaborated layout is
the static case of the geometry-evolution contract, served by the static geometry arrays as node
positions at every rank (C1; models_static_geometry.py writes the two graphs of C1 below the
surface).

* ``build_graded_ring``: a scalar on the periodic ring 0 <= x < 1 whose nodes are
  s - A sin(2 pi s) / (2 pi), dense at the seam and sparse at x = 1/2, with the two-point diffusion
  or the Rusanov advection of ``pops.lib.numerics.schemes`` and the SSPRK2 of
  ``pops.lib.time.methods``;
* ``build_graded_plane``: the Rusanov advection of a scalar at the velocity (1, 1/2) on the
  periodic plane, x graded as the ring and y uniform;
* ``build_graded_box``: the two-point diffusion of a scalar on the periodic box of rank three, x
  and z graded, y uniform, z on [-1, 1]: the test of rank three that the note asks of L1 (the
  widths and the diffusion inventory of a graded box);
* ``build_wall_layer``: the case of probe 3 (section 6.3 of the note) on its graded layout, the
  nodes tanh(3 s) / tanh(3) of the wall at x = 1 written with exp (tanh is a word of N11), marched
  in time by SSPRK2 where the probe solves its steady layer once with gmres (N9). Its layout
  elaborates and its arrays nest; the case is refused at validation, as on its uniform reference,
  until G04 realizes a value relation under a diffusion.

Each builder returns the case with its evolved quantity and its datum; the initial values and the
oracles live in the tests that run them.
"""
from __future__ import annotations

import pops.surface as pops
from pops.lib.numerics.boundaries import ghosts
from pops.lib.numerics.schemes import rusanov, two_point
from pops.lib.quadrature import GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.surface.discrete import FaceFlux
from pops.surface.math import ddt, div, dot, exp, grad, pi, sin, vector

# The ring and its two terms.
RING_CELLS = 32
RING_GRADING = 0.6          # the widths range from (1 - 0.6) / 32 at the seam to (1 + 0.6) / 32
CONDUCTIVITY = 0.01
RING_SPEED = 1.0
RING_DIFFUSION_DT = 2.0e-3
RING_ADVECTION_DT = 5.0e-3
RING_STEPS = 10

# The plane: x graded as the ring, y uniform.
PLANE_CELLS = (16, 8)
PLANE_VELOCITY = (1.0, 0.5)
PLANE_DT = 5.0e-3
PLANE_STEPS = 8

# The box of rank three: x and z graded, y uniform.
BOX_CELLS = (8, 6, 4)
BOX_Z_GRADING = 0.5
BOX_DT = 1.0e-2
BOX_STEPS = 6

# Probe 3: the wall layer of width LAYER_WIDTH / WALL_SPEED at x = 1.
WALL_CELLS = 32
WALL_GRADING = 3
LAYER_WIDTH = 1.0e-2
WALL_SPEED = 1.0
WALL_DT = 1.0e-5


def ring_nodes(s):
    """The nodes of the ring: dense at the seam x = 0 = 1, sparse at x = 1/2."""
    return s - RING_GRADING * sin(2 * pi * s) / (2 * pi)


def box_z_nodes(s):
    """The nodes of z on [-1, 1]: sparse at the seam, dense at z = 0."""
    return -1 + 2 * (s + BOX_Z_GRADING * sin(2 * pi * s) / (2 * pi))


def wall_nodes(s):
    """tanh(3 s) / tanh(3), written with exp: the nodes cluster at the wall x = 1."""
    growth = exp(-2 * WALL_GRADING * s)
    edge = exp(-2 * WALL_GRADING)
    return (1 - growth) * (1 + edge) / ((1 + growth) * (1 - edge))


def build_graded_ring(term="diffusion", cells=RING_CELLS):
    """A scalar on the graded ring, diffused by two_point() or advected by rusanov()."""
    ring = pops.Box(x=pops.Periodic(0, 1))
    model = pops.Model("graded_ring_" + term)
    u = model.quantity("u", pops.Scalar(ring))
    u0 = model.data("u0", pops.Scalar(ring))
    if term == "diffusion":
        flux = model.diffusive_flux(of=u, value=-CONDUCTIVITY * grad(u))
        scheme, dt = two_point(), RING_DIFFUSION_DT
    else:
        flux = model.flux(of=u, value=RING_SPEED * u)
        scheme, dt = rusanov(), RING_ADVECTION_DT
    model.balance(ddt(u) == -div(flux))
    numerics = pops.Numerics(represent={u: pops.CellAverage()}, schemes={flux: scheme})
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=dt), initial={u: u0})
    layout = pops.Graded(base=pops.Uniform(cells=cells), nodes={ring.x: ring_nodes})
    case = pops.Case(model=model, numerics=numerics, program=program, layout={ring: layout})
    return case, u, u0


def build_graded_plane(cells=PLANE_CELLS):
    """A scalar advected by rusanov() on the periodic plane, x graded and y uniform."""
    plane = pops.Box(x=pops.Periodic(0, 1), y=pops.Periodic(0, 1))
    model = pops.Model("graded_plane")
    u = model.quantity("u", pops.Scalar(plane))
    u0 = model.data("u0", pops.Scalar(plane))
    a = model.constant("a", value=PLANE_VELOCITY[0])
    b = model.constant("b", value=PLANE_VELOCITY[1])
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    numerics = pops.Numerics(represent={u: pops.CellAverage()}, schemes={advection: rusanov()})
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=PLANE_DT),
                           initial={u: u0})
    layout = pops.Graded(base=pops.Uniform(cells=cells), nodes={plane.x: ring_nodes})
    case = pops.Case(model=model, numerics=numerics, program=program, layout={plane: layout})
    return case, u, u0


def build_graded_box(cells=BOX_CELLS):
    """A scalar diffused by two_point() on the periodic box of rank three, x and z graded."""
    box = pops.Box(x=pops.Periodic(0, 1), y=pops.Periodic(0, 2), z=pops.Periodic(-1, 1))
    model = pops.Model("graded_box")
    u = model.quantity("u", pops.Scalar(box))
    u0 = model.data("u0", pops.Scalar(box))
    diffusion = model.diffusive_flux(of=u, value=-CONDUCTIVITY * grad(u))
    model.balance(ddt(u) == -div(diffusion))
    numerics = pops.Numerics(represent={u: pops.CellAverage()}, schemes={diffusion: two_point()})
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=BOX_DT), initial={u: u0})
    layout = pops.Graded(base=pops.Uniform(cells=cells),
                         nodes={box.x: ring_nodes, box.z: box_z_nodes})
    case = pops.Case(model=model, numerics=numerics, program=program, layout={box: layout})
    return case, u, u0


@pops.scheme
def centred(face, term):
    """The mean of the two normal fluxes, without dissipation (section 6.2 of the note)."""
    q = term.quantity
    left, right = face.left(q), face.right(q)
    return FaceFlux((dot(term.law(states={q: left}), face.normal)
                     + dot(term.law(states={q: right}), face.normal)) / 2)


def build_wall_layer(cells=WALL_CELLS, graded=True):
    """The case of probe 3 on the wall nodes, marched in time: -eps u'' + a u' = 0 with u(0) = 0
    and u(1) = 1, a centred face and the two-point diffusion; ``graded=False`` gives the uniform
    reference of the probe, the same case on pops.Uniform(cells=cells)."""
    sheath = pops.Box(x=(0, 1))
    model = pops.Model("wall_layer")
    u = model.quantity("u", pops.Scalar(sheath))
    advection = model.flux(of=u, value=WALL_SPEED * u)
    diffusion = model.diffusive_flux(of=u, value=-LAYER_WIDTH * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    model.on(sheath.boundary.x_min, u == 0)
    model.on(sheath.boundary.x_max, u == 1)
    numerics = pops.Numerics(represent={u: pops.CellAverage()},
                             schemes={advection: centred(), diffusion: two_point()},
                             boundary=ghosts())
    program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=WALL_DT),
                           initial={u: pops.project(sheath.x, quadrature=GaussLegendre(1))})
    layout = pops.Uniform(cells=cells)
    if graded:
        layout = pops.Graded(base=layout, nodes={sheath.x: wall_nodes})
    case = pops.Case(model=model, numerics=numerics, program=program, layout={sheath: layout})
    return case, u, None
