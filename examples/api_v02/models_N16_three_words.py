"""Cases of N16 (ROM-109), the lot that lifts the table for three words: Burgers, with or without
a decay source, under each representation of one constant mode that the lowering serves (N16 5.1,
A1), on two layouts:

- ``"graded"``: the 32 graded cells of N17's segment (widths in a ratio of about 92), FaceFlux
  only, since a mapped layout refuses a named source today (``_static_geometry_consumers.py``);
- ``"uniform"``: 16 cells of [0, 3], μ(K) = 3/16, neither 1 nor a logical spacing, FaceFlux and the
  decay source, so the scale μ(K) of a total under CellRate is seen.

``represent`` names the row that the builder adds with ``plan.representations.add``:

- ``"total"``: ``cell_total(1)``, U_K = ∫_K u, c = 1/cell.measure;
- ``"sample"``: ``point_value(1)``, the value at the centre of the cell, c = 1;
- ``"rule"``: ``cell_average(1, quadrature=GaussLegendre(3))``, the average under a stated rule;
- ``"average"``: ``cell_average(1)``, today's average;
- None: no row, the average that every state has without one.

The rate is ``ddt(U) == -div(flux) + source``: ``FaceFlux`` is the Rusanov flux of Burgers and
``CellRate`` the density S(u) = -DECAY u. Under one constant mode the Riesz map 1/(c² μ(K)) and
the weight c of the test compose into 1/(c μ(K)) (N16 5.1, A1), so the update written by hand is

    dU_K/dt = -(F_+ - F_-) + μ(K) S(U_K/μ(K))       for a total,
    du_K/dt = -(F_+ - F_-)/μ(K) + S(u_K)            for an average, a sample and a stated rule.

The initial value is the encode of each representation, computed here from the closed form of
u0 = 1 + 0.5 cos(pi x/L) and bound as data (``BindArray``): the exact integral for a total, the
sample at the centre, the three-point Gauss-Legendre average for the rule, the exact average for
the average. The analytic route of b6 projects cell averages only, so the projection by encode
waits for N16 lot 2 (driver 13/11 12:10, « projection initiale encode(Q(e)) calculée par
l'exemple, puis liée »).
"""
from __future__ import annotations

import math

import numpy as np
import pops
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.quadrature import GaussLegendre
from pops.mesh import CartesianGrid, MovingControlVolumes
from pops.numerics import DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.representations.definition import cell_average, cell_total, point_value

from examples.api_v02.models_G03_totals import _finite_volume, _ssprk2
from examples.api_v02.models_static_geometry import GRADED_CELLS, graded_map, graded_nodes

DECAY = 2.0
GRADED_DT = 2.0e-4
GRADED_STEPS = 10
UNIFORM_CELLS = 16
UNIFORM_LENGTH = 3.0
UNIFORM_DT = 0.01
UNIFORM_STEPS = 12
RULE_POINTS = 3

REPRESENTATIONS = {
    "total": cell_total(1),
    "sample": point_value(1),
    "rule": cell_average(1, quadrature=GaussLegendre(RULE_POINTS)),
    "average": cell_average(1),
    None: None,
}


def u0(x, length=1.0):
    return 1.0 + 0.5*np.cos(math.pi*np.asarray(x, dtype=float)/length)


def _u0_integral(x, length):
    x = np.asarray(x, dtype=float)
    return x + 0.5*length*np.sin(math.pi*x/length)/math.pi


def layout_nodes(layout):
    """The nodes of the cells of ``layout``, from its closed form."""
    if layout == "graded":
        return graded_nodes()
    if layout == "uniform":
        return np.linspace(0.0, UNIFORM_LENGTH, UNIFORM_CELLS + 1)
    raise ValueError("layout is 'graded' or 'uniform'")


def encoded_initial(represent, layout="graded"):
    """The coefficients of u0 under ``represent``, from its closed form, independently of PoPS."""
    nodes = layout_nodes(layout)
    length = nodes[-1] - nodes[0]
    lower, upper = nodes[:-1], nodes[1:]
    measures = upper - lower
    integrals = _u0_integral(upper, length) - _u0_integral(lower, length)
    if represent == "total":
        return integrals
    if represent == "sample":
        return u0(0.5*(lower + upper), length)
    if represent == "rule":
        points, weights = np.polynomial.legendre.leggauss(RULE_POINTS)
        centres, halves = 0.5*(lower + upper), 0.5*measures
        return sum(0.5*w*u0(centres + p*halves, length)
                   for p, w in zip(points, weights, strict=True))
    if represent in ("average", None):
        return integrals/measures
    raise ValueError("represent is 'total', 'sample', 'rule', 'average' or None")


def build_words(*, represent="total", layout="graded", source=None, initial="bound",
                coefficients=None, name=None):
    """Burgers on ``layout`` with outflow ends, under the row that ``represent`` names, with the
    decay source -DECAY u on the uniform layout (``source`` defaults to it there). The step is
    fixed. ``initial="analytic"`` projects u0 on b6's analytic route instead, which a total, a
    sample and a stated rule refuse (it projects cell averages). ``coefficients`` binds the given
    coefficients in place of the encode of u0, and ``name`` names the case in place of its
    representation and layout."""
    if represent not in REPRESENTATIONS:
        raise ValueError("represent is 'total', 'sample', 'rule', 'average' or None")
    nodes = layout_nodes(layout)
    source = (layout == "uniform") if source is None else source
    frame = CartesianDomain("n16_words_%s" % layout, (0.0,), (float(nodes[-1]),)).frame(
        Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("N16_words_%s_burgers" % layout, frame=frame)
    U = model.state("u", components=("u",))
    (u,) = U
    flux = model.flux("burgers_flux", state=U, frame=frame, components={x: (0.5*u*u,)},
                      waves={x: (u,)})
    if source:
        decay = model.source("decay", on=U, value=(-DECAY*u,))
        rate = model.rate("burgers_decay", equation=pm.ddt(U) == -pm.div(flux) + decay)
    else:
        rate = model.rate("burgers", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case(name or "N16_words_%s_%s%s" % (layout, represent, "_decay" if source else ""))
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, _finite_volume(flux, U))
    if REPRESENTATIONS[represent] is not None:
        plan.representations.add(U, REPRESENTATIONS[represent])
    boundaries = frame.boundaries
    plan.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Outflow(state=block[U]),
        boundaries.x_max: Outflow(state=block[U]),
    }))
    case.numerics(plan, block=block)
    if layout == "graded":
        _ssprk2(case, block, U, rate, GRADED_DT, "N16_words_SSPRK2")
        grid = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(GRADED_CELLS,))),
                                    evolution=graded_map(frame))
    else:
        _ssprk2(case, block, U, rate, UNIFORM_DT, "N16_words_SSPRK2")
        grid = Uniform(CartesianGrid(frame=frame, cells=(UNIFORM_CELLS,)))
    coefficients = (encoded_initial(represent, layout) if coefficients is None
                    else np.asarray(coefficients, dtype=float))[None]
    if initial == "analytic":
        from pops.analytic import coordinates, cos
        from pops.lib.initial import Analytic

        (xc,) = coordinates(frame)
        case.initials.add(InitialCondition(
            state=block[U], value=Analytic(frame=frame, components=(
                1.0 + 0.5*cos(math.pi*xc/float(nodes[-1])),)),
            projection=ConservativeCellAverage()))
        return case, grid, {}, coefficients
    case.initials.add(InitialCondition(state=block[U], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    return case, grid, {block[U]: np.ascontiguousarray(coefficients)}, coefficients


__all__ = ["DECAY", "GRADED_DT", "GRADED_STEPS", "REPRESENTATIONS", "RULE_POINTS", "UNIFORM_CELLS",
           "UNIFORM_DT", "UNIFORM_LENGTH", "UNIFORM_STEPS", "build_words", "encoded_initial",
           "layout_nodes", "u0"]
