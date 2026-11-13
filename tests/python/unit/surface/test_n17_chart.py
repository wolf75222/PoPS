"""N17 L2a (ROM-110, C2): the charted support at the surface, ``σ.chart(x=..., y=...)`` (D0b, DN1).

A support of rank two states its chart with one keyword per axis of the Cartesian frame, each an
expression of the coordinates of the support. The keywords name the frame axes, which key the
components of a flux and of a vector; the coordinates of the support stay the logical axes, which
name the mesh axes, the boundary pieces and the node functions of ``pops.Graded`` (section 3.2 of
the design note). The layout states the quadrature rule of the measures with ``geometry=``,
required on a charted support and refused on a box (DN4, DN29). The elaborated layout is the chart
of C1, a clock-free coordinate map of rank two with its rule, which the static geometry arrays serve
since L1a. A chart of rank one, or one that only moves the nodes along each axis, is refused
naming ``pops.Graded`` (DN1, DN27); a piece on which the chart does not depend on the other
coordinate is degenerate and leaves the boundary, and a case on that support waits for lot L2c
with ``identify`` of G09 (DN28); a coordinate read on a charted support waits for lot L2b, which
reads it at the centre of the cell and at the centroid of the face (DN7, DN30), and so do a value
relation, whose ghost is the reflection of a cell across a face of the chart, and a reconstruction
above the cell values (DN30), refused where the case is built. On the physical normal of a chart
the face input s of rusanov() is the bound sum_k |n_k| rho(A_k) of b6, which pops.inspect names as a
transitional route (DN31), and an array that a run published on another support is not a datum
of the chart (DN34). The run is the native test test_n17_chart_annulus_native.py.

Since lot L2b-1 a diffusive flux served by ``two_point()`` on a charted support resolves, as one
law on every logical axis (DN35), and ``pops.inspect`` states that its faces take the line of the
two centroids for the normal (K-orthogonality), which the arrays bound at ``pops.bind`` measure
(DN36); the run is test_n17_chart_two_point_native.py.

At the base a support had no reading ``chart`` and ``pops.Uniform`` no keyword ``geometry``;
every test below fails there (CF15).
"""
from __future__ import annotations

import types

import numpy
import pytest

import pops.surface as surface
from examples.api_v02.models_charted_supports import (
    ANNULUS_CELLS, ANNULUS_RADII, build_chart_annulus, build_chart_diffusion, radial_nodes,
)
from pops.codegen._static_geometry_consumers import static_geometry_layouts
from pops.lib.numerics import rusanov
from pops.lib.numerics.boundaries import ghosts
from pops.lib.quadrature import GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.mesh import MovingControlVolumes
from pops.surface import _run
from pops.surface._refusal import (
    AmbiguousElaboration, NotYetElaborated, ReservedName, SupportMismatch,
)
from pops.surface.math import cos, ddt, div, exp, expm1, outflow, pi, sin, vector


def _ring():
    return surface.Box(r=(1, 2), theta=surface.Periodic(0, 2 * pi))


def _annulus():
    ring = _ring()
    return ring, ring.chart(x=ring.r * cos(ring.theta), y=ring.r * sin(ring.theta))


def _case(support, layout=None, value=None, reconstruction=None, inflow=None):
    """A scalar advected by rusanov() on ``support``, laid out by ``layout``, ``outflow`` on each
    piece of its boundary, or ``u == inflow`` on its first piece; building the case elaborates it,
    so a refusal is raised here."""
    model = surface.Model("charted")
    u = model.quantity("u", surface.Scalar(support))
    u0 = model.data("u0", surface.Scalar(support))
    advection = model.flux(of=u, value=vector(1.0, 0.5) * u if value is None else value(u))
    model.balance(ddt(u) == -div(advection))
    pieces = [getattr(support.boundary, name) for name in support.piece_names()]
    if inflow is not None:
        model.on([pieces[0]], u == inflow)
    if pieces[1 if inflow is not None else 0:]:
        model.on(pieces[1 if inflow is not None else 0:], outflow(u))
    face = rusanov() if reconstruction is None else rusanov(reconstruction=reconstruction)
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: face},
                                boundary=ghosts() if pieces else None)
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-2),
                              initial={u: u0})
    if layout is None:
        layout = surface.Uniform(cells=(8, 16), geometry=GaussLegendre(2))
    return surface.Case(model=model, numerics=numerics, program=program, layout={support: layout})


def _reading(case):
    report = surface.inspect(case)
    (reading,) = static_geometry_layouts(report.resolved.layout_plan).values()
    return report, reading


def _layout_row(report):
    (row,) = [row for row in report.rows if row.construct.startswith("layout=")]
    return row


# --- the spelling and the readings (section 3.2) ------------------------------------------------

def test_chart_returns_a_support_whose_frame_axes_are_its_keywords():
    """The keywords name the frame axes; the coordinates of the box stay the logical axes."""
    ring, annulus = _annulus()
    assert annulus is not ring
    assert [axis.name for axis in annulus.axes] == ["x", "y"]
    assert annulus.axes[0] is annulus.x and annulus.axes[1] is annulus.y
    assert annulus.r is ring.r and annulus.theta is ring.theta
    assert annulus.rank == 2
    assert annulus.x.expression is not None and annulus.x.support is annulus
    assert repr(annulus) == repr(ring) + ".chart(x=r * cos(theta), y=r * sin(theta))"
    with pytest.raises(AttributeError, match="frame axes x, y and the logical axes r, theta"):
        annulus.z  # noqa: B018


def test_the_boundary_of_a_charted_support_has_its_logical_pieces():
    """The pieces are those of the logical box: r_min and r_max, theta being periodic."""
    _, annulus = _annulus()
    assert annulus.piece_names() == ["r_min", "r_max"]
    assert annulus.boundary.r_min.support is annulus
    assert annulus.boundary.r_max.name == "r_max"
    with pytest.raises(ValueError, match="periodic"):
        annulus.boundary.theta_min  # noqa: B018
    with pytest.raises(AttributeError, match="r_min, r_max"):
        annulus.boundary.x_min  # noqa: B018


def test_chart_and_weighted_are_readings_that_no_axis_may_take():
    """SU8, as section 13 of the note amends it: chart and weighted are readings of a support."""
    for name in ("chart", "weighted"):
        with pytest.raises(ReservedName, match=name):
            surface.Box(**{name: (0, 1), "y": (0, 1)})


# --- the refusals of the spelling (DN1, DN26, DN27) --------------------------------------------

def test_a_chart_of_rank_one_is_refused_naming_graded():
    segment = surface.Box(s=(0, 1))
    with pytest.raises(SupportMismatch, match="pops.Graded"):
        segment.chart(x=exp(segment.s))


@pytest.mark.parametrize("components", [
    lambda b: {"x": 2 * b.s + 1, "y": b.t * b.t},
    lambda b: {"x": b.t, "y": exp(b.s)},
], ids=["diagonal", "permuted"])
def test_a_chart_that_only_moves_the_nodes_is_refused_naming_graded(components):
    """A chart whose components each read one coordinate, a different one, is graded spacing."""
    box = surface.Box(s=(0, 1), t=(0, 1))
    with pytest.raises(SupportMismatch, match="pops.Graded") as refusal:
        box.chart(**components(box))
    assert "nodes" in str(refusal.value)


def test_a_keyword_of_the_chart_may_not_name_an_axis_of_the_support():
    box = surface.Box(x=(1, 2), t=(0, 1))
    with pytest.raises(SupportMismatch, match="x"):
        box.chart(x=box.x * cos(box.t), y=box.x * sin(box.t))


def test_a_chart_takes_one_keyword_per_axis_of_the_support():
    ring = _ring()
    with pytest.raises(SupportMismatch, match="2"):
        ring.chart(x=ring.r * cos(ring.theta))


def test_a_frame_not_named_x_and_y_waits_for_the_supports_of_g03():
    ring = _ring()
    with pytest.raises(NotYetElaborated, match="G03"):
        ring.chart(u=ring.r * cos(ring.theta), v=ring.r * sin(ring.theta))


def test_a_chart_of_rank_three_is_planned_not_covered():
    box = surface.Box(a=(1, 2), b=(0, 1), c=(0, 1))
    with pytest.raises(NotYetElaborated, match="rank 3"):
        box.chart(x=box.a * cos(box.b), y=box.a * sin(box.b), z=box.c + box.b)


def test_one_chart_per_support_and_identify_before_chart():
    _, annulus = _annulus()
    with pytest.raises(SupportMismatch, match="one chart"):
        annulus.chart(x=annulus.r * cos(annulus.theta), y=annulus.r * sin(annulus.theta))
    with pytest.raises(SupportMismatch, match="identify"):
        annulus.identify  # noqa: B018


def test_weighted_waits_for_lot_l3():
    ring, annulus = _annulus()
    for support in (ring, annulus):
        with pytest.raises(NotYetElaborated, match="L3"):
            support.weighted  # noqa: B018


def test_a_chart_reads_the_coordinates_of_its_support_numbers_and_pi():
    ring, other = _ring(), _ring()
    with pytest.raises(SupportMismatch, match="coordinates of"):
        ring.chart(x=other.r * cos(ring.theta), y=ring.r * sin(ring.theta))
    with pytest.raises(NotYetElaborated, match="N11"):
        ring.chart(x=ring.r * cos(ring.theta), y=expm1(ring.r) * sin(ring.theta))
    with pytest.raises(TypeError, match="expression"):
        ring.chart(x="r cos(theta)", y=ring.r * sin(ring.theta))


def test_a_chart_whose_image_has_no_area_is_refused():
    """A component that reads no coordinate, or a coordinate that no component reads."""
    ring = _ring()
    with pytest.raises(SupportMismatch, match="reads no coordinate"):
        ring.chart(x=2.0, y=ring.r * sin(ring.theta))
    with pytest.raises(SupportMismatch, match="theta in none"):
        ring.chart(x=ring.r, y=2 * ring.r + 0 * ring.theta)


def test_the_logical_axes_name_the_nodes_of_a_graded_chart_not_the_frame_axes():
    _, annulus = _annulus()
    with pytest.raises(SupportMismatch, match="logical axis"):
        surface.Graded(base=surface.Uniform(cells=(8, 16), geometry=GaussLegendre(2)),
                       nodes={annulus.x: radial_nodes})


# --- the layout: geometry= (DN4, DN29) ----------------------------------------------------------

def test_a_charted_support_elaborates_to_the_chart_of_c1_with_its_rule():
    """The layout is the static case of the geometry-evolution contract, read at resolution as a
    chart of rank two with the stated rule, and pops.inspect reports the chart and the rule."""
    case, _, _ = build_chart_annulus()
    assert isinstance(case.elaborated.layout, MovingControlVolumes)
    report, reading = _reading(case)
    assert (reading.kind, reading.rank) == ("chart", 2)
    assert reading.rule is not None and len(reading.rule["nodes"]) == 2
    row = _layout_row(report)
    assert row.elaborated.startswith("MovingControlVolumes(Uniform(CartesianGrid(")
    assert "GeometryEvolution((r * cos(theta), r * sin(theta)))" in row.elaborated
    choices = {choice.name: choice for choice in row.choices}
    assert choices["cells"].value == ANNULUS_CELLS
    assert choices["chart"].origin == "stated"
    assert choices["chart"].value == "x = r * cos(theta), y = r * sin(theta)"
    assert choices["geometry"].origin == "stated"
    assert choices["degenerate pieces"].value == "none"


def test_a_charted_support_needs_its_rule():
    _, annulus = _annulus()
    with pytest.raises(AmbiguousElaboration, match="geometry="):
        _case(annulus, layout=surface.Uniform(cells=(8, 16)))


def test_geometry_takes_a_rule_on_a_bounded_interval():
    from pops.lib.quadrature import GaussHermite

    with pytest.raises(TypeError, match="quadrature rule"):
        surface.Uniform(cells=(8, 16), geometry=2)
    with pytest.raises(ValueError, match="bounded interval"):
        surface.Uniform(cells=(8, 16), geometry=GaussHermite(4))


def test_a_box_and_node_positions_refuse_a_rule():
    box = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    with pytest.raises(SupportMismatch, match="products of its spacings"):
        _case(box, layout=surface.Uniform(cells=(8, 8), geometry=GaussLegendre(2)))
    graded = surface.Graded(base=surface.Uniform(cells=(8, 8), geometry=GaussLegendre(2)),
                            nodes={box.x: lambda s: s * s})
    with pytest.raises(SupportMismatch, match="node positions"):
        _case(box, layout=graded)


def test_a_graded_chart_composes_its_nodes_with_the_chart():
    """pops.Graded on a charted support states the rule on its base; the radial nodes are f(s),
    and the coordinate map is the chart of the graded logical coordinate, still a chart."""
    case, _, _ = build_chart_annulus(graded=radial_nodes)
    report, reading = _reading(case)
    assert (reading.kind, reading.rank) == ("chart", 2)
    row = _layout_row(report)
    assert row.elaborated.endswith(
        "GeometryEvolution((r * cos(theta), r * sin(theta))), geometry=rule) where "
        "r = 1 + s * (1 + s) / 2 with s = r - 1.0")
    names = [choice.name for choice in row.choices]
    assert {"nodes of r", "ends of r", "chart", "geometry"} <= set(names)
    assert "geometry of r" not in names


# --- degenerate pieces (DN28) and coordinate reads (DN30) ---------------------------------------

def test_a_degenerate_piece_leaves_the_boundary_and_waits_for_identify():
    polar = surface.Box(r=(0, 1), theta=surface.Periodic(0, 2 * pi))
    disk = polar.chart(x=polar.r * cos(polar.theta), y=polar.r * sin(polar.theta))
    assert disk.piece_names() == ["r_max"]
    with pytest.raises(SupportMismatch, match="the pieces r_max; r_min is degenerate"):
        disk.boundary.r_min  # noqa: B018
    with pytest.raises(NotYetElaborated, match="identify"):
        _case(disk)
    # the products by an exact zero are folded: sin(0) is zero, cos(0) is not
    wedge = surface.Box(r=(0, 1), t=(0, 1))
    assert wedge.chart(x=wedge.r * cos(wedge.t), y=sin(wedge.r) * wedge.t).piece_names() \
        == ["r_max", "t_min", "t_max"]


def test_a_coordinate_read_on_a_charted_support_waits_for_lot_l2b():
    _, annulus = _annulus()
    with pytest.raises(NotYetElaborated, match="L2b"):
        _case(annulus, value=lambda u: vector(-annulus.y, annulus.x) * u)
    with pytest.raises(NotYetElaborated, match="L2b"):
        _case(annulus, value=lambda u: vector(annulus.r, 1.0) * u)


def test_a_value_relation_on_a_chart_waits_for_lot_l2b():
    """DN30: a value relation is refused where the case is built, naming the relation, its piece
    and lot L2b, whose reflected ghost serves it; outflow(u), which extends the cell, is served."""
    _, annulus = _annulus()
    with pytest.raises(NotYetElaborated, match="L2b") as caught:
        _case(annulus, inflow=0.1)
    message = str(caught.value)
    assert "the value relation model.on(r_min, u == 0.1) of the charted support" in message
    assert "reflects the ghost of a value relation across the face of a chart" in message


def test_a_reconstruction_above_the_cell_values_on_a_chart_waits_for_lot_l2b():
    """DN30: the traces of a face on a chart are the cell values; muscl, whose slopes read the
    logical stencil of a cell, is refused where the case is built, naming the scheme and lot L2b."""
    from pops.lib.numerics import muscl
    from pops.lib.numerics.limiters import vanleer

    _, annulus = _annulus()
    with pytest.raises(NotYetElaborated, match="L2b") as caught:
        _case(annulus, reconstruction=muscl(limiter=vanleer))
    message = str(caught.value)
    assert "rusanov(reconstruction=muscl(" in message
    assert "reconstructed to the formal order 3 from the logical stencil of a cell" in message


def test_rusanov_on_a_chart_states_that_its_speed_is_the_bound_of_its_face():
    """DN31: on a physical normal the face input s is b6's bound sum_k |n_k| rho(A_k), not the
    exact radius of spectral_radius; pops.inspect names it on the row of the scheme as a
    transitional route, and the same scheme on a box states nothing of it."""
    case, _, _ = build_chart_annulus()
    report = surface.inspect(case)
    (row,) = [row for row in report.rows if row.construct.startswith("schemes={")]
    choices = {choice.name: choice for choice in row.choices}
    assert choices["speed on the chart"].origin == "derived"
    value = choices["speed on the chart"].value
    assert "the bound sum_k |n_k| rho(A_k)" in value and "not the exact radius" in value
    assert "lot L2d of N17 (DN31)" in value
    box = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    (row,) = [row for row in surface.inspect(_case(box, layout=surface.Uniform(cells=(8, 8))))
              .rows if row.construct.startswith("schemes={")]
    assert "speed on the chart" not in [choice.name for choice in row.choices]


# --- the representation of a charted quantity (DN34) ---------------------------------------------

def _artifact(case):
    """What bind reads before it calls the native bind (as test_published_representation.py)."""
    bindings = (types.SimpleNamespace(subject="u"),)
    compiled = types.SimpleNamespace(plan=types.SimpleNamespace(
        initial_condition_plan=types.SimpleNamespace(bindings=bindings)))
    return _run.Artifact(case, compiled, surface.OpenMP())


def _published_on(case):
    """An array of 8 x 16 cells as a run of ``case`` publishes it in result.state[u]."""
    (representation,) = case.elaborated.representations.values()
    return _run.publish(numpy.zeros((16, 8)), representation)


def test_an_array_published_on_another_support_is_not_a_datum_of_the_annulus():
    """DN34: the representation of a quantity on a charted support carries its logical coordinates
    and the components of its chart. An array of the same 8 x 16 cells that a run published on the
    box [1, 2] x [0, 2 pi), or on the same logical ring under another chart, is refused by
    pops.bind as the datum of the annulus, before the native bind, naming both supports; the array
    that a run of the annulus published is read back under its identity."""
    _, annulus = _annulus()
    on_annulus = _case(annulus)
    (u0,) = on_annulus.model.data_fields
    on_box = _case(surface.Box(x=(1, 2), y=surface.Periodic(0, 2 * pi)),
                   layout=surface.Uniform(cells=(8, 16)))
    ring = surface.Box(r=(1, 2), theta=surface.Periodic(0, 2 * pi))
    on_ellipse = _case(ring.chart(x=ring.r * cos(ring.theta), y=2 * ring.r * sin(ring.theta)))
    chart = "x in chart r * cos(theta) x y in chart r * sin(theta)"
    for other, written in ((on_box, "x in [1.0, 2.0] x y in periodic[0.0, 6.283185307179586)"),
                           (on_ellipse, "x in chart r * cos(theta) x y in chart 2 * r * sin(theta)")):
        with pytest.raises(SupportMismatch, match="equal shapes do not make supports "
                                                  "interchangeable") as caught:
            surface.bind(_artifact(on_annulus), data={u0: _published_on(other)})
        message = str(caught.value)
        assert "a run published on the support %s" % written in message.replace(
            "r in [1.0, 2.0] x theta in periodic[0.0, 6.283185307179586) x ", "")
        assert ("as the datum of a quantity on the support r in [1.0, 2.0] x theta in "
                "periodic[0.0, 6.283185307179586) x %s" % chart) in message
    (representation,) = on_annulus.elaborated.representations.values()
    _run.require_published_under(_published_on(on_annulus), representation, "pops.bind")


# --- two-point faces on a chart (DN35, DN36) ---------------------------------------------------

def _scheme_row(report):
    (row,) = [row for row in report.rows if row.construct.startswith("schemes={")]
    return row


@pytest.mark.parametrize("chart", ["turned", "sheared"])
def test_two_point_on_a_chart_states_that_its_faces_assume_k_orthogonality(chart):
    """DN36: K-orthogonality is measured, not required. The diffusion of -k grad(u) by two_point()
    on a charted square resolves (one law on every logical axis, DN35), and pops.inspect states on
    the row of its scheme that a face takes the line of its two centroids for its normal, and that
    the arrays bound at pops.bind report the largest sine of the two."""
    case, _, _ = build_chart_diffusion(chart)
    report, reading = _reading(case)
    assert (reading.kind, reading.rank) == ("chart", 2)
    choices = {choice.name: choice for choice in _scheme_row(report).choices}
    assert choices["faces"].origin == "derived"
    assert "K-orthogonality" in choices["faces"].value
    assert "the line of its two centroids" in choices["faces"].value
    assert "the largest sine" in choices["faces"].value


def test_two_point_on_a_box_states_no_assumption_of_its_faces():
    case, _, _ = build_chart_diffusion(None)
    names = [choice.name for choice in _scheme_row(surface.inspect(case)).choices]
    assert "faces" not in names


def test_the_annulus_bounds_and_cells_are_logical():
    """cells= follows the logical axes (r, theta); the reference box is [1, 2] x [0, 2 pi)."""
    case, _, _ = build_chart_annulus()
    geometry = case.elaborated.layout.normalized_geometry()
    assert tuple(geometry.cells) == ANNULUS_CELLS
    assert tuple(geometry.lower) == (ANNULUS_RADII[0], 0.0)
    assert tuple(geometry.upper)[0] == ANNULUS_RADII[1]
