"""N17 L1b (ROM-110, C1 and C2): graded spacing at the surface, ``pops.Graded`` (D0b, DN3).

``pops.Graded(base=pops.Uniform(cells=c), nodes={Ω.x: f})`` places node k of an axis at f(s),
s = k / c the logical coordinate of the node in [0, 1], the quotient correctly rounded. f is a
plain function, passed uncalled, which PoPS calls once with s and which returns an expression of s
(SU5, as section 13 of the design note N17 states it). The support stays a box: f(0) and f(1) are
its bounds within 4 ulp, then set to them, and f increases strictly at the nodes, or the case is
refused with ``SupportMismatch`` naming the ends or the first pair that does not increase. The
elaborated layout is the static case of the geometry-evolution contract: a coordinate map that
reads no clock and whose component a reads the logical coordinate a alone, which the static
geometry arrays serve as node positions at every rank (C1). The runs are the native tests
test_n17_graded_{ring,plane,box}_native.py; probe 3 is test_n17_probe_3_wall_layer_native.py.

At the base the word was not in the surface (``SymbolNotInSurface``) and a static map of rank three
was refused at resolution (``static_geometry_rank_unavailable``); both flip in this lot (CF15).
"""
from __future__ import annotations

import math

import pytest

import pops.surface as surface
from examples.api_v02.models_graded_spacing import (
    BOX_CELLS, RING_CELLS, build_graded_box, build_graded_plane, build_graded_ring,
    build_wall_layer, ring_nodes,
)
from pops._report import DiagnosticError
from pops.codegen._static_geometry_consumers import static_geometry_layouts
from pops.lib.numerics.schemes import two_point
from pops.lib.time.methods import ssprk2
from pops.mesh import MovingControlVolumes
from pops.surface.math import ddt, div, erf, expm1, grad, pi, sin, tanh


def _ring_case(nodes, cells=RING_CELLS, support=None):
    """The diffusion of the graded ring with the node function ``nodes``."""
    ring = support or surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("ring")
    u = model.quantity("u", surface.Scalar(ring))
    u0 = model.data("u0", surface.Scalar(ring))
    diffusion = model.diffusive_flux(of=u, value=-1e-2 * grad(u))
    model.balance(ddt(u) == -div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    layout = surface.Graded(base=surface.Uniform(cells=cells), nodes={ring.x: nodes})
    return surface.Case(model=model, numerics=numerics, program=program, layout={ring: layout})


def _reading(case):
    report = surface.inspect(case)
    (reading,) = static_geometry_layouts(report.resolved.layout_plan).values()
    return report, reading


def _layout_row(report):
    (row,) = [row for row in report.rows if row.construct.startswith("layout=")]
    return row


def test_graded_is_a_word_of_the_surface():
    """C2 lands pops.Graded with C1 (note, section 10, L1); the base refused the name."""
    assert "Graded" in surface.__all__
    assert isinstance(surface.Graded, type)


def test_a_graded_ring_elaborates_to_node_positions():
    """The layout is the static case of the geometry-evolution contract, read at resolution as
    node positions of rank one, and pops.inspect reports the node function, its ends and the
    arrays."""
    case, _, _ = build_graded_ring("diffusion")
    assert isinstance(case.elaborated.layout, MovingControlVolumes)
    report, reading = _reading(case)
    assert (reading.kind, reading.rank) == ("nodes", 1)
    row = _layout_row(report)
    assert row.elaborated.startswith("MovingControlVolumes(Uniform(CartesianGrid(")
    choices = {choice.name: choice for choice in row.choices}
    assert choices["nodes of x"].origin == "stated"
    assert choices["nodes of x"].value == (
        "ring_nodes(s) = s - 0.6 * sin(2 * pi * s) / (2 * pi)")
    assert choices["ends of x"].origin == "derived"
    assert "set to the bounds 0.0 and 1.0" in choices["ends of x"].value


@pytest.mark.parametrize("build,rank", [(build_graded_plane, 2), (build_graded_box, 3)])
def test_graded_boxes_of_rank_two_and_three_are_node_positions(build, rank):
    """A graded box of every rank to three is read as node positions; an axis that no function
    grades keeps its uniform nodes, the identity of its logical coordinate."""
    case, _, _ = build()
    _, reading = _reading(case)
    assert (reading.kind, reading.rank) == ("nodes", rank)


def test_the_node_function_is_called_once_with_the_logical_coordinate():
    """SU5: a plain function passed uncalled, which PoPS calls once, at tracing, with s."""
    calls = []

    def nodes(s):
        calls.append(str(s))
        return s * s * 0.5 + s * 0.5

    case = _ring_case(nodes)
    assert calls == ["s"]
    report, reading = _reading(case)
    assert reading.kind == "nodes" and calls == ["s"]


def test_the_ends_are_set_to_the_bounds_within_four_ulp():
    """f(1) two ulp above the bound is set to it; eight ulp above is refused, naming the ends."""
    def near(s):
        return s * (1 + 2 * 2.0 ** -52)

    report, _ = _reading(_ring_case(near))
    ends = {choice.name: choice for choice in _layout_row(report).choices}["ends of x"]
    assert ends.value.startswith("f(0) = 0.0, f(1) = %r" % (1 + 2 * 2.0 ** -52))

    def far(s):
        return s * (1 + 8 * 2.0 ** -52)

    with pytest.raises(surface.SupportMismatch) as caught:
        _ring_case(far)
    assert "sends s = 0 and s = 1 to 0.0 and %r, not to the bounds 0.0 and 1.0" % (
        1 + 8 * 2.0 ** -52) in str(caught.value)

    def shifted(s):
        return s + 1e-3

    with pytest.raises(surface.SupportMismatch, match="graded nodes move inside the box"):
        _ring_case(shifted)


def test_a_node_function_that_does_not_increase_names_the_first_pair():
    """DN3: strictly increasing at the nodes; the refusal names the first pair that is not."""
    def folded(s):
        return s + 0.3 * sin(2 * pi * s)

    values = [k / RING_CELLS + 0.3 * math.sin(2 * math.pi * (k / RING_CELLS))
              for k in range(RING_CELLS + 1)]
    first = next(k for k in range(RING_CELLS) if not values[k + 1] > values[k])
    with pytest.raises(surface.SupportMismatch) as caught:
        _ring_case(folded)
    assert ("does not increase strictly at the nodes of %d cells: node %d is at %r and node %d at "
            "%r, the first pair that does not increase"
            % (RING_CELLS, first, values[first], first + 1, values[first + 1])
            in str(caught.value))


def test_a_node_function_that_is_not_finite_at_a_node_is_refused():
    def singular(s):
        return s + 0 * (1 / s)

    with pytest.raises(surface.SupportMismatch,
                       match=r"is not finite at the node 0 of 32 \(s = 0.0\)"):
        _ring_case(singular)


def test_a_node_function_reads_its_coordinate_numbers_pi_and_constants():
    """The nodes enter the identity of the artifact: a function of s, numbers, pi and the
    constants of the model, never the coordinate x, a parameter or a quantity."""
    ring = surface.Box(x=surface.Periodic(0, 1))

    def reads_x(s):
        return s + 0 * ring.x

    with pytest.raises(surface.SupportMismatch, match="reads x; a node function returns"):
        _ring_case(reads_x, support=ring)

    def returns_text(s):
        return "s"

    with pytest.raises(surface.SupportMismatch, match="returns 's'"):
        _ring_case(returns_text)


@pytest.mark.parametrize("word,owner", [("expm1", "N11"), ("erf", "N11")])
def test_a_word_that_the_analytic_programs_lack_waits_for_n11(word, owner):
    """The arrays are generated at bind from the analytic programs of the map, which have exp,
    log, sqrt, sin, cos and abs; erf and erfc stay words of initial values (pops.math)."""
    function = {"expm1": expm1, "erf": erf}[word]

    def nodes(s):
        return s + 0 * function(s)

    with pytest.raises(surface.NotYetElaborated) as caught:
        _ring_case(nodes)
    assert owner in str(caught.value) and "node function" in str(caught.value)


def test_tanh_is_refused_at_its_call_by_pops_math():
    def nodes(s):
        return tanh(3 * s) / tanh(3)

    with pytest.raises(surface.NotYetElaborated, match="N11"):
        _ring_case(nodes)


def test_a_power_whose_exponent_is_not_a_number_waits_for_n11():
    def nodes(s):
        return s ** (1 + 0 * s)

    with pytest.raises(surface.NotYetElaborated, match="exponent is not a number"):
        _ring_case(nodes)


def test_the_spelling_of_graded_is_checked_when_it_is_written():
    ring = surface.Box(x=surface.Periodic(0, 1))
    other = surface.Box(x=(0, 1))
    with pytest.raises(TypeError, match="takes pops.Uniform"):
        surface.Graded(base=None, nodes={ring.x: ring_nodes})
    with pytest.raises(TypeError, match="keyed by the axes"):
        surface.Graded(base=surface.Uniform(cells=8), nodes={"x": ring_nodes})
    with pytest.raises(TypeError, match="a layout that grades no axis is pops.Uniform"):
        surface.Graded(base=surface.Uniform(cells=8), nodes={})
    with pytest.raises(TypeError, match="plain function of the logical coordinate"):
        surface.Graded(base=surface.Uniform(cells=8), nodes={ring.x: 3})
    with pytest.raises(surface.SupportMismatch, match="grades the axes of several supports"):
        surface.Graded(base=surface.Uniform(cells=8), nodes={ring.x: ring_nodes,
                                                             other.x: ring_nodes})


def test_a_graded_layout_grades_an_axis_of_its_own_support():
    other = surface.Box(x=surface.Periodic(0, 1))
    ring = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("ring")
    u = model.quantity("u", surface.Scalar(ring))
    u0 = model.data("u0", surface.Scalar(ring))
    diffusion = model.diffusive_flux(of=u, value=-1e-2 * grad(u))
    model.balance(ddt(u) == -div(diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.SupportMismatch, match="grades an axis of"):
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={ring: surface.Graded(base=surface.Uniform(cells=8),
                                                  nodes={other.x: ring_nodes})})


def test_the_box_of_rank_three_grades_two_axes_and_keeps_the_third():
    case, _, _ = build_graded_box()
    report, _ = _reading(case)
    row = _layout_row(report)
    names = [choice.name for choice in row.choices]
    assert names[0] == "cells" and row.choices[0].value == BOX_CELLS
    assert "nodes of x" in names and "nodes of z" in names and "nodes of y" not in names
    # z on [-1, 1]: s = (z + 1) / 2, as the arrays compute it
    assert "with s = (z + 1.0) / 2.0" in row.elaborated


def test_probe_3_elaborates_its_wall_layout_and_waits_on_g04_for_its_relations():
    """Probe 3 (note, 6.3), marched in time: its layout, pops.Graded with the wall nodes
    tanh(3 s) / tanh(3), elaborates to node positions (their arrays nest from 32 to 256 cells in
    test_n17_graded_ring_native.py); the case is refused at validation, as its uniform reference
    is, because a value relation under a diffusion has no realization yet (G04). The pin flips
    when G04 lands."""
    refusal = "a resolved spatial method must expose boundary_stencil_requirements()"
    case, _, _ = build_wall_layer()
    assert isinstance(case.elaborated.layout, MovingControlVolumes)
    row = _layout_row_of(case)
    assert row.elaborated.startswith("MovingControlVolumes(Uniform(CartesianGrid(")
    with pytest.raises(DiagnosticError) as caught:
        surface.inspect(case)
    assert refusal in str(caught.value)
    reference, _, _ = build_wall_layer(graded=False)
    with pytest.raises(DiagnosticError) as caught:
        surface.inspect(reference)
    assert refusal in str(caught.value)


def test_probe_3_steady_layer_waits_on_gmres_of_n9():
    """The steady layer of probe 3 is solved once with gmres(restart=64), which N9 elaborates;
    the pin flips with it."""
    from pops.lib.solve import gmres

    with pytest.raises(surface.NotYetElaborated, match="N9"):
        gmres(restart=64)


def _layout_row_of(case):
    (row,) = [row for row in case.elaborated.rows if row.construct.startswith("layout=")]
    return row
