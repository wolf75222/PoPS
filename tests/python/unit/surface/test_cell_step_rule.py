"""N8 « dt= as a function of cell » (ROM-89), the surface trace (design note
``n8/cell_step_rule/design_note.md``, D1 and D3 (a)).

``pops.Advance(method, dt=f)`` takes a plain function of ``cell`` (SU5). The elaborator runs it once
with the cell of a step rule, whose reads are ``value``, ``width`` and ``speed``, and lowers the
bound it returns to a b6 expression over the cell reads of the block state and the widths of the
cell. ``cell.speed(F, axis)`` is the spectral radius of F at the cell value (R9; D3 (a)). The
elaborated Advance carries the bound as its first proposal, under the Courant budget that the
function declares: 1, that of a fixed step, for a function written by hand (C2, review rev 3 P3).
"""
import types

import pytest

import pops.surface as surface
from pops.lib.numerics import rusanov
from pops.lib.time.methods import ssprk2
from pops.mesh._amr.refinement_criterion import CellRead, CellWidth, criterion_inputs
from pops.surface import _elaborate
from pops.surface._refusal import NotYetElaborated
from pops.surface.math import ddt, div, vector

CFL = 0.45


def _parts():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    return types.SimpleNamespace(domain=domain, model=model, u=u, x=domain.x, y=domain.y,
                                 advection=advection)


def _case(parts, dt):
    numerics = surface.Numerics(represent={parts.u: surface.CellAverage()},
                                schemes={parts.advection: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=parts.model), dt=dt),
                              initial={parts.u: 1.0})
    return surface.Case(model=parts.model, numerics=numerics, program=program,
                        layout={parts.domain: surface.Uniform(cells=(16, 8))})


def _limit(parts):
    def limit(cell):
        return CFL * surface.math.smallest([cell.width(parts.x), cell.width(parts.y)]) / \
            surface.math.largest([cell.speed(parts.advection, parts.x),
                                  cell.speed(parts.advection, parts.y)])
    return limit


def _bound(build, monkeypatch):
    """The bound and its b6 expression, as the elaborator builds them (``surface.Case`` elaborates
    the case it builds)."""
    seen = {}
    original = _elaborate._Elaboration.step_bound

    def spy(self, block, states):
        seen["result"] = original(self, block, states)
        return seen["result"]

    monkeypatch.setattr(_elaborate._Elaboration, "step_bound", spy)
    build()
    return seen["result"]


def _leaves(expression, kind):
    """The leaves of ``kind`` that N12's generated cell reductions take (``criterion_inputs``)."""
    return [leaf for leaf in criterion_inputs((expression,)) if isinstance(leaf, kind)]


def test_a_function_of_cell_is_a_step_rule_of_the_advance():
    parts = _parts()
    limit = _limit(parts)
    advance = surface.Advance(ssprk2(rate=parts.model), dt=limit)
    assert advance.dt is limit


def test_the_bound_lowers_to_the_cell_reads_and_widths(monkeypatch):
    parts = _parts()
    bound, lowered = _bound(lambda: _case(parts, _limit(parts)), monkeypatch)
    text = surface._expr.text(bound)
    assert "cell.width(x)" in text and "cell.width(y)" in text
    assert "cell.value(u)" not in text  # a constant velocity: dF/du reads no value
    widths = {leaf.axis for leaf in _leaves(lowered, CellWidth)}
    assert widths == {0, 1}
    assert not _leaves(lowered, CellRead)


def test_a_speed_that_reads_the_value_lowers_to_a_cell_read(monkeypatch):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("burgers")
    u = model.quantity("u", surface.Scalar(domain))
    flux = model.flux(of=u, value=u * u / 2)
    model.balance(ddt(u) == -div(flux))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={flux: rusanov()})

    def limit(cell):
        return CFL * cell.width(domain.x) / cell.speed(flux, domain.x)

    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=limit),
                              initial={u: 1.0})
    bound, lowered = _bound(lambda: surface.Case(
        model=model, numerics=numerics, program=program,
        layout={domain: surface.Uniform(cells=(16,))}), monkeypatch)
    assert "cell.value(u)" in surface._expr.text(bound)
    (read,) = {(leaf.handle.qualified_id, leaf.component, leaf.offset)
               for leaf in _leaves(lowered, CellRead)}
    assert read[2] == ()


def test_a_function_written_by_hand_takes_the_budget_of_a_fixed_step():
    parts = _parts()
    case = _case(parts, _limit(parts))
    bound = case.elaborated.case._time._advance_step_bound
    assert (bound.fraction, bound.origin) == (1.0, "default")
    assert case.elaborated.case._time._advance_control.step_bound[0] == 1.0
    rows = {choice.name: choice for row in case.elaborated.rows for choice in row.choices}
    assert rows["fraction"].origin == "default"
    assert "that of a fixed step" in str(rows["fraction"].value)
    assert "limit, the smallest over the cells of" in str(rows["dt"].value)


def test_a_declared_budget_is_read_from_the_rule():
    parts = _parts()
    limit = _limit(parts)
    limit.step_fraction = 0.5
    bound = _case(parts, limit).elaborated.case._time._advance_step_bound
    assert (bound.fraction, bound.origin) == (0.5, "stated")
    for wrong, error in ((1.5, ValueError), (0.0, ValueError), (float("nan"), ValueError),
                         (True, TypeError), ("0.5", TypeError)):
        limit.step_fraction = wrong
        with pytest.raises(error, match=r"pops.Advance\(dt=limit\): the step_fraction of a "
                                        r"step rule"):
            _case(parts, limit)


@pytest.mark.parametrize("read", ["shift", "center", "measure", "load", "level"])
def test_a_read_of_a_later_lot_is_refused_with_its_lot(read):
    parts = _parts()

    def limit(cell):
        getattr(cell, read)
        return cell.width(parts.x)

    with pytest.raises(NotYetElaborated, match="cell.%s in dt= of pops.Advance" % read):
        _case(parts, limit)


def test_a_bound_that_reads_no_cell_is_a_number():
    parts = _parts()
    with pytest.raises(TypeError, match="reads no cell: write dt="):
        _case(parts, lambda cell: 1e-3)


def test_the_speed_of_a_flux_needs_a_transport_flux():
    parts = _parts()
    with pytest.raises(TypeError, match="takes a transport flux"):
        _case(parts, lambda cell: cell.speed(parts.u, parts.x))


def test_an_axis_of_another_support_is_refused():
    parts = _parts()
    other = surface.Box(z=surface.Periodic(0, 1))
    with pytest.raises(TypeError, match="cell.width takes an axis of the support"):
        _case(parts, lambda cell: cell.width(other.z))
