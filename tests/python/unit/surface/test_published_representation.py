"""G03 lot E (ROM-21, K4, K7): the surface keeps the elaborated identity of each state.

b6's state carries neither the support nor the representation of the quantity it elaborates, so
two quantities of equal shape on two supports, or under two representations, were interchangeable
once elaborated. The elaboration now keeps ``Representation(space, definition)`` of contract @2
beside each state (``surface/_elaborate.py``); a run publishes its arrays under it
(``result.state[u]``, the publications of an Output), and ``bind(data=...)`` refuses a published
array as the datum of a quantity on another support before it calls the native bind. Nothing of it
enters b6, so the b6 case, its plan and its identity are those of the base. The run that publishes
is checked in ``tests/python/integration/runtime/test_surface_lot_g_run_native.py``.
"""
import types

import numpy
import pytest

import pops.surface as surface
from pops.lib.numerics.schemes import two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.representations.definition import (
    ElaboratedRepresentation, cell_average, cell_total,
)
from pops.surface import _run
from pops.surface._elaborate import elaborated_representation
from pops.surface.math import ddt, div, grad


def _case(domain, cells=16, name="advection_diffusion", quantity="u"):
    model = surface.Model(name)
    u = model.quantity(quantity, surface.Scalar(domain))
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=1.0 * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3, attempts=2),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=cells)})
    return types.SimpleNamespace(case=case, u=u, u0=u0)


def _artifact(case):
    """What bind reads before it calls the native bind (as test_lot_g_run.py)."""
    bindings = (types.SimpleNamespace(subject="u"),)
    compiled = types.SimpleNamespace(plan=types.SimpleNamespace(
        initial_condition_plan=types.SimpleNamespace(bindings=bindings)))
    return _run.Artifact(case, compiled, surface.OpenMP())


def _published(parts, values=None):
    """An array as a run of ``parts`` publishes it in result.state[u]."""
    elaborated = parts.case.elaborated
    array = numpy.zeros(16) if values is None else values
    return _run.publish(array, elaborated.representations[parts.u])


def test_each_elaborated_state_keeps_its_representation():
    parts = _case(surface.Box(x=surface.Periodic(0, 1)))
    elaborated = parts.case.elaborated
    (representation,) = elaborated.representations.values()
    assert set(elaborated.representations) == set(elaborated.states)
    assert isinstance(representation, ElaboratedRepresentation)
    assert representation.definition.identity == cell_average(1).identity
    assert representation.space.support.coordinates == (("x", "periodic[0.0, 1.0)"),)
    assert representation.identity.startswith("pops.representation:sha256:")


def test_the_identity_reads_the_support_not_the_names():
    """P10: the name of the model, of the quantity and of the Box object are presentation."""
    a = _case(surface.Box(x=surface.Periodic(0, 1)))
    b = _case(surface.Box(x=surface.Periodic(0, 1)), name="other", quantity="rho")
    c = _case(surface.Box(x=surface.Periodic(0, 2)))
    identity = [next(iter(parts.case.elaborated.representations.values())).identity
                for parts in (a, b, c)]
    identity.append(elaborated_representation(None, surface.Box(x=(0, 1))).identity)
    identity.append(elaborated_representation(None, surface.Box(y=surface.Periodic(0, 1))).identity)
    assert identity[0] == identity[1]
    assert len(set(identity[:1] + identity[2:])) == 4


def test_the_b6_state_is_unchanged():
    """Nothing enters b6: the state space and its representation are the base's."""
    parts = _case(surface.Box(x=surface.Periodic(0, 1)))
    (state,) = parts.case.elaborated.states.values()
    space = state.space
    assert space.representation == "conservative" and space.representation_data is None


def test_a_published_array_is_rebound_on_its_own_support():
    parts = _case(surface.Box(x=surface.Periodic(0, 1)))
    again = _case(surface.Box(x=surface.Periodic(0, 1)))
    array = _published(parts)
    expected = again.case.elaborated.representations[again.u]
    _run.require_published_under(array, expected, "pops.bind(data={u0: array})")
    assert _run.published_representation(array) == expected


def test_a_published_array_is_refused_on_another_support_of_equal_shape():
    """K4: same size, other support, refused before publication (here, before the bind)."""
    a = _case(surface.Box(x=surface.Periodic(0, 1)))
    b = _case(surface.Box(x=surface.Periodic(0, 2)))
    array = _published(a)
    with pytest.raises(surface.SupportMismatch,
                       match=r"published on the support x in periodic\[0.0, 1.0\).*"
                             r"x in periodic\[0.0, 2.0\)") as refusal:
        surface.bind(_artifact(b.case), data={b.u0: array})
    assert refusal.value.rule == "E2" and refusal.value.phase == "validate"


def test_an_unpublished_or_copied_array_is_an_authoring_assertion():
    """A bare array carries no identity; a copy of a published one neither (K7: an observation
    gains no authority, and none is taken from it)."""
    a = _case(surface.Box(x=surface.Periodic(0, 1)))
    b = _case(surface.Box(x=surface.Periodic(0, 2)))
    expected = b.case.elaborated.representations[b.u]
    array = _published(a)
    for value in (numpy.zeros(16), numpy.array(array), array + 0.0, array[:]):
        assert _run.published_representation(value) is None
        _run.require_published_under(value, expected, "pops.bind(data={u0: array})")


def test_the_identity_is_forgotten_with_its_array():
    parts = _case(surface.Box(x=surface.Periodic(0, 1)))
    array = _published(parts)
    key = id(array)
    assert key in _run._PUBLISHED
    del array
    assert key not in _run._PUBLISHED


def test_another_representation_on_one_support_is_refused():
    """Equal shapes on one support under two representations: lot R's require_same. The surface
    elaborates CellAverage() only until N16 lot 1, so the record is built below the surface."""
    parts = _case(surface.Box(x=surface.Periodic(0, 1)))
    average = parts.case.elaborated.representations[parts.u]
    total = ElaboratedRepresentation(average.space, cell_total(1))
    array = _run.publish(numpy.zeros(16), total)
    with pytest.raises(TypeError, match="equal shapes do not make them interchangeable"):
        _run.require_published_under(array, average, "pops.bind(data={u0: array})")


def test_the_support_of_rank_two_names_both_axes():
    domain = surface.Box(x=surface.Periodic(0, 1), y=(0, 2))
    representation = elaborated_representation(None, domain)
    assert representation.space.support.coordinates == (
        ("x", "periodic[0.0, 1.0)"), ("y", "[0.0, 2.0]"))
    assert representation.definition.identity == cell_average(2).identity
