"""Lot B of N15 (ROM-98, OD31): the boundary relations of the surface and the policy ``ghosts()``.

``model.on(pieces, relation)`` states a relation of an evolved quantity on boundary pieces of its
support (2.6.2, SU14, D9); this lot elaborates the forms ``Value`` (``u == g``, g free of the
quantities) and ``NoData`` (``outflow(u)``), and refuses the others with ``NotYetElaborated``.
``pops.Numerics(boundary=ghosts())`` realizes each relation: the elaborator runs the policy under
each relation at the depths that b6 reads, checks that the ghost it gives is a ghost law of b6
(2 g - u_k is ``Inflow``, u_1 is ``Outflow``) and adds the ``TransportBoundarySet`` of the block
to the plan (T5). Each case gives the case written by hand for b6 (SU-T3: program IR, artifact
without provenance, semantic identity, C++). The compiled run is checked by
``tests/python/integration/runtime/test_surface_lot_b_boundaries_native.py``.
"""
import json
import types

import pytest

from examples.api_v02 import b6_numerics

import pops
import pops.surface as surface
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics.boundaries import ghosts
from pops.lib.numerics import rusanov, upwind
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import Ghost, forms
from pops.surface.math import ddt, div, grad, outflow, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method


def _n8_method():
    @method
    def ssprk2(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)

    return ssprk2


_n8_ssprk2 = _n8_method()


# --- the cases ------------------------------------------------------------------------------------

def _parts(periodic_x=False):
    x = surface.Periodic(0, 1) if periodic_x else (0.0, 1.0)
    domain = surface.Box(x=x, y=(0.0, 1.0))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    return types.SimpleNamespace(domain=domain, sides=domain.boundary, model=model, u=u, a=a,
                                 b=b, u0=u0, advection=advection)


def _case(parts, scheme=None, boundary="ghosts"):
    policy = ghosts() if boundary == "ghosts" else boundary
    numerics = surface.Numerics(represent={parts.u: surface.CellAverage()},
                                schemes={parts.advection: scheme or rusanov()},
                                boundary=policy)
    program = surface.Program(advance=surface.Advance(ssprk2(rate=parts.model), dt=1e-3),
                              initial={parts.u: parts.u0})
    return surface.Case(model=parts.model, numerics=numerics, program=program,
                        layout={parts.domain: surface.Uniform(cells=(16, 8))})


def _inflow_outflow(parts, value=0.05):
    inlet = parts.model.on([parts.sides.x_min, parts.sides.y_min], parts.u == value)
    outlet = parts.model.on([parts.sides.x_max, parts.sides.y_max], outflow(parts.u))
    return inlet, outlet


def _hand_case(laws, periodic_x=False, inlet=0.05):
    """The case written for b6 and N8: ``laws(boundaries, state, inlet)`` gives the transport
    boundaries of the block, keyed by the boundaries of the frame."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],)}, waves={x: (a,), y: (b,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, b6_numerics.rusanov(flux_u, u))
    value = inlet(model) if callable(inlet) else inlet
    case = pops.Case("advection")
    block = case.block("advection", model)
    plan.boundaries.add(TransportBoundarySet(
        laws(frame.boundaries, block[u], value),
        periodic=PeriodicAxes((x,)) if periodic_x else None))
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        return selection(state)

    elaborate(Advance(_n8_ssprk2(rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(16, 8),
                                       periodic=PeriodicAxes((x,)) if periodic_x else None))


def _four_sides(boundaries, state, value):
    return {boundaries.x_min: Inflow(state=state, value=value),
            boundaries.y_min: Inflow(state=state, value=value),
            boundaries.x_max: Outflow(state=state),
            boundaries.y_max: Outflow(state=state)}


def _resolved(case, layout):
    return pops.resolve(pops.validate(case), layout=layout)


def _cpp(resolved):
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _without_provenance(value):
    if isinstance(value, dict):
        return {key: _without_provenance(item) for key, item in value.items()
                if key != "provenance"}
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def _artifact(resolved):
    return json.loads(resolved.snapshot._artifact_canonical_json)


def _assert_same(case, hand):
    elaborated = case.elaborated
    assert elaborated.case._time._serialize(include_provenance=False) == hand[0]._time._serialize(
        include_provenance=False)
    resolved = _resolved(elaborated.case, elaborated.layout)
    by_hand = _resolved(*hand)
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(by_hand))
    assert resolved.snapshot.semantic_identity == by_hand.snapshot.semantic_identity
    assert _cpp(resolved) == _cpp(by_hand)


# --- SU-T3: the elaborated case is the case written by hand --------------------------------------

def test_a_value_and_no_data_are_the_inflow_and_outflow_of_b6():
    parts = _parts()
    _inflow_outflow(parts)
    _assert_same(_case(parts), _hand_case(_four_sides))


def test_the_data_of_a_value_may_read_a_constant_of_the_model():
    parts = _parts()
    g = parts.model.constant("g", value=0.125)
    parts.model.on([parts.sides.x_min, parts.sides.y_min], parts.u == 2 * g)
    parts.model.on([parts.sides.x_max, parts.sides.y_max], outflow(parts.u))

    def inlet(model):
        return 2 * model.value(model.param(ConstParam("g", 0.125)))

    _assert_same(_case(parts), _hand_case(_four_sides, inlet=inlet))


def test_the_data_of_a_value_may_read_a_runtime_parameter_bound_at_bind():
    parts = _parts()
    g = parts.model.parameter("g", default=0.05)
    parts.model.on([parts.sides.x_min, parts.sides.y_min], parts.u == g)
    parts.model.on([parts.sides.x_max, parts.sides.y_max], outflow(parts.u))
    case = _case(parts)
    assert list(case.elaborated.params) == [g]

    def inlet(model):
        return model.value(model.param(RuntimeParam("g", default=0.05)))

    _assert_same(case, _hand_case(_four_sides, inlet=inlet))


def test_a_periodic_axis_and_relations_on_the_other():
    parts = _parts(periodic_x=True)
    parts.model.on(parts.sides.y_min, parts.u == 0.05)
    parts.model.on(parts.sides.y_max, outflow(parts.u))

    def laws(boundaries, state, value):
        return {boundaries.y_min: Inflow(state=state, value=value),
                boundaries.y_max: Outflow(state=state)}

    _assert_same(_case(parts), _hand_case(laws, periodic_x=True))


def test_inspect_reports_the_form_and_the_ghost_of_each_relation():
    parts = _parts()
    _inflow_outflow(parts)
    report = str(surface.inspect(_case(parts)))
    assert "Inflow(state=block[u], value=0.05) on x_min, y_min" in report
    assert "form = Value (derived)" in report
    assert "ghost = 2 g - u_k (FixedState), by ghosts() (derived)" in report
    assert "Outflow(state=block[u]) on x_max, y_max" in report
    assert "ghost = u_1 (Extrapolate), by ghosts() (derived)" in report
    assert "periodic=None" in report


# --- model.on -------------------------------------------------------------------------------------

def test_model_on_returns_a_handle_with_its_form_and_data():
    parts = _parts()
    inlet, outlet = _inflow_outflow(parts)
    assert inlet.kind == "BoundaryRelation" and inlet.form == "Value" and inlet.data == 0.05
    assert repr(inlet) == "model.on([x_min, y_min], u == 0.05)"
    assert outlet.form == "NoData" and outlet.data is None
    assert repr(outlet) == "model.on([x_max, y_max], outflow(u))"
    assert repr(outflow(parts.u)) == "outflow(u)"
    with pytest.raises(TypeError, match="no truth value"):
        bool(outflow(parts.u))
    with pytest.raises(TypeError, match="evolved quantity"):
        outflow(parts.u0)


def test_model_on_checks_its_pieces():
    parts = _parts()
    with pytest.raises(TypeError, match="a tuple is a point"):
        parts.model.on((parts.sides.x_min, parts.sides.y_min), parts.u == 0.0)
    with pytest.raises(TypeError, match="boundary piece"):
        parts.model.on("x_min", parts.u == 0.0)
    with pytest.raises(ValueError, match="names a piece twice"):
        parts.model.on([parts.sides.x_min, parts.sides.x_min], parts.u == 0.0)
    other = surface.Box(x=(0.0, 2.0), y=(0.0, 1.0))
    with pytest.raises(ValueError, match="not a piece of the support of u"):
        parts.model.on(other.boundary.x_min, parts.u == 0.0)
    with pytest.raises(TypeError, match="takes a relation"):
        parts.model.on(parts.sides.x_min, 0.0)


def test_a_second_relation_of_the_same_field_on_a_piece_is_ambiguous():
    parts = _parts()
    parts.model.on(parts.sides.x_min, parts.u == 0.0)
    with pytest.raises(surface.AmbiguousElaboration, match="already has the relation"):
        parts.model.on([parts.sides.y_min, parts.sides.x_min], outflow(parts.u))


def test_the_other_forms_and_options_name_their_lot():
    parts = _parts()
    with pytest.raises(surface.NotYetElaborated, match="NormalFlux"):
        parts.model.on(parts.sides.x_min, -grad(parts.u) == 0.0)
    with pytest.raises(surface.NotYetElaborated, match="datum u0"):
        parts.model.on(parts.sides.x_min, parts.u == parts.u0)
    with pytest.raises(surface.NotYetElaborated, match="closes=B"):
        parts.model.on(parts.sides.x_min, parts.u == 0.0, closes=parts.model)


# --- the boundary policy --------------------------------------------------------------------------

def test_a_piece_without_a_relation_is_a_missing_realization():
    parts = _parts()
    parts.model.on([parts.sides.x_min, parts.sides.y_min], parts.u == 0.05)
    with pytest.raises(surface.MissingBoundaryRealization,
                       match="crosses the boundary pieces x_max, y_max|"
                             "cross the boundary pieces x_max, y_max") as caught:
        _case(parts).elaborated  # noqa: B018
    assert "model.on(Ω.boundary.x_max" in caught.value.write


def test_relations_without_a_policy_are_a_missing_realization():
    parts = _parts()
    _inflow_outflow(parts)
    with pytest.raises(surface.MissingBoundaryRealization,
                       match="the numerics state no boundary policy"):
        _case(parts, boundary=None).elaborated  # noqa: B018


def test_the_policy_is_a_called_scheme_over_face():
    parts = _parts()
    with pytest.raises(surface.NotAPopsOperation, match="uncalled"):
        surface.Numerics(represent={parts.u: surface.CellAverage()}, boundary=ghosts)
    with pytest.raises(surface.NotYetElaborated):
        surface.Numerics(represent={parts.u: surface.CellAverage()},
                         boundary={parts.sides.x_min: ghosts()})


def test_a_policy_whose_ghost_is_not_a_law_of_b6_names_its_lot():
    @surface.scheme
    def constant_ghost(face, term, depth):
        return Ghost(term.data)

    parts = _parts()
    _inflow_outflow(parts)
    with pytest.raises(surface.NotYetElaborated, match="not a ghost law of b6"):
        _case(parts, boundary=constant_ghost()).elaborated  # noqa: B018


def test_a_policy_written_as_the_library_is_the_library_ghosts():
    @surface.scheme
    def mirror(face, term, depth):
        if term.form is forms.Value:
            return Ghost(2 * term.data - face.inner(depth).value(term.quantity))
        return Ghost(face.inner(1).value(term.quantity))

    parts = _parts()
    _inflow_outflow(parts)
    _assert_same(_case(parts, boundary=mirror()), _hand_case(_four_sides))


def test_upwind_with_boundaries():
    parts = _parts()
    _inflow_outflow(parts)
    case = _case(parts, scheme=upwind())
    resolved = _resolved(case.elaborated.case, case.elaborated.layout)
    assert resolved.snapshot.semantic_identity


def test_a_policy_that_departs_from_the_laws_deeper_than_one_cell_is_refused():
    @surface.scheme
    def deep(face, term, depth):
        if term.form is forms.Value:
            return Ghost(2 * term.data - face.inner(depth).value(term.quantity))
        return Ghost(face.inner(depth).value(term.quantity))   # u_k, not u_1, past depth 1

    parts = _parts()
    _inflow_outflow(parts)
    with pytest.raises(surface.NotYetElaborated, match="not a ghost law of b6"):
        _case(parts, boundary=deep()).elaborated  # noqa: B018


def test_boundary_data_that_is_not_finite_is_refused_at_the_relation():
    parts = _parts()
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not finite"):
            parts.model.on(parts.sides.x_min, parts.u == value)
