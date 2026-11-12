"""N3 lot 1 (ROM-87): the scheme words of ``pops.lib.numerics`` are bodies of ``@pops.scheme``.

``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()`` (``python/pops/lib/numerics/fluxes.py``)
replace the bridges of ``pops.lib.numerics.schemes``, which returned b6's native routes.

* E1, P1: each word gives, operation by operation, the case written by hand with its b6 oracle
  (``examples/api_v02/b6_numerics.py``, D5): the same program IR, artifact, semantic identity and
  generated C++. ``upwind()`` gives the face of the b6 ``rusanov`` (D1), ``upwind(speed=f)`` the
  b6 ``upwind(speed=)``, and ``two_point()`` a ``StencilFlux`` written by hand (D7, D10).
* E3, CF4: a copy of the library's bodies written in a case elaborates into the same case as the
  library's words.
* G10 (O2 of the review of the lot note): a diffusive face of four points is not elaborated, and
  names its lot; a face of the form k (w_R - w_L) / face.distance that is not -k grad(w) neither.
* O3: the refusals that a diffusive face meets name their owners: G04 for a wall, N12 for a
  hierarchy.
"""
import inspect
import json

import pops
import pops.lib.numerics.fluxes as library
import pops.surface as surface
import pytest
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import hll, rusanov, two_point, upwind
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.numerics.stencil_flux import StencilFlux
from pops.params import Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import FaceFlux
from pops.surface.math import ddt, div, dot, grad
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

from examples.api_v02 import b6_numerics

CELLS = 32


def _n8_method():
    """ssprk2 written for N8, as ``tests/python/unit/surface/test_su_t3_parity.py`` writes it."""
    @method
    def ssprk2(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)

    return ssprk2


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


def _assert_same(case, other_case, other_layout):
    elaborated = case.elaborated
    assert elaborated.case._time._serialize(include_provenance=False) == \
        other_case._time._serialize(include_provenance=False)
    resolved = _resolved(elaborated.case, elaborated.layout)
    other = _resolved(other_case, other_layout)
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(other))
    assert resolved.snapshot.semantic_identity == other.snapshot.semantic_identity
    assert _cpp(resolved) == _cpp(other)


# --- the cases ---------------------------------------------------------------------------------

LINEAR = (lambda u: 2 * u, lambda u: 2 * u[0], lambda u: 2.0)
BURGERS = (lambda u: u**2 / 2, lambda u: u[0] ** 2 / 2, lambda u: u[0])


def _surface_transport(flux, scheme):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("transport")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    term = model.flux(of=u, value=flux[0](u))
    model.balance(ddt(u) == -div(term))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={term: scheme(u)})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})


def _frame():
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    return frame, frame.axes[0]


def _finish(model, plan, block_name, u, transport, frame, x):
    case = pops.Case(block_name)
    block = case.block(block_name, model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_method()(rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(CELLS,), periodic=PeriodicAxes((x,))))


def _hand_transport(flux, oracle, signed=False):
    """The case written for b6 with the face ``oracle(flux_u, u)`` of b6_numerics."""
    frame, x = _frame()
    model = pops.Model("transport", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (flux[1](u),)},
                        waves={x: (flux[2](u),)})
    if signed:
        model.wave_speeds(flux_u, frame=frame, values={x: (flux[2](u), flux[2](u))})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, oracle(flux_u, u))
    return _finish(model, plan, "transport", u, lambda state: selection(state), frame, x)


def _surface_heat(scheme):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("heat")
    u = model.quantity("u", surface.Scalar(domain))
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    term = model.diffusive_flux(of=u, value=-nu * grad(u))
    model.balance(ddt(u) == -div(term))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={term: scheme()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})


def _hand_heat():
    """The heat equation of nu, a runtime parameter, written for b6: the state is stored with one
    ghost layer, and each evaluation of the rate is the StencilFlux of the two-point face
    -nu (u_R - u_L) / h, of frequency (nu / h) / h (D7, D10)."""
    frame, x = _frame()
    model = pops.Model("heat", frame=frame)
    nu = model.value(model.param(RuntimeParam("nu", default=0.01, domain=Positive())))
    u = model.state("u", components=("u",))
    model.diffusive_flux("diffusive_flux_u", state=u, value=nu * pm.grad(u[0]))
    plan = DiscretizationPlan()
    plan.states.add(u, StateStorage(ghost_depth=1))

    def body(sample, spacing, axis):
        h = spacing[axis]
        u_left = sample((-1,), 0)
        u_right = sample((0,), 0)
        return ((-nu) * ((u_right - u_left) / h),), (nu / h) / h

    flux = StencilFlux(dimension=1, components=1, radius=1, body=body)

    def transport(state):
        program = state.prog
        evaluation = sum(1 for value in program._values
                         if value.op == "stencil_flux_divergence")
        return program.stencil_flux_divergence("diffusive_flux_u_%d" % evaluation, state,
                                               flux=flux)

    return _finish(model, plan, "heat", u, transport, frame, x)


# --- E1, P1: each word is the case written with its b6 oracle ----------------------------------

@pytest.mark.parametrize("flux, word, oracle, signed", [
    (LINEAR, lambda u: upwind(), b6_numerics.rusanov, False),
    (BURGERS, lambda u: rusanov(), b6_numerics.rusanov, False),
    (BURGERS, lambda u: hll(), b6_numerics.hll, True),
    (LINEAR, lambda u: upwind(speed=lambda face: 2.0 * face.normal),
     lambda flux_u, u: b6_numerics.upwind(flux_u, u, speed=lambda face: 2.0 * face.normal[0]),
     False),
], ids=["upwind", "rusanov", "hll", "upwind-speed"])
def test_each_word_is_its_b6_oracle_operation_by_operation(flux, word, oracle, signed):
    _assert_same(_surface_transport(flux, word), *_hand_transport(flux, oracle, signed))


def test_two_point_is_the_stencil_flux_written_by_hand():
    _assert_same(_surface_heat(two_point), *_hand_heat())


def test_a_balance_of_stencils_alone_states_its_storage():
    elaborated = _surface_heat(two_point).elaborated
    report = surface.inspect(_surface_heat(two_point))
    assert "no rate of b6: in the program the StencilFlux of diffusive_flux_u" in str(report)
    assert [value.op for value in elaborated.case._time._values].count(
        "stencil_flux_divergence") == 2


# --- E3, CF4: a copy of each body written in a case is the library's word ----------------------

def _copied_words():
    """The source of ``pops.lib.numerics.fluxes``, run as a case runs it: new schemes, the same
    text."""
    namespace = {"__name__": "a_case"}
    exec(compile(inspect.getsource(library), __file__, "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("flux, name, options", [
    (LINEAR, "upwind", {}), (BURGERS, "rusanov", {}), (BURGERS, "hll", {}),
    (LINEAR, "upwind", {"speed": lambda face: 2.0 * face.normal}),
], ids=["upwind", "rusanov", "hll", "upwind-speed"])
def test_a_copy_of_a_body_in_a_case_is_the_library_word(flux, name, options):
    copy = _copied_words()[name]
    assert copy is not getattr(library, name)
    case = _surface_transport(flux, lambda u: copy(**options))
    word = _surface_transport(flux, lambda u: getattr(library, name)(**options)).elaborated
    _assert_same(case, word.case, word.layout)


def test_a_copy_of_two_point_in_a_case_is_the_library_word():
    copy = _copied_words()["two_point"]
    word = _surface_heat(two_point).elaborated
    _assert_same(_surface_heat(copy), word.case, word.layout)


# --- G10: the forms of a diffusive face that this lot does not elaborate -------------------------

def test_a_diffusive_face_of_four_points_names_its_lot():
    @surface.scheme
    def four_point(face, term):
        q = term.quantity
        wide = face.left_cell.shift(-1).value(q)
        return FaceFlux(wide - face.left(q))

    with pytest.raises(surface.NotYetElaborated, match="stencil faces of N7a"):
        _surface_heat(four_point)


def test_a_diffusive_face_of_another_coefficient_names_its_lot():
    @surface.scheme
    def doubled(face, term):
        q = term.quantity
        law = term.law(states={q: face.average(q)},
                       gradients={q: face.normal_gradient(q) * face.normal})
        return FaceFlux(2 * dot(law, face.normal))

    with pytest.raises(surface.NotYetElaborated, match="coupled and drift realizations of G05"):
        _surface_heat(doubled)


# --- O3: the refusals of a diffusive face name their owners -------------------------------------

def test_a_diffusive_face_at_a_wall_names_g04():
    domain = surface.Box(x=(0, 1))
    model = surface.Model("heat")
    u = model.quantity("u", surface.Scalar(domain))
    term = model.diffusive_flux(of=u, value=-1e-2 * grad(u))
    model.balance(ddt(u) == -div(term))
    u0 = model.data("u0", surface.Scalar(domain))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={term: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.MissingBoundaryRealization) as caught:
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: surface.Uniform(cells=CELLS)})
    message = str(caught.value)
    assert "the diffusive flux" in message and "StencilFlux, which is periodic" in message
    assert "the boundary policies of a diffusive face are elaborated by G04" in message


def test_a_diffusive_face_on_a_hierarchy_names_n12():
    with pytest.raises(surface.NotYetElaborated, match="hierarchies of N12"):
        surface.AMR()
