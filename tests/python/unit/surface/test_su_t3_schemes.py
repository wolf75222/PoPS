"""SU-T3 and CF4 (spec 0.3, 2.6.7) for a scheme of one's own, lot 2 of N15 (ROM-98).

A case whose numerics name a scheme written with ``@pops.scheme`` elaborates into the case written
by hand for b6 with the same ``riemann.User`` body (and the same ``reconstruction.User``): the
same program IR, the same artifact once the source lines of its provenance are set aside, the same
semantic identity and the same generated C++. ``upwind(speed=f)``, f a plain function of ``face``,
gives the b6 ``upwind`` (``examples/api_v02/b6_numerics.py``) face of the case written by hand.

CF4: a scheme of one's own written in the arithmetic of b6 ``rusanov``
gives that face node for node, its ``largest`` of the two spectral radii being the face input
``s`` of ``riemann.User``; the native body-swap parity of N3
(``test_lib_face_body_swap_native.py``) ties that body to the native Rusanov.
"""
import json

import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from examples.api_v02 import b6_numerics
from pops.lib.numerics import rusanov, upwind
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import FaceFlux, Reconstruct, spectral_radius
from pops.surface.math import ddt, div, dot, largest, smallest, where
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


# --- the schemes, as a case writes them ---------------------------------------------------------

@surface.scheme
def local_lax_friedrichs(face, term, reconstruction=None):
    q = term.quantity
    left, right = face.left(q), face.right(q)
    s = largest([spectral_radius(term, left, face.normal),
                 spectral_radius(term, right, face.normal)])
    mean = (dot(term.law(states={q: left}), face.normal)
            + dot(term.law(states={q: right}), face.normal)) / 2
    return FaceFlux(mean - s * (right - left) / 2, reconstruction=reconstruction, speed=s)


@surface.scheme
def rusanov_as_the_library(face, term):
    q = term.quantity
    n = face.normal
    left, right = face.left(q), face.right(q)
    s = largest([spectral_radius(term, left, n), spectral_radius(term, right, n)])
    flux_left = dot(term.law(states={q: left}), n)
    flux_right = dot(term.law(states={q: right}), n)
    return FaceFlux(0.5 * (flux_left + flux_right) - 0.5 * s * (right - left), speed=s)


@surface.scheme
def central(face, term):
    """No speed=: the face records the spectral radius s of R9 (review N1)."""
    q = term.quantity
    return FaceFlux((dot(term.law(states={q: face.left(q)}), face.normal)
                     + dot(term.law(states={q: face.right(q)}), face.normal)) / 2)


def minmod(a, b):
    return where(a * b > 0, where(abs(a) < abs(b), a, b), 0)


@surface.scheme
def muscl(cell, term, axis, limiter):
    q = term.quantity
    centre = cell.value(q)
    backward = centre - cell.shift(axis, -1).value(q)
    forward = cell.shift(axis, 1).value(q) - centre
    return Reconstruct(centre + limiter(backward, forward) / 2)


# --- the two forms of a Burgers case ------------------------------------------------------------

def _surface_burgers(scheme):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("burgers")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    burgers = model.flux(of=u, value=u**2 / 2)
    model.balance(ddt(u) == -div(burgers))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={burgers: scheme(u)})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=64)})


def _hand_burgers(face, recon=lambda u: reconstruction.FirstOrder()):
    """The case written for b6 and N8; ``face(flux_u, u)`` is its finite volume's face."""
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("burgers", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (u[0] ** 2 / 2,)},
                        waves={x: (u[0],)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    realization = face(flux_u, u)
    if not isinstance(realization, FiniteVolume):
        realization = FiniteVolume(flux=flux_u, variables=variables.Conservative(u),
                                   reconstruction=recon(u), riemann=realization)
    plan.rates.add(selection, realization)
    case = pops.Case("burgers")
    block = case.block("burgers", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        return selection(state)

    elaborate(Advance(_n8_ssprk2(rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(64,), periodic=PeriodicAxes((x,))))


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


# --- the faces written by hand -------------------------------------------------------------------

def _llf_face(flux_u, u):
    def body(uL, uR, fL, fR, s, *, n):
        return ((fL[0] + fR[0]) / 2 - s * (uR[0] - uL[0]) / 2,)

    def stability(uL, uR, fL, fR, s, *, n):
        return s

    return riemann.User(body=body, state=u, stability=stability)


def _central_face(flux_u, u):
    def body(uL, uR, fL, fR, s, *, n):
        return ((fL[0] + fR[0]) / 2,)

    def stability(uL, uR, fL, fR, s, *, n):
        return s

    return riemann.User(body=body, state=u, stability=stability)


def _muscl(u):
    def limiter(a, b):
        return pm.where(a * b > 0, lambda: pm.where(abs(a) < abs(b), lambda: a, lambda: b),
                        lambda: 0)

    def body(sample):
        minus, centre, plus = sample(-1)[0], sample(0)[0], sample(1)[0]
        return (centre + limiter(centre - minus, plus - centre) / 2,)

    return reconstruction.User(body, state=u, formal_order=3, name="muscl")


# --- SU-T3 ---------------------------------------------------------------------------------------

def test_a_face_scheme_gives_the_riemann_user_face_written_by_hand():
    _assert_same(_surface_burgers(lambda u: local_lax_friedrichs()), _hand_burgers(_llf_face))


def test_a_face_scheme_without_speed_records_the_spectral_radius_of_r9():
    _assert_same(_surface_burgers(lambda u: central()), _hand_burgers(_central_face))


def test_a_cell_scheme_gives_the_reconstruction_user_written_by_hand():
    case = _surface_burgers(lambda u: local_lax_friedrichs(reconstruction=muscl(limiter=minmod)))
    _assert_same(case, _hand_burgers(_llf_face, recon=_muscl))


def test_a_library_word_takes_a_cell_scheme_as_its_reconstruction():
    case = _surface_burgers(lambda u: rusanov(reconstruction=muscl(limiter=minmod)))
    _assert_same(case, _hand_burgers(lambda flux, u: b6_numerics.rusanov(
        flux, u, reconstruction=_muscl(u))))


def test_upwind_with_a_stated_speed_gives_the_library_upwind_face():
    def surface_speed(u):
        def speed(face):
            return (face.left(u) + face.right(u)) / 2

        return upwind(speed=speed)

    def hand(flux_u, u):
        return b6_numerics.upwind(flux_u, u, speed=lambda face: (face.left(u)[0]
                                                            + face.right(u)[0]) / 2)

    _assert_same(_surface_burgers(surface_speed), _hand_burgers(hand))


# --- CF4 -----------------------------------------------------------------------------------------

def test_a_scheme_in_the_arithmetic_of_the_library_is_the_library_rusanov_node_for_node():
    _assert_same(_surface_burgers(lambda u: rusanov_as_the_library()),
                 _hand_burgers(lambda flux_u, u: b6_numerics.rusanov(flux_u, u)))


# --- pops.inspect --------------------------------------------------------------------------------

def test_inspect_reports_the_scheme_its_options_and_the_order_bound_of_its_stencil():
    case = _surface_burgers(lambda u: local_lax_friedrichs(reconstruction=muscl(limiter=minmod)))
    report = surface.inspect(case)
    choices = [(choice.name, choice.value, choice.origin) for choice in report.choices()]
    assert ("option reconstruction", "muscl(limiter=minmod)", "stated") in choices
    text = str(report)
    assert "reconstruction.User(the Reconstruct of muscl(limiter=minmod), formal_order=3, the " \
           "bound of its stencil, not claimed)" in text
    assert "riemann.User(body=the FaceFlux of local_lax_friedrichs" in text


def test_inspect_reports_a_default_option():
    report = surface.inspect(_surface_burgers(lambda u: local_lax_friedrichs()))
    choices = [(choice.name, choice.value, choice.origin) for choice in report.choices()]
    assert ("option reconstruction", "None", "default") in choices


@pytest.mark.parametrize("offsets, order", [((0,), 1), ((-1, 0), 2), ((-1, 0, 1), 3),
                                            ((-2, -1, 0, 1, 2), 5)])
def test_the_formal_order_is_the_number_of_cells_read(offsets, order):
    """Review of the lot note, N2 and Q1: the bound that the stencil proves, never short of the
    order, so never short of the AMR lookahead that b6 derives from it."""
    from pops.surface._scheme import SchemeLeaf, formal_order

    value = sum(SchemeLeaf(("sample", k), None, "c%d" % k) for k in offsets)
    assert formal_order(value) == order


# --- the boundary of the input s, and largest and smallest as b6 maximum and minimum ---------------

@surface.scheme
def radius_at(face, term, at):
    """speed= a radius at another value than the two bare traces: |dF/du(v)·n| (delta review)."""
    q = term.quantity
    left, right = face.left(q), face.right(q)
    s = spectral_radius(term, at(left, right), face.normal)
    return FaceFlux(dot(term.law(states={q: left}), face.normal) - s * (right - left), speed=s)


def _mean(left, right):
    return (left + right) / 2


def _left(left, right):
    return left


@pytest.mark.parametrize("at, radius", [
    (_mean, lambda uL, uR, n: abs((uL[0] + uR[0]) / 2 * n[0])),
    (_left, lambda uL, uR, n: abs(uL[0] * n[0])),
], ids=["mean-state", "one-trace"])
def test_a_radius_at_another_value_is_its_expression(at, radius):
    def face(flux_u, u):
        def body(uL, uR, fL, fR, s, *, n):
            return (fL[0] - radius(uL, uR, n) * (uR[0] - uL[0]),)

        def stability(uL, uR, fL, fR, s, *, n):
            return radius(uL, uR, n)

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_burgers(lambda u: radius_at(at=at)), _hand_burgers(face))


@surface.scheme
def bounded(face, term):
    """largest and smallest of other values lower to b6 maximum and minimum (review Q2, A1)."""
    q = term.quantity
    left, right = face.left(q), face.right(q)
    s = largest([abs(left), abs(right)])
    return FaceFlux(smallest([left, right]) * s, speed=s)


def test_largest_and_smallest_lower_to_b6_maximum_and_minimum():
    def face(flux_u, u):
        def body(uL, uR, fL, fR, s, *, n):
            return (pm.minimum(uL[0], uR[0]) * pm.maximum(abs(uL[0]), abs(uR[0])),)

        def stability(uL, uR, fL, fR, s, *, n):
            return pm.maximum(abs(uL[0]), abs(uR[0]))

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_burgers(lambda u: bounded()), _hand_burgers(face))


# --- the stability of a stated speed (review of #760, R2) ----------------------------------------

@surface.scheme
def signed(face, term):
    """A signed speed reaches b6 as its absolute value."""
    q = term.quantity
    s = face.left(q) * face.normal
    return FaceFlux(s * face.left(q), speed=s)


@surface.scheme
def unsigned(face, term):
    """A speed nonnegative by its form reaches b6 unchanged."""
    q = term.quantity
    s = abs(face.left(q))
    return FaceFlux(s * face.left(q), speed=s)


@pytest.mark.parametrize("scheme, bound", [
    (signed, lambda uL, n: abs(uL[0] * n[0])),
    (unsigned, lambda uL, n: abs(uL[0])),
], ids=["signed", "nonnegative"])
def test_a_stated_speed_is_the_stability_bound_in_absolute_value(scheme, bound):
    def face(flux_u, u):
        def body(uL, uR, fL, fR, s, *, n):
            return (speed(uL, n) * uL[0],)

        def speed(uL, n):
            return uL[0] * n[0] if scheme is signed else abs(uL[0])

        def stability(uL, uR, fL, fR, s, *, n):
            return bound(uL, n)

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_burgers(lambda u: scheme()), _hand_burgers(face))


# --- the average of a cell (review of #760, R1) -----------------------------------------------------

@surface.scheme
def cell_average(face, term, reconstruction=None):
    q = term.quantity
    return FaceFlux((face.left_cell.value(q) + face.right_cell.value(q)) / 2,
                    reconstruction=reconstruction)


def test_the_average_of_a_cell_is_the_trace_of_a_face_without_reconstruction():
    def face(flux_u, u):
        def body(uL, uR, fL, fR, s, *, n):
            return ((uL[0] + uR[0]) / 2,)

        def stability(uL, uR, fL, fR, s, *, n):
            return s

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_burgers(lambda u: cell_average()), _hand_burgers(face))
