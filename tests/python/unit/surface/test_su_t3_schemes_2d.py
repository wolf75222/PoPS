"""SU-T3, CF4 and SU-T2 (spec 0.3, 2.6.7) for a scheme of one's own on a support of rank two and three, lot
2b of N15 (ROM-98).

A flux of a support of rank d is a vector of d components, ``model.flux(of=u, value=vector(a, b)
* u)``, which the elaborator lowers on the dictionary per axis of b6, with the waves dF_a/du of
each component (R9). A face scheme runs once per axis; ``dot(term.law(...), face.normal)`` is the
normal component of the flux, the face input fL or fR at a bare trace and c_1 n_1 + ... + c_d n_d
over the canonical normal elsewhere, which is one body for every axis. Each case gives the case
written by hand for b6 (program IR, artifact without provenance, semantic identity, C++).
"""
import json

import pytest

import pops
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import fluxes
from pops.lib.numerics.schemes import rusanov, upwind
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import FaceFlux, Reconstruct, spectral_radius
from pops.surface.math import ddt, div, dot, exp, largest, vector, where
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


# --- the schemes ---------------------------------------------------------------------------------

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
def linear_upwind(face, term):
    """The upwind of the gallery without speed=: s = F(1)·n, F = F(1) u checked by term.expect."""
    q = term.quantity
    unit_flux = term.law(states={q: 1})
    term.expect(unit_flux * q)
    s = dot(unit_flux, face.normal)
    return FaceFlux(where(s > 0, s * face.left(q), s * face.right(q)), speed=s)


@surface.scheme
def rusanov_as_the_library(face, term):
    q = term.quantity
    n = face.normal
    left, right = face.left(q), face.right(q)
    s = largest([spectral_radius(term, left, n), spectral_radius(term, right, n)])
    flux_left = dot(term.law(states={q: left}), n)
    flux_right = dot(term.law(states={q: right}), n)
    return FaceFlux(0.5 * (flux_left + flux_right) - 0.5 * s * (right - left), speed=s)


def minmod(a, b):
    return where(a * b > 0, where(abs(a) < abs(b), a, b), 0)


@surface.scheme
def muscl(cell, term, axis, limiter):
    q = term.quantity
    centre = cell.value(q)
    backward = centre - cell.shift(axis, -1).value(q)
    forward = cell.shift(axis, 1).value(q) - centre
    return Reconstruct(centre + limiter(backward, forward) / 2)


# --- the two forms of a 2D advection case ----------------------------------------------------------

def _surface_case(scheme, flux=lambda a, b, u: vector(a, b) * u, declare=None):
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.5)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=flux(a, b, u))
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: scheme(a, b)})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=(16, 16))})


def _hand_case(face, recon=lambda u: reconstruction.FirstOrder(), waves=True):
    """The case written for b6 and N8; ``face(flux_u, u, a, b)`` is its finite volume's face."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.5)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],)},
                        waves={x: (a,), y: (b,)} if waves else None)
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    realization = face(flux_u, u, a, b)
    if not isinstance(realization, FiniteVolume):
        realization = FiniteVolume(flux=flux_u, variables=variables.Conservative(u),
                                   reconstruction=recon(u), riemann=realization)
    plan.rates.add(selection, realization)
    case = pops.Case("advection")
    block = case.block("advection", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        return selection(state)

    elaborate(Advance(_n8_ssprk2(rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(16, 16),
                                       periodic=PeriodicAxes((x, y))))


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


def _llf_face(flux_u, u, a, b):
    def body(uL, uR, fL, fR, s, *, n):
        return ((fL[0] + fR[0]) / 2 - s * (uR[0] - uL[0]) / 2,)

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


# --- SU-T3 on a support of rank two ----------------------------------------------------------------

def test_the_flux_is_lowered_on_the_dictionary_per_axis_with_its_waves():
    _assert_same(_surface_case(lambda a, b: rusanov()),
                 _hand_case(lambda flux_u, u, a, b: riemann.Rusanov()))


def test_a_face_scheme_on_a_support_of_rank_two():
    _assert_same(_surface_case(lambda a, b: local_lax_friedrichs()), _hand_case(_llf_face))


def test_a_cell_scheme_on_a_support_of_rank_two():
    case = _surface_case(lambda a, b: local_lax_friedrichs(reconstruction=muscl(limiter=minmod)))
    _assert_same(case, _hand_case(_llf_face, recon=_muscl))


def test_the_normal_component_of_the_flux_at_a_state_is_its_dot_with_the_canonical_normal():
    def face(flux_u, u, a, b):
        def speed(n):
            return a * 1 * n[0] + b * 1 * n[1]

        def body(uL, uR, fL, fR, s, *, n):
            sn = speed(n)
            return (pm.where(sn > 0, lambda: sn * uL[0], lambda: sn * uR[0]),)

        def stability(uL, uR, fL, fR, s, *, n):
            return abs(speed(n))

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_case(lambda a, b: linear_upwind()), _hand_case(face))


def test_upwind_with_a_stated_velocity_dotted_with_the_normal():
    def surface_scheme(a, b):
        def speed(face):
            return dot(vector(a, b), face.normal)

        return upwind(speed=speed)

    def hand(flux_u, u, a, b):
        return fluxes.upwind(flux_u, u, speed=lambda face: a * face.normal[0]
                             + b * face.normal[1])

    _assert_same(_surface_case(surface_scheme), _hand_case(hand))


def test_cf4_a_scheme_in_the_arithmetic_of_the_library_is_the_library_rusanov():
    _assert_same(_surface_case(lambda a, b: rusanov_as_the_library()),
                 _hand_case(lambda flux_u, u, a, b: fluxes.rusanov(flux_u, u)))


def test_inspect_reports_the_components_and_the_waves_per_axis():
    report = surface.inspect(_surface_case(lambda a, b: rusanov()))
    choices = [(choice.name, choice.value, choice.origin) for choice in report.choices()]
    assert ("waves", "dF/du = vector(a, b)", "derived") in choices
    assert "components={x: ((a * u[0]),), y: ((b * u[0]),)}, waves={x: (a,), y: (b,)}" in \
        str(report)


# --- SU-T3 on a support of rank three (R2 of the review of #763) ---------------------------------

def _surface_case3(scheme):
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1),
                         z=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.5)
    c = model.constant("c", value=0.25)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(a, b, c) * u)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: scheme})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=(8, 8, 8))})


def _hand_case3(face):
    frame = CartesianDomain("box", lower=(0.0, 0.0, 0.0),
                            upper=(1.0, 1.0, 1.0)).frame(Cartesian3D())
    x, y, z = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.5)))
    c = model.value(model.param(ConstParam("c", 0.25)))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],), z: (c * u[0],)},
                        waves={x: (a,), y: (b,), z: (c,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    realization = face(flux_u, u, a, b, c)
    if not isinstance(realization, FiniteVolume):
        realization = FiniteVolume(flux=flux_u, variables=variables.Conservative(u),
                                   reconstruction=reconstruction.FirstOrder(),
                                   riemann=realization)
    plan.rates.add(selection, realization)
    case = pops.Case("advection")
    block = case.block("advection", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        return selection(state)

    elaborate(Advance(_n8_ssprk2(rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(8, 8, 8),
                                       periodic=PeriodicAxes((x, y, z))))


def test_the_normal_component_at_a_state_on_a_support_of_rank_three():
    """dot(term.law(states={q: 1}), face.normal) at rank 3 is a n0 + b n1 + c n2 written by hand:
    each of the three components meets its component of the canonical normal."""
    def face(flux_u, u, a, b, c):
        def speed(n):
            return a * 1 * n[0] + b * 1 * n[1] + c * 1 * n[2]

        def body(uL, uR, fL, fR, s, *, n):
            sn = speed(n)
            return (pm.where(sn > 0, lambda: sn * uL[0], lambda: sn * uR[0]),)

        def stability(uL, uR, fL, fR, s, *, n):
            return abs(speed(n))

        return riemann.User(body=body, state=u, stability=stability)

    _assert_same(_surface_case3(linear_upwind()), _hand_case3(face))



def test_upwind_with_a_stated_velocity_on_a_support_of_rank_three():
    """upwind(speed=f), f(face) = dot(vector(a, b, c), face.normal), against fluxes.upwind with
    a n0 + b n1 + c n2 written by hand."""
    def speed(face):
        return dot(vector(1.0, 0.5, 0.25), face.normal)

    def hand(flux_u, u, a, b, c):
        return fluxes.upwind(flux_u, u, speed=lambda face: 1.0 * face.normal[0]
                             + 0.5 * face.normal[1] + 0.25 * face.normal[2])

    _assert_same(_surface_case3(upwind(speed=speed)), _hand_case3(hand))

# --- SU-T2 on a support of rank two ----------------------------------------------------------------

def test_a_scalar_flux_on_a_support_of_rank_two_names_vector():
    with pytest.raises(surface.SupportMismatch, match=r"Write model.flux\(of=u, "
                                                      r"value=vector\(\.\.\., \.\.\.\)\)"):
        _surface_case(lambda a, b: rusanov(), flux=lambda a, b, u: a * u)


def test_a_vector_of_another_rank_is_a_support_mismatch():
    with pytest.raises(surface.SupportMismatch, match="has 3 components"):
        _surface_case(lambda a, b: rusanov(), flux=lambda a, b, u: vector(a, b, a) * u)


def test_a_vector_plus_a_scalar_is_a_support_mismatch():
    with pytest.raises(surface.SupportMismatch, match="adds a vector and a scalar"):
        _surface_case(lambda a, b: rusanov(), flux=lambda a, b, u: vector(a, b) * u + u)


def test_other_vector_algebra_belongs_to_n11():
    with pytest.raises(surface.NotYetElaborated, match="N11"):
        _surface_case(lambda a, b: rusanov(), flux=lambda a, b, u: exp(vector(a, b)) * u)


def test_the_normal_where_a_scalar_is_taken_is_a_support_mismatch():
    @surface.scheme
    def bare(face, term):
        q = term.quantity
        return FaceFlux(face.left(q) * face.normal)

    with pytest.raises(surface.SupportMismatch, match="a vector of 2 components"):
        _surface_case(lambda a, b: bare())


def test_a_face_that_differs_by_axis_names_its_lot():
    @surface.scheme
    def by_axis(face, term):
        q = term.quantity
        return FaceFlux(face.left(q) if face.axis is face.axis.support.axes[0] else face.right(q))

    with pytest.raises(surface.NotYetElaborated, match="face lowered per axis"):
        _surface_case(lambda a, b: by_axis())


def test_a_cell_scheme_that_differs_by_axis_names_its_lot():
    """R1 of the review of #763: the runs of a cell scheme on the axes are compared on every axis,
    so a trace that differs by axis is refused, never the trace of the first axis on all."""
    @surface.scheme
    def by_axis_cell(cell, term, axis):
        q = term.quantity
        first = axis is axis.support.axes[0]
        return Reconstruct(cell.value(q) if first else cell.shift(axis, 1).value(q))

    with pytest.raises(surface.NotYetElaborated, match="reconstruction lowered per axis"):
        _surface_case(lambda a, b: local_lax_friedrichs(reconstruction=by_axis_cell()))
