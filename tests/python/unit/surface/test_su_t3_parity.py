"""SU-T3 (spec 0.3, 2.6.7): a case of the surface elaborates into the forms that the same case
written by hand for b6 and N8 gives, lot 1 of N15 (ROM-98).

The L1 case of §17.1 is taken at level 1: its model, its numerics and its program, with a fixed step
``dt=1e-3`` and two attempts. Its claim (§13), its cellwise step rule ``dt=combined_limit`` (PF8),
``pops.compile`` and ``pops.bind`` belong to later lots, which SU-T2 checks are refused by name.
The hand-written form is the b6 model, plan and case of ``examples/api_v02/models_M01_M14.py``
with the native Rusanov route that §17.1 « Core » names for ``upwind()``, and the method written
for N8. Both forms give the same program IR, the same artifact once the source lines of its
provenance are set aside, the same semantic identity and the same generated C++. The artifact
hash of b6 digests those source lines, so the two plan identities differ by them alone (P10,
C47): the provenance of a surface case names the user's ``pops.Case`` line.

Each method of ``pops.lib.time.methods`` gives the program of the same method written for N8,
and ``rusanov()`` and ``hll()`` give the b6 realization of a non-linear flux with the waves dF/du
that R9 derives. SU12: printing a value shows its mathematics, and ``pops.inspect`` shows its
elaborated form, its level and the origin of each choice.
"""
import json
from fractions import Fraction

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
from pops.lib.numerics import hll, rusanov, two_point, upwind
from pops.lib.time.methods import forward_euler, rk4, ssprk2, ssprk3
from pops.mesh import CartesianGrid, PeriodicAxes
from examples.api_v02 import b6_numerics
from pops.numerics import DiscretizationPlan
from pops.numerics.stencil_flux import StencilFlux
from pops.params import ConstParam, Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface._derive import derivative
from pops.surface.math import cos, ddt, div, exp, expm1, grad, log, sin, sqrt
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time._program.traced_method import Candidate, method

LIBRARY = {"forward_euler": forward_euler, "ssprk2": ssprk2, "ssprk3": ssprk3, "rk4": rk4}


def _n8_methods():
    """The methods of ``pops.lib.time.methods`` written for N8: ``B`` the step, ``p.dt``."""

    @method
    def forward_euler(B, p, rate):
        return Candidate(B.state + p.dt * rate(B.state, B.time), B.time + p.dt)

    @method
    def ssprk2(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)

    @method
    def ssprk3(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        second = (3 * u + first + h * rate(first, t + h)) / 4
        return Candidate((u + 2 * (second + h * rate(second, t + h / 2))) / 3, t + h)

    @method
    def rk4(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        k1 = rate(u, t)
        k2 = rate(u + h / 2 * k1, t + h / 2)
        k3 = rate(u + h / 2 * k2, t + h / 2)
        k4 = rate(u + h * k3, t + h)
        return Candidate(u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + h)

    return {"forward_euler": forward_euler, "ssprk2": ssprk2, "ssprk3": ssprk3, "rk4": rk4}


N8 = _n8_methods()


# --- the two forms of the L1 case -------------------------------------------------------------

def _l1_balance(advection, diffusion):
    return -div(advection + diffusion)


def _surface_l1(word, component="rate", balance=_l1_balance):
    """§17.1 at level 1, as the user writes it; ``component`` names the rate option of its
    method."""
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    nu = model.parameter("nu", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    diffusion = model.diffusive_flux(of=u, value=-nu * grad(u))
    transport = model.balance(ddt(u) == balance(advection, diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    advance = surface.Advance(word(**{component: model}), dt=1e-3, attempts=2)
    program = surface.Program(advance=advance, initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=256)})
    return case, transport, advection


def _hand_l1_equation(u, flux_u):
    return pm.ddt(u) == -pm.div(flux_u)


def _two_point_body(nu):
    """b6's two-point diffusion of -nu grad(u) written by hand as the body of a ``StencilFlux``
    (N3 lot 1, D7): the face flux -nu (u_R - u_L) / h, in the arithmetic of the face of
    ``two_point()``, and its incident frequency (nu / h) / h (D10)."""
    def body(sample, spacing, axis):
        h = spacing[axis]
        u_left = sample(tuple(-1 if d == axis else 0 for d in range(1)), 0)
        u_right = sample((0,), 0)
        return ((-nu) * ((u_right - u_left) / h),), (nu / h) / h

    return body


def _hand_stencil(flux, name, sign):
    """``sign`` times -div_h of the face flux of ``flux``, named ``<name>_<k>`` in the k-th
    evaluation of the rate, as the surface enters a diffusive term (D7)."""
    def term(state):
        program = state.prog
        evaluation = sum(1 for value in program._values
                         if value.op == "stencil_flux_divergence"
                         and value.name.startswith(name + "_"))
        rate = program.stencil_flux_divergence("%s_%d" % (name, evaluation), state, flux=flux)
        return rate if sign == 1 else -rate

    return term


def _hand_l1(n8_method, component="rate", equation=_hand_l1_equation, diffusion_first=False):
    """The same case written for b6 and N8 with the oracles of N3 lot 1: the face of the b6
    ``rusanov`` of ``examples/api_v02/b6_numerics.py`` (D1, D5) and a ``StencilFlux`` written by
    hand that reads the runtime parameter nu (D7). ``component`` names the component of the
    method, ``equation`` gives the balance of b6 and ``diffusion_first`` the order of the terms in
    the rate."""
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("advection_diffusion", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    nu = model.value(model.param(RuntimeParam("nu", default=0.01, domain=Positive())))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (a * u[0],)},
                        waves={x: (a,)})
    # The model declares F^d (R7) so that nu, read by a formula of the model, has a slot in the
    # block's RuntimeParams table, which the StencilFlux reads (CF15).
    model.diffusive_flux("diffusive_flux_u", state=u, value=nu * pm.grad(u[0]))
    rate = model.rate("u", equation=equation(u, flux_u))
    plan = DiscretizationPlan()
    advection = rate.select(flux_u)
    plan.rates.add(advection, b6_numerics.rusanov(flux_u, u))
    diffusion = _hand_stencil(StencilFlux(dimension=1, components=1, radius=1,
                                          body=_two_point_body(nu)), "diffusive_flux_u", 1)
    case = pops.Case("advection_diffusion")
    block = case.block("advection_diffusion", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        if diffusion_first:
            return diffusion(state) + advection(state)
        return advection(state) + diffusion(state)

    elaborate(Advance(n8_method(**{component: transport}), retry=Retry(max_attempts=2)),
              case=case, state=block[u], proposal=Proposal(0.001))
    layout = Uniform(CartesianGrid(frame=frame, cells=(256,), periodic=PeriodicAxes((x,))))
    return case, layout


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


@pytest.fixture(scope="module")
def l1():
    case, transport, advection = _surface_l1(ssprk2)
    hand_case, hand_layout = _hand_l1(N8["ssprk2"])
    elaborated = case.elaborated
    return {"case": case, "transport": transport, "advection": advection,
            "elaborated": elaborated, "hand": hand_case,
            "resolved": _resolved(elaborated.case, elaborated.layout),
            "hand_resolved": _resolved(hand_case, hand_layout)}


# --- SU-T3: the L1 case of §17.1 ---------------------------------------------------------------

def test_l1_gives_the_program_ir_of_the_hand_written_case(l1):
    program = l1["elaborated"].case._time
    assert program._serialize(include_provenance=False) == l1["hand"]._time._serialize(
        include_provenance=False)


def test_l1_gives_the_artifact_and_semantic_identity_of_the_hand_written_case(l1):
    resolved, hand = l1["resolved"], l1["hand_resolved"]
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(hand))
    assert resolved.snapshot.semantic_identity == hand.snapshot.semantic_identity


def test_l1_provenance_names_the_line_of_the_surface_case(l1):
    # N11 lot 2a (CC2, C48): source spans are provenance, kept in the full snapshot and out of the
    # artifact projection.
    full = json.loads(l1["resolved"].snapshot._canonical_json)
    operators = full["blocks"]["advection_diffusion"]["model"]["manifest"]["operators"]
    assert operators
    # N11 lot 2b-i (CC2, C47): the artifact projection names the block by its position.
    assert all("provenance" not in operator for operator in _artifact(l1["resolved"])[
        "payload"]["blocks"]["$block0"]["model"]["manifest"]["operators"])
    for operator in operators:
        primary = operator["provenance"]["primary"]
        assert (primary["file"], int(primary["line"]["$scalar"]["value"])) == (
            l1["case"].site.file, l1["case"].site.line)


def test_l1_gives_the_generated_cpp_of_the_hand_written_case(l1):
    source = _cpp(l1["resolved"])
    assert source == _cpp(l1["hand_resolved"])
    assert "pops_program_advance_max_attempts() { return 2; }" in source


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_a_library_method_gives_the_program_of_the_method_written_for_n8(name):
    case, _, _ = _surface_l1(LIBRARY[name])
    hand_case, _ = _hand_l1(N8[name])
    program = case.elaborated.case._time
    assert program.name == hand_case._time.name
    assert program._serialize(include_provenance=False) == hand_case._time._serialize(
        include_provenance=False)


def _surface_rate_p():
    @surface.method
    def forward_euler(step, p):
        return surface.Candidate(step.state + step.dt * p(step.state, step.time))

    return forward_euler


def _n8_rate_p():
    @method
    def forward_euler(B, q, p):
        return Candidate(B.state + q.dt * p(B.state, B.time), B.time + q.dt)

    return forward_euler


def test_a_rate_option_takes_the_name_of_a_parameter_of_n8():
    """A rate option keeps its name as the component of N8, even the name ``p`` that N8 gives the
    proposal in ``f(B, p, *components)``."""
    case, _, _ = _surface_l1(_surface_rate_p(), component="p")
    hand_case, _ = _hand_l1(_n8_rate_p(), component="p")
    assert case.elaborated.case._time._serialize(include_provenance=False) == \
        hand_case._time._serialize(include_provenance=False)


def test_python_sum_of_fluxes_is_the_sum_of_its_terms():
    """``div(sum([F, G]))``, a finite sum of 2.6.3, is ``div(F + G)``."""
    case, _, _ = _surface_l1(ssprk2, balance=lambda advection, diffusion:
                             -div(sum([advection, diffusion])))
    plain, _, _ = _surface_l1(ssprk2)
    assert case.elaborated.case._time._serialize(include_provenance=False) == \
        plain.elaborated.case._time._serialize(include_provenance=False)


def test_a_later_term_of_weight_minus_one_is_subtracted():
    """``-div(D + F)``, the diffusive flux first: the flux F, a later term of weight -1, enters the
    balance of b6 as ``- div(F)``."""
    case, _, _ = _surface_l1(ssprk2, balance=lambda advection, diffusion:
                             -div(diffusion + advection))
    hand_case, hand_layout = _hand_l1(N8["ssprk2"], equation=lambda u, flux:
                                      pm.ddt(u) == -pm.div(flux), diffusion_first=True)
    elaborated = case.elaborated
    resolved = _resolved(elaborated.case, elaborated.layout)
    hand = _resolved(hand_case, hand_layout)
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(hand))
    assert resolved.snapshot.semantic_identity == hand.snapshot.semantic_identity


# --- a non-linear flux: rusanov() and hll() with the derived waves ------------------------------

def _surface_burgers(word):
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("burgers")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    burgers = model.flux(of=u, value=u**2 / 2)
    model.balance(ddt(u) == -div(burgers))
    numerics = surface.Numerics(represent={u: surface.CellAverage()}, schemes={burgers: word()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=64)})


def _hand_burgers(word):
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("burgers", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (u[0] ** 2 / 2,)},
                        waves={x: (u[0],)})
    if word is b6_numerics.hll:
        model.wave_speeds(flux_u, frame=frame, values={x: (u[0], u[0])})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, word(flux_u, u))
    case = pops.Case("burgers")
    block = case.block("burgers", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
        return selection(state)

    elaborate(Advance(N8["ssprk2"](rate=transport)), case=case, state=block[u],
              proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(64,), periodic=PeriodicAxes((x,))))


@pytest.mark.parametrize("word, oracle", [(rusanov, b6_numerics.rusanov), (hll, b6_numerics.hll)],
                         ids=["rusanov", "hll"])
def test_a_non_linear_flux_takes_the_waves_that_r9_derives(word, oracle):
    """N3 lot 1: rusanov() and hll() are bodies of @pops.scheme; each gives the face of its b6
    oracle (examples/api_v02/b6_numerics.py, D5), with the waves that R9 derives, and hll() the
    signed pair (dF/du, dF/du) that its face input speeds reads."""
    case = _surface_burgers(word)
    elaborated = case.elaborated
    resolved = _resolved(elaborated.case, elaborated.layout)
    hand = _resolved(*_hand_burgers(oracle))
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(hand))
    assert resolved.snapshot.semantic_identity == hand.snapshot.semantic_identity
    report = surface.inspect(case)
    assert ("waves", "dF/du = u", "derived") in [
        (choice.name, choice.value, choice.origin) for choice in report.choices()]


# --- the derivatives of R9, which give the waves of rusanov() and hll() ---------------------------

def _scalar_quantity():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("waves")
    return model.quantity("u", surface.Scalar(domain)), model.constant("a", value=3.0)


@pytest.mark.parametrize("flux, waves", [
    (lambda u, a: exp(u), "exp(u)"),
    (lambda u, a: expm1(u), "exp(u)"),
    (lambda u, a: log(u), "1 / u"),
    (lambda u, a: sqrt(u), "1 / (2 * sqrt(u))"),
    (lambda u, a: sin(u), "cos(u)"),
    (lambda u, a: cos(u), "-sin(u)"),
    (lambda u, a: 1 / u, "(-1) / u**2"),
    (lambda u, a: a / (1 + u), "-a / (1 + u)**2"),
    (lambda u, a: u ** Fraction(3, 2), "3/2 * u**(1/2)"),
    (lambda u, a: u ** 3 / 3, "u**2"),
    (lambda u, a: exp(2 * u), "exp(2 * u) * 2"),
    (lambda u, a: log(1 + u ** 2), "1 / (1 + u**2) * 2 * u"),
], ids=["exp", "expm1", "log", "sqrt", "sin", "cos", "reciprocal", "quotient", "fraction-power",
        "cube", "chain", "chain-log"])
def test_the_waves_are_the_exact_derivative_of_the_flux(flux, waves):
    """R9: the waves of a flux F(u) are dF/du, derived exactly by each rule, the chain rule
    included."""
    u, a = _scalar_quantity()
    assert str(derivative(flux(u, a), u)) == waves


def test_an_unsigned_coefficient_is_accepted_and_its_dissipation_is_not_established():
    """R2 refuses only what the declarations derive to be anti-diffusive; a parameter without an
    admissible set gives a coefficient of unknown sign, which pops.inspect reports (K7)."""
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("heat")
    u = model.quantity("u", surface.Scalar(domain))
    nu = model.parameter("nu", default=1e-2)
    kappa = model.parameter("kappa", default=1e-2, admissible=surface.Positive())
    u0 = model.data("u0", surface.Scalar(domain))
    unsigned = model.diffusive_flux(of=u, value=nu * grad(u))
    signed = model.diffusive_flux(of=u, value=-kappa * grad(u))
    model.balance(ddt(u) == -div(unsigned + signed))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={unsigned: two_point(), signed: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=16)})
    choices = [(choice.name, choice.value, choice.origin)
               for choice in surface.inspect(case).choices()]
    assert ("dissipation", "not established: the sign of -nu is not derived", "derived") in choices
    assert ("dissipation", "established: kappa > 0", "derived") in choices


# --- SU12: print shows the mathematics, inspect the elaborated form ------------------------------

def test_print_shows_the_mathematics_of_a_balance(l1):
    assert str(l1["transport"]) == "ddt(u) == -div(a * u + -nu * grad(u))"
    assert str(l1["advection"]) == "a * u"


def test_a_named_term_prints_by_its_name():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("drift")
    v = model.quantity("v", surface.Scalar(domain))
    drift = model.flux(of=v, value=2 * v, name="drift")
    assert str(model.balance(ddt(v) == -div(drift))) == "ddt(v) == -div(drift)"


def test_inspect_reports_each_choice_and_its_origin(l1):
    report = surface.inspect(l1["case"])
    assert report.level == "1 (model, numerics, program)"
    assert report.resolved is not None
    text = str(report)
    assert text.startswith("pops.inspect(case 'advection_diffusion'), level 1 (model, numerics, "
                           "program): validate and resolve passed, nothing compiled")
    choices = [(choice.name, choice.value, choice.origin) for choice in report.choices()]
    for expected in [("periodic axes", "x", "stated"), ("name", "flux_u", "derived"),
                     ("waves", "dF/du = a", "derived"), ("name", "diffusive_flux_u", "derived"),
                     ("k", "nu", "derived"), ("speed", "s = F(1)·n = a", "derived"),
                     ("dt", 0.001, "stated"), ("attempts", 2, "stated"),
                     ("accept", "finite()", "default"), ("cells", (256,), "stated")]:
        assert expected in choices
    assert "FiniteVolume(variables=Conservative(u), reconstruction=FirstOrder(), " \
           "riemann=riemann.User(body=the FaceFlux of upwind" in text
    assert "StencilFlux(radius=1, body=the FaceFlux of two_point, frequency=nu / face.distance " \
           "/ face.distance, derived)" in text
    assert "and in the program the StencilFlux of diffusive_flux_u" in text


def test_inspect_of_a_term_shows_its_kind_and_its_model(l1):
    report = surface.inspect(l1["advection"])
    assert str(report).splitlines() == [
        "pops.inspect(a * u), level model",
        "[model] flux of u: a * u",
        "    -> a term of pops.Model('advection_diffusion')"]
