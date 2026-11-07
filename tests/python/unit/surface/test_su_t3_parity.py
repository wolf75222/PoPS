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
from pops.lib.numerics.schemes import hll, rusanov, two_point, upwind
from pops.lib.time.methods import forward_euler, rk4, ssprk2, ssprk3
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import ConstParam, Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface.math import ddt, div, grad
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

def _surface_l1(word, component="rate"):
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
    transport = model.balance(ddt(u) == -div(advection + diffusion))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    advance = surface.Advance(word(**{component: model}), dt=1e-3, attempts=2)
    program = surface.Program(advance=advance, initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=256)})
    return case, transport, advection


def _hand_l1(n8_method, component="rate"):
    """The same case written for b6 and N8; ``component`` names the component of the method."""
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("advection_diffusion", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    nu = model.value(model.param(RuntimeParam("nu", default=0.01, domain=Positive())))
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (a * u[0],)},
                        waves={x: (a,)})
    diffusive_flux_u = model.diffusive_flux("diffusive_flux_u", state=u,
                                            value=nu * pm.grad(u[0]))
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u) + pm.div(diffusive_flux_u))
    plan = DiscretizationPlan()
    advection = rate.select(flux_u)
    plan.rates.add(advection, FiniteVolume(
        flux=flux_u, variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    diffusion = rate.select(diffusive_flux_u)
    plan.rates.add(diffusion, Diffusion(flux=diffusive_flux_u))
    case = pops.Case("advection_diffusion")
    block = case.block("advection_diffusion", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))

    def transport(state):
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
    operators = _artifact(l1["resolved"])["payload"]["blocks"]["advection_diffusion"]["model"][
        "manifest"]["operators"]
    assert operators
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


def _hand_burgers(face):
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("burgers", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (u[0] ** 2 / 2,)},
                        waves={x: (u[0],)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, FiniteVolume(
        flux=flux_u, variables=variables.Conservative(u),
        reconstruction=reconstruction.FirstOrder(), riemann=face))
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


@pytest.mark.parametrize("word, face", [(rusanov, riemann.Rusanov), (hll, riemann.HLL)],
                         ids=["rusanov", "hll"])
def test_a_non_linear_flux_takes_the_waves_that_r9_derives(word, face):
    case = _surface_burgers(word)
    elaborated = case.elaborated
    resolved = _resolved(elaborated.case, elaborated.layout)
    hand = _resolved(*_hand_burgers(face()))
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(hand))
    assert resolved.snapshot.semantic_identity == hand.snapshot.semantic_identity
    report = surface.inspect(case)
    assert ("waves", "dF/du = u", "derived") in [
        (choice.name, choice.value, choice.origin) for choice in report.choices()]


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
           "riemann=Rusanov())" in text
    assert "Diffusion(flux=F^d)" in text


def test_inspect_of_a_term_shows_its_kind_and_its_model(l1):
    report = surface.inspect(l1["advection"])
    assert str(report).splitlines() == [
        "pops.inspect(a * u), level model",
        "[model] flux of u: a * u",
        "    -> a term of pops.Model('advection_diffusion')"]
