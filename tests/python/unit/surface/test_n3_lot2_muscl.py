"""N3 lot 2 (ROM-87): ``muscl(limiter=)`` and its four limiters are written with the surface.

``muscl`` is a ``@pops.scheme`` over ``cell``, ``term`` and ``axis`` that returns ``Reconstruct``,
and ``minmod``, ``vanleer``, ``mc`` and ``superbee`` are plain functions over the expressions of the
surface (``python/pops/lib/numerics/limiters.py``). Their b6 bodies are the parity oracles of
``examples/api_v02/b6_reconstructions.py`` (D5).

* E1, P1: with each limiter, ``rusanov(reconstruction=muscl(limiter=X))`` gives the case written
  by hand with the b6 limiter X in the body of a ``reconstruction.User``: the same program IR,
  artifact, semantic identity and generated C++. The trace is the b6 ``muscl(Uh, X)`` node for
  node, and its generated C++ is the one of the b6 policy line for line; the formal order differs,
  3 (the stencil bound of the surface, Q1 of N15 lot 2) against 2 (N4), and so do the artifact and
  the identity, which carry it.
* E2: the generated policy of the surface word, compiled beside master's native limiter of the
  same name, gives the native trace bit for bit wherever it is finite, with the two known
  exceptions of N3 (the cell value -0 on the negative face, vanleer's product beyond the largest
  double), and refuses elsewhere.
* P4: ``equal``, ``all_of``, ``any_of`` and ``none_of`` in the body of a scheme are b6's ``==``,
  the left folds of ``&`` and ``|``, and ``~``.
* E3, CF4: a copy of the library's module written in a case gives the same case as the words.
* N4: on the AMR case of the scalar advection, master's MUSCL gives the plan of master, and the
  ported ``muscl``, lowered to ``reconstruction.User``, is refused for want of a typed AMR gradient
  stencil; the surface itself refuses a hierarchy, naming N12, on whose hierarchies N4 lifts it.
"""
import inspect
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pops
import pops.lib.numerics.limiters as library
import pops.surface as surface
import pytest
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.numerics import mc, minmod, muscl, rusanov, superbee, vanleer
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction
from pops.projection import ConservativeCellAverage
from pops.surface.discrete import Reconstruct
from pops.surface.math import all_of, any_of, ddt, div, equal, none_of, where
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

from examples.api_v02 import b6_numerics, b6_reconstructions

REPO = Path(__file__).resolve().parents[4]
CELLS = 32
LIMITERS = {"minmod": (minmod, "Minmod"), "vanleer": (vanleer, "VanLeer"), "mc": (mc, "MC"),
            "superbee": (superbee, "Superbee")}


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


def _assert_same(case, hand):
    elaborated = case.elaborated
    assert elaborated.case._time._serialize(include_provenance=False) == \
        hand[0]._time._serialize(include_provenance=False)
    resolved = _resolved(elaborated.case, elaborated.layout)
    other = _resolved(*hand)
    assert _without_provenance(_artifact(resolved)) == _without_provenance(_artifact(other))
    assert resolved.snapshot.semantic_identity == other.snapshot.semantic_identity
    assert _cpp(resolved) == _cpp(other)


# --- the cases: Burgers on a periodic line, its face a Rusanov over a reconstruction -----------

def _surface_burgers(reconstruction_scheme, namespace=None):
    words = namespace or {"rusanov": rusanov}
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("transport")
    u = model.quantity("u", surface.Scalar(domain))
    u0 = model.data("u0", surface.Scalar(domain))
    term = model.flux(of=u, value=u**2 / 2)
    model.balance(ddt(u) == -div(term))
    numerics = surface.Numerics(
        represent={u: surface.CellAverage()},
        schemes={term: words["rusanov"](reconstruction=reconstruction_scheme)})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})


def _hand_burgers(recon):
    """The case written for b6 with the b6 rusanov face over the reconstruction ``recon(u)``."""
    frame = CartesianDomain("box", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())
    x = frame.axes[0]
    model = pops.Model("transport", frame=frame)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame, components={x: (u[0] ** 2 / 2,)},
                        waves={x: (u[0],)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, b6_numerics.rusanov(flux_u, u, reconstruction=recon(u)))
    case = pops.Case("transport")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_method()(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(CELLS,), periodic=PeriodicAxes((x,))))


def _user(body, name="muscl"):
    """A reconstruction of the cell written by hand for b6 over the samples -1, 0 and 1, with the
    order bound of its stencil, as the surface realizes a ``Reconstruct``."""
    def recon(u):
        def sampled(sample):
            minus, centre, plus = sample(-1)[0], sample(0)[0], sample(1)[0]
            return (body(minus, centre, plus),)

        return reconstruction.User(sampled, state=u, formal_order=3, name=name)

    return recon


def _hand_muscl(limiter):
    return _user(lambda minus, centre, plus: centre + .5*limiter(centre - minus, plus - centre))


def _realized(case):
    """The reconstruction.User that the elaborated case gives its face."""
    (plan,) = case.elaborated.case._numerics.values()
    ((_, finite_volume),) = plan.rates.items()
    return finite_volume.reconstruction


# --- E1, P1 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_muscl_with_each_limiter_is_the_case_written_with_its_b6_limiter(name):
    word = LIMITERS[name][0]
    _assert_same(_surface_burgers(muscl(limiter=word)),
                 _hand_burgers(_hand_muscl(getattr(b6_reconstructions, name))))


def _policy_lines(descriptor):
    """The generated lines that compute the trace of a policy, its samples spelled by their
    offset, and the values whose finiteness it checks."""
    policy = emit_user_reconstruction_policy(SimpleNamespace(_user_reconstruction=descriptor))
    lines = [line.strip() for line in policy.splitlines()]
    computed = [re.sub(r"sample\((-?\d+)(, 0)?\)", r"sample(\1)", line) for line in lines
                if line.startswith(("pops::Real ", "const pops::Real "))]
    checked = set(re.findall(r"std::isfinite\((\w+)\)", policy))
    return computed, checked


@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_the_trace_is_the_b6_muscl_node_for_node_and_claims_the_bound_of_its_stencil(name):
    """N4: the same C++ as the b6 ``muscl(Uh, X)``; the formal order is 3 against 2."""
    realized = _realized(_surface_burgers(muscl(limiter=LIMITERS[name][0])))
    oracle = b6_reconstructions.muscl(realized.options["state"],
                                      getattr(b6_reconstructions, name))
    assert _policy_lines(realized) == _policy_lines(oracle)
    assert (realized.options["formal_order"], oracle.options["formal_order"]) == (3, 2)
    assert realized.options["ghost_depth"] == oracle.options["ghost_depth"] == 2


# --- E2: the generated policy against master's native limiter -------------------------------------

# Differences and values met by a face trace, as in test_lib_numerics_reconstructions.py.
_VALUES = (0., -0., 5e-324, 1e-323, 2.5e-323, -3e-310, 1e-160, -2e-158, 0.3, -0.7, 1., 2.5, -4.,
           1e150, -3e153, 1e200, -1e300, 1.7e308, -1.7e308, math.inf, -math.inf, math.nan)


def _cpp_double(value):
    if math.isnan(value):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(value):
        return "%sstd::numeric_limits<double>::infinity()" % ("-" if value < 0 else "")
    return float.hex(value)


def _program(name):
    policy = emit_user_reconstruction_policy(SimpleNamespace(
        _user_reconstruction=_realized(_surface_burgers(muscl(limiter=LIMITERS[name][0])))))
    values = ", ".join(_cpp_double(v) for v in _VALUES)
    return "\n".join((
        "#include <array>", "#include <cmath>", "#include <cstdint>", "#include <cstdio>",
        "#include <cstring>", "#include <limits>", "#include <pops/numerics/fv/reconstruction.hpp>",
        policy,
        "struct Sampler { const double* centre; int orientation;",
        "  double operator()(int offset, int) const { return centre[orientation*offset]; } };",
        "static std::uint64_t bits(double x) { std::uint64_t b; std::memcpy(&b, &x, 8); return b; }",
        "int main() {",
        "  const double values[] = {%s};" % values,
        "  const int n = sizeof(values)/sizeof(values[0]);",
        "  const pops_generated::UserReconstructionPolicy generated{};",
        "  const pops::%s native{};" % LIMITERS[name][1],
        "  long equal = 0, refused = 0, product_refusals = 0, signed_zeros = 0;",
        "  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {",
        "    const double cells[3] = {values[i], values[j], values[k]};",
        "    for (int orientation : {1, -1}) {",
        "      const double centre = cells[1];",
        "      const double slope = native.limited_slope(centre - cells[0], cells[2] - centre);",
        "      const double expected = centre + double(0.5)*double(orientation)*slope;",
        "      const double got = generated.stencil_face_state(Sampler{cells + 1, orientation})[0];",
        "      if (std::isfinite(expected)) {",
        "        if (bits(got) == bits(expected)) { ++equal; continue; }",
        "        if (orientation == -1 && got == 0. && expected == 0. && centre == 0. &&",
        "            std::signbit(centre)) {",
        "          ++signed_zeros; continue; }",
        "        const double a = centre - cells[0], b = cells[2] - centre;",
        "        if (std::isnan(got) && std::isfinite(a) && std::isfinite(b) &&",
        "            !std::isfinite(a*b)) { ++product_refusals; continue; }",
        "        std::printf(\"MISMATCH %a %a %a o=%d native=%a generated=%a\\n\",",
        "                    cells[0], cells[1], cells[2], orientation, expected, got);",
        "        return 1;",
        "      }",
        "      if (!std::isnan(got)) {",
        "        std::printf(\"ACCEPTED %a %a %a o=%d native=%a generated=%a\\n\",",
        "                    cells[0], cells[1], cells[2], orientation, expected, got);",
        "        return 1;",
        "      }",
        "      ++refused;",
        "    }",
        "  }",
        "  std::printf(\"%ld %ld %ld %ld\\n\", equal, refused, product_refusals, signed_zeros);",
        "  return 0;",
        "}",
    ))


@pytest.mark.compiler
@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_the_generated_policy_of_the_word_is_the_native_limiter_bit_for_bit(
        name, native_cxx, tmp_path, record_property):
    source = tmp_path / ("muscl_%s.cpp" % name)
    source.write_text(_program(name))
    binary = tmp_path / ("muscl_%s" % name)
    compiled = subprocess.run(
        [native_cxx, "-std=c++20", "-O2", "-I", str(REPO / "include"), str(source),
         "-o", str(binary)], capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stderr[-4000:]
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout[-2000:]
    equal, refused, product_refusals, signed_zeros = map(int, ran.stdout.split())
    record_property("bitwise_equal_traces", equal)
    record_property("signed_zero_traces", signed_zeros)
    record_property("refused_by_both", refused)
    record_property("vanleer_product_refusals", product_refusals)
    assert equal + refused + product_refusals + signed_zeros == 2*len(_VALUES)**3
    assert equal > len(_VALUES)**3
    if name != "vanleer":
        assert product_refusals == 0
    else:
        assert 0 < product_refusals < refused


# --- P4: the predicates of a scheme body -------------------------------------------------------

def _reconstruct(trace):
    @surface.scheme
    def predicated(cell, term, axis):
        q = term.quantity
        return Reconstruct(trace(cell.shift(axis, -1).value(q), cell.value(q),
                                 cell.shift(axis, 1).value(q)))

    return predicated()


@pytest.mark.parametrize("surface_trace, hand_trace", [
    (lambda m, c, p: where(equal(m, p), c, m),
     lambda m, c, p: pm.where(m == p, lambda: c, lambda: m)),
    (lambda m, c, p: where(all_of([m < c, c < p, m > 0]), c, p),
     lambda m, c, p: pm.where(((m < c) & (c < p)) & (m > 0), lambda: c, lambda: p)),
    (lambda m, c, p: where(any_of([m < c, c < p, m > 0]), c, p),
     lambda m, c, p: pm.where(((m < c) | (c < p)) | (m > 0), lambda: c, lambda: p)),
    (lambda m, c, p: where(none_of([m < c]), c, p),
     lambda m, c, p: pm.where(~(m < c), lambda: c, lambda: p)),
    (lambda m, c, p: where(none_of([m < c, equal(c, p)]), c, p),
     lambda m, c, p: pm.where(~((m < c) | (c == p)), lambda: c, lambda: p)),
], ids=["equal", "all_of", "any_of", "none_of", "none_of-two"])
def test_the_predicates_of_a_body_are_the_b6_operators(surface_trace, hand_trace):
    _assert_same(_surface_burgers(_reconstruct(surface_trace)),
                 _hand_burgers(_user(hand_trace, name="predicated")))


# --- E3, CF4: a copy of the library -------------------------------------------------------------

@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_a_copy_of_the_library_is_the_library(name):
    namespace = {}
    exec(compile(inspect.getsource(library), library.__file__, "exec"), namespace)  # noqa: S102
    copy = namespace["muscl"](limiter=namespace[name])
    word = muscl(limiter=LIMITERS[name][0])
    case, by_words = _surface_burgers(copy), _surface_burgers(word)
    _assert_same(case, (by_words.elaborated.case, by_words.elaborated.layout))


# --- refusals -----------------------------------------------------------------------------------

def test_the_b6_signature_and_a_limiter_that_is_not_a_function_are_refused():
    """The b6 call muscl(Uh, vanleer) is refused by the rule of the options (SU5); a limiter is a
    plain function, which the body of muscl calls as the gallery writes it (D8)."""
    with pytest.raises(TypeError, match=r"muscl takes its options by keyword, "
                                        r"muscl\(limiter=\.\.\.\)"):
        muscl(object(), vanleer)
    with pytest.raises(TypeError, match=r"muscl needs the options limiter"):
        _surface_burgers(muscl())
    with pytest.raises(TypeError, match="'float' object is not callable"):
        _surface_burgers(muscl(limiter=0.5))
    with pytest.raises(TypeError, match=r"takes a scheme that returns Reconstruct, called with "
                                        r"its options, such as muscl\(limiter=minmod\)"):
        _surface_burgers(muscl)


# --- N4: the AMR case of the scalar advection -----------------------------------------------------

def _amr_resolution():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "amr_resolution", REPO / "tests" / "python" / "unit" / "amr" /
        "test_public_amr_resolution.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_on_a_hierarchy_master_s_muscl_keeps_its_plan_and_the_ported_one_is_refused(monkeypatch):
    """N4 of the lot note: the plan of master with the native MUSCL (buffer (2, 2), look-ahead 1
    at each transition), and the refusal of the ported muscl lowered to reconstruction.User, which
    carries no typed AMR gradient stencil. The lowered trace is stated on the state of the AMR
    case, which the surface does not reach (it refuses a hierarchy, naming N12): the b6 body of
    the trace, which is the word node for node (E1), with the order bound of the surface."""
    amr = _amr_resolution()
    _, _, _, authorities = amr._resolved_target()
    plan = authorities.hierarchy.plan
    assert [(row.buffer, row.lookahead) for row in plan.transitions] == [((2, 2), 1)] * 2

    def lowered(*args, **kwargs):
        limiter = b6_reconstructions.vanleer

        def sampled(sample):
            minus, centre, plus = sample(-1), sample(0), sample(1)
            return centre + .5*limiter(centre - minus, plus - centre)

        return reconstruction.User(sampled, formal_order=3, name="muscl")

    monkeypatch.setattr(reconstruction, "MUSCL", lowered)
    with pytest.raises(NotImplementedError,
                       match="reconstruction 'muscl' does not provide a typed AMR gradient "
                             "stencil"):
        amr._resolved_target()


def test_the_surface_refuses_a_hierarchy_naming_n12():
    with pytest.raises(surface.NotYetElaborated, match="hierarchies of N12"):
        surface.AMR()
