"""N3 lot 3 (ROM-87): ``weno5(epsilon=)`` is written with the surface.

``weno5`` is a ``@pops.scheme`` over ``cell``, ``term`` and ``axis`` that returns ``Reconstruct``
(``python/pops/lib/numerics/reconstructions.py``). Its b6 body, with the b6 ``Reconstruct``, is the
parity oracle of ``examples/api_v02/b6_reconstructions.py`` (D5); the b6 ``Reconstruct`` is not
ported (D4).

* E1, P1: ``rusanov(reconstruction=weno5())``, and with ``epsilon=1e-6``, gives the case written by
  hand with the b6 WENO5-Z body in a ``reconstruction.User``: the same program IR, artifact,
  semantic identity and generated C++. The trace is the b6 ``weno5(Uh)`` node for node, its
  generated C++ the one of the b6 policy line for line, with the same order (5) and ghost depth (3).
* E2: the generated policy of the surface word, compiled beside master's native ``pops::Weno5``
  (``weno5z``), gives the native trace bit for bit wherever it is finite, and refuses elsewhere.
* P4: the body shares its subexpressions, and the elaboration lowers each shared node once: the
  lowered trace has as many nodes as the b6 body, and the walks of the surface visit each node
  once.
* E3, CF4: a copy of the library's module written in a case gives the same case as the word.
* The options: ``epsilon`` is a finite positive Python number; ``weno5`` takes it by keyword (SU5).
"""
import inspect
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pops
import pops.lib.numerics.reconstructions as library
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
from pops.lib.numerics import rusanov, weno5
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction
from pops.projection import ConservativeCellAverage
from pops.surface import _elaborate
from pops.surface._expr import Expr, Predicate, walk
from pops.surface.math import ddt, div
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method

from examples.api_v02 import b6_numerics, b6_reconstructions

REPO = Path(__file__).resolve().parents[4]
CELLS = 32


# --- the cases (as test_n3_lot2_muscl.py): Burgers on a periodic line, a Rusanov face ------------

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


def _realized(case):
    """The reconstruction.User that the elaborated case gives its face."""
    (plan,) = case.elaborated.case._numerics.values()
    ((_, finite_volume),) = plan.rates.items()
    return finite_volume.reconstruction


def _policy_lines(descriptor):
    """The generated lines that compute the trace of a policy, its samples spelled by their
    offset, and the values whose finiteness it checks."""
    policy = emit_user_reconstruction_policy(SimpleNamespace(_user_reconstruction=descriptor))
    lines = [line.strip() for line in policy.splitlines()]
    computed = [re.sub(r"sample\((-?\d+)(, 0)?\)", r"sample(\1)", line) for line in lines
                if line.startswith(("pops::Real ", "const pops::Real "))]
    checked = set(re.findall(r"std::isfinite\((\w+)\)", policy))
    return computed, checked


def _cpp_double(value):
    if math.isnan(value):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(value):
        return "%sstd::numeric_limits<double>::infinity()" % ("-" if value < 0 else "")
    return float.hex(value)


def _hand_weno5(epsilon):
    """The reconstruction written by hand for b6 over the samples -2 to 2 with the b6 WENO5-Z
    body, the order bound of its stencil, as the surface realizes a ``Reconstruct``."""
    def recon(u):
        def sampled(sample):
            samples = tuple(sample(k)[0] for k in (-2, -1, 0, 1, 2))
            return (b6_reconstructions._weno5z(samples, epsilon),)

        return reconstruction.User(sampled, state=u, formal_order=5, name="weno5")

    return recon


# --- E1, P1 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("epsilon", (None, 1e-6))
def test_weno5_is_the_case_written_with_the_b6_body(epsilon):
    word = weno5() if epsilon is None else weno5(epsilon=epsilon)
    _assert_same(_surface_burgers(word),
                 _hand_burgers(_hand_weno5(1e-40 if epsilon is None else epsilon)))


@pytest.mark.parametrize("epsilon", (None, 1e-6))
def test_the_trace_is_the_b6_weno5_node_for_node(epsilon):
    word = weno5() if epsilon is None else weno5(epsilon=epsilon)
    realized = _realized(_surface_burgers(word))
    oracle = b6_reconstructions.weno5(realized.options["state"], epsilon=epsilon)
    assert _policy_lines(realized) == _policy_lines(oracle)
    assert len(realized.options["body"]["nodes"]) == len(oracle.options["body"]["nodes"])
    for option in ("formal_order", "ghost_depth", "stencil_min_offset", "stencil_max_offset"):
        assert realized.options[option] == oracle.options[option]
    assert (realized.options["formal_order"], realized.options["ghost_depth"]) == (5, 3)


# --- E2: the generated policy against master's native WENO5 --------------------------------------

# Five-sample stencils over zeros of both signs, subnormal, tiny, ordinary, values on both sides of
# the native rescaling bound 1e37, huge and non-finite samples, as test_lib_numerics_reconstructions.
_VALUES = (0., -0., 5e-324, -1e-300, 0.3, 1., -1., 2.5, 1e30, 9e36, -2e37, 1e300, math.inf,
           math.nan)


def _program(epsilon):
    word = weno5() if epsilon is None else weno5(epsilon=epsilon)
    policy = emit_user_reconstruction_policy(SimpleNamespace(
        _user_reconstruction=_realized(_surface_burgers(word))))
    values = ", ".join(_cpp_double(v) for v in _VALUES)
    native_eps = "" if epsilon is None else "  native.eps = %s;" % float.hex(float(epsilon))
    return "\n".join((
        "#include <array>", "#include <cmath>", "#include <cstdint>", "#include <cstdio>",
        "#include <cstring>", "#include <limits>", "#include <pops/numerics/fv/reconstruction.hpp>",
        policy,
        "struct Sampler { const double* centre; int orientation;",
        "  double operator()(int offset) const { return centre[orientation*offset]; }",
        "  double operator()(int offset, int) const { return centre[orientation*offset]; } };",
        "static std::uint64_t bits(double x) { std::uint64_t b; std::memcpy(&b, &x, 8); return b; }",
        "static long equal = 0, refused = 0;",
        "static bool compare(const pops::Weno5& native,",
        "                    const pops_generated::UserReconstructionPolicy& generated,",
        "                    const double* cells) {",
        "  for (int orientation : {1, -1}) {",
        "    const double expected = native.stencil_face_value(Sampler{cells + 2, orientation});",
        "    const double got = generated.stencil_face_state(Sampler{cells + 2, orientation})[0];",
        "    if (std::isfinite(expected) ? bits(got) == bits(expected) : std::isnan(got)) {",
        "      ++(std::isfinite(expected) ? equal : refused);",
        "      continue;",
        "    }",
        "    std::printf(\"MISMATCH %a %a %a %a %a o=%d native=%a generated=%a\\n\", cells[0],",
        "                cells[1], cells[2], cells[3], cells[4], orientation, expected, got);",
        "    return false;",
        "  }",
        "  return true;",
        "}",
        "int main() {",
        "  const double values[] = {%s};" % values,
        "  const int n = sizeof(values)/sizeof(values[0]);",
        "  const pops_generated::UserReconstructionPolicy generated{};",
        "  pops::Weno5 native{};",
        native_eps,
        "  double cells[5];",
        "  for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int c = 0; c < n; ++c)",
        "  for (int d = 0; d < n; ++d) for (int e = 0; e < n; ++e) {",
        "    cells[0] = values[a]; cells[1] = values[b]; cells[2] = values[c];",
        "    cells[3] = values[d]; cells[4] = values[e];",
        "    if (!compare(native, generated, cells)) return 1;",
        "  }",
        "  // Smooth and rough profiles at magnitudes from 1e-30 to 1e49, across the rescaling bound:",
        "  // a sine with a shift and a jump, sampled at a deterministic sequence of phases.",
        "  std::uint64_t state = 88172645463325252ull;",
        "  for (int trial = 0; trial < 200000; ++trial) {",
        "    state ^= state << 13; state ^= state >> 7; state ^= state << 17;",
        "    const double phase = double(state % 100000) / 100000.0 * 6.283185307179586;",
        "    const double magnitude = std::pow(10.0, double(int(state >> 20) % 80) - 30.0);",
        "    const int jump = int((state >> 40) % 6);",
        "    for (int k = 0; k < 5; ++k)",
        "      cells[k] = magnitude * (std::sin(phase + 0.3 * k) + (k >= jump ? 0.5 : 0.0));",
        "    if (!compare(native, generated, cells)) return 1;",
        "  }",
        "  std::printf(\"%ld %ld\\n\", equal, refused);",
        "  return 0;",
        "}",
    ))


@pytest.mark.compiler
@pytest.mark.parametrize("epsilon", (None, 1e-6))
def test_the_generated_policy_of_the_word_is_the_native_weno5z_bit_for_bit(
        epsilon, native_cxx, tmp_path, record_property):
    source = tmp_path / "weno5.cpp"
    source.write_text(_program(epsilon))
    binary = tmp_path / "weno5"
    compiled = subprocess.run(
        [native_cxx, "-std=c++20", "-O2", "-I", str(REPO / "include"), str(source),
         "-o", str(binary)], capture_output=True, text=True, check=False)
    assert compiled.returncode == 0, compiled.stderr[-4000:]
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout[-2000:]
    equal, refused = map(int, ran.stdout.split())
    record_property("bitwise_equal_traces", equal)
    record_property("refused_by_both", refused)
    assert equal + refused == 2*(len(_VALUES)**5 + 200000)
    assert equal > len(_VALUES)**5


# --- P4: each shared node is lowered once, and walked once ---------------------------------------

def test_each_shared_node_of_the_body_is_lowered_once(monkeypatch):
    """Under the inputs of one body (the samples of the trace, the traces of a face), each node
    is lowered once; the face's own nodes are lowered once per body of the face."""
    lowered = []
    original = _elaborate._Elaboration._lower

    def counting(self, value, inputs):
        if inputs is not None and isinstance(value, (Expr, Predicate)):
            lowered.append((value, inputs))
        return original(self, value, inputs)

    monkeypatch.setattr(_elaborate._Elaboration, "_lower", counting)
    _realized(_surface_burgers(weno5()))
    trace = [value for value, inputs in lowered if ("sample", 0) in inputs]
    assert len(trace) > 300
    assert len({(id(value), id(inputs)) for value, inputs in lowered}) == len(lowered)


def test_a_walk_visits_a_shared_node_once():
    word = weno5()
    _realized(_surface_burgers(word))
    ((_, traced),) = word._traced
    visited = [node for node in walk(traced.builder.value) if isinstance(node, (Expr, Predicate))]
    assert len({id(node) for node in visited}) == len(visited)
    # the unfolded tree of the WENO5-Z body has millions of nodes; its graph a few hundred
    assert len(visited) < 2000


# --- E3, CF4: a copy of the library ---------------------------------------------------------------

def test_a_copy_of_the_library_is_the_library():
    namespace = {}
    exec(compile(inspect.getsource(library), library.__file__, "exec"), namespace)  # noqa: S102
    _assert_same(_surface_burgers(namespace["weno5"](epsilon=1e-6)),
                 _hand_burgers(_hand_weno5(1e-6)))


# --- the options ----------------------------------------------------------------------------------

@pytest.mark.parametrize("bad, error", [
    ("1e-6", TypeError), (True, TypeError), (0.0, ValueError), (-1e-6, ValueError),
    (math.inf, ValueError), (math.nan, ValueError)])
def test_the_regulariser_is_a_finite_positive_number(bad, error):
    with pytest.raises(error, match=r"weno5\(epsilon=\)"):
        _realized(_surface_burgers(weno5(epsilon=bad)))


def test_weno5_takes_its_options_by_keyword():
    with pytest.raises(TypeError, match="by keyword"):
        weno5(1e-6)
    with pytest.raises(TypeError, match="has the options epsilon"):
        weno5(eps=1e-6)


def test_the_b6_reconstruct_is_not_a_word_of_the_library():
    """D4: a case writes ``Reconstruct`` from ``pops.surface.discrete``."""
    import pops.lib.numerics as package

    assert not hasattr(package, "Reconstruct") and "Reconstruct" not in package.__all__
    assert not hasattr(library, "Reconstruct") or library.Reconstruct is surface.discrete.Reconstruct
