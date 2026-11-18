"""CF17 for L1 (spec 0.3, CF17 and 17.1; exit of phase 1, OD31): the swaps of L1 pass.

``examples/surface/litmus/l1/`` holds L1-a (``upwind()``, ``two_point()``, SSPRK2) and its three
swaps: L1-b puts ``rusanov()`` in place of ``upwind()``, L1-c the upwind face written in the case
with ``@pops.scheme``, and L1-d IMEX Euler with ``cg()`` in place of SSPRK2. Each swap changes only
the lines of the swapped function (its import, its definition and the statement that uses it):
the test reads the diff of the code of each file against L1-a, past the module docstrings. Each
file passes the lint SU-T1, and the four are the CF1 case ``L1-swaps`` (frozen engine unchanged).
Each runs natively in Dim1 from its generated C++ to t = 1 with no Python call inside the native
run (CF9), at the level of L1-a (K6): the integral of u is kept, u stays in [0, 1], and the
distance to the exact cell averages stays within the bound that L1-a fixes; L1-b and L1-c compute
L1-a's face in the same order, so they give it bit for bit (CC7).
"""
from __future__ import annotations

import ast
import difflib
import importlib.util
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
L1 = ROOT / "examples" / "surface" / "litmus" / "l1"
VARIANTS = ("a", "b", "c", "d")

# The lines that each swap removes from L1-a and adds, in the order of the file (CF17).
SWAPS = {
    "b": (["from pops.lib.numerics import upwind, two_point",
           "                         schemes={advection: upwind(), diffusion: two_point()})"],
          ["from pops.lib.numerics import rusanov, two_point",
           "                         schemes={advection: rusanov(), diffusion: two_point()})"]),
    "c": (["from pops.lib.numerics import upwind, two_point"],
          ["from pops.lib.numerics import two_point",
           "from pops.surface.discrete import FaceFlux",
           "",
           "",
           "@pops.scheme",
           "def upwind(face, term):",
           '    """The upwind face written in the case: the mean of the two fluxes minus |s| '
           "(u_R - u_L) / 2,",
           '    s = F(1).n, which for a linear flux is s u_L where s > 0 and s u_R elsewhere."""',
           "    q = term.quantity",
           "    term.expect(term.law(states={q: 1}) * q)",
           "    s = abs(dot(term.law(states={q: 1}), face.normal))",
           "    left = dot(term.law(states={q: face.left(q)}), face.normal)",
           "    right = dot(term.law(states={q: face.right(q)}), face.normal)",
           "    return FaceFlux((left + right) / 2 - s * (face.right(q) - face.left(q)) / 2, "
           "speed=s)",
           ""]),
    "d": (["from pops.lib.time import ssprk2",
           "DT = 1 / (abs(A) * NX + 2 * NU * NX**2)",
           "program = pops.Program(advance=pops.Advance(ssprk2(rate=model), dt=DT, attempts=2),",
           "                       initial={u: u0})"],
          ["from pops.lib.time import imex_euler",
           "from pops.lib.solve import cg",
           "DT = 1 / (abs(A) * NX)",
           "imex = imex_euler(explicit=advection, implicit=diffusion)",
           "program = pops.Program(advance=pops.Advance(imex, dt=DT, attempts=2,",
           "                                            tolerance=pops.Tolerance(rtol=1e-10, "
           "atol=1e-12)),",
           "                       initial={u: u0}, solvers={imex: cg()})"]),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code(variant):
    """The lines of the file past its module docstring."""
    text = (L1 / ("l1_%s.py" % variant)).read_text(encoding="utf-8")
    docstring = ast.parse(text).body[0]
    assert isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant), variant
    return text.splitlines()[docstring.end_lineno:]


@pytest.mark.parametrize("variant", ("b", "c", "d"))
def test_a_swap_changes_only_the_lines_of_the_swapped_function(variant):
    removed, added = [], []
    for line in difflib.ndiff(_code("a"), _code(variant)):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    assert (removed, added) == SWAPS[variant]


def test_the_methods_import_from_pops_lib_time_as_17_1_writes_it():
    """``pops.lib.time`` re-exports the methods of ``pops.lib.time.methods``: the same objects,
    so the files import them as spec 17.1 does (driver 14:12)."""
    import pops.lib.time as package
    from pops.lib.time import imex_euler, ssprk2
    from pops.lib.time import methods

    assert ssprk2 is methods.ssprk2 and imex_euler is methods.imex_euler
    assert {"ssprk2", "imex_euler"} <= set(package.__all__)


def test_the_cases_pass_the_lint():
    lint = _load("surface_lint", ROOT / "scripts" / "surface_lint.py")
    for variant in VARIANTS:
        path = L1 / ("l1_%s.py" % variant)
        assert [str(finding) for finding in lint.lint_file(
            path, lint.load_words(), name=str(path.relative_to(ROOT)))] == [], variant


def _exact(nx, a, nu, t, modes=4000):
    """The exact cell averages of u_t + a u_x = nu u_xx on the unit period from the indicator of
    |x - 1/2| < 1/4: 1/2 + sum_k b_k exp(-nu (2 pi k)^2 t) sinc(k h) cos(2 pi k (x - a t - 1/2)),
    b_k = 2 sin(pi k / 2) / (pi k)."""
    h = 1.0 / nx
    x = (np.arange(nx) + 0.5) * h
    value = np.full(nx, 0.5)
    for k in range(1, modes + 1):
        b = 2 * math.sin(math.pi * k / 2) / (math.pi * k)
        if b == 0.0:
            continue
        decay = math.exp(-nu * (2 * math.pi * k) ** 2 * t)
        if decay < 1e-300:
            break
        sinc = math.sin(math.pi * k * h) / (math.pi * k * h)
        value += b * decay * sinc * np.cos(2 * math.pi * k * (x - a * t - 0.5))
    return value


def _run(variant, tmp_path, monkeypatch):
    """Run the file in a fresh directory, counting the Python frames of its native run (CF9)."""
    from pops.runtime import _simulation as _n8

    calls, runs = [], []
    initialize = _n8.Simulation.__init__

    def counted_init(self, *arguments, **options):
        initialize(self, *arguments, **options)
        self._runtime.state_global("advection_diffusion")     # pybind11's NumPy check, once
        native = self._run_native

        def run_native(*values):
            def profile(frame, event, _):
                if event == "call":
                    calls.append(frame.f_code.co_name)
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                report = native(*values)
                _probe()
            finally:
                sys.setprofile(previous)
            runs.append(report)
            return report
        self._run_native = run_native

    directory = tmp_path / variant
    directory.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(_n8.Simulation, "__init__", counted_init)
        patch.chdir(directory)
        namespace = runpy.run_path(str(L1 / ("l1_%s.py" % variant)), run_name="__main__")
    assert runs and calls == ["_probe"] * len(runs), (variant, calls[:20])
    return namespace


def _probe():
    return None


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cf17_l1")
    with pytest.MonkeyPatch.context() as patch:
        return {variant: _run(variant, tmp_path, patch) for variant in VARIANTS}


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_the_swaps_run_natively_at_the_level_of_l1a(runs):
    first = runs["a"]
    nx, a, nu, t_end = first["NX"], first["A"], first["NU"], first["T_END"]
    exact = _exact(nx, a, nu, t_end)
    states, errors = {}, {}
    for variant, namespace in runs.items():
        result = namespace["result"]
        assert result.failure is None and result.time == t_end, variant
        state = np.asarray(result.state[namespace["u"]], dtype=np.float64)
        assert state.shape == (nx,), variant
        states[variant] = state
        mass = float(np.sum(state)) / nx
        assert abs(mass - 0.5) <= (1e-9 if variant == "d" else 1e-13), (variant, mass)
        assert state.min() >= -1e-9 and state.max() <= 1 + 1e-9, variant
        errors[variant] = float(np.sum(np.abs(state - exact))) / nx
    # L1-b and L1-c compute L1-a's face in the same order (CC7)
    for variant in ("b", "c"):
        assert np.array_equal(states[variant], states["a"]), variant
        assert runs[variant]["result"].steps == runs["a"]["result"].steps, variant
    # the level of L1-a (K6): its distance to the exact averages is that of a first-order upwind
    # face at the Courant number 0.16, whose numerical diffusion a h (1 - c) / 2 adds about 16 % to
    # nu (measured 0.0210); no swap is farther from the exact averages than L1-a (L1-d, explicit at
    # c = 1 where the upwind face carries no numerical diffusion, measured 1.4e-4)
    assert 0.015 < errors["a"] < 0.025, errors
    assert max(errors.values()) == errors["a"], errors
    # L1-d on its own: the explicit face is an exact shift at c = 1, so its distance is that of
    # backward Euler on the diffusion (measured 1.4e-4); the bound catches a tenfold growth that
    # L1-a's level would not see (review O1)
    assert errors["d"] < 1e-3, errors
