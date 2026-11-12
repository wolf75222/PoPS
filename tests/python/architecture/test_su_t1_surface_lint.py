"""SU-T1 (spec 0.3, 2.6.7; SD79): the lint of the user surface, over the gallery and the library.

The gallery of the surface (``examples/surface/gallery``, the cases and library drafts of
``/mnt/project-files/dsl/cases/after/`` as R12c left them, with the quadrature rules imported from
``pops.lib.quadrature`` since the review of 11/11 (P3, D0); pinned here by hash) and every module
of ``pops.lib`` that is written with the surface pass the lint of ``scripts/surface_lint.py``. One
module is excepted where it reaches b6: ``pops.lib.numerics.schemes`` returns the b6 realization
of each scheme word until the words are written with ``@pops.scheme`` (A4, the bridge of N15
lot 1). The test fails on a new finding and on an exception that no longer holds.

Each rule fails or warns on a statement written for it, so that a lint that sees nothing fails
here. The words of the lint come from the spec (``--words``), whose hash the word file records.

The test reads the source tree only; it imports neither ``pops`` nor a case.
"""
from __future__ import annotations

import hashlib
import importlib.util
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("surface_lint", ROOT / "scripts" / "surface_lint.py")
assert SPEC is not None and SPEC.loader is not None
lint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint)

GALLERY = ROOT / "examples" / "surface" / "gallery"
GALLERY_SHA256 = {
    "amr_advection.py": "1455340c4808d75ed69256b137e6ec3d30c09c115fe49b194e81fcafadcf70fb",
    "cahn_hilliard.py": "9652af2a0422745809c79937c9ece4d4a1fac2c385b44edfea25d532c5d9d3f6",
    "euler_gravity.py": "94d3b6b61349584c0e9321466ee9733f7dabb33f9c56610781c36cf8a259964a",
    "l1_advection_diffusion.py": "7d56e57c4610500d1a6cb09e9472b366936b3735e300c996f707e8e1998c4a26",
    "l2_guiding_centre.py": "302dc251f447e7425b92d447afcc8d9040876cf1eee99bfa0fea4edc86d8d4fa",
    "l4_two_fluid_drag.py": "8b1a4dd29f75ba8632df993fcfbe2e43fa86c470b6833d9c573d02a008bf173b",
    "lib_methods.py": "a3f35db05eca230fef657f5e11b57d29c0d5b47aa65e39271d2dfb1c3b9fd8ae",
    "lib_models.py": "bfe1d3b06ba66b93d1850d32994b9d1357624adf0c3ead13d9ff9ece3a8be774",
    "lib_schemes.py": "7bc16f74237133755f7847966178d46a3cc90f035b831868083bdca500282e13",
    "m10_drift_diffusion.py": "e90c3636f7ac4c65721937fa0b1520b797024dd3b3d886f306da2e796e9f1cec",
    "m19_vlasov_bgk.py": "5aa4f38164b3fc1df744ad578d90e479ff688abe7cde4ad13ad2ad4578ba961a",
    "m26_aggregation_diffusion.py":
        "0b765124007990d7449a21484c7b7fc052eaaa6c1feb6b5d5715c8ba2c328caf",
    "t02_advection.py": "bd01e13a2fa9f47b3b2ac8cc6c721faefa5ee7792c90efb2f7e586b69c46c48e",
    "t05_moment_closure.py": "5a1d193a9db9392ef41807d8e8f0282ba3d6fd98e41c7910304dbd38dcd4c615",
}
LIBRARY = ROOT / "python" / "pops" / "lib"
# The bridge of N15 lot 1 (python/pops/surface/_numerics.py): the words that it imports to
# return a b6 realization, each a finding of the rule "word" (2.6.9, A4).
BRIDGE = "python/pops/lib/numerics/schemes.py"
BRIDGE_WORDS = {("Diffusion", "pops.numerics"), ("riemann", "pops.numerics"),
                ("FiniteVolume", "pops.numerics.spatial"), ("NotYetElaborated", "pops.surface"),
                ("SchemeValue", "pops.surface"), ("TermView", "pops.surface"),
                # lot 2: upwind(speed=f) is realized by fluxes.upwind, and f is checked to be a
                # plain function of face (SU5)
                ("fluxes", "pops.lib.numerics"), ("check_plain_function", "pops.surface")}
# The cg() bridge of MG9 (lot « solves », python/pops/lib/solve/__init__.py): it returns b6's
# native CG descriptor and refuses the words of 2.6.9 that N9 elaborates with the NotYetElaborated
# of the surface; both findings leave with the bridge, when N9 writes the reference CG (SV9).
SOLVE_BRIDGE = "python/pops/lib/solve/__init__.py"
SOLVE_BRIDGE_WORDS = {("CG", "pops.solvers"), ("NotYetElaborated", "pops.surface")}


@pytest.fixture(scope="module")
def words():
    return lint.load_words()


def _findings(source, words):
    return [(finding.line, finding.kind, finding.rule)
            for finding in lint.lint_source(textwrap.dedent(source), "<case>", words)]


def test_the_gallery_is_the_gallery_of_r12c():
    files = sorted(path.name for path in GALLERY.glob("*.py"))
    assert files == sorted(GALLERY_SHA256)
    for name, digest in GALLERY_SHA256.items():
        assert hashlib.sha256((GALLERY / name).read_bytes()).hexdigest() == digest, name


def test_the_gallery_passes_the_lint(words):
    findings = [str(finding) for path in sorted(GALLERY.glob("*.py"))
                for finding in lint.lint_file(path, words, name=path.name)]
    assert findings == []


def _surface_modules():
    return sorted(path for path in LIBRARY.rglob("*.py")
                  if "import pops.surface" in path.read_text(encoding="utf-8")
                  or "from pops.surface" in path.read_text(encoding="utf-8"))


def test_the_library_written_with_the_surface_passes_the_lint(words):
    modules = [path.relative_to(ROOT).as_posix() for path in _surface_modules()]
    assert {"python/pops/lib/time/methods.py", BRIDGE} <= set(modules)
    bridges = {BRIDGE: set(), SOLVE_BRIDGE: set()}
    for name in modules:
        for finding in lint.lint_file(ROOT / name, words, name=name):
            if name in bridges and finding.rule == "word" and finding.kind == "F":
                word, module = finding.text.split(" of ", 1)[0][len("the word "):], \
                    finding.text.split(" of ", 1)[1].split(",", 1)[0]
                bridges[name].add((word, module))
                continue
            raise AssertionError(str(finding))
    assert bridges == {BRIDGE: BRIDGE_WORDS, SOLVE_BRIDGE: SOLVE_BRIDGE_WORDS}


@pytest.mark.parametrize("source, expected", [
    # F lambda (SU5)
    ("""\
     def rule(cell):
         return 1
     step_rule = lambda cell: 1
     """, [(3, "F", "lambda")]),
    # F string-key (SU8)
    ("""\
     import pops
     numerics = pops.Numerics(schemes={"u": 1})
     """, [(2, "F", "string-key")]),
    # F word: an attribute of pops, an import from a module of pops, a read of an entity
    ("""\
     import pops
     domain = pops.Rectangle(x=(0, 1))
     """, [(2, "F", "word")]),
    ("""\
     from pops.math import maximum
     """, [(1, "F", "word")]),
    ("""\
     import pops

     @pops.method
     def euler(step, rate):
         request = step.request
         return pops.Candidate(step.state + step.dt * rate(step.state, step.time))
     """, [(5, "F", "word")]),
    # F word: a name of the star import that pops.math does not list
    ("""\
     from pops.math import *
     limit = minimum(1, 2)
     """, [(2, "F", "word")]),
    # F star-import
    ("""\
     from pops.lib.numerics import *
     """, [(1, "F", "star-import")]),
    # F math-rebinding: at module level, and a parameter
    ("""\
     from pops.math import *
     grad = 1
     """, [(2, "F", "math-rebinding")]),
    ("""\
     def flux(div):
         return div
     """, [(1, "F", "math-rebinding")]),
    # F float (SD79): literals, module-level names; exact arithmetic passes
    ("""\
     third = 1 / 3
     """, [(1, "F", "float")]),
    ("""\
     from fractions import Fraction
     LENGTH = 2
     HALF = LENGTH / 4
     EXACT = Fraction(LENGTH, 4)
     CELLS = 4 * 16
     """, [(3, "F", "float")]),
    ("""\
     cfl = 0.5 * 2
     """, [(1, "F", "float")]),
    # F pow (SD79)
    ("""\
     square = pow(2, 3)
     """, [(1, "F", "pow")]),
    # F membership (SD79): in and .index on a list or a tuple of handles; a set compares by
    # identity, and a list of strings holds no handle
    ("""\
     import pops
     model = pops.Model("m")
     u = model.quantity("u", 1)
     w = model.quantity("w", 1)
     found = w in [u, w]
     where_w = (u, w).index(w)
     by_identity = w in {u, w}
     named = "u" in ["u", "w"]
     """, [(5, "F", "membership"), (6, "F", "membership")]),
    # W order (SU6)
    ("""\
     import pops
     model = pops.Model("m")
     domain = pops.Box(x=(0, 1))
     """, [(3, "W", "order")]),
    # W vector (2.6.3)
    ("""\
     from pops.math import *
     velocity = vector(1)
     """, [(2, "W", "vector")]),
    # W lag of the state (2.6.8, SD55)
    ("""\
     import pops
     from pops.lib.time import lag

     @pops.method
     def two_step(step, rate):
         return pops.Candidate(lag(step.state, steps=1))
     """, [(6, "W", "lag")]),
])
def test_each_rule_of_the_lint_fires(source, expected, words):
    assert _findings(source, words) == expected


def test_the_transitional_namespace_reads_as_pops(words):
    source = """\
    import pops.surface as pops
    from pops.surface.math import *

    @pops.method
    def euler(step, rate):
        return pops.Candidate(step.state + step.dt * rate(step.state, step.time))
    """
    assert _findings(source, words) == []


def test_the_words_are_those_of_spec_0_3(words):
    source = words["source"]
    assert source["spec"] == "pops_api_spec_v0_3.md"
    assert len(source["sha256"]) == 64
    assert [section["section"] for section in source["sections"]] == list(lint.SECTIONS)
    assert source["skipped_columns"] == list(lint.SKIPPED_COLUMNS)
    # The constructs and the words of the tables; not the spellings of earlier texts.
    assert {"Box", "Periodic", "Numerics", "Candidate", "upwind", "two_point", "ssprk2", "width",
            "state", "dt"} <= words["words"]
    assert not {"Rectangle", "ConstParam", "RuntimeParam", "Realize", "Retry", "Proposal",
                "maximum", "laplacian"} & words["words"]
    assert {"Fraction", "pi", "ddt", "grad", "div", "where", "vector", "outflow"} <= words["math"]
    assert not set(lint.MENTIONED) & words["math"]
    assert words["entities"][:3] == ("cell", "face", "edge")
