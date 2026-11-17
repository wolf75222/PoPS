"""MG9 (spec 0.3, section 18.2; N3 lot 1, ROM-87): the registry of the bridges of ``pops.lib``.

``tests/gates/mg9_bridges.toml`` lists each word of ``pops.lib`` that returns a native route
without a Python body (``[[bridge]]``) and each module still written with a b6 signature
(``[[earlier]]``). The test checks it against the tree with ``scripts/mg9_bridges.py``:

1. every module of ``pops.lib`` is a module of the surface, a module of ``[[earlier]]`` or a
   package that only re-exports its submodules, and a module of the surface passes the lint SU-T1
   without a finding (SD94);
2. the bridges found in the source are exactly the entries of ``[[bridge]]``; an entry whose bridge
   is gone is stale and fails;
3. the registry does not grow against its base (``--check-base``, MG6), which CI runs on every
   event against the first parent;
4. the lint SU-T1 reads its exceptions here (``test_su_t1_surface_lint.py``).

Each rule fails on a tree written for it, so that a check that sees nothing fails here. The test
reads the source tree only; it imports neither ``pops`` nor a case.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mg9 = _load("mg9_bridges", ROOT / "scripts" / "mg9_bridges.py")
lint = _load("surface_lint", ROOT / "scripts" / "surface_lint.py")

# The schemes of pops.lib.numerics.schemes before N3 lot 1, in the form of their realizations:
# each returned a native route of b6 (SD87).
SCHEMES_BEFORE_N3 = """\
from pops.surface import SchemeValue


class _Rusanov(SchemeValue):
    def realize(self, view):
        from pops.numerics import reconstruction, riemann, variables
        from pops.numerics.spatial import FiniteVolume

        return FiniteVolume(flux=view.elaborated, variables=variables.Conservative(view.state),
                            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())


def two_point():
    from pops.numerics import Diffusion

    return Diffusion(flux=None)


def cg(apply):
    from pops.solvers import CG

    return CG(apply=apply)


def face(view):
    from pops.numerics import riemann
    from pops.numerics.stencil_flux import StencilFlux

    return riemann.User(body=None), StencilFlux(dimension=1, components=1, radius=1, body=None)
"""


def test_the_registry_holds_every_module_and_every_bridge_of_pops_lib():
    assert mg9.check(ROOT) == []


def test_no_word_of_n3_is_a_bridge_after_lot_1():
    registry = mg9.read(ROOT)
    # the entries are the ghost laws of the boundary policy of lot B (#774) and cfl_limit() of
    # part D of lot D-E (#788); no word of pops.lib.numerics.fluxes, and no longer cg() of the N15
    # lot « solves » (#776), which the N9 lot « La référence sous les solves de relation » runs as
    # the reference
    assert [(entry["word"], entry["module"], "arrives" in entry) for entry in registry["bridge"]] \
        == [("realize", "pops.lib.numerics.ghost_laws", False),
            ("cfl_limit", "pops.lib.time.limits", False)]
    assert registry["bridge"][0]["owner"].startswith("G04")
    assert {(word, route) for word, module, route in mg9.found_bridges(ROOT)
            if module == "pops.lib.solve"} == set()
    assert {word for word, module, _ in mg9.found_bridges(ROOT)
            if module.startswith("pops.lib.numerics")} == {"realize"}
    kinds = mg9.classify(ROOT, registry)
    assert kinds["pops.lib.numerics.ghost_laws"] == "bridge"
    assert kinds["pops.lib.numerics.fluxes"] == "surface"
    assert kinds["pops.lib.numerics"] == "package"
    assert "pops.lib.numerics.schemes" not in kinds
    earlier = {module: entry["owner"] for entry in registry["earlier"]
               for module in entry["modules"]}
    assert earlier["pops.lib.numerics.face_flux"].startswith("N7")
    assert "pops.lib.numerics.reconstructions" not in earlier
    assert kinds["pops.lib.numerics.limiters"] == "surface"
    assert kinds["pops.lib.numerics.reconstructions"] == "surface"


def test_the_modules_of_the_surface_pass_the_lint():
    words = lint.load_words()
    kinds = mg9.classify(ROOT)
    modules = mg9.library_modules(ROOT)
    surface = sorted(name for name, kind in kinds.items() if kind == "surface")
    assert {"pops.lib.numerics.fluxes", "pops.lib.output", "pops.lib.quadrature",
            "pops.lib.time.methods"} <= set(surface)
    assert kinds["pops.lib.time.when"] == "earlier"
    # a bridge imports the native route that it returns: its findings of the rule "word" leave
    # with it (as in the lint SU-T1)
    bridges = {entry["module"] for entry in mg9.read(ROOT)["bridge"]}
    for name in surface:
        path = modules[name]
        rel = path.relative_to(ROOT).as_posix()
        findings = [str(finding) for finding in lint.lint_file(path, words, name=rel)
                    if not (name in bridges and finding.rule == "word" and finding.kind == "F")]
        assert findings == [], findings


def test_a_step_rule_value_is_a_bridge_under_its_public_word():
    """cfl_limit() of N15 lot D-E part D returns a StepRuleValue whose realize is b6's
    AdaptiveCFL: the bridge is the public word, owned by N8 until dt= is a function of cell."""
    registry = mg9.read(ROOT)
    assert {(word, route) for word, module, route in mg9.found_bridges(ROOT)
            if module == "pops.lib.time.limits"} == {("cfl_limit", "StepRuleValue")}
    entry = [entry for entry in registry["bridge"] if entry["word"] == "cfl_limit"][0]
    assert entry["owner"].startswith("N8, which elaborates dt= as a function of cell")
    assert mg9.classify(ROOT, registry)["pops.lib.time.limits"] == "surface"
    found = mg9.bridges_of(textwrap.dedent("""\
        from pops.surface import StepRuleValue


        class _Limit(StepRuleValue):
            pass


        def limit(flux):
            return _Limit(flux)
        """), "pops.lib.time.limits")
    assert found == {("limit", "pops.lib.time.limits", "StepRuleValue")}


def test_a_native_route_is_found_and_a_body_is_not():
    found = mg9.bridges_of(SCHEMES_BEFORE_N3, "pops.lib.numerics.schemes")
    assert found == {
        ("_Rusanov", "pops.lib.numerics.schemes", "SchemeValue"),
        ("_Rusanov", "pops.lib.numerics.schemes", "reconstruction.FirstOrder"),
        ("_Rusanov", "pops.lib.numerics.schemes", "riemann.Rusanov"),
        ("_Rusanov", "pops.lib.numerics.schemes", "pops.numerics.spatial.FiniteVolume"),
        ("two_point", "pops.lib.numerics.schemes", "pops.numerics.Diffusion"),
        ("cg", "pops.lib.numerics.schemes", "pops.solvers.CG"),
    }


def _tree(tmp_path, registry, modules):
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts", ignore=shutil.ignore_patterns(
        "__pycache__"))
    (tmp_path / "tests" / "gates").mkdir(parents=True)
    (tmp_path / mg9.REGISTRY).write_text(textwrap.dedent(registry), encoding="utf-8")
    for name, source in modules.items():
        path = tmp_path / "python" / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return tmp_path


HEADER = """\
schema_version = 1
gate = "mg9-bridges"
"""
PACKAGE = '"""The package."""\nfrom . import numerics\n\n__all__ = ["numerics"]\n'


def test_an_unlisted_module_and_a_stale_earlier_entry_fail(tmp_path):
    root = _tree(tmp_path, HEADER + """\
        bridge = []

        [[earlier]]
        modules = ["pops.lib.gone"]
        words = ["w"]
        owner = "N0"
        """, {"pops.lib.__init__": PACKAGE,
              "pops.lib.numerics.__init__": "from pops.numerics import riemann\n\nVALUE = 1\n"})
    assert mg9.check(root) == [
        "pops.lib.numerics is neither a module of the surface, nor [[earlier]], nor a package "
        "that re-exports its submodules",
        "[[earlier]] lists pops.lib.gone, which pops.lib no longer has: remove it"]


def test_a_module_that_imports_no_module_of_pops_is_of_the_surface_unless_earlier(tmp_path):
    """A writer in plain Python (pops.lib.output) holds no b6 signature; a body over b6's
    continuation without an import (pops.lib.time.when) stays where [[earlier]] lists it."""
    plain = "import numpy as np\n\n\ndef npz(path):\n    return np.asarray(path)\n"
    registry = HEADER + """\
        bridge = []

        [[earlier]]
        modules = ["pops.lib.when"]
        words = ["every"]
        owner = "N8"
        """
    root = _tree(tmp_path, registry, {"pops.lib.__init__": PACKAGE, "pops.lib.numerics": plain,
                                      "pops.lib.when": plain,
                                      "pops.lib.relative": "from . import numerics\nVALUE = 1\n"})
    kinds = mg9.classify(root)
    assert kinds["pops.lib.numerics"] == "surface" and kinds["pops.lib.when"] == "earlier"
    assert kinds["pops.lib.relative"] == "unlisted"
    assert mg9.check(root) == [
        "pops.lib.relative is neither a module of the surface, nor [[earlier]], nor a package "
        "that re-exports its submodules"]


SOLVE = """\
import pops.surface as pops


class _CG:
    def realize(self):
        from pops.solvers import CG

        return CG(max_iter=2000)


def cg(*, max_iterations=2000):
    return _CG()
"""
CG_ENTRY = """\
    [[bridge]]
    word = "cg"
    module = "pops.lib.solve"
    route = "pops.solvers.CG"
    owner = "N9"
    refusal = "tests/test_refusals.py::test_a_parameter_names_the_bridge"
    exit_test = "L1-d of 17.1 with nu a model.parameter"
    """


def test_a_bridge_recorded_before_its_lot_lands_carries_the_conditions_of_its_ruling(tmp_path):
    """cg() of the N15 lot « solves »: arrives= while pops.lib.solve is absent, then the arriving
    lot removes it; the refusal named by the entry must be a test of the tree."""
    pending = HEADER + textwrap.dedent(CG_ENTRY) + 'arrives = "N15 lot solves"\n'
    surface = "import pops.surface\n"
    root = _tree(tmp_path / "before", pending, {"pops.lib.__init__": PACKAGE,
                                                "pops.lib.numerics.__init__": surface})
    assert mg9.check(root) == []
    assert mg9.check(_tree(tmp_path / "unstated", HEADER + textwrap.dedent(CG_ENTRY), {
        "pops.lib.__init__": PACKAGE, "pops.lib.numerics.__init__": "import pops.surface\n"})) == [
        "[[bridge]] lists cg of pops.lib.solve, which pops.lib does not have: remove it, or state "
        "the lot that brings it with arrives="]
    landed = {"pops.lib.__init__": PACKAGE, "pops.lib.numerics.__init__": "import pops.surface\n",
              "pops.lib.solve.__init__": SOLVE}
    assert mg9.bridges_of(SOLVE, "pops.lib.solve") == {("cg", "pops.lib.solve", "pops.solvers.CG")}
    root = _tree(tmp_path / "arrived", pending, landed)
    assert mg9.check(root) == [
        "[[bridge]] cg of pops.lib.solve has arrived: remove its arrives=",
        "[[bridge]] cg of pops.lib.solve names the refusal "
        "tests/test_refusals.py::test_a_parameter_names_the_bridge, which the tree does not have"]
    root = _tree(tmp_path / "after", HEADER + textwrap.dedent(CG_ENTRY), landed)
    (root / "tests" / "test_refusals.py").write_text(
        "def test_a_parameter_names_the_bridge():\n    pass\n", encoding="utf-8")
    assert mg9.check(root) == []
    with pytest.raises(mg9.RegistryError, match="may have arrives, refusal and exit_test"):
        mg9.load(HEADER + textwrap.dedent(CG_ENTRY) + 'reason = "x"\n')


def test_a_bridge_must_be_listed_and_a_stale_bridge_fails(tmp_path):
    bridge = "import pops.surface as pops\n\n\ndef rusanov():\n" \
             "    from pops.numerics import riemann\n\n    return riemann.Rusanov()\n"
    root = _tree(tmp_path, HEADER + "bridge = []\n",
                 {"pops.lib.__init__": PACKAGE, "pops.lib.numerics.__init__": bridge})
    assert mg9.check(root) == [
        "rusanov of pops.lib.numerics returns a native route and is not a [[bridge]]"]
    root = _tree(tmp_path / "stale", HEADER + """\
        [[bridge]]
        word = "rusanov"
        module = "pops.lib.numerics"
        route = "riemann.Rusanov"
        owner = "N3"
        """, {"pops.lib.__init__": PACKAGE,
              "pops.lib.numerics.__init__": "import pops.surface as pops\n"})
    assert mg9.check(root) == [
        "[[bridge]] lists rusanov of pops.lib.numerics, which no longer returns a native route: "
        "remove it"]


def test_the_registry_only_shrinks_against_its_base(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        *args], check=True, stdout=subprocess.DEVNULL)

    base = HEADER + """\
        bridge = []

        [[earlier]]
        modules = ["pops.lib.numerics"]
        words = ["muscl"]
        owner = "N3"
        """
    _tree(tmp_path, base, {"pops.lib.__init__": PACKAGE, "pops.lib.numerics.__init__": "x = 1\n"})
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    assert mg9.check_base("HEAD", tmp_path) == []
    grown = HEADER + """\
        [[bridge]]
        word = "cg"
        module = "pops.lib.numerics"
        route = "pops.solvers.CG"
        owner = "N9"

        [[earlier]]
        modules = ["pops.lib.numerics", "pops.lib.other"]
        words = ["muscl", "weno5"]
        owner = "N3"
        """
    (tmp_path / mg9.REGISTRY).write_text(textwrap.dedent(grown), encoding="utf-8")
    assert mg9.check_base("HEAD", tmp_path) == [
        "[[bridge]] cg of pops.lib.numerics is not in the base",
        "[[earlier]] pops.lib.numerics gains the words weno5",
        "[[earlier]] pops.lib.other is not in the base"]
    assert mg9.main(["check", "--source", str(tmp_path), "--check-base", "HEAD"]) == 1


def test_a_base_without_the_registry_records_nothing(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "base"], check=True)
    _tree(tmp_path, HEADER + "bridge = []\n", {"pops.lib.__init__": '"""Empty."""\n'})
    assert mg9.check_base("HEAD", tmp_path) == []
    with pytest.raises(mg9.RegistryError, match="not in the repository"):
        mg9.check_base("0" * 40, tmp_path)


def test_a_builder_handed_to_a_word_at_import_is_that_word_s_bridge():
    """N9 lot 4 bound the native CG to the @pops.solver cg at import, until the N9 lot « La
    référence sous les solves de relation »: ``cg.bind_relation_native(_native_cg, ...)`` made cg
    the word of the bridge, and the gate still reads such a binding."""
    source = textwrap.dedent("""\
        from .krylov import cg


        def _native_cg(options, request):
            from pops.solvers import CG

            return CG(max_iter=options["max_iterations"])


        cg.bind_relation_native(_native_cg, requires=("symmetric_positive_definite",))
        """)
    assert mg9.bridges_of(source, "pops.lib.solve") == {("cg", "pops.lib.solve",
                                                         "pops.solvers.CG")}
    unbound = source.replace("cg.bind_relation_native(_native_cg, ", "print(")
    assert mg9.bridges_of(unbound, "pops.lib.solve") == {("_native_cg", "pops.lib.solve",
                                                          "pops.solvers.CG")}


def test_a_module_that_only_a_bridge_holds_is_scanned_and_leaves_with_it(tmp_path):
    """The ghost laws of lot B: a module that imports only the b6 boundary conditions is a bridge
    module while a [[bridge]] lists it, unlisted otherwise, and its entry is stale once it no
    longer builds the route."""
    laws = ("def realize(form, state, value):\n"
            "    from pops.boundary.transport import Inflow, Outflow\n\n"
            "    return Inflow(state=state, value=value) if form == 'Value' else "
            "Outflow(state=state)\n")
    entry = HEADER + """\
        [[bridge]]
        word = "realize"
        module = "pops.lib.numerics.ghost_laws"
        route = "pops.boundary.transport.Inflow, pops.boundary.transport.Outflow"
        owner = "G04"
        """
    modules = {"pops.lib.__init__": PACKAGE, "pops.lib.numerics.__init__": "import pops.surface\n",
               "pops.lib.numerics.ghost_laws": laws}
    assert mg9.check(_tree(tmp_path / "listed", entry, modules)) == []
    assert mg9.check(_tree(tmp_path / "unlisted", HEADER + "bridge = []\n", modules)) == [
        "pops.lib.numerics.ghost_laws is neither a module of the surface, nor [[earlier]], nor a "
        "package that re-exports its submodules"]
    modules["pops.lib.numerics.ghost_laws"] = "from pops.boundary import transport\n"
    assert mg9.check(_tree(tmp_path / "stale", entry, modules)) == [
        "[[bridge]] lists realize of pops.lib.numerics.ghost_laws, which no longer returns a "
        "native route: remove it"]
