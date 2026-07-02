"""ADC-621: the CI import-closure test selector picks the right impacted tests.

These are SOURCE-ONLY tests (no ``pops`` / ``_pops`` import): they exercise the
static import-graph selector in ``scripts/ci_import_closure.py`` and its wiring into
``scripts/ci_select_tests.py`` against the REAL source tree, so a wrong closure that
would silently drop a test's coverage fails the gate here.

Ground-truth edges asserted below were verified by reading the source:

* ``python/pops/runtime/_bound_sim.py`` is imported (function scope) by
  ``test_bind_adapters.py`` and ``test_freeze_lifecycle.py``;
* ``python/pops/numerics/riemann/waves.py`` is imported by
  ``test_wave_speed_providers.py``;
* the DSL cross-test helpers: ``test_dsl_block.py`` imports ``test_dsl_brick.py``
  (bare sibling import), and ``test_dsl_brick.py`` is a shared helper of five tests;
* the LAZY (function-scope) edge ``pops.codegen.orchestration`` ->
  ``pops.runtime._bind_adapters`` must be captured, so a ``_bind_adapters`` change
  reaches the orchestration-dependent tests.
"""
import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cic = _load("ci_import_closure")


# --------------------------------------------------------------------------- #
# Graph builder                                                                #
# --------------------------------------------------------------------------- #
def test_module_graph_has_expected_shape():
    graph = cic.build_module_graph(REPO_ROOT)
    # Every pops sub-package we sampled is a node.
    for name in ("pops", "pops.runtime._bound_sim", "pops.numerics.riemann.waves"):
        assert name in graph, f"{name} missing from the module graph"
    # The graph is non-trivial (hundreds of real edges over the package).
    assert sum(len(v) for v in graph.values()) > 500


def test_relative_import_edge_is_resolved():
    """A relative ``from ._facade_compile import ...`` must become a real edge.

    Regression guard: the relative-import anchor for a plain module drops the module's
    own leaf before applying the level, so ``pops.physics.facade`` importing
    ``._facade_compile`` resolves to ``pops.physics._facade_compile`` (not
    ``pops.physics.facade._facade_compile``).
    """
    graph = cic.build_module_graph(REPO_ROOT)
    assert "pops.physics._facade_compile" in graph.get("pops.physics.facade", set())


def test_lazy_function_scope_edge_is_captured():
    """The function-scope ``pops.codegen.orchestration`` -> ``_bind_adapters`` edge.

    A module-scope-only walker would miss it (orchestration imports ``_bind_adapters``
    only inside functions to keep the import-graph architecture gate green).
    """
    graph = cic.build_module_graph(REPO_ROOT)
    assert "pops.runtime._bind_adapters" in graph.get("pops.codegen.orchestration", set())


# --------------------------------------------------------------------------- #
# Impact query -- ground-truth module -> tests                                 #
# --------------------------------------------------------------------------- #
def test_bound_sim_change_selects_bind_and_freeze_tests():
    sel = cic.impacted_tests(["python/pops/runtime/_bound_sim.py"], repo_root=REPO_ROOT)
    assert "tests/python/integration/bindings/test_bind_adapters.py" in sel
    assert "tests/python/integration/runtime/test_freeze_lifecycle.py" in sel


def test_waves_change_selects_wave_speed_providers_test():
    sel = cic.impacted_tests(
        ["python/pops/numerics/riemann/waves.py"], repo_root=REPO_ROOT
    )
    assert "tests/python/unit/physics/test_wave_speed_providers.py" in sel


def test_bind_adapters_change_selects_bind_adapters_test():
    """The lazy orchestration edge means a ``_bind_adapters`` change reaches its tests."""
    sel = cic.impacted_tests(
        ["python/pops/runtime/_bind_adapters.py"], repo_root=REPO_ROOT
    )
    assert "tests/python/integration/bindings/test_bind_adapters.py" in sel


# --------------------------------------------------------------------------- #
# Cross-test edge closure (both directions)                                    #
# --------------------------------------------------------------------------- #
def test_cross_test_forward_pulls_shared_brick_helper():
    """Selecting ``test_dsl_block`` pulls the ``test_dsl_brick`` helper it imports."""
    _, edges = cic.test_imports(REPO_ROOT)
    assert edges.get("tests/python/unit/codegen/test_dsl_block.py") == {
        "tests/python/unit/codegen/test_dsl_brick.py"
    }
    selected = {"tests/python/unit/codegen/test_dsl_block.py"}
    cic._close_cross_test(selected, edges)
    assert "tests/python/unit/codegen/test_dsl_brick.py" in selected


def test_cross_test_reverse_pulls_dependents_of_shared_helper():
    """Selecting the shared ``test_dsl_brick`` helper pulls every test importing it."""
    _, edges = cic.test_imports(REPO_ROOT)
    selected = {"tests/python/unit/codegen/test_dsl_brick.py"}
    cic._close_cross_test(selected, edges)
    for dependent in (
        "tests/python/unit/codegen/test_dsl_block.py",
        "tests/python/unit/codegen/test_dsl_cse.py",
        "tests/python/unit/codegen/test_dsl_dynamic.py",
        "tests/python/unit/codegen/test_dsl_recon.py",
    ):
        assert dependent in selected


# --------------------------------------------------------------------------- #
# Fallbacks                                                                    #
# --------------------------------------------------------------------------- #
def test_unknown_new_pops_file_is_off_graph():
    """A pops file the graph has never seen triggers the ALL fail-safe."""
    import pytest

    with pytest.raises(cic.OffGraphChange):
        cic.impacted_tests(
            ["python/pops/brand_new_module_not_on_graph.py"], repo_root=REPO_ROOT
        )


def test_non_pops_change_selects_nothing_from_closure():
    """A change outside ``python/pops`` contributes no closure seeds."""
    sel = cic.impacted_tests(["docs/whatever.md", "CHANGELOG.md"], repo_root=REPO_ROOT)
    assert sel == set()


# --------------------------------------------------------------------------- #
# End-to-end wiring in ci_select_tests.plan_python                             #
# --------------------------------------------------------------------------- #
def _run_plan_python(tmp_path, changed_lines):
    sel = _load("ci_select_tests")
    changed = tmp_path / "changed.txt"
    changed.write_text("".join(f"{c}\n" for c in changed_lines), encoding="utf-8")
    tests_file = tmp_path / "tests.txt"
    out = tmp_path / "gh_out.txt"

    class Args:
        pass

    args = Args()
    args.changed_files = str(changed)
    args.tests_file = str(tests_file)
    args.shard_index = None
    args.shard_total = None
    args.github_output = str(out)
    args.force_all = False
    sel.plan_python(args)
    outputs = {}
    for line in out.read_text().splitlines():
        key, _, value = line.partition("=")
        outputs[key] = value
    selected = [t for t in tests_file.read_text().splitlines() if t]
    return outputs, selected


def test_plan_python_leaf_change_is_a_strict_subset_with_smoke(tmp_path):
    """A leaf pops change selects a strict subset that includes the smoke tests."""
    outputs, selected = _run_plan_python(
        tmp_path, ["python/pops/diagnostics/__init__.py"]
    )
    total = int(outputs["python_total"])
    count = int(outputs["python_count"])
    assert outputs["python_mode"] == "subset"
    assert 0 < count < total, "leaf change must be a strict subset"
    assert "import-closure" in outputs["python_why"]
    assert "tests/python/unit/runtime/test_diagnostics_typed.py" in selected
    for smoke in (
        "tests/python/integration/bindings/test_bindings.py",
        "tests/python/unit/runtime/test_capabilities.py",
    ):
        assert smoke in selected


def test_plan_python_direct_test_edit_pulls_cross_test_family(tmp_path):
    """A direct edit of ``test_dsl_block`` pulls its cross-test family + smoke."""
    outputs, selected = _run_plan_python(
        tmp_path, ["tests/python/unit/codegen/test_dsl_block.py"]
    )
    assert outputs["python_mode"] == "subset"
    assert "direct-test" in outputs["python_why"]
    assert "tests/python/unit/codegen/test_dsl_block.py" in selected
    assert "tests/python/unit/codegen/test_dsl_brick.py" in selected  # helper it imports


def test_plan_python_broad_file_runs_all(tmp_path):
    """A broad Python file (the package ``__init__``) forces the whole suite."""
    outputs, _ = _run_plan_python(tmp_path, ["python/pops/__init__.py"])
    assert outputs["python_mode"] == "all"
    assert "broad-file" in outputs["python_why"]


def test_plan_python_bindings_change_runs_all(tmp_path):
    """A ``python/bindings`` change (extension-affecting) forces the whole suite."""
    outputs, _ = _run_plan_python(tmp_path, ["python/bindings/system/init_system.cpp"])
    assert outputs["python_mode"] == "all"


def test_plan_python_changelog_only_selects_none(tmp_path):
    """A CHANGELOG-only change gracefully selects nothing (mode ``none``)."""
    outputs, selected = _run_plan_python(tmp_path, ["CHANGELOG.md"])
    assert outputs["python_mode"] == "none"
    assert outputs["python_count"] == "0"
    assert selected == []


def test_plan_python_new_pops_file_fails_safe_to_all(tmp_path):
    """An off-graph (new) pops file falls back to ALL rather than under-selecting."""
    outputs, _ = _run_plan_python(
        tmp_path, ["python/pops/a_freshly_added_leaf_module.py"]
    )
    assert outputs["python_mode"] == "all"
    assert "off-graph-pops-file" in outputs["python_why"]
