"""ADC-629: the C++ test selector narrows a header change by include-graph impact.

These are SOURCE-ONLY tests (no ``pops`` / ``_pops`` import): they exercise the
``#include``-graph model in ``scripts/ci_include_graph.py`` and its wiring into
``scripts/ci_select_tests.py`` against the REAL source tree. A wrong closure that would
silently drop a suite's coverage -- or fail to escalate a shared header to FULL -- fails
the gate here.

The selection contract asserted below:

* HEADER IMPACT: when every changed file is a project header under ``include/pops/``, each
  suite is selected iff its source ``#include`` closure reaches a changed header.
* GLOBAL INCLUDERS: a changed header in the transitive closure of the heavy shared TUs /
  seam templates / codegen emitter / ``tests/cpp/support`` selects ALL suites (it is compiled
  into or linked by effectively every target).
* FAIL-OPEN: a changed header absent from the tree, a mixed (non-pure-header) change, or any
  graph anomaly selects ALL suites -- never a subset.

The pruning anchor (``splitting.hpp`` -> ``test_splitting``) is a hand-verified leaf; the
strict-subset and existence checks are computed from the graph itself so they stay
self-maintaining as the tree drifts.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


graph = _load("ci_include_graph")
sel = _load("ci_select_tests")


def _serial_suites():
    """The manifest's serial (non-MPI) C++ suites, the selection universe."""
    return sel.manifest_cpp_suites(sel.load_manifest())


def _suite_impacted_by(header):
    """Ground-truth: serial suite names whose source closure reaches ``header``."""
    hit = set()
    for suite in _serial_suites():
        closure = set()
        for source in suite["sources"]:
            closure |= graph.source_closure(source)
        if header in closure:
            hit.add(suite["name"])
    return hit


# --------------------------------------------------------------------------- #
# Synthetic-repo graph logic                                                   #
# --------------------------------------------------------------------------- #
def test_pops_includes_parses_angle_directives_only():
    """Only ``#include <pops/...>`` is an edge; quoted / system includes are ignored."""
    text = (
        '#include <pops/a/b.hpp>\n'
        '#include <pops/c/d.hpp>  // trailing comment\n'
        '#include "pops/not_an_edge.hpp"\n'
        '#include <vector>\n'
        '# include < pops/spaced.hpp >\n'
    )
    assert graph.pops_includes(text) == {
        "pops/a/b.hpp",
        "pops/c/d.hpp",
        "pops/spaced.hpp",
    }


def test_transitive_closure_follows_header_edges(tmp_path, monkeypatch):
    """A BFS over a synthetic ``include/pops`` tree returns the full reachable set."""
    include = tmp_path / "include"
    (include / "pops" / "x").mkdir(parents=True)
    (include / "pops" / "x" / "root.hpp").write_text(
        "#include <pops/x/mid.hpp>\n", encoding="utf-8"
    )
    (include / "pops" / "x" / "mid.hpp").write_text(
        "#include <pops/x/leaf.hpp>\n", encoding="utf-8"
    )
    (include / "pops" / "x" / "leaf.hpp").write_text("// no edges\n", encoding="utf-8")
    monkeypatch.setattr(graph, "INCLUDE_DIR", include)
    closure = graph.transitive_closure(["pops/x/root.hpp"])
    assert closure == {"pops/x/root.hpp", "pops/x/mid.hpp", "pops/x/leaf.hpp"}


def test_transitive_closure_tolerates_a_dangling_include(tmp_path, monkeypatch):
    """A ``#include`` of a header that does not exist is a node with no out-edges."""
    include = tmp_path / "include"
    (include / "pops").mkdir(parents=True)
    (include / "pops" / "root.hpp").write_text(
        "#include <pops/missing.hpp>\n", encoding="utf-8"
    )
    monkeypatch.setattr(graph, "INCLUDE_DIR", include)
    closure = graph.transitive_closure(["pops/root.hpp"])
    assert closure == {"pops/root.hpp", "pops/missing.hpp"}
    assert graph.header_exists("pops/root.hpp")
    assert not graph.header_exists("pops/missing.hpp")


def test_source_closure_raises_on_missing_source():
    """A suite source that is not on disk is an anomaly (fail-open signal)."""
    with pytest.raises(graph.GraphError):
        graph.source_closure("tests/cpp/does/not/exist.cpp")


# --------------------------------------------------------------------------- #
# Real-tree graph facts                                                        #
# --------------------------------------------------------------------------- #
def test_global_includer_closure_contains_the_heavy_system_header():
    """``pops/runtime/system.hpp`` is reachable from the bound system TU -> global."""
    assert "pops/runtime/system.hpp" in graph.global_includer_closure()


def test_leaf_header_hits_exactly_its_one_suite_and_is_not_global():
    """Hand-anchored pruning: ``splitting.hpp`` is included by exactly ``test_splitting``.

    This is the real-tree fact the subset selection depends on. If a future edit makes
    ``splitting.hpp`` reach more suites (or become global), this pins the change loudly.
    """
    header = "pops/numerics/time/schemes/splitting.hpp"
    assert graph.header_exists(header)
    assert header not in graph.global_includer_closure()
    assert _suite_impacted_by(header) == {"test_splitting"}


def test_at_least_one_non_global_leaf_header_exists():
    """The graph must expose real pruning: some header hits a strict subset of suites.

    Computed from the graph so it stays self-maintaining. If EVERY header became global the
    impact selection would degenerate to FULL and this fence would catch the regression.
    """
    suites = _serial_suites()
    all_names = {s["name"] for s in suites}
    global_closure = graph.global_includer_closure()
    pruning = []
    for suite in suites:
        for source in suite["sources"]:
            for header in graph.source_closure(source):
                if header in global_closure:
                    continue
                impacted = _suite_impacted_by(header)
                if 0 < len(impacted) < len(all_names):
                    pruning.append(header)
                    break
            if pruning:
                break
        if pruning:
            break
    assert pruning, "no non-global header prunes the suite set -- impact selection is a no-op"


# --------------------------------------------------------------------------- #
# End-to-end wiring in ci_select_tests.plan_cpp                                #
# --------------------------------------------------------------------------- #
def _run_plan_cpp(tmp_path, changed_lines):
    changed = tmp_path / "changed.txt"
    changed.write_text("".join(f"{c}\n" for c in changed_lines), encoding="utf-8")
    out = tmp_path / "gh_out.txt"

    class Args:
        pass

    args = Args()
    args.changed_files = str(changed)
    args.github_output = str(out)
    args.explain_file = None
    args.force_all = False
    sel.plan_cpp(args)
    outputs = {}
    for line in out.read_text().splitlines():
        key, _, value = line.partition("=")
        outputs[key] = value
    targets = [t for t in outputs.get("cpp_targets", "").split(" ") if t]
    return outputs, targets


def test_leaf_header_selects_a_strict_subset_containing_its_suite(tmp_path):
    """A leaf-header change prunes to a strict subset that includes the impacted suite."""
    outputs, targets = _run_plan_cpp(
        tmp_path, ["include/pops/numerics/time/schemes/splitting.hpp"]
    )
    total = int(outputs["cpp_total"])
    count = int(outputs["cpp_count"])
    assert outputs["cpp_mode"] == "subset"
    assert 0 < count < total, "leaf header must select a strict subset"
    assert "test_splitting" in targets
    for smoke in sel.CPP_SMOKE_TARGETS:
        assert smoke in targets


def test_global_includer_header_selects_all(tmp_path):
    """A header in the heavy-TU closure escalates to FULL (soundness rule)."""
    outputs, _ = _run_plan_cpp(tmp_path, ["include/pops/runtime/system.hpp"])
    assert outputs["cpp_mode"] == "all"
    assert outputs["cpp_count"] == outputs["cpp_total"]


def test_nonexistent_header_fails_open_to_all(tmp_path):
    """A changed header absent from the tree cannot be reasoned about -> FULL."""
    outputs, _ = _run_plan_cpp(tmp_path, ["include/pops/does/not/exist.hpp"])
    assert outputs["cpp_mode"] == "all"


def test_mixed_header_and_cmake_change_selects_all(tmp_path):
    """A leaf header prunes, but an unmapped cmake file in the same change forces FULL.

    ADC-646: the header contributes its include-closure to the union, yet ``cmake/toolchain.cmake``
    is an unmapped build input whose per-file impact is ALL, so the union escalates to FULL.
    """
    outputs, _ = _run_plan_cpp(
        tmp_path,
        ["include/pops/numerics/time/schemes/splitting.hpp", "cmake/toolchain.cmake"],
    )
    assert outputs["cpp_mode"] == "all"


def test_core_header_stays_broad_full(tmp_path):
    """A ``core`` header is a broad-prefix change -> FULL before impact logic runs."""
    outputs, _ = _run_plan_cpp(tmp_path, ["include/pops/core/state/state.hpp"])
    assert outputs["cpp_mode"] == "all"


def test_two_leaf_headers_select_the_union(tmp_path):
    """Two non-global leaf headers select the UNION of their impacted suites."""
    outputs, targets = _run_plan_cpp(
        tmp_path,
        [
            "include/pops/numerics/time/schemes/splitting.hpp",
            "include/pops/numerics/time/integrators/ssprk.hpp",
        ],
    )
    assert outputs["cpp_mode"] == "subset"
    assert {"test_splitting", "test_diffusion"} <= set(targets)


def test_every_selected_suite_name_exists_in_the_manifest(tmp_path):
    """Whatever a leaf-header change selects, every target is a real serial suite."""
    _outputs, targets = _run_plan_cpp(
        tmp_path, ["include/pops/numerics/time/schemes/splitting.hpp"]
    )
    manifest_names = {s["name"] for s in _serial_suites()}
    unknown = sorted(set(targets) - manifest_names)
    assert not unknown, f"selected targets not in the manifest: {unknown}"


def test_empty_change_selects_none(tmp_path):
    """No changed files is unchanged behavior: mode ``none`` (nothing to build)."""
    outputs, targets = _run_plan_cpp(tmp_path, [])
    assert outputs["cpp_mode"] == "none"
    assert targets == []


# --------------------------------------------------------------------------- #
# ADC-646: compositional per-file impact (union of per-file impacts)           #
# --------------------------------------------------------------------------- #
def _run_plan_cpp_explain(tmp_path, changed_lines):
    """Run ``plan_cpp`` and return ``(outputs, targets, plan)`` including the explain JSON."""
    changed = tmp_path / "changed.txt"
    changed.write_text("".join(f"{c}\n" for c in changed_lines), encoding="utf-8")
    out = tmp_path / "gh_out.txt"
    plan_path = tmp_path / "plan.json"

    class Args:
        pass

    args = Args()
    args.changed_files = str(changed)
    args.github_output = str(out)
    args.explain_file = str(plan_path)
    args.force_all = False
    sel.plan_cpp(args)
    outputs = {}
    for line in out.read_text().splitlines():
        key, _, value = line.partition("=")
        outputs[key] = value
    targets = [t for t in outputs.get("cpp_targets", "").split(" ") if t]
    plan = json.loads(plan_path.read_text())
    return outputs, targets, plan


def test_binding_tu_selects_only_its_object_lib_consumers(tmp_path):
    """A binding TU maps to the test targets compiling it, NOT the broad label group.

    ``system_fields.cpp`` is one of the ``pops_runtime_system`` OBJECT-lib TUs; the selection is
    exactly that lib's serial consumers (+ smoke), a strict subset far below the old bindings
    label group. The precise consumer set is read from ``tests/CMakeLists.txt``.
    """
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path, ["python/bindings/system/base/system_fields.cpp"]
    )
    assert outputs["cpp_mode"] == "subset"
    _sources, consumers = sel._binding_object_lib_map()
    serial = {s["name"] for s in _serial_suites()}
    expected = {t for t in consumers["pops_runtime_system"] if t in serial}
    assert expected, "no serial consumers parsed for pops_runtime_system"
    assert expected <= set(targets)
    for smoke in sel.CPP_SMOKE_TARGETS:
        assert smoke in targets
    entry = plan["impact"]["python/bindings/system/base/system_fields.cpp"]
    assert entry["kind"] == "binding-tu-targets"
    assert entry["object_libs"] == ["pops_runtime_system"]


def test_binding_private_header_maps_to_the_same_object_lib(tmp_path):
    """A binding private header impacts the OBJECT lib whose TUs ``#include`` it.

    ``system_impl.hpp`` is included by the ``pops_runtime_system`` TUs, so a change to it selects
    the same consumers as a change to one of those TUs.
    """
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path, ["python/bindings/system/base/system_impl.hpp"]
    )
    assert outputs["cpp_mode"] == "subset"
    entry = plan["impact"]["python/bindings/system/base/system_impl.hpp"]
    assert entry["kind"] == "binding-tu-targets"
    assert entry["object_libs"] == ["pops_runtime_system"]
    assert set(entry["targets"]) <= set(targets)


def test_codegen_emitter_maps_to_codegen_label_group_only(tmp_path):
    """A ``python/pops/codegen/**`` emitter selects the codegen / native-loader group only."""
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path, ["python/pops/codegen/program_emit_ops.py"]
    )
    assert outputs["cpp_mode"] == "subset"
    codegen_labels = sel.expand_area_labels(sel.CPP_CODEGEN_AREAS)
    expected = {s["name"] for s in _serial_suites() if s["labels"] & codegen_labels}
    assert expected, "codegen area selected no suite"
    assert expected <= set(targets)
    entry = plan["impact"]["python/pops/codegen/program_emit_ops.py"]
    assert entry["kind"] == "codegen-labels"


def test_pops_non_codegen_python_has_zero_cpp_impact(tmp_path):
    """A non-codegen ``python/pops`` edit + docs + CHANGELOG selects no C++ suite."""
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path,
        [
            "python/pops/time/program.py",
            "docs/whatever.md",
            "CHANGELOG.md",
            "tests/python/unit/time/test_time_condensed_schur.py",
        ],
    )
    assert outputs["cpp_mode"] == "none"
    assert targets == []
    assert all(entry["kind"] == "none" for entry in plan["impact"].values())


def test_compositional_union_prunes_a_mixed_change(tmp_path):
    """A leaf header + binding TU + codegen + zero-impact files select their UNION, not ALL.

    The core ADC-646 win: none of these files is a global includer or an unmapped build input, so
    the change prunes to the union of the leaf-header closure, the binding-TU consumers and the
    codegen group -- a strict subset -- instead of collapsing to coarse labels or ALL.
    """
    changed = [
        "CHANGELOG.md",
        "include/pops/numerics/time/schemes/splitting.hpp",
        "python/bindings/system/base/system_fields.cpp",
        "python/pops/codegen/program_emit_control.py",
        "python/pops/time/program.py",
        "tests/python/unit/time/test_time_condensed_schur.py",
    ]
    outputs, targets, plan = _run_plan_cpp_explain(tmp_path, changed)
    assert outputs["cpp_mode"] == "subset"
    assert 0 < int(outputs["cpp_count"]) < int(outputs["cpp_total"])
    # The union must contain the leaf-header suite, a binding consumer and a codegen suite.
    assert "test_splitting" in targets
    kinds = {f: v["kind"] for f, v in plan["impact"].items()}
    assert kinds["include/pops/numerics/time/schemes/splitting.hpp"] == "include-impact"
    assert kinds["python/bindings/system/base/system_fields.cpp"] == "binding-tu-targets"
    assert kinds["python/pops/codegen/program_emit_control.py"] == "codegen-labels"
    assert kinds["python/pops/time/program.py"] == "none"


def test_global_header_in_a_mixed_change_still_forces_all(tmp_path):
    """A global-includer header anywhere in the change escalates the union to FULL (soundness).

    This is the literal ADC-427 shape: ``system.hpp`` / the program-context headers are global
    includers (compiled into every target via the binding TUs and the emitter), so the sound
    selection is ALL -- the plan spells out the per-file reason for each.
    """
    changed = [
        "CHANGELOG.md",
        "include/pops/runtime/program/amr_program_context.hpp",
        "include/pops/runtime/program/program_context.hpp",
        "include/pops/runtime/system.hpp",
        "python/bindings/system/base/system_fields.cpp",
        "python/pops/codegen/program_emit_control.py",
        "python/pops/time/program.py",
        "tests/python/unit/time/test_time_condensed_schur.py",
    ]
    outputs, _targets, plan = _run_plan_cpp_explain(tmp_path, changed)
    assert outputs["cpp_mode"] == "all"
    assert outputs["cpp_count"] == outputs["cpp_total"]
    for header in (
        "include/pops/runtime/system.hpp",
        "include/pops/runtime/program/program_context.hpp",
        "include/pops/runtime/program/amr_program_context.hpp",
    ):
        assert plan["impact"][header]["kind"] == "all"
        assert plan["impact"][header]["reason"] == "header-in-global-includer-closure"
    # The narrow files still carry their real per-file impact in the plan (auditable).
    assert plan["impact"]["python/bindings/system/base/system_fields.cpp"]["kind"] == (
        "binding-tu-targets"
    )


def test_unmapped_path_fails_safe_to_all(tmp_path):
    """A single unmapped path (a script) forces ALL, with a per-file ``unmapped-path`` reason."""
    outputs, _targets, plan = _run_plan_cpp_explain(
        tmp_path, ["scripts/some_helper.py"]
    )
    assert outputs["cpp_mode"] == "all"
    assert plan["impact"]["scripts/some_helper.py"]["reason"] == "unmapped-path"


def test_seam_template_is_a_build_input_selecting_all(tmp_path):
    """A ``python/bindings`` seam ``.cpp.in`` template is a build input -> FULL."""
    outputs, _targets, plan = _run_plan_cpp_explain(
        tmp_path, ["python/bindings/templates/system_flux_seam.cpp.in"]
    )
    assert outputs["cpp_mode"] == "all"
    assert plan["impact"]["python/bindings/templates/system_flux_seam.cpp.in"]["reason"] == (
        "binding-build-input"
    )


def test_explain_plan_has_per_file_impact_for_every_changed_file(tmp_path):
    """The explain plan maps EVERY changed file to an impact kind (auditability)."""
    changed = [
        "include/pops/numerics/time/schemes/splitting.hpp",
        "python/bindings/system/base/system_fields.cpp",
        "python/pops/codegen/program_emit_ops.py",
        "docs/x.md",
    ]
    _outputs, _targets, plan = _run_plan_cpp_explain(tmp_path, changed)
    assert set(plan["impact"]) == set(changed)
    for entry in plan["impact"].values():
        assert entry["kind"] in {
            "include-impact",
            "binding-tu-targets",
            "binding-labels",
            "codegen-labels",
            "test-target",
            "none",
            "all",
        }


def test_binding_object_lib_map_is_parsed_from_tests_cmake():
    """The binding OBJECT-lib source/consumer map is discoverable from ``tests/CMakeLists.txt``."""
    sources, consumers = sel._binding_object_lib_map()
    assert "python/bindings/system/base/system_fields.cpp" in sources["pops_runtime_system"]
    assert "python/bindings/amr/amr_system.cpp" in sources["pops_runtime_amr"]
    assert consumers["pops_runtime_system"], "no system consumers parsed"
    assert consumers["pops_runtime_amr"], "no amr consumers parsed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
