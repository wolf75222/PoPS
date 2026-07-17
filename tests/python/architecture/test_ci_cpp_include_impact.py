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
import re
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
# Duration-balanced C++ matrix partition                                      #
# --------------------------------------------------------------------------- #
def _run_plan_cpp_shard(tmp_path, changed_lines, shard_index, shard_total=6):
    changed = tmp_path / f"changed-{shard_index}.txt"
    changed.write_text("".join(f"{c}\n" for c in changed_lines), encoding="utf-8")
    out = tmp_path / f"gh-out-{shard_index}.txt"

    class Args:
        pass

    args = Args()
    args.changed_files = str(changed)
    args.github_output = str(out)
    args.explain_file = None
    args.force_all = False
    args.shard_index = shard_index
    args.shard_total = shard_total
    sel.plan_cpp(args)
    outputs = {}
    for line in out.read_text().splitlines():
        key, _, value = line.partition("=")
        outputs[key] = value
    return outputs


def test_cpp_target_shards_are_deterministic_duration_balanced_exact_cover():
    targets = sorted(suite["name"] for suite in _serial_suites())
    measured_build = sel.ci_shard_binpack.load_durations(sel.CPP_BUILD_DURATIONS_JSON)
    measured_test = sel.ci_shard_binpack.load_durations(sel.CPP_DURATIONS_JSON)
    assert set(measured_build) == set(targets), (
        "C++ build-duration catalog must exactly mirror the manifest"
    )
    assert set(measured_test) == set(targets), (
        "C++ test-duration catalog must exactly mirror the manifest"
    )
    weights = sel.cpp_target_weights(targets)

    first = sel.cpp_target_shards(targets, 4)
    assert sel.cpp_target_shards(list(reversed(targets)), 4) == first
    flat = [target for shard in first for target in shard]
    assert set(flat) == set(targets)
    assert len(flat) == len(set(flat)), "a C++ target was duplicated across shards"

    loads = [sum(weights[target] for target in shard) for shard in first]
    lower_bound = max(max(weights.values()), sum(weights.values()) / len(first))
    lpt_bound = (4.0 / 3.0 - 1.0 / (3.0 * len(first))) * lower_bound
    assert max(loads) <= lpt_bound + 1.0e-9


def test_cpp_cold_build_catalog_separates_five_minute_template_targets():
    build = sel.ci_shard_binpack.load_durations(sel.CPP_BUILD_DURATIONS_JSON)
    very_heavy = sorted(target for target, seconds in build.items() if seconds >= 240.0)
    assert len(very_heavy) >= 7, "cold-CI catalog lost the known five-minute AMR TUs"

    shards = sel.cpp_target_shards(very_heavy, 6)
    sel.ci_shard_binpack.verify_partition(very_heavy, shards, excluded=())
    assert max(len(shard) for shard in shards) == 2
    assert sum(len(shard) == 2 for shard in shards) == len(very_heavy) - 6

    # The unavoidable seventh five-minute TU shares one shard, but LPT must leave enough room
    # around that pair: <= 650 s of target compilation plus the observed ~6 min shared runtime
    # build stays comfortably inside the workflow's 30 min watchdog.
    full_shards = sel.cpp_target_shards(sorted(build), 6)
    build_loads = [sum(build[target] for target in shard) for shard in full_shards]
    assert max(build_loads) <= 650.0


def test_cpp_ctest_selection_uses_target_labels_not_gtest_suite_names():
    cmake = (REPO_ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    assert 'cpp-target:${ARG_NAME}' in cmake
    regex = sel.cpp_target_label_regex(["test_brick_catalog", "test_program_runtime"])
    assert re.search(regex, "cpp-target:test_brick_catalog")
    assert re.search(regex, "cpp-target:test_program_runtime")
    assert not re.search(regex, "BrickCatalog.EntryRoundTripsAllElevenRows")


def test_cpp_ctest_registration_avoids_runtime_discovery_file_fanout():
    """Ordinary suites stay source-registered; runtime discovery is explicit and rare."""
    cmake = (REPO_ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    assert re.search(
        r"gtest_add_tests\(\s*TARGET \$\{ARG_NAME\}\s+"
        r"SOURCES \$\{ARG_SOURCES\}\s+TEST_LIST _pops_discovered_tests\)",
        cmake,
    )
    assert "DISCOVERY_MODE PRE_TEST" not in cmake

    # No current registration opts into the one-include-per-executable escape
    # hatch.  A future parameterized/generated suite must make that cost and
    # contract explicit instead of silently restoring the CTest file fanout.
    registrations = cmake.split("function(pops_add_test name)", maxsplit=1)[1]
    assert "RUNTIME_DISCOVERY" not in registrations

    runtime_only = re.compile(
        r"\b(?:TEST_P|TYPED_TEST|TYPED_TEST_P|INSTANTIATE_TEST_SUITE_P)\s*\("
    )
    test_declaration = re.compile(r"\b(?:TEST|TEST_F)\s*\(")
    conditional_start = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b")
    conditional_end = re.compile(r"^\s*#\s*endif\b")
    offenders = []
    conditional_offenders = []
    for source in (REPO_ROOT / "tests/cpp").rglob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        if runtime_only.search(text):
            offenders.append(source.relative_to(REPO_ROOT).as_posix())
        conditional_depth = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if conditional_start.match(line):
                conditional_depth += 1
            elif conditional_end.match(line):
                conditional_depth -= 1
            elif conditional_depth and test_declaration.search(line):
                conditional_offenders.append(
                    f"{source.relative_to(REPO_ROOT).as_posix()}:{line_number}"
                )
    assert not offenders, (
        "parameterized GoogleTests require an explicit RUNTIME_DISCOVERY suite: "
        + ", ".join(offenders)
    )
    assert not conditional_offenders, (
        "conditionally compiled GoogleTests require explicit RUNTIME_DISCOVERY: "
        + ", ".join(conditional_offenders)
    )


def test_full_cpp_plan_six_shards_preserves_every_cpp_target(tmp_path):
    outputs = [
        _run_plan_cpp_shard(tmp_path, ["CMakeLists.txt"], shard_index)
        for shard_index in range(6)
    ]
    selected = set(outputs[0]["cpp_targets"].split())
    sharded = [output["cpp_shard_targets"].split() for output in outputs]
    flat = [target for shard in sharded for target in shard]

    assert outputs[0]["cpp_mode"] == "all"
    assert int(outputs[0]["cpp_count"]) == int(outputs[0]["cpp_total"])
    assert set(flat) == selected
    assert len(flat) == len(set(flat)), "matrix duplicates a C++ target"
    assert all(
        output["cpp_shard_counts"] == outputs[0]["cpp_shard_counts"]
        for output in outputs
    )

    assert not any(
        re.search(output["cpp_shard_label_regex"], "test_component_catalog_generated")
        for output in outputs
    ), "the generated catalog is a pure-Python architecture test, not a C++ shard"


def test_subset_cpp_plan_six_shards_preserves_selected_union(tmp_path):
    changed = ["include/pops/numerics/time/schemes/splitting.hpp"]
    outputs = [
        _run_plan_cpp_shard(tmp_path, changed, shard_index)
        for shard_index in range(6)
    ]
    selected = set(outputs[0]["cpp_targets"].split())
    flat = [
        target
        for output in outputs
        for target in output["cpp_shard_targets"].split()
    ]

    assert outputs[0]["cpp_mode"] == "subset"
    assert set(flat) == selected
    assert len(flat) == len(set(flat)), "subset matrix duplicates a C++ target"
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


def test_runtime_tu_selects_only_its_object_lib_consumers(tmp_path):
    """A runtime TU maps to the test targets compiling it, NOT the broad label group.

    ``system_fields.cpp`` is one of the ``pops_runtime_system`` OBJECT-lib TUs; the selection is
    exactly that lib's serial consumers (+ smoke), a strict subset far below the old bindings
    label group. The precise consumer set is read from ``tests/CMakeLists.txt``.
    """
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path, ["src/runtime/system/system_fields.cpp"]
    )
    assert outputs["cpp_mode"] == "subset"
    _sources, consumers = sel._runtime_object_lib_map()
    serial = {s["name"] for s in _serial_suites()}
    expected = {t for t in consumers["pops_runtime_system"] if t in serial}
    assert expected, "no serial consumers parsed for pops_runtime_system"
    assert expected <= set(targets)
    for smoke in sel.CPP_SMOKE_TARGETS:
        assert smoke in targets
    entry = plan["impact"]["src/runtime/system/system_fields.cpp"]
    assert entry["kind"] == "runtime-tu-targets"
    assert entry["object_libs"] == ["pops_runtime_system"]


def test_runtime_private_header_maps_to_the_same_object_lib(tmp_path):
    """A runtime-private header impacts the OBJECT lib whose TUs ``#include`` it.

    ``system_impl.hpp`` is included by the ``pops_runtime_system`` TUs, so a change to it selects
    the same consumers as a change to one of those TUs.
    """
    outputs, targets, plan = _run_plan_cpp_explain(
        tmp_path, ["src/runtime/system/system_impl.hpp"]
    )
    assert outputs["cpp_mode"] == "subset"
    entry = plan["impact"]["src/runtime/system/system_impl.hpp"]
    assert entry["kind"] == "runtime-tu-targets"
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
            "python/pops/time/_program/api.py",
            "docs/whatever.md",
            "CHANGELOG.md",
            "tests/python/unit/time/test_time_condensed_schur.py",
        ],
    )
    assert outputs["cpp_mode"] == "none"
    assert targets == []
    assert all(entry["kind"] == "none" for entry in plan["impact"].values())


def test_compositional_union_prunes_a_mixed_change(tmp_path):
    """A leaf header + runtime TU + codegen + zero-impact files select their UNION, not ALL.

    The core ADC-646 win: none of these files is a global includer or an unmapped build input, so
    the change prunes to the union of the leaf-header closure, the runtime-TU consumers and the
    codegen group -- a strict subset -- instead of collapsing to coarse labels or ALL.
    """
    changed = [
        "CHANGELOG.md",
        "include/pops/numerics/time/schemes/splitting.hpp",
        "src/runtime/system/system_fields.cpp",
        "python/pops/codegen/program_emit_control.py",
        "python/pops/time/_program/api.py",
        "tests/python/unit/time/test_time_condensed_schur.py",
    ]
    outputs, targets, plan = _run_plan_cpp_explain(tmp_path, changed)
    assert outputs["cpp_mode"] == "subset"
    assert 0 < int(outputs["cpp_count"]) < int(outputs["cpp_total"])
    # The union must contain the leaf-header suite, a binding consumer and a codegen suite.
    assert "test_splitting" in targets
    kinds = {f: v["kind"] for f, v in plan["impact"].items()}
    assert kinds["include/pops/numerics/time/schemes/splitting.hpp"] == "include-impact"
    assert kinds["src/runtime/system/system_fields.cpp"] == "runtime-tu-targets"
    assert kinds["python/pops/codegen/program_emit_control.py"] == "codegen-labels"
    assert kinds["python/pops/time/_program/api.py"] == "none"


def test_global_header_in_a_mixed_change_still_forces_all(tmp_path):
    """A global-includer header anywhere in the change escalates the union to FULL (soundness).

    This is the literal ADC-427 shape: ``system.hpp`` / the program-context headers are global
    includers (compiled into every target via the runtime TUs and the emitter), so the sound
    selection is ALL -- the plan spells out the per-file reason for each.
    """
    changed = [
        "CHANGELOG.md",
        "include/pops/runtime/program/amr_program_context.hpp",
        "include/pops/runtime/program/program_context.hpp",
        "include/pops/runtime/system.hpp",
        "src/runtime/system/system_fields.cpp",
        "python/pops/codegen/program_emit_control.py",
        "python/pops/time/_program/api.py",
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
    assert plan["impact"]["src/runtime/system/system_fields.cpp"]["kind"] == (
        "runtime-tu-targets"
    )


def test_unmapped_path_fails_safe_to_all(tmp_path):
    """A single unmapped path (a script) forces ALL, with a per-file ``unmapped-path`` reason."""
    outputs, _targets, plan = _run_plan_cpp_explain(
        tmp_path, ["scripts/some_helper.py"]
    )
    assert outputs["cpp_mode"] == "all"
    assert plan["impact"]["scripts/some_helper.py"]["reason"] == "unmapped-path"


def test_seam_template_is_a_build_input_selecting_all(tmp_path):
    """A runtime-builder seam ``.cpp.in`` template is a build input -> FULL."""
    outputs, _targets, plan = _run_plan_cpp_explain(
        tmp_path, ["src/runtime/builders/templates/system_flux_seam.cpp.in"]
    )
    assert outputs["cpp_mode"] == "all"
    assert plan["impact"]["src/runtime/builders/templates/system_flux_seam.cpp.in"]["reason"] == (
        "runtime-build-input"
    )


def test_explain_plan_has_per_file_impact_for_every_changed_file(tmp_path):
    """The explain plan maps EVERY changed file to an impact kind (auditability)."""
    changed = [
        "include/pops/numerics/time/schemes/splitting.hpp",
        "src/runtime/system/system_fields.cpp",
        "python/pops/codegen/program_emit_ops.py",
        "docs/x.md",
    ]
    _outputs, _targets, plan = _run_plan_cpp_explain(tmp_path, changed)
    assert set(plan["impact"]) == set(changed)
    for entry in plan["impact"].values():
        assert entry["kind"] in {
            "include-impact",
            "runtime-tu-targets",
            "binding-labels",
            "codegen-labels",
            "test-target",
            "none",
            "all",
        }


def test_runtime_object_lib_map_uses_central_sources_and_test_consumers():
    """Sources come from ``src/CMakeLists.txt`` and consumers from ``tests/CMakeLists.txt``."""
    sources, consumers = sel._runtime_object_lib_map()
    assert "src/runtime/system/system_fields.cpp" in sources["pops_runtime_system"]
    assert "src/runtime/amr/amr_system.cpp" in sources["pops_runtime_amr"]
    assert consumers["pops_runtime_system"], "no system consumers parsed"
    assert consumers["pops_runtime_amr"], "no amr consumers parsed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
