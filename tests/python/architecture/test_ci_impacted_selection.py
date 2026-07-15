"""ADC-621: the CI import-closure test selector picks the right impacted tests.

These are SOURCE-ONLY tests (no ``pops`` / ``_pops`` import): they exercise the
static import-graph selector in ``scripts/ci_import_closure.py`` and its wiring into
``scripts/ci_select_tests.py`` against the REAL source tree, so a wrong closure that
would silently drop a test's coverage fails the gate here.

Ground-truth edges asserted below were verified by reading the source:

* ``python/pops/runtime/_bind_adapters.py`` is imported directly by the final typed bind gate;
* ``python/pops/numerics/riemann/waves.py`` is imported by
  ``test_wave_speed_providers.py``;
* the codegen cross-test helper: ``test_dsl_cse.py`` imports sibling ``test_dsl_brick.py``;
* the LAZY (function-scope) edge ``pops.codegen._phases`` ->
  ``pops.runtime._bind_adapters`` must be captured, so a ``_bind_adapters`` change
  reaches the orchestration-dependent tests.
"""
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cic = _load("ci_import_closure")
route_mode = _load("ci_route_mode")


# --------------------------------------------------------------------------- #
# Graph builder                                                                #
# --------------------------------------------------------------------------- #
def test_module_graph_has_expected_shape():
    graph = cic.build_module_graph(REPO_ROOT)
    # Every pops sub-package we sampled is a node.
    for name in ("pops", "pops.runtime._runtime_instance", "pops.numerics.riemann.waves"):
        assert name in graph, f"{name} missing from the module graph"
    # The graph is non-trivial (hundreds of real edges over the package).
    assert sum(len(v) for v in graph.values()) > 500


def test_relative_import_edge_is_resolved():
    """A relative ``from ._facade_compile import ...`` must become a real edge.

    Regression guard: the relative-import anchor for a plain module drops the module's
    own leaf before applying the level, so ``pops.physics._facade`` importing
    ``._facade_compile`` resolves to ``pops.physics._facade_compile`` (not
    ``pops.physics._facade._facade_compile``).
    """
    graph = cic.build_module_graph(REPO_ROOT)
    assert "pops.physics._facade_compile" in graph.get("pops.physics._facade", set())


def test_lazy_function_scope_edge_is_captured():
    """The function-scope canonical phase module -> runtime adapter edge.

    A module-scope-only walker would miss it (orchestration imports ``_bind_adapters``
    only inside functions to keep the import-graph architecture gate green).
    """
    graph = cic.build_module_graph(REPO_ROOT)
    assert "pops.runtime._bind_adapters" in graph.get("pops.codegen._phases", set())


# --------------------------------------------------------------------------- #
# Impact query -- ground-truth module -> tests                                 #
# --------------------------------------------------------------------------- #
def test_runtime_instance_change_selects_final_runtime_gate():
    sel = cic.impacted_tests(
        ["python/pops/runtime/_runtime_instance.py"], repo_root=REPO_ROOT)
    assert "tests/python/unit/runtime/test_runtime_instance_gate.py" in sel


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
    assert "tests/python/unit/runtime/test_typed_bind_phase.py" in sel


# --------------------------------------------------------------------------- #
# Cross-test edge closure (both directions)                                    #
# --------------------------------------------------------------------------- #
def test_cross_test_forward_pulls_shared_brick_helper():
    """Selecting the CSE test pulls the sibling brick helper it imports."""
    _, edges = cic.test_imports(REPO_ROOT)
    importer = "tests/python/unit/codegen/test_dsl_cse.py"
    helper = "tests/python/unit/codegen/test_dsl_brick.py"
    assert edges.get(importer) == {helper}
    selected = {importer}
    cic._close_cross_test(selected, edges)
    assert helper in selected


def test_cross_test_reverse_pulls_dependents_of_shared_helper():
    """Selecting the brick helper pulls its remaining sibling dependent."""
    _, edges = cic.test_imports(REPO_ROOT)
    selected = {"tests/python/unit/codegen/test_dsl_brick.py"}
    cic._close_cross_test(selected, edges)
    assert "tests/python/unit/codegen/test_dsl_cse.py" in selected


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
        "tests/python/integration/bindings/test_m1_scalar_advection_pipeline.py",
        "tests/python/unit/runtime/test_capabilities.py",
    ):
        assert smoke in selected


def test_plan_python_direct_test_edit_pulls_cross_test_family(tmp_path):
    """A direct edit of the CSE test pulls its sibling helper + smoke."""
    importer = "tests/python/unit/codegen/test_dsl_cse.py"
    outputs, selected = _run_plan_python(
        tmp_path, [importer]
    )
    assert outputs["python_mode"] == "subset"
    assert "direct-test" in outputs["python_why"]
    assert importer in selected
    assert "tests/python/unit/codegen/test_dsl_brick.py" in selected


def test_plan_python_nested_suite_test_is_in_the_selection_universe(tmp_path):
    """A manifest suite owns nested test directories, not only its immediate children."""
    nested = "tests/python/unit/mesh/amr/test_hierarchy_contract.py"
    outputs, selected = _run_plan_python(tmp_path, [nested])
    assert outputs["python_mode"] == "subset"
    assert "direct-test" in outputs["python_why"]
    assert nested in selected


def test_plan_python_broad_file_runs_all(tmp_path):
    """A broad Python file (the package ``__init__``) forces the whole suite."""
    outputs, _ = _run_plan_python(tmp_path, ["python/pops/__init__.py"])
    assert outputs["python_mode"] == "all"
    assert "broad-file" in outputs["python_why"]


def test_plan_python_bindings_change_runs_all(tmp_path):
    """A ``python/bindings`` change (extension-affecting) forces the whole suite."""
    outputs, _ = _run_plan_python(tmp_path, ["python/bindings/core/init/init_system.cpp"])
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


def test_manifest_cpp_suites_exclude_mpi_only_targets():
    """The serial C++ gate must never select an MPI-only suite.

    MPI suites build only in the ci-mpi job; if one reached the serial selection the
    build step would hit ``ninja: unknown target`` (seen live on
    test_amr_regrid_mpi_parity, whose ``mpi`` segment is an INFIX the old ``test_mpi_``
    prefix filter missed -- #435). Convention: every MPI-only suite carries an ``mpi``
    name segment (and an ``mpi`` label / ``mpi_nproc``); the manifest-driven selector
    must drop it. This asserts the same intent against the manifest API that replaced
    the CMake target scraper.
    """
    sel = _load("ci_select_tests")
    suites = sel.manifest_cpp_suites(sel.load_manifest())
    names = [s["name"] for s in suites]
    assert names, "no serial C++ suites resolved from tests/test_manifest.toml"
    offenders = [n for n in names if "mpi" in n.split("_")]
    assert not offenders, f"MPI-only suites leaked into the serial selection: {offenders}"
    assert "test_splitting" in names


def test_manifest_projects_exact_mpi_targets_for_dedicated_job():
    sel = _load("ci_select_tests")
    manifest = sel.load_manifest()
    targets = sel.cpp_targets_with_label(manifest, "mpi")
    all_suites = sel.manifest_cpp_suites(manifest, include_mpi=True)
    expected = sorted(
        suite["name"]
        for suite in all_suites
        if "mpi" in suite["labels"] or suite["mpi_variants"]
    )
    assert targets == expected
    variant_targets = {
        suite["name"]: suite["mpi_variants"]
        for suite in all_suites
        if suite["mpi_variants"]
    }
    assert variant_targets == {
        "test_amr_system_bz_multibox": (2, 4),
        "test_copy_schedule_cache": (1, 2, 4),
        "test_fill_boundary_cache": (1, 2, 4),
        "test_krylov_solver": (1, 2, 4),
    }
    serial_targets = {
        suite["name"] for suite in sel.manifest_cpp_suites(manifest)
    }
    assert set(variant_targets) <= serial_targets
    assert set(targets) - set(variant_targets) == {
        suite["name"] for suite in all_suites if "mpi" in suite["labels"]
    }
    assert sel.cpp_mpi_ctest_count(manifest) == sum(
        len(suite["mpi_nproc"]) + len(suite["mpi_variants"])
        for suite in all_suites
    ) == 61


def test_cpp_target_label_fence_requires_each_selected_target(tmp_path):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "ctest.json"
    inventory.write_text(json.dumps({
        "tests": [
            {
                "name": "Suite.One",
                "properties": [
                    {"name": "LABELS", "value": ["unit", "cpp-target:test_one"]},
                ],
            },
            {
                "name": "Suite.Two",
                "properties": [
                    {"name": "LABELS", "value": ["fast", "cpp-target:test_two"]},
                ],
            },
        ],
    }))

    args = SimpleNamespace(
        ctest_json=str(inventory),
        targets=["test_one", "test_two"],
    )
    assert sel.verify_cpp_target_labels(args) == 0

    args.targets.append("test_missing")
    with pytest.raises(SystemExit, match="test_missing"):
        sel.verify_cpp_target_labels(args)


def test_manifest_projects_exact_python_mpi_entrypoints():
    """Script-style MPI contracts carry their launcher ranks in the manifest, not CI YAML."""
    sel = _load("ci_select_tests")
    assert sel.manifest_python_mpi_entrypoints(sel.load_manifest()) == [
        {
            "suite": "pops_python_integration_mpi",
            "path": "tests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
            "nproc": 2,
        },
        {
            "suite": "pops_python_integration_mpi",
            "path": "tests/python/integration/mpi/test_amr_history_mpi.py",
            "nproc": 2,
        },
    ]


def test_python_mpi_plan_is_ranked_and_manifest_owned(tmp_path):
    sel = _load("ci_select_tests")

    class Args:
        plan_file = str(tmp_path / "python-mpi-plan.tsv")
        github_output = str(tmp_path / "github-output.txt")
        explain_file = str(tmp_path / "python-mpi-plan.json")

    assert sel.plan_python_mpi(Args()) == 0
    assert (tmp_path / "python-mpi-plan.tsv").read_text().splitlines() == [
        "2\ttests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
        "2\ttests/python/integration/mpi/test_amr_history_mpi.py",
    ]
    outputs = dict(
        line.partition("=")[::2]
        for line in (tmp_path / "github-output.txt").read_text().splitlines()
    )
    assert outputs["python_mpi_count"] == "2"


@pytest.mark.parametrize(
    ("event", "inputs", "expected"),
    (
        ("pull_request", {"cpp_paths": True},
         (False, True, True, True, False, False)),
        ("pull_request", {"python_paths": True},
         (False, False, True, True, False, False)),
        ("pull_request", {"architecture_paths": True},
         (False, False, False, True, False, False)),
        ("pull_request", {},
         (False, False, False, False, False, False)),
        ("pull_request", {"ci_kokkos": True},
         (False, True, True, True, False, False)),
        ("pull_request", {"ci_full": True},
         (True, True, True, True, True, True)),
        ("pull_request", {"force_full": True},
         (True, True, True, True, True, True)),
        ("push", {},
         (False, True, True, True, False, False)),
        ("push", {"full_paths": True},
         (True, True, True, True, True, True)),
        ("workflow_call", {},
         (True, True, True, True, True, True)),
    ),
)
def test_ci_route_authority_covers_pr_labels_push_and_full(event, inputs, expected):
    decision = route_mode.decide_routes(event_name=event, **inputs)
    assert (
        decision.full,
        decision.cpp_required,
        decision.python_required,
        decision.architecture_required,
        decision.mpi_required,
        decision.openmp_required,
    ) == expected


def test_ci_gate_verdict_is_fail_closed_but_allows_an_unrouted_skip():
    route_mode.validate_gate_result("required", "success", True)
    route_mode.validate_gate_result("optional-success", "success", False)
    route_mode.validate_gate_result("optional-skip", "skipped", False)
    with pytest.raises(route_mode.RouteModeError, match="required"):
        route_mode.validate_gate_result("required", "skipped", True)
    with pytest.raises(route_mode.RouteModeError, match="optional"):
        route_mode.validate_gate_result("optional", "failure", False)
    with pytest.raises(route_mode.RouteModeError, match="true or false"):
        route_mode.validate_gate_result("missing-route", "skipped", "")


def test_ci_required_gate_aggregates_full_matrix_and_mpi_path_changes():
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "mpi: ${{ steps.filter.outputs.mpi }}" in workflow
    for mpi_input in (
        "include/pops/parallel/**",
        "include/pops/runtime/**",
        "src/runtime/**",
        "python/bindings/core/init/init_amr.cpp",
        "python/bindings/core/init/init_core.cpp",
        "python/bindings/core/init/init_system.cpp",
        "python/pops/_platform_contracts.py",
        "python/pops/codegen/_compiled_artifact.py",
        "python/pops/runtime/_component_execution_context.py",
        "python/pops/runtime/_platform_manifest.py",
        "python/pops/runtime/_runtime_authorities.py",
        "python/pops/runtime/_runtime_executor.py",
        "python/pops/runtime/_system_unified_install.py",
        "scripts/ci_select_tests.py",
        "tests/**/mpi/**",
    ):
        assert mpi_input in workflow
    for selector_input in (
        "schemas/**",
        "tests/cpp/test_durations.json",
        "tests/test_manifest.toml",
        "scripts/generate_component_catalog.py",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        ".github/workflows/ci.yml",
    ):
        assert selector_input in workflow

    filter_text = workflow.split("\n          filters: |\n", 1)[1].split(
        "\n  # Autorite unique de routage", 1)[0]
    cpp_filter = filter_text.split("            cpp:\n", 1)[1].split(
        "\n            python:\n", 1)[0]
    python_filter = filter_text.split("            python:\n", 1)[1].split(
        "\n            python_arch:\n", 1)[0]
    for cpp_control in (
        "tests/cpp/test_sources.cmake",
        "tests/cpp/test_durations.json",
        "tests/test_manifest.toml",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        "scripts/ci_include_graph.py",
        "scripts/ci_route_mode.py",
        ".github/actions/setup-kokkos/**",
        ".github/workflows/ci.yml",
    ):
        assert cpp_control in cpp_filter
    for python_control in (
        "tests/python/test_durations.json",
        "tests/test_manifest.toml",
        "pyproject.toml",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        "scripts/ci_import_closure.py",
        "scripts/ci_route_mode.py",
        ".github/actions/setup-kokkos/**",
        ".github/workflows/ci.yml",
    ):
        assert python_control in python_filter

    cpp_shards_block = workflow.split("\n  gate-cpp-shards:\n", 1)[1].split(
        "\n  # Check historique", 1)[0]
    assert "ctest --preset ci-kokkos -N --show-only=json-v1" in cpp_shards_block
    assert "scripts/ci_select_tests.py verify-cpp-target-labels" in cpp_shards_block
    assert '--targets "${cpp_targets[@]}"' in cpp_shards_block
    assert cpp_shards_block.index("verify-cpp-target-labels") < cpp_shards_block.index(
        "ctest --preset ci-kokkos -L"
    )

    gate_block = workflow.split("\n  gate:\n", 1)[1].split("\n  mpi:\n", 1)[0]
    assert "needs: [changes, set-mode," in gate_block
    assert "mpi, kokkos-openmp" in gate_block
    assert "needs.set-mode.result" in gate_block
    assert "scripts/ci_route_mode.py check" in gate_block
    assert "needs.set-mode.outputs.architecture_required" in gate_block
    assert "needs.set-mode.outputs.python_required" in gate_block
    assert "needs.set-mode.outputs.mpi_required" in gate_block
    assert "needs.set-mode.outputs.openmp_required" in gate_block
    assert "needs.mpi.result" in gate_block
    assert "needs.kokkos-openmp.result" in gate_block

    mpi_block = workflow.split("\n  mpi:\n", 1)[1].split("\n  kokkos-openmp:\n", 1)[0]
    assert "needs: [set-mode, changes]" in mpi_block
    assert "if: needs.set-mode.outputs.mpi_required == 'true'" in mpi_block
    assert "-DPOPS_BUILD_PYTHON=ON" in mpi_block
    assert "scripts/ci_select_tests.py cpp-label" in mpi_block
    assert "--label mpi" in mpi_block
    assert "scripts/ci_select_tests.py python-mpi" in mpi_block
    assert "build-mpi/python-mpi-plan.tsv" in mpi_block
    assert '--target _pops "${mpi_targets[@]}"' in mpi_block
    assert "steps.mpi-test-plan.outputs.cpp_label_ctest_count" in mpi_block
    assert '"${mpi_n:-0}" != "${mpi_expected:-0}"' in mpi_block
    assert mpi_block.count("-L '^mpi$' -LE '^python$'") == 2
    assert "POPS_REQUIRE_MPI_TESTS: \"1\"" in mpi_block
    assert "mpiexec -n \"$mpi_ranks\" /usr/bin/python3 \"$mpi_test\"" in mpi_block
    assert "test_amr_clean_route_program_mpi.py" not in mpi_block
    assert "test_amr_history_mpi.py" not in mpi_block
    assert "cmake --build --preset ci-mpi\n" not in mpi_block
    assert "build-mpi/python-package" in mpi_block
    assert "collective HDF5 lifecycle requires an MPI-enabled _pops" in mpi_block
    assert "This writer is pure Python" not in mpi_block

    set_mode_block = workflow.split("\n  set-mode:\n", 1)[1].split(
        "\n  # GATE C++", 1)[0]
    for output in (
        "cpp_required", "python_required", "architecture_required",
        "mpi_required", "openmp_required",
    ):
        assert f"{output}: ${{{{ steps.decide.outputs.{output} }}}}" in set_mode_block
    assert "python3 scripts/ci_route_mode.py decide" in set_mode_block

    cpp_verdict = workflow.split("\n  gate-cpp:\n", 1)[1].split(
        "\n  # GATE PYTHON ARCHITECTURE", 1)[0]
    assert "scripts/ci_route_mode.py check" in cpp_verdict
    assert "needs.set-mode.outputs.cpp_required" in cpp_verdict
    assert "success|skipped" not in cpp_verdict

    architecture_block = workflow.split(
        "\n  gate-python-architecture:\n", 1)[1].split(
            "\n  gate-python-build:\n", 1)[0]
    assert "if: needs.set-mode.outputs.architecture_required == 'true'" in architecture_block
    assert "python3 scripts/generate_component_catalog.py --check" in architecture_block

    python_build_block = workflow.split(
        "\n  gate-python-build:\n", 1)[1].split("\n  gate-python:\n", 1)[0]
    python_shards_block = workflow.split(
        "\n  gate-python:\n", 1)[1].split("\n  gate-python-compile-cache:\n", 1)[0]
    python_cache_block = workflow.split(
        "\n  gate-python-compile-cache:\n", 1)[1].split("\n  gate:\n", 1)[0]
    assert "timeout-minutes: 20" in python_build_block
    assert "timeout-minutes: 19" in python_build_block
    assert "timeout --signal=TERM --kill-after=30s 18m" in python_build_block
    assert "exceeded its 18-minute cache-safe watchdog" in python_build_block
    assert "exit \"$build_status\"" in python_build_block
    assert "-DPOPS_HEAVY_MODULE_TU_POOL=4" in python_build_block
    assert "uses: actions/cache/restore@v6" in python_build_block
    assert "uses: actions/cache/save@v6" in python_build_block
    assert "always() && steps.kokkos.outcome == 'success'" in python_build_block
    assert "github.run_attempt" in python_build_block
    assert "timeout-minutes: 30" in python_shards_block
    assert "timeout-minutes: 30" in python_cache_block
    for block in (python_build_block, python_shards_block, python_cache_block):
        assert "if: needs.set-mode.outputs.python_required == 'true'" in block

    # GitHub rejects `runner.*` in a job-level `env` mapping before creating any job.  Keep the
    # runner-specific cache prefix at step scope and the compile-cache temporary workspace-owned.
    assert "CCACHE_CACHE_KEY: ccache-${{ runner.os }}" not in workflow
    assert "COMPILE_CACHE_TMP: ${{ runner.temp }}" not in workflow
    assert "key: ccache-${{ runner.os }}-${{ env.CCACHE_CACHE_KEY }}" in workflow
    assert "COMPILE_CACHE_TMP: ${{ github.workspace }}/.pops-ci/compile-cache-test" in workflow
    assert 'mkdir -p "$COMPILE_CACHE_TMP"' in python_cache_block


def test_ci_control_plane_inputs_force_full_functional_selection():
    selector = _load("ci_select_tests")
    for path in (
        ".github/workflows/ci.yml",
        "scripts/ci_route_mode.py",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        "tests/test_manifest.toml",
    ):
        assert path in selector.CPP_BROAD_FILES
        assert path in selector.PYTHON_BROAD_FILES
    assert "tests/cpp/test_sources.cmake" in selector.CPP_BROAD_FILES
    assert "tests/cpp/test_durations.json" in selector.CPP_BROAD_FILES
    assert "scripts/ci_include_graph.py" in selector.CPP_BROAD_FILES
    assert "tests/python/test_durations.json" in selector.PYTHON_BROAD_FILES
    assert "scripts/ci_import_closure.py" in selector.PYTHON_BROAD_FILES
