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


@pytest.mark.parametrize(
    "header",
    [
        "include/pops/numerics/elliptic/linear/krylov_method_provider.hpp",
        "include/pops/core/identity/prepared_provider_options.hpp",
        "include/pops/mesh/layout/field_distribution.hpp",
        "include/pops/mesh/storage/field_replica_consensus.hpp",
    ],
)
def test_plan_python_native_elliptic_protocol_runs_all_external_provider_e2es(
    tmp_path, header
):
    outputs, selected = _run_plan_python(tmp_path, [header])
    assert outputs["python_mode"] == "subset"
    assert "elliptic-native-provider-contract" in outputs["python_why"]
    assert {
        "tests/python/integration/native_loader/test_prepared_krylov_method_component.py",
        "tests/python/integration/native_loader/test_prepared_nullspace_component.py",
        "tests/python/integration/native_loader/test_prepared_preconditioner_component.py",
    } <= set(selected)


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
    name segment (and an ``mpi`` label / exact MPI launch contract); the manifest-driven selector
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
        "test_geometric_mg": (2,),
        "test_generic_krylov": (1, 2, 4),
        "test_krylov_workspace_reentrancy": (2,),
        "test_pure_field_algebra_extreme_dot": (2,),
        "test_world_communicator": (1, 2),
    }
    serial_targets = {
        suite["name"] for suite in sel.manifest_cpp_suites(manifest)
    }
    assert set(variant_targets) <= serial_targets
    assert set(targets) - set(variant_targets) == {
        suite["name"] for suite in all_suites if "mpi" in suite["labels"]
    }
    expected_count = sum(
        len(suite["mpi_nproc"])
        + bool(suite["mpi_rank_parity"])
        + len(suite["mpi_variants"])
        for suite in all_suites
    )
    ctest_plan = sel.cpp_mpi_ctest_plan(manifest)
    assert len(ctest_plan) == sel.cpp_mpi_ctest_count(manifest) == expected_count == 72
    assert ctest_plan["test_mpi_external_lifecycle_np1"] == 1
    assert ctest_plan["test_mpi_hdf5_collective_np2"] == 2
    assert ctest_plan["test_mpi_amr_compiled_parity_rank_parity"] == 4
    assert ctest_plan["test_mpi_amr_distributed_coarse_rank_parity"] == 4
    assert ctest_plan["test_mpi_amr_program_reflux_np4"] == 4
    assert ctest_plan["test_mpi_system_analytic_level_set_np2"] == 2
    assert ctest_plan["test_world_communicator_np2"] == 2


def test_manifest_rejects_ambiguous_mpi_only_launch_contracts():
    sel = _load("ci_select_tests")
    suite = {
        "name": "test_ambiguous_mpi",
        "sources": ["tests/cpp/test_ambiguous_mpi.cpp"],
        "labels": ["backend", "mpi"],
        "mpi_nproc": [1],
        "mpi_rank_parity": [1, 2],
    }
    with pytest.raises(SystemExit, match="exactly one of mpi_nproc or mpi_rank_parity"):
        sel.manifest_cpp_suites({"cpp": {"suite": [suite]}}, include_mpi=True)


def test_manifest_rejects_rank_parity_without_an_mpi_label():
    sel = _load("ci_select_tests")
    suite = {
        "name": "test_unlabelled_parity",
        "sources": ["tests/cpp/test_unlabelled_parity.cpp"],
        "labels": ["backend"],
        "mpi_rank_parity": [1, 2],
    }
    with pytest.raises(SystemExit, match="exactly one of mpi_nproc or mpi_rank_parity"):
        sel.manifest_cpp_suites({"cpp": {"suite": [suite]}}, include_mpi=True)


def _write_mpi_ctest_inventory(path, plan):
    path.write_text(json.dumps({
        "tests": [
            {
                "name": name,
                "properties": [
                    {"name": "LABELS", "value": ["backend", "mpi", "medium"]},
                    {"name": "TIMEOUT", "value": 600.0},
                ] + (
                    [] if nproc == 1
                    else [{"name": "PROCESSORS", "value": nproc}]
                ),
            }
            for name, nproc in plan.items()
        ],
    }))


def test_cpp_mpi_ctest_fence_authenticates_exact_names_labels_and_ranks(tmp_path):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "mpi-ctest.json"
    plan = sel.cpp_mpi_ctest_plan(sel.load_manifest())
    _write_mpi_ctest_inventory(inventory, plan)

    args = SimpleNamespace(ctest_json=str(inventory))
    assert sel.verify_cpp_mpi_ctests(args) == 0


def test_cpp_mpi_ctest_fence_rejects_same_count_identity_drift(tmp_path):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "mpi-ctest-drift.json"
    plan = sel.cpp_mpi_ctest_plan(sel.load_manifest())
    plan.pop("test_mpi_hdf5_collective_np2")
    plan["test_mpi_hdf5_collective_np4"] = 4
    _write_mpi_ctest_inventory(inventory, plan)

    with pytest.raises(SystemExit, match="missing=.*np2.*unexpected=.*np4"):
        sel.verify_cpp_mpi_ctests(SimpleNamespace(ctest_json=str(inventory)))


def test_cpp_mpi_ctest_fence_rejects_processors_drift(tmp_path):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "mpi-ctest-rank-drift.json"
    plan = sel.cpp_mpi_ctest_plan(sel.load_manifest())
    plan["test_world_communicator_np2"] = 4
    _write_mpi_ctest_inventory(inventory, plan)

    with pytest.raises(SystemExit, match="PROCESSORS=.*manifest=2:ctest=4"):
        sel.verify_cpp_mpi_ctests(SimpleNamespace(ctest_json=str(inventory)))


def test_cpp_mpi_processor_groups_are_an_exact_disjoint_cover():
    sel = _load("ci_select_tests")
    manifest = sel.load_manifest()
    plan = sel.cpp_mpi_ctest_plan(manifest)
    groups = sel.cpp_mpi_ctest_groups(manifest)
    assert [group["processors"] for group in groups] == sorted(set(plan.values()))
    flattened = [name for group in groups for name in group["names"]]
    assert sorted(flattened) == sorted(plan)
    assert len(flattened) == len(set(flattened))
    for group in groups:
        assert all(plan[name] == group["processors"] for name in group["names"])


def test_cpp_mpi_ctest_fence_rejects_missing_per_test_timeout(tmp_path):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "mpi-ctest-unbounded.json"
    plan = sel.cpp_mpi_ctest_plan(sel.load_manifest())
    _write_mpi_ctest_inventory(inventory, plan)
    payload = json.loads(inventory.read_text())
    payload["tests"][0]["properties"] = [
        prop for prop in payload["tests"][0]["properties"]
        if prop["name"] != "TIMEOUT"
    ]
    inventory.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="must expose one numeric TIMEOUT"):
        sel.verify_cpp_mpi_ctests(SimpleNamespace(ctest_json=str(inventory)))


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
                    {"name": "LABELS", "value": "cpp-target:test_two"},
                ],
            },
            {
                "name": "Suite.Three",
                "properties": [
                    {
                        "name": "LABELS",
                        "value": [
                            "integration", "amr", "medium", "cpp-target:test_three"
                        ],
                    },
                ],
            },
        ],
    }))

    args = SimpleNamespace(
        ctest_json=str(inventory),
        targets=["test_one", "test_two", "test_three"],
    )
    assert sel.verify_cpp_target_labels(args) == 0

    for missing in ("test", "test_missing"):
        args.targets.append(missing)
        with pytest.raises(SystemExit, match=rf"cpp-target:{missing}"):
            sel.verify_cpp_target_labels(args)
        args.targets.pop()


@pytest.mark.parametrize("encoded_labels", [
    "integration;amr;medium;cpp-target:test_three",
    ["integration;amr;medium;cpp-target:test_three"],
])
def test_cpp_target_label_fence_rejects_overescaped_ctest_labels(
    tmp_path, encoded_labels,
):
    sel = _load("ci_select_tests")
    inventory = tmp_path / "ctest-overescaped.json"
    inventory.write_text(json.dumps({
        "tests": [
            {
                "name": "Suite.Three",
                "properties": [
                    {
                        "name": "LABELS",
                        "value": encoded_labels,
                    },
                ],
            },
        ],
    }))

    args = SimpleNamespace(
        ctest_json=str(inventory),
        targets=["test_three"],
    )
    with pytest.raises(SystemExit, match="non-atomic LABELS"):
        sel.verify_cpp_target_labels(args)


def test_manifest_projects_exact_python_mpi_entrypoints():
    """Script-style MPI contracts carry their launcher ranks in the manifest, not CI YAML."""
    sel = _load("ci_select_tests")
    assert sel.manifest_python_mpi_entrypoints(sel.load_manifest()) == [
        {
            "suite": "pops_python_integration_io",
            "path": "tests/python/integration/io/test_hdf5_parallel.py",
            "nproc": 2,
        },
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
        {
            "suite": "pops_python_integration_mpi",
            "path": "tests/python/integration/mpi/test_scientific_output_mpi.py",
            "nproc": 2,
        },
        {
            "suite": "pops_python_integration_mpi",
            "path": "tests/python/integration/mpi/test_uniform_history_checkpoint_mpi.py",
            "nproc": 2,
        },
    ]


def test_python_mpi_entrypoints_are_excluded_from_serial_pytest_suites():
    sel = _load("ci_select_tests")
    manifest = sel.load_manifest()
    mpi_paths = {row["path"] for row in sel.manifest_python_mpi_entrypoints(manifest)}
    serial_paths = {
        path for suite in sel.manifest_python_suites(manifest) for path in suite["files"]
    }
    assert mpi_paths.isdisjoint(serial_paths)


def test_python_mpi_plan_is_ranked_and_manifest_owned(tmp_path):
    sel = _load("ci_select_tests")

    class Args:
        plan_file = str(tmp_path / "python-mpi-plan.tsv")
        github_output = str(tmp_path / "github-output.txt")
        explain_file = str(tmp_path / "python-mpi-plan.json")

    assert sel.plan_python_mpi(Args()) == 0
    assert (tmp_path / "python-mpi-plan.tsv").read_text().splitlines() == [
        "2\ttests/python/integration/io/test_hdf5_parallel.py",
        "2\ttests/python/integration/mpi/test_amr_clean_route_program_mpi.py",
        "2\ttests/python/integration/mpi/test_amr_history_mpi.py",
        "2\ttests/python/integration/mpi/test_scientific_output_mpi.py",
        "2\ttests/python/integration/mpi/test_uniform_history_checkpoint_mpi.py",
    ]
    outputs = dict(
        line.partition("=")[::2]
        for line in (tmp_path / "github-output.txt").read_text().splitlines()
    )
    assert outputs["python_mpi_count"] == "5"


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
        "src/CMakeLists.txt",
        "python/CMakeLists.txt",
        "python/bindings/**",
        "python/pops/_native_collectives.py",
        "python/pops/_platform_contracts.py",
        "python/pops/codegen/**",
        "python/pops/output/**",
        "python/pops/runtime/**",
        "python/pops/runtime_environment.py",
        "scripts/ci_select_tests.py",
        "tests/**/mpi/**",
    ):
        assert mpi_input in workflow
    for selector_input in (
        "schemas/**",
        "tests/cpp/build_durations.json",
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
        "tests/cpp/build_durations.json",
        "tests/cpp/test_durations.json",
        "tests/test_manifest.toml",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        "scripts/ci_include_graph.py",
        "scripts/ci_python_module_objects.py",
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
        "scripts/ci_python_module_objects.py",
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
    assert "timeout-minutes: 60" in cpp_shards_block
    assert "timeout-minutes: 50" in cpp_shards_block
    assert cpp_shards_block.count("run_with_heartbeat() {") == 1
    assert 'run_with_heartbeat "Kokkos Serial shard ${{ matrix.shard }} build" 38m' in cpp_shards_block
    assert "test_watchdog=7m" in cpp_shards_block
    assert (
        'run_with_heartbeat "Kokkos Serial shard ${{ matrix.shard }} tests" '
        '"$test_watchdog"' in cpp_shards_block
    )
    assert "timeout --signal=TERM --kill-after=30s" in cpp_shards_block
    assert "mem_available=" in cpp_shards_block
    assert "NINJA_STATUS='[%f/%t elapsed=%es active=%r] '" in cpp_shards_block

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
    assert "--ctest-groups-file build-mpi/mpi-ctest-groups.tsv" in mpi_block
    assert "scripts/ci_select_tests.py python-mpi" in mpi_block
    assert "build-mpi/python-mpi-plan.tsv" in mpi_block
    assert '--target _pops "${mpi_targets[@]}"' in mpi_block
    assert "scripts/ci_select_tests.py verify-cpp-mpi-ctests" in mpi_block
    assert "ctest --preset ci-mpi -N --show-only=json-v1" in mpi_block
    assert "steps.mpi-test-plan.outputs.cpp_label_ctest_count" in mpi_block
    assert "mpi_expected=" not in mpi_block
    # One filtered dry-run authenticates the exact group size; the second invocation executes it.
    assert mpi_block.count("-L '^mpi$' -LE '^python$'") == 2
    assert mpi_block.count("-L '^backend$'") == 2
    assert "ctest --preset ci-mpi --output-on-failure" in mpi_block
    assert "POPS_REQUIRE_MPI_TESTS: \"1\"" in mpi_block
    assert "MPIEXEC_PREFLAGS=--mca;orte_abort_on_non_zero_status;1" in mpi_block
    assert "grep -Eqi 'Open MPI|OpenRTE'" in mpi_block
    assert "mpi_failfast_args+=(--mca orte_abort_on_non_zero_status 1)" in mpi_block
    assert 'run_mpi "Python MPI contract ${mpi_test}"' in mpi_block
    assert 'run_mpi "collective HDF5 writer"' not in mpi_block
    assert "timeout --signal=TERM --kill-after=30s 20m" in mpi_block
    assert "timeout --signal=TERM --kill-after=30s 4m" not in mpi_block
    assert "read -r processors expected regex <&3" in mpi_block
    assert "done 3< build-mpi/mpi-ctest-groups.tsv" in mpi_block
    assert mpi_block.count("</dev/null") == 2
    assert "MPI CTest processor group ${processors} failed" in mpi_block
    assert "selected_count=$(python3 -c" in mpi_block
    assert "selected ${selected_count}/${expected} launches" in mpi_block
    assert "ctest --preset ci-mpi --output-on-failure --parallel 4 --no-tests=error" in mpi_block
    assert "timeout-minutes: 70" in mpi_block
    assert "timeout-minutes: 35" in mpi_block
    assert '/usr/bin/python3 -u "$mpi_test"' in mpi_block
    assert "mpiexec -n \"$mpi_ranks\"" not in mpi_block
    assert "test_amr_clean_route_program_mpi.py" not in mpi_block
    assert "test_amr_history_mpi.py" not in mpi_block
    assert "test_scientific_output_mpi.py" not in mpi_block
    assert "cmake --build --preset ci-mpi\n" not in mpi_block
    assert "build-mpi/python-package" in mpi_block
    assert "collective HDF5 lifecycle requires an MPI-enabled _pops" in mpi_block
    assert "This writer is pure Python" not in mpi_block

    openmp_block = workflow.split("\n  kokkos-openmp:\n", 1)[1]
    assert "name: ubuntu-latest / Kokkos (OpenMP, ${{ matrix.lane }})" in openmp_block
    assert "timeout-minutes: 70" in openmp_block
    assert "fail-fast: false" in openmp_block
    assert openmp_block.count("- lane: cpp-") == 6
    for shard in range(6):
        assert (
            f"- lane: cpp-{shard}\n"
            "            kind: cpp\n"
            f"            shard: {shard}\n"
            "            shard_total: 6\n"
            "            ccache_maxsize: 512M"
        ) in openmp_block
    assert (
        "- lane: python\n"
        "            kind: python\n"
        "            shard: 0\n"
        "            shard_total: 1\n"
        "            ccache_maxsize: 2G"
    ) in openmp_block
    assert openmp_block.count("if: matrix.kind == 'cpp'") == 4
    assert openmp_block.count("if: matrix.kind == 'python'") == 6
    assert "CCACHE_MAXSIZE: ${{ matrix.ccache_maxsize }}" in openmp_block
    assert "uses: actions/cache/restore@v6" in openmp_block
    assert "uses: actions/cache/save@v6" in openmp_block
    assert "github.run_attempt" in openmp_block
    assert "id: openmp-cpp-plan" in openmp_block
    assert "scripts/ci_select_tests.py cpp" in openmp_block
    assert "--changed-files /dev/null" in openmp_block
    assert "--force-all" in openmp_block
    assert '--shard-index "${{ matrix.shard }}"' in openmp_block
    assert '--shard-total "${{ matrix.shard_total }}"' in openmp_block
    assert "openmp-cpp-test-plan-shard-${{ matrix.shard }}" in openmp_block
    assert openmp_block.count("run_with_heartbeat() {") == 2
    openmp_cpp_build = openmp_block[
        openmp_block.index("- name: Configure + build (backend Kokkos OpenMP)"):
        openmp_block.index("- name: Test (ctest, backend Kokkos OpenMP)")
    ]
    assert "if: matrix.kind == 'cpp'" in openmp_cpp_build
    assert "timeout-minutes: 43" in openmp_cpp_build
    assert (
        'run_with_heartbeat "Kokkos OpenMP C++ shard ${{ matrix.shard }} build" 38m'
        in openmp_block
    )
    assert 'cmake --build --preset ci-kokkos --target "${cpp_targets[@]}"' in openmp_block
    assert "cmake --build --preset ci-kokkos\n" not in openmp_block
    assert "ctest --preset ci-kokkos -N --show-only=json-v1" in openmp_block
    assert "scripts/ci_select_tests.py verify-cpp-target-labels" in openmp_block
    assert '--targets "${cpp_targets[@]}"' in openmp_block
    assert "steps.openmp-cpp-plan.outputs.cpp_shard_label_regex" in openmp_block
    assert "ctest --preset ci-kokkos --parallel 2" in openmp_block
    assert openmp_block.index("verify-cpp-target-labels") < openmp_block.index(
        "steps.openmp-cpp-plan.outputs.cpp_shard_label_regex"
    )
    assert "test_watchdog=7m" in openmp_block
    assert 'timeout --signal=TERM --kill-after=30s "$test_watchdog"' in openmp_block
    assert 'run_with_heartbeat "Kokkos OpenMP Python module build" 45m' in openmp_block
    assert "-DPOPS_HEAVY_MODULE_TU_POOL=2" in openmp_block
    assert "-DPOPS_HEAVY_MODULE_TU_POOL=1" not in openmp_block
    assert "-DCMAKE_LINKER_TYPE=MOLD" in openmp_block
    assert '-DCMAKE_CXX_FLAGS="-ffile-prefix-map=${{ github.workspace }}=."' in openmp_block
    assert "name: Cache exact OpenMP Python module" in openmp_block
    assert "id: openmp-python-module-cache" in openmp_block
    assert "pops-module-openmp-${{ runner.os }}" in openmp_block
    assert "id: openmp-python" in openmp_block
    assert "steps.openmp-python.outputs.python-version" in openmp_block
    assert "hashFiles('include/**', 'src/**', 'python/bindings/**'" in openmp_block
    assert "'.github/workflows/ci.yml')" in openmp_block
    openmp_module_cache_block = openmp_block.split(
        "\n      - name: Cache exact OpenMP Python module", 1
    )[1].split("\n      - name:", 1)[0]
    assert "restore-keys:" not in openmp_module_cache_block
    assert (
        "if: matrix.kind == 'python' && "
        "steps.openmp-python-module-cache.outputs.cache-hit == 'true'"
    ) in openmp_block
    assert (
        "if: matrix.kind == 'python' && "
        "steps.openmp-python-module-cache.outputs.cache-hit != 'true'"
    ) in openmp_block
    assert "rsync -aI --delete" in openmp_block
    assert "exact OpenMP module cache hit but no _pops*.so present" in openmp_block
    assert openmp_block.index("Cache exact OpenMP Python module") < openmp_block.index(
        "Restore ccache (Kokkos OpenMP"
    )
    assert openmp_block.index("Build + install Kokkos (OpenMP)") < openmp_block.index(
        "Restore ccache (Kokkos OpenMP"
    )
    assert (
        "matrix.kind != 'python' || "
        "steps.openmp-python-module-cache.outputs.cache-hit != 'true'"
    ) in openmp_block
    assert openmp_block.count("NINJA_STATUS='[%f/%t elapsed=%es active=%r] '") == 2
    openmp_native_test_block = openmp_block.split(
        "\n      - name: Test ABI natif", 1)[1].split("\n      - name:", 1)[0]
    assert 'POPS_REQUIRE_NATIVE_TESTS: "1"' in openmp_native_test_block
    assert "cache-hit" not in openmp_native_test_block
    for native_test in (
        "test_native_abi_std", "test_dsl_production", "test_dsl_production_amr",
    ):
        assert native_test in openmp_native_test_block

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
            "\n  gate-python-prewarm:\n", 1)[0]
    assert "if: needs.set-mode.outputs.architecture_required == 'true'" in architecture_block
    assert "python3 scripts/generate_component_catalog.py --check" in architecture_block

    python_prewarm_block = workflow.split(
        "\n  gate-python-prewarm:\n", 1)[1].split("\n  gate-python-build:\n", 1)[0]
    python_build_block = workflow.split(
        "\n  gate-python-build:\n", 1)[1].split("\n  gate-python:\n", 1)[0]
    python_shards_block = workflow.split(
        "\n  gate-python:\n", 1)[1].split("\n  gate-python-compile-cache:\n", 1)[0]
    python_cache_block = workflow.split(
        "\n  gate-python-compile-cache:\n", 1)[1].split("\n  gate:\n", 1)[0]
    assert "timeout-minutes: 45" in python_build_block
    assert "timeout-minutes: 42" in python_build_block
    assert "timeout --signal=TERM --kill-after=30s 40m" in python_build_block
    assert "exceeded its 40-minute cold-build watchdog" in python_build_block
    assert "exit \"$build_status\"" in python_build_block
    assert "-DPOPS_HEAVY_MODULE_TU_POOL=4" in python_build_block
    assert "ccache --zero-stats" in python_build_block
    assert "uses: actions/cache/restore@v6" in python_build_block
    assert "uses: actions/cache/save@v6" in python_build_block
    assert "always() && steps.kokkos.outcome == 'success'" in python_build_block
    assert "github.run_attempt" in python_build_block
    assert "needs: [changes, set-mode, gate-python-prewarm]" in python_build_block
    assert "steps.modcache.outputs.cache-hit == 'true'" in python_build_block
    assert "github.event_name == 'pull_request'" not in python_build_block
    assert "-py${{ steps.python.outputs.python-version }}-" in python_build_block
    prewarm_module_key = next(
        line for line in python_prewarm_block.splitlines()
        if "key: pops-module-" in line
    )
    build_module_key = next(
        line for line in python_build_block.splitlines()
        if "key: pops-module-" in line
    )
    for cache_input in (
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".github/actions/setup-kokkos/**",
        "scripts/ci_python_module_objects.py",
    ):
        assert repr(cache_input) in prewarm_module_key
        assert repr(cache_input) in build_module_key
    assert "actions/download-artifact@v8" in python_build_block
    assert "--verify-contracts" in python_build_block
    assert "test \"${#cache_archives[@]}\" -eq 3" in python_build_block
    assert "test \"${#compile_contracts[@]}\" -eq 3" in python_build_block
    assert "matrix.lane: [system, amr-block, amr-compiled]" not in python_prewarm_block
    assert "lane: [system, amr-block, amr-compiled]" in python_prewarm_block
    assert "timeout-minutes: 22" in python_prewarm_block
    assert "lookup-only: true" in python_prewarm_block
    assert "scripts/ci_python_module_objects.py" in python_prewarm_block
    assert "--contract-file" in python_prewarm_block
    assert "compression-level: 0" in python_prewarm_block
    assert "-DPOPS_HEAVY_MODULE_TU_POOL=4" in python_prewarm_block
    assert "-DCMAKE_CXX_FLAGS=\"-ffile-prefix-map=${{ github.workspace }}=.\"" in python_prewarm_block
    assert python_prewarm_block.count("run_with_heartbeat() {") == 1
    assert 'run_with_heartbeat "Python prewarm ${{ matrix.lane }}" 18m' \
        in python_prewarm_block
    assert "mem_available=${mem_available_mib}MiB" in python_prewarm_block
    assert 'if [ "${{ matrix.lane }}" = "amr-block" ]; then' in python_prewarm_block
    assert 'lane_parallelism=2' in python_prewarm_block
    assert '--parallel "$lane_parallelism"' in python_prewarm_block
    # Lanes publish only their new, disjoint entries. Restoring the same historical cache in all
    # three would upload its payload three times and erase the cold-build wall-time gain.
    assert "Restore prewarm ccache" not in python_prewarm_block
    assert "Save prewarm ccache" not in python_prewarm_block
    assert "CCACHE_CACHE_KEY" not in python_prewarm_block
    assert "timeout-minutes: 40" in python_shards_block
    assert 'POPS_REQUIRE_NATIVE_TESTS: "1"' in python_shards_block
    assert "timeout-minutes: 30" in python_cache_block
    assert 'POPS_REQUIRE_NATIVE_TESTS: "1"' in python_cache_block
    for block in (python_build_block, python_shards_block, python_cache_block):
        assert "if: needs.set-mode.outputs.python_required == 'true'" in block

    # GitHub rejects `runner.*` in a job-level `env` mapping before creating any job.  Keep the
    # runner-specific cache prefix at step scope and the compile-cache temporary workspace-owned.
    assert "CCACHE_CACHE_KEY: ccache-${{ runner.os }}" not in workflow
    assert "COMPILE_CACHE_TMP: ${{ runner.temp }}" not in workflow
    assert "key: ccache-${{ runner.os }}-${{ env.CCACHE_CACHE_KEY }}" in workflow
    assert "COMPILE_CACHE_TMP: ${{ github.workspace }}/.pops-ci/compile-cache-test" in workflow
    assert 'mkdir -p "$COMPILE_CACHE_TMP"' in python_cache_block


def test_native_required_lane_cannot_reclassify_subprocess_failure_as_skip():
    from tests.python import conftest as process_runner

    diagnostic = "RuntimeError: required native test unavailable: Kokkos introuvable"
    assert process_runner._missing_process_requirement_for_environment(
        diagnostic, {}) == "native compile requires POPS_KOKKOS_ROOT/Kokkos_ROOT"
    assert process_runner._missing_process_requirement_for_environment(
        diagnostic, {"POPS_REQUIRE_NATIVE_TESTS": "1"}) is None


@pytest.mark.parametrize(
    "relative_path",
    (
        "tests/python/integration/native_loader/test_native_abi_std.py",
        "tests/python/integration/native_loader/test_dsl_production.py",
        "tests/python/integration/native_loader/test_dsl_production_amr.py",
    ),
)
def test_openmp_native_scripts_share_the_fail_closed_requirement_policy(relative_path):
    source = (REPO_ROOT / relative_path).read_text()
    assert "missing_native_compile_requirement" in source
    assert "require_native_or_skip" in source
    assert "if not cxx or not os.path.isdir(INCLUDE)" not in source
    assert "OK (rien a compiler)" not in source


def test_ci_control_plane_inputs_force_full_functional_selection():
    selector = _load("ci_select_tests")
    for path in (
        ".github/workflows/ci.yml",
        "scripts/ci_route_mode.py",
        "scripts/ci_select_tests.py",
        "scripts/ci_shard_binpack.py",
        "scripts/ci_python_module_objects.py",
        "tests/test_manifest.toml",
    ):
        assert path in selector.CPP_BROAD_FILES
        assert path in selector.PYTHON_BROAD_FILES
    assert "tests/cpp/test_sources.cmake" in selector.CPP_BROAD_FILES
    assert "tests/cpp/test_durations.json" in selector.CPP_BROAD_FILES
    assert "scripts/ci_include_graph.py" in selector.CPP_BROAD_FILES
    assert "tests/python/test_durations.json" in selector.PYTHON_BROAD_FILES
    assert "scripts/ci_import_closure.py" in selector.PYTHON_BROAD_FILES
