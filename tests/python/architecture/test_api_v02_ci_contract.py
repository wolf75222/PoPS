"""Source-only registration checks; these do not execute or qualify native profiles."""
import ast
import json
from pathlib import Path
import tomllib
import copy
import pytest

from scripts.ci_python_dimensions import partition


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = "tests/python/integration/runtime/"


def _registration():
    timings = json.loads((ROOT / "tests/python/test_durations.json").read_text())
    return timings, timings["_meta"]["api_v02_20261013"]["files"]


def test_api_v02_receivers_have_explicit_disjoint_native_dimensions():
    _, rows = _registration()
    native = {path: row for path, row in rows.items() if row["kind"] == "native_profile"}
    discovered = {str(path.relative_to(ROOT))
                  for path in (ROOT / RUNTIME).glob("test_api_v02*.py")}
    discovered.update(str(path.relative_to(ROOT))
                      for path in (ROOT / RUNTIME).glob("test_*v14_native.py"))
    discovered.update(str(path.relative_to(ROOT))
                      for path in (ROOT / RUNTIME).glob("test_*_v15_native.py"))
    discovered.update(RUNTIME + name for name in (
        "test_computed_scratch_ssa_native.py", "test_stencil_flux_v02_native.py",
        "test_relations_v09_native.py", "test_continuation_v10_native.py",
        "test_collective_control_preparation_v10.py", "test_finite_structural_depth_native.py",
        "test_diffusion_device_publication_native.py", "test_distributed_balanced_domain_native.py"))
    assert set(native) == discovered
    contract = json.loads((ROOT / "tests/python/native_dimensions.json").read_text())
    for path, row in native.items():
        assert contract["files"][path] == row["native_dimension"]
    groups = partition(sorted(native), contract)
    assert groups[1] == [RUNTIME + "test_api_v02_ale_native.py",
                         RUNTIME + "test_api_v02_hyqmom_orders_native.py"]
    assert set(groups[2]) == set(native) - set(groups[1])
    assert sum(map(len, groups.values())) == len(native)


def test_api_v02_registered_files_have_existing_suites_and_estimated_weights():
    timings, rows = _registration()
    manifest = tomllib.loads((ROOT / "tests/test_manifest.toml").read_text())
    suites = [ROOT / row["path"] for row in manifest["python"]["suite"]]
    for path, row in rows.items():
        file = ROOT / path
        assert file.is_file(), path
        assert sum(file.is_relative_to(suite) for suite in suites) == 1, path
        assert row["kind"] in ("native_profile", "source_contract")
        assert type(row["estimated_seconds"]) is int and 0 < row["estimated_seconds"] <= 1200
        if row.get("execution_lane") == "mpi_only":
            assert path not in timings
            assert path not in timings["_meta"]["estimated_files"]
        else:
            assert timings[path] == row["estimated_seconds"]
            assert path in timings["_meta"]["estimated_files"]
    assert timings["_meta"]["estimated_count"] == len(timings["_meta"]["estimated_files"])


def _test_case_count(relative):
    tree = ast.parse((ROOT / relative).read_text())
    result = {}
    fixture_cases = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "fixture"):
                params = next((item.value for item in decorator.keywords if item.arg == "params"), None)
                if params is not None:
                    assert isinstance(params, (ast.Tuple, ast.List))
                    fixture_cases[node.name] = len(params.elts)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        cases = 1
        for argument in node.args.args:
            cases *= fixture_cases.get(argument.arg, 1)
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "parametrize"):
                assert isinstance(decorator.args[1], (ast.Tuple, ast.List))
                cases *= len(decorator.args[1].elts)
        result[node.name] = cases
    return result


def test_ale_and_relaxation_receivers_keep_all_cases_and_no_test_module_imports():
    assert _test_case_count(RUNTIME + "test_api_v02_ale_native.py") == {
        "test_actual_ale_volumes": 3}
    assert _test_case_count(RUNTIME + "test_api_v02_continuum_native.py") == {
        "test_continuum_profile": 4,
        "test_hall_zero_and_ohmic_are_different_physics_in_same_native_primitive": 1,
        "test_bn_authored_relaxation_changes_state_and_conserves_inventory": 3}
    helper = "tests/python/support/api_v02_native.py"
    assert not _test_case_count(helper)
    for path in (RUNTIME + "test_api_v02_ale_native.py",
                 RUNTIME + "test_api_v02_continuum_native.py", helper):
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not any(part.startswith("test_") for part in (node.module or "").split("."))


def test_hyqmom_orders_are_six_dim1_cases_outside_the_dim2_model_receiver():
    assert _test_case_count(RUNTIME + "test_api_v02_hyqmom_orders_native.py") == {
        "test_native_hyqmom_multiple_orders_and_regimes": 6}
    receivers = ("test_api_v02_models_M15_M21.py", "test_api_v02_models_M19_M21.py")
    models = []
    for receiver in receivers:
        tree = ast.parse((ROOT / (RUNTIME + receiver)).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                assert node.value != "M15", "M15 must not select the Dim1 ABI in a Dim2 test process"
                if isinstance(node.value, str) and node.value in {"M16", "M17", "M18", "M19", "M20", "M21"}:
                    models.append(node.value)
    # M16 appears once as a successful profile and once as the oblique refusal.
    assert sorted(models) == ["M16", "M16", "M17", "M18", "M19", "M20", "M21"]
    assert sum(sum(_test_case_count(RUNTIME + receiver).values()) for receiver in receivers) == 7
    helper = "tests/python/support/api_v02_moment_profiles.py"
    for path in [*(RUNTIME + receiver for receiver in receivers), helper]:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not any(part.startswith("test_") for part in (node.module or "").split("."))


def test_v09_native_receivers_match_their_declared_case_counts():
    timings, rows = _registration()
    registered = {path: row for path, row in rows.items()
                  if "declared_cases" in row and row.get("registration", "v09") == "v09"}
    summary = timings["_meta"]["api_v02_20261013"]["v09_registration"]
    assert len(registered) == summary["native_files"]
    assert sum(row["declared_cases"] for row in registered.values()) == summary["native_cases"]
    for path, row in registered.items():
        assert row["kind"] == "native_profile"
        assert sum(_test_case_count(path).values()) == row["declared_cases"], path


def test_v10_receivers_have_exact_cardinality_and_ranked_pytest_entrypoints():
    timings, rows = _registration()
    native = {path: row for path, row in rows.items()
              if row.get("registration") == "v10" and row["kind"] == "native_profile"}
    summary = timings["_meta"]["api_v02_20261013"]["v10_registration"]
    assert len(native) == summary["native_files"]
    assert sum(row["declared_cases"] for row in native.values()) == summary["native_cases"]
    manifest = tomllib.loads((ROOT / "tests/test_manifest.toml").read_text())
    entries = {row["path"]: row for suite in manifest["python"]["suite"]
               for row in suite.get("mpi_entrypoints", [])}
    found = {}
    mpi_directory = ROOT / "tests/python/integration/mpi"
    wrappers = [*mpi_directory.glob("test_api_v02_*_mpi2.py"),
                mpi_directory / "test_finite_structural_depth_native_mpi2.py",
                mpi_directory / "test_diffusion_device_publication_native_mpi2.py",
                mpi_directory / "test_distributed_balanced_domain_native_mpi2.py"]
    for wrapper in wrappers:
        tree = ast.parse(wrapper.read_text())
        assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                       for node in tree.body), wrapper
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "run"]
        assert len(calls) == 1
        receiver = ast.literal_eval(calls[0].args[0])
        options = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        assert receiver not in found
        found[receiver] = options
        entry = entries[str(wrapper.relative_to(ROOT))]
        assert entry["nproc"] == 2 and entry["native_dimension"] == 2
    mixed = {path for path, row in rows.items()
             if row.get("registration") == "ci_mpi_20261014"}
    assert mixed == {RUNTIME + "test_api_v02_finite_arrays_native.py",
                     RUNTIME + "test_api_v02_regions_native.py"}
    completion = {path: row for path, row in rows.items()
                  if row.get("registration") == "completion_20261015"
                  and row["kind"] == "native_profile"}
    assert len(completion) == 38
    assert set(found) == set(native) | mixed | set(completion)
    for path, row in completion.items():
        assert sum(_test_case_count(path).values()) == row["declared_cases"], path
        assert found[path] == {"expected_cases": row["declared_cases"], "ranks": 2}
        assert row["ci_ranks"] == [1, 2] and row["planned_local_reception_ranks"] == [1, 2, 3]
        assert row["execution_lane"] == "serial_and_mpi"
    for path, row in native.items():
        assert sum(_test_case_count(path).values()) == row["declared_cases"], path
        assert found[path] == {"expected_cases": row["declared_cases"], "ranks": 2}
        assert row["ci_ranks"] == [2] and row["local_reception_ranks"] == [2, 3]


def test_v14_receivers_have_exact_counts_and_unchanged_bounded_mpi2_launches():
    from scripts import ci_select_tests as selection
    from scripts import ci_python_mpi as mpi
    timings, rows = _registration()
    native = {path: row for path, row in rows.items()
              if row.get("registration") == "v14" and row["kind"] == "native_profile"}
    source = {path: row for path, row in rows.items()
              if row.get("registration") == "v14" and row["kind"] == "source_contract"}
    summary = timings["_meta"]["api_v02_20261013"]["v14_registration"]
    assert len(native) == summary["native_files"]
    assert len(source) == summary["source_files"]
    assert sum(row["declared_cases"] for row in native.values()) == summary["native_cases"]
    assert sum(row["declared_cases"] for row in source.values()) == summary["source_cases"]
    assert set(source) == {str(path.relative_to(ROOT))
                           for path in (ROOT / "tests/python/unit").rglob("test_*v14.py")}
    manifest = selection.load_manifest()
    entries = {row["path"]: row for row in selection.manifest_python_mpi_entrypoints(manifest)}
    found = {}
    for wrapper in (ROOT / "tests/python/integration/mpi").glob("test_api_v14_*_mpi2.py"):
        tree = ast.parse(wrapper.read_text())
        assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                       for node in tree.body), wrapper
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "run"]
        assert len(calls) == 1
        receiver = ast.literal_eval(calls[0].args[0])
        options = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        assert receiver not in found
        found[receiver] = options
        entry = entries[str(wrapper.relative_to(ROOT))]
        assert entry["nproc"] == 2 and entry["native_dimension"] == 2
    assert len(found) == summary["mpi2_entrypoints"]
    assert set(found) == set(native)
    for path, row in {**source, **native}.items():
        assert sum(_test_case_count(path).values()) == row["declared_cases"], path
    for path, row in native.items():
        assert found[path] == {"expected_cases": row["declared_cases"], "ranks": 2}
        assert row["ci_ranks"] == [2]
        assert row["local_reception_ranks"] in ([1, 2, 3], [2, 3])
    for group in mpi.make_plan(manifest)["groups"]:
        for launch in group["launches"]:
            if Path(launch["path"]).name.startswith("test_api_v14_"):
                assert launch["timeout_seconds"] == 1200


def test_api_v02_mpi_only_modules_have_exact_exclusive_receivers_without_serial_skips():
    from scripts import ci_select_tests as selection
    _, rows = _registration()
    manifest = selection.load_manifest()
    receivers = selection.manifest_python_mpi_receivers(manifest)
    expected = {RUNTIME + "test_" + name + "_v14_native.py" for name in (
        "bundle_repartition", "distributed_algebra", "distributed_failure", "distributed_gc")}
    expected.add(RUNTIME + "test_api_v02_distributed_program_native.py")
    assert {row["path"] for row in receivers} == expected
    serial = {path for suite in selection.manifest_python_suites(manifest) for path in suite["files"]}
    assert not serial & expected
    for receiver in receivers:
        row = rows[receiver["path"]]
        assert row["execution_lane"] == "mpi_only"
        assert receiver["entrypoint"] == row["entrypoint"]
        assert receiver["expected_cases"] == row["declared_cases"]
        assert receiver["nproc"] == 2


@pytest.mark.parametrize("corruption", ["missing_entrypoint", "wrong_receiver", "duplicate_receiver"])
def test_mpi_only_routing_refuses_loss_or_misdirection(corruption):
    from scripts import ci_select_tests as selection
    manifest = copy.deepcopy(selection.load_manifest())
    owner = next(suite for suite in manifest["python"]["suite"] if suite.get("mpi_receivers"))
    row = owner["mpi_receivers"][0]
    if corruption == "missing_entrypoint":
        for suite in manifest["python"]["suite"]:
            if "mpi_entrypoints" in suite:
                suite["mpi_entrypoints"] = [entry for entry in suite["mpi_entrypoints"]
                                            if entry["path"] != row["entrypoint"]]
    elif corruption == "wrong_receiver":
        row["entrypoint"] = owner["mpi_receivers"][1]["entrypoint"]
    else:
        owner["mpi_receivers"].append(dict(row))
    with pytest.raises(SystemExit, match="MPI receiver"):
        selection.manifest_python_suites(manifest)


def test_v15_inventory_and_closure_registration_has_exact_counts_and_mpi2_route():
    from scripts import ci_select_tests as selection
    from scripts import ci_python_mpi as mpi
    timings, rows = _registration()
    native = {path: row for path, row in rows.items()
              if row.get("registration") == "v15" and row["kind"] == "native_profile"}
    source = {path: row for path, row in rows.items()
              if row.get("registration") == "v15" and row["kind"] == "source_contract"}
    summary = timings["_meta"]["api_v02_20261013"]["v15_registration"]
    assert set(native) == {str(path.relative_to(ROOT))
                           for path in (ROOT / RUNTIME).glob("test_*_v15_native.py")}
    assert set(source) == {str(path.relative_to(ROOT))
                           for path in (ROOT / "tests/python/unit").rglob("test_*_v15.py")}
    assert len(native) == summary["native_files"]
    assert len(source) == summary["source_files"]
    assert sum(row["declared_cases"] for row in native.values()) == summary["native_cases"]
    assert sum(row["declared_cases"] for row in source.values()) == summary["source_cases"]
    for path, row in {**native, **source}.items():
        assert sum(_test_case_count(path).values()) == row["declared_cases"], path
    manifest = selection.load_manifest()
    entries = {row["path"]: row for row in selection.manifest_python_mpi_entrypoints(manifest)}
    found = {}
    for wrapper in (ROOT / "tests/python/integration/mpi").glob("test_api_v15_*_mpi2.py"):
        tree = ast.parse(wrapper.read_text())
        assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                       for node in tree.body), wrapper
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "run"]
        assert len(calls) == 1
        receiver = ast.literal_eval(calls[0].args[0])
        options = {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords}
        assert receiver not in found
        found[receiver] = options
        entry = entries[str(wrapper.relative_to(ROOT))]
        assert entry["nproc"] == 2 and entry["native_dimension"] == 2
    assert set(found) == set(native)
    assert len(found) == summary["mpi2_entrypoints"]
    for path, row in native.items():
        assert found[path] == {"expected_cases": row["declared_cases"], "ranks": 2}
        assert row["ci_ranks"] == [2] and row["local_reception_ranks"] == [1, 2]
    launches = [launch for group in mpi.make_plan(manifest)["groups"]
                for launch in group["launches"]
                if Path(launch["path"]).name.startswith("test_api_v15_")]
    assert len(launches) == summary["mpi2_entrypoints"]
    assert all(launch["timeout_seconds"] == 1200 for launch in launches)


def test_distributed_program_has_one_mandatory_mpi_owner_and_no_serial_assignment():
    from scripts import ci_select_tests as selection
    from scripts import ci_python_mpi as mpi
    path = RUNTIME + "test_api_v02_distributed_program_native.py"
    entrypoint = "tests/python/integration/mpi/test_api_v02_distributed_program_mpi2.py"
    manifest = selection.load_manifest()
    owners = [row for row in selection.manifest_python_mpi_receivers(manifest)
              if row["path"] == path]
    assert owners == [{"path": path, "entrypoint": entrypoint, "expected_cases": 4, "nproc": 2}]
    assert _test_case_count(path) == {
        "test_distributed_program_checks_native_data_before_commit": 3,
        "test_corrupt_import_unhandled_failure_preserves_payload_and_rolls_back": 1,
    }
    assert all(path not in suite["files"] for suite in selection.manifest_python_suites(manifest))
    launches = [launch for group in mpi.make_plan(manifest)["groups"]
                for launch in group["launches"] if launch["path"] == entrypoint]
    assert len(launches) == 1
    assert launches[0]["nproc"] == 2 and launches[0]["timeout_seconds"] == 1200


@pytest.mark.parametrize("name,cases", [("finite_arrays", 8), ("regions", 6)])
def test_mixed_api_v02_collective_receivers_keep_serial_and_exact_mpi_coverage(name, cases):
    from scripts import ci_select_tests as selection
    from scripts import ci_python_mpi as mpi
    path = RUNTIME + "test_api_v02_" + name + "_native.py"
    entrypoint = "tests/python/integration/mpi/test_api_v02_" + name + "_mpi2.py"
    manifest = selection.load_manifest()
    serial = {path for suite in selection.manifest_python_suites(manifest) for path in suite["files"]}
    assert path in serial
    timings, rows = _registration()
    assert timings[path] == rows[path]["estimated_seconds"] == 600
    assert rows[path]["declared_cases"] == sum(_test_case_count(path).values()) == cases
    tree = ast.parse((ROOT / entrypoint).read_text())
    assert any(isinstance(node, ast.ImportFrom)
               and node.module == "tests.python.support.mpi_pytest_entrypoint"
               and [(alias.name, alias.asname) for alias in node.names] == [("run", None)]
               for node in tree.body)
    call, = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "run"]
    assert [ast.literal_eval(arg) for arg in call.args] == [path]
    assert {item.arg: ast.literal_eval(item.value) for item in call.keywords} == {
        "expected_cases": cases, "ranks": 2}
    launch, = [launch for group in mpi.make_plan(manifest)["groups"]
               for launch in group["launches"] if launch["path"] == entrypoint]
    assert launch["nproc"] == 2 and launch["timeout_seconds"] == 1200
