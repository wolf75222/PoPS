"""ADC-625 fence: every Python test file is claimed by a manifest [[python.suite]] path.

Phase 5 added new test directories (``tests/python/unit/problem``,
``tests/python/unit/numerics``) that were NOT listed in ``tests/test_manifest.toml``, so
``scripts/ci_select_tests.py`` -- which selects the Python suites by DIRECTORY -- never ran them
in any CI lane. This source-only fence makes that failure LOUD: it parses the manifest's
``[[python.suite]]`` paths and asserts that every active ``tests/python/**/test_*.py`` file lives
under one of them (a directory-prefix match), so a future new test directory fails here instead
of silently never running.

Seven byte-pinned tests were introduced without their dependencies before M0. Their
closed catalogue records historical unavailability, not successful execution. Any
change to a test or return of a dependency requires explicit reclassification.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import hashlib
import json
import pathlib
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "tests" / "test_manifest.toml"
UNAVAILABLE_CATALOG = pathlib.Path("tests/python/unavailable_verification_tests.json")
# Pin the catalogue independently of its contents: editing a digest, dependency,
# or path cannot silently broaden this historical disposition.
UNAVAILABLE_CATALOG_SHA256 = "adc9a63419d00b23d5885cf5d60e696ebb0d15da9e2276aa54788f7c63928129"


def _historically_unavailable_paths(repo_root, suite_dirs):
    catalog_bytes = (repo_root / UNAVAILABLE_CATALOG).read_bytes()
    assert hashlib.sha256(catalog_bytes).hexdigest() == UNAVAILABLE_CATALOG_SHA256, (
        "historical unavailable catalogue changed; reclassify explicitly, never absorb new tests")
    catalog = json.loads(catalog_bytes)
    historical = set()
    for record in catalog["tests"]:
        relative = record["path"]
        test_file = repo_root / relative
        unchanged = (test_file.is_file()
                     and hashlib.sha256(test_file.read_bytes()).hexdigest() == record["sha256"])
        assert unchanged, (
            f"historically unavailable test changed or disappeared: {relative}; reclassify explicitly")
        assert not any(test_file.resolve().is_relative_to(directory) for directory in suite_dirs), (
            f"historically unavailable test claimed by an active suite: {relative}; "
            "restore its dependencies and reclassify before activation")
        for asset in record["missing_assets"]:
            asset_path = repo_root / asset
            assert not asset_path.exists() and not asset_path.is_symlink(), (
                f"historical dependency appeared: {asset}; reclassify or reactivate affected tests")
        historical.add(relative)
    return historical


def _uncovered_test_paths(repo_root, suite_paths):
    suite_dirs = [(repo_root / path).resolve() for path in suite_paths]
    historical = _historically_unavailable_paths(repo_root, suite_dirs)
    return [str(test_file.relative_to(repo_root))
            for test_file in sorted((repo_root / "tests/python").rglob("test_*.py"))
            if str(test_file.relative_to(repo_root)) not in historical
            and not any(test_file.resolve().is_relative_to(directory) for directory in suite_dirs)]


def _suite_paths():
    manifest = tomllib.loads(MANIFEST.read_text())
    return [str(suite["path"]) for suite in manifest.get("python", {}).get("suite", [])
            if suite.get("path")]


def test_manifest_lists_python_suite_paths():
    paths = _suite_paths()
    assert paths, "tests/test_manifest.toml must declare [[python.suite]] entries with a path"
    for path in paths:
        assert (REPO_ROOT / path).is_dir(), (
            "manifest [[python.suite]] path %r is not an existing directory" % path)


def test_every_python_test_file_is_covered_by_a_suite_path():
    uncovered = _uncovered_test_paths(REPO_ROOT, _suite_paths())
    assert not uncovered, (
        "these test files are under no [[python.suite]] path in tests/test_manifest.toml, so "
        "scripts/ci_select_tests.py never runs them -- add a [[python.suite]] entry (and a "
        "ci_select_tests.py route + area alias) for their directory:\n  " + "\n  ".join(uncovered))


@pytest.fixture
def historical_tree(tmp_path):
    catalog = REPO_ROOT / UNAVAILABLE_CATALOG
    files = [UNAVAILABLE_CATALOG] + [pathlib.Path(item["path"])
                                    for item in json.loads(catalog.read_text())["tests"]]
    for relative in files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())
    return tmp_path


def test_historical_disposition_covers_only_the_seven_original_files(historical_tree):
    assert len(_historically_unavailable_paths(historical_tree, [])) == 7
    assert _uncovered_test_paths(historical_tree, []) == []


@pytest.mark.parametrize("mutation", [
    "test_bytes", "test_missing", "asset_returned", "catalog_expanded", "catalog_rehashed",
])
def test_historical_disposition_refuses_drift(historical_tree, mutation):
    catalog_path = historical_tree / UNAVAILABLE_CATALOG
    catalog = json.loads(catalog_path.read_text())
    record = catalog["tests"][0]
    test_file = historical_tree / record["path"]
    if mutation == "test_bytes":
        test_file.write_bytes(test_file.read_bytes() + b"\n# changed\n")
    elif mutation == "test_missing":
        test_file.unlink()
    elif mutation == "asset_returned":
        asset = historical_tree / record["missing_assets"][0]
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("restored dependency\n")
    elif mutation == "catalog_expanded":
        catalog["tests"].append({**record, "path": "tests/python/verification/test_new.py"})
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    else:
        test_file.write_text("# changed test\n")
        record["sha256"] = hashlib.sha256(test_file.read_bytes()).hexdigest()
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    with pytest.raises(AssertionError, match="reclassify"):
        _uncovered_test_paths(historical_tree, [])


def test_historical_disposition_refuses_implicit_suite_activation(historical_tree):
    with pytest.raises(AssertionError, match="claimed by an active suite"):
        _uncovered_test_paths(historical_tree, ["tests/python/verification"])


@pytest.mark.parametrize("directory", ["verification", "new_module"])
def test_new_tests_remain_uncovered_beside_historical_files(historical_tree, directory):
    relative = pathlib.Path("tests/python") / directory / "test_new.py"
    test_file = historical_tree / relative
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_new(): pass\n")
    assert _uncovered_test_paths(historical_tree, []) == [str(relative)]


def test_active_suite_still_covers_other_tests(historical_tree):
    directory = historical_tree / "tests/python/active"
    directory.mkdir()
    (directory / "test_active.py").write_text("def test_active(): pass\n")
    assert _uncovered_test_paths(historical_tree, ["tests/python/active"]) == []
