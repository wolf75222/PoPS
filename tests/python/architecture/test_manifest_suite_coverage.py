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
change to a test or return of a dependency requires explicit reclassification. The
catalogue may only shrink: a record added or rewritten fails even when the catalogue
pin is updated with it.

The weekly manifest check of quality.yml (``docs/gen_test_counts.py --check-matrix``) applies the
same disposition: it exempts these seven files while their bytes match the catalogue, and fails
when a suite claims one of them.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "tests" / "test_manifest.toml"
UNAVAILABLE_CATALOG = pathlib.Path("tests/python/unavailable_verification_tests.json")
# Pin the catalogue independently of its contents: editing a digest, dependency,
# or path cannot silently broaden this historical disposition.
UNAVAILABLE_CATALOG_SHA256 = "adc9a63419d00b23d5885cf5d60e696ebb0d15da9e2276aa54788f7c63928129"
# The catalogue may only shrink. Its header and its seven records as first catalogued (1298dc52),
# each as the SHA-256 of its canonical JSON: a record added or rewritten fails even when the pin
# above is updated with it; a record may only be dropped, once its test is reclassified into a suite.
ORIGINAL_UNAVAILABLE_HEADER_SHA256 = "7c50c1fbeef28ce4586d9667800a512a1138b55a4f5f4e27d8de6107d19b2126"
ORIGINAL_UNAVAILABLE_RECORDS = {
    "tests/python/verification/test_check_verification_manifest.py":
        "936f16e506a88496647de3b9308230a4e36fa648dbd88332b9d8415d7c2785bf",
    "tests/python/verification/test_reference_errors.py":
        "5811aff8c7db5103540862660b8be929c7d3507a1ee529dbd4d81131e5bd84fd",
    "tests/python/verification/test_run_verification.py":
        "603024823cced758741b9725bc2c6e7716832889d91851be79857a00900ddf07",
    "tests/python/verification/test_verification_manifest_schema.py":
        "8b8a998396a4b9f110aa37d84efcb6e36e6309b0f00842b003b402abca66093e",
    "tests/python/verification/test_verification_metrics_schema.py":
        "4e8e0237de079bfb8e644803afb5bbb15ae0806e587df4e996ae0f57d37f6d78",
    "tests/python/verification/test_verification_provenance_schema.py":
        "0682d933a32c3ffd770cc261f3c5a8f8d02dc14d73434871d71addb579722d24",
    "tests/python/verification/test_verification_report_schema.py":
        "6da37074dcd1bf55083e0ba3b409ab634011fca7032f0e05b127efe4835b7679",
}
GEN_TEST_COUNTS = pathlib.Path("docs/gen_test_counts.py")


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _historically_unavailable_paths(repo_root, suite_dirs,
                                    catalog_sha256=UNAVAILABLE_CATALOG_SHA256):
    catalog_bytes = (repo_root / UNAVAILABLE_CATALOG).read_bytes()
    assert hashlib.sha256(catalog_bytes).hexdigest() == catalog_sha256, (
        "historical unavailable catalogue changed; reclassify explicitly, never absorb new tests")
    catalog = json.loads(catalog_bytes)
    header = {key: value for key, value in catalog.items() if key != "tests"}
    assert _canonical_sha256(header) == ORIGINAL_UNAVAILABLE_HEADER_SHA256, (
        "historical unavailable catalogue may only shrink: its header changed; "
        "reclassify explicitly")
    historical = set()
    for record in catalog["tests"]:
        relative = record["path"]
        assert (relative not in historical
                and ORIGINAL_UNAVAILABLE_RECORDS.get(relative) == _canonical_sha256(record)), (
            f"historical unavailable catalogue may only shrink: {relative} added or rewritten; "
            "reclassify explicitly, never absorb new tests")
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


def _uncovered_test_paths(repo_root, suite_paths, catalog_sha256=UNAVAILABLE_CATALOG_SHA256):
    suite_dirs = [(repo_root / path).resolve() for path in suite_paths]
    historical = _historically_unavailable_paths(repo_root, suite_dirs, catalog_sha256)
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


def _rewrite_and_repin(historical_tree, catalog):
    catalog_path = historical_tree / UNAVAILABLE_CATALOG
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return hashlib.sha256(catalog_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [
    "record_added", "record_duplicated", "record_rehashed", "assets_dropped", "header_rewritten",
])
def test_historical_catalogue_never_grows_even_when_repinned(historical_tree, mutation):
    catalog = json.loads((historical_tree / UNAVAILABLE_CATALOG).read_text())
    record = catalog["tests"][0]
    if mutation == "record_added":
        new_test = historical_tree / "tests/python/verification/test_new.py"
        new_test.write_text("def test_new(): pass\n")
        catalog["tests"].append({**record, "path": str(new_test.relative_to(historical_tree)),
                                 "sha256": hashlib.sha256(new_test.read_bytes()).hexdigest()})
    elif mutation == "record_duplicated":
        catalog["tests"].append(dict(record))
    elif mutation == "record_rehashed":
        test_file = historical_tree / record["path"]
        test_file.write_text("# changed test\n")
        record["sha256"] = hashlib.sha256(test_file.read_bytes()).hexdigest()
    elif mutation == "assets_dropped":
        record["missing_assets"] = record["missing_assets"][1:]
    else:
        catalog["reason"] = "Any test may wait here until its dependencies return."
    pin = _rewrite_and_repin(historical_tree, catalog)
    with pytest.raises(AssertionError, match="may only shrink"):
        _uncovered_test_paths(historical_tree, [], catalog_sha256=pin)


def test_historical_catalogue_may_shrink_when_repinned(historical_tree):
    catalog = json.loads((historical_tree / UNAVAILABLE_CATALOG).read_text())
    dropped = catalog["tests"].pop(0)["path"]
    pin = _rewrite_and_repin(historical_tree, catalog)
    assert len(_historically_unavailable_paths(historical_tree, [], catalog_sha256=pin)) == 6
    # The dropped test is active again, so a suite must now claim it.
    assert _uncovered_test_paths(historical_tree, [], catalog_sha256=pin) == [dropped]


def _check_matrix(repo_root):
    return subprocess.run([sys.executable, str(repo_root / GEN_TEST_COUNTS), "--check-matrix"],
                          capture_output=True, text=True, check=False)


def test_quality_manifest_check_exempts_exactly_the_historical_files():
    result = _check_matrix(REPO_ROOT)
    assert result.returncode == 0, result.stdout
    assert "Historically unavailable, never run: 7 Python files" in result.stdout
    assert "MISSING" not in result.stdout


@pytest.fixture
def matrix_tree(historical_tree):
    (historical_tree / "docs").mkdir()
    shutil.copy(REPO_ROOT / GEN_TEST_COUNTS, historical_tree / GEN_TEST_COUNTS)
    (historical_tree / "tests/cpp").mkdir(parents=True)
    active = historical_tree / "tests/python/active"
    active.mkdir()
    (active / "test_active.py").write_text("def test_active(): pass\n")
    (historical_tree / "tests/test_manifest.toml").write_text(
        '[[python.suite]]\nname = "active"\npath = "tests/python/active"\n')
    return historical_tree


def test_quality_manifest_check_passes_beside_the_historical_files(matrix_tree):
    result = _check_matrix(matrix_tree)
    assert result.returncode == 0, result.stdout
    assert "Historically unavailable, never run: 7 Python files" in result.stdout


@pytest.mark.parametrize("mutation", ["test_bytes", "new_test", "suite_claims"])
def test_quality_manifest_check_refuses_drift(matrix_tree, mutation):
    first = json.loads((matrix_tree / UNAVAILABLE_CATALOG).read_text())["tests"][0]["path"]
    if mutation == "test_bytes":
        test_file = matrix_tree / first
        test_file.write_bytes(test_file.read_bytes() + b"\n# changed\n")
        expected = f"MISSING {first}"
    elif mutation == "new_test":
        (matrix_tree / "tests/python/verification/test_new.py").write_text("def test_new(): pass\n")
        expected = "MISSING tests/python/verification/test_new.py"
    else:
        with (matrix_tree / "tests/test_manifest.toml").open("a") as manifest:
            manifest.write('\n[[python.suite]]\nname = "verification"\n'
                           'path = "tests/python/verification"\n')
        expected = f"CLAIMED {first}"
    result = _check_matrix(matrix_tree)
    assert result.returncode == 1, result.stdout
    assert expected in result.stdout
