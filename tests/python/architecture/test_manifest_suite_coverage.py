"""ADC-625 fence: every Python test file is claimed by a manifest [[python.suite]] path.

Phase 5 added new test directories (``tests/python/unit/problem``,
``tests/python/unit/numerics``) that were NOT listed in ``tests/test_manifest.toml``, so
``scripts/ci_select_tests.py`` -- which selects the Python suites by DIRECTORY -- never ran them
in any CI lane. This source-only fence makes that failure LOUD: it parses the manifest's
``[[python.suite]]`` paths and asserts that EVERY ``tests/python/**/test_*.py`` file lives under one
of them (a directory-prefix match), so a future new test directory fails here instead of silently
never running.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "tests" / "test_manifest.toml"
TESTS_ROOT = REPO_ROOT / "tests" / "python"


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
    suite_dirs = [(REPO_ROOT / path).resolve() for path in _suite_paths()]
    uncovered = []
    for test_file in sorted(TESTS_ROOT.rglob("test_*.py")):
        resolved = test_file.resolve()
        if not any(resolved.is_relative_to(suite_dir) for suite_dir in suite_dirs):
            uncovered.append(str(test_file.relative_to(REPO_ROOT)))
    assert not uncovered, (
        "these test files are under no [[python.suite]] path in tests/test_manifest.toml, so "
        "scripts/ci_select_tests.py never runs them -- add a [[python.suite]] entry (and a "
        "ci_select_tests.py route + area alias) for their directory:\n  " + "\n  ".join(uncovered))
