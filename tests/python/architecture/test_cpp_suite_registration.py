"""ADC-550 fence: every ``[[cpp.suite]]`` in the manifest is registered in ``tests/CMakeLists.txt``.

The C++ mirror of ``test_manifest_suite_coverage.py`` (which guards the Python suites). A C++ suite
can be declared in ``tests/test_manifest.toml`` and carry a source row in ``tests/cpp/test_sources.cmake``
yet be registered in NO CMake list, so it never builds and never runs -- a silent coverage hole
(``test_coupling_operator_contract`` was exactly that: a valid header-only gtest, in the manifest and
test_sources.cmake, registered nowhere). This source-only fence makes that LOUD: it parses the
manifest's ``[[cpp.suite]]`` names and asserts each is registered in ``tests/CMakeLists.txt``, where
"registered" is one of:

* a bare word in the ``set(POPS_CPP_STANDARD_TESTS ...)`` list block;
* a bare word in the ``set(POPS_CPP_MPI_ONLY_TESTS ...)`` list block;
* an explicit ``pops_add_gtest_suite(NAME <name> ...)`` / ``pops_add_mpi_gtest_suite(<name> ...)`` /
  ``pops_add_test(<name>)`` call with a literal name.

The test reads the source tree only (tomllib + regex over CMake text); it imports neither ``pops``
nor ``_pops`` and builds nothing, so it runs in the CI architecture lane before the extension exists.
"""
import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "tests" / "test_manifest.toml"
CMAKELISTS = REPO_ROOT / "tests" / "CMakeLists.txt"
SOURCES_CMAKE = REPO_ROOT / "tests" / "cpp" / "test_sources.cmake"

_SET_BLOCK = re.compile(r"set\(\s*(POPS_CPP_STANDARD_TESTS|POPS_CPP_MPI_ONLY_TESTS)\b(.*?)\)",
                        re.DOTALL)
_MPI_VARIANT_BLOCK = re.compile(
    r"set\(\s*POPS_CPP_MPI_VARIANT_TESTS\b(.*?)\)", re.DOTALL
)
_MPI_RANKS = re.compile(r"set\(\s*POPS_MPI_RANKS_(test_[A-Za-z0-9_]+)\s+([0-9 ]+)\)")
_TEST_WORD = re.compile(r"\btest_[A-Za-z0-9_]+\b")
_ADD_GTEST = re.compile(r"pops_add_gtest_suite\(\s*NAME\s+(test_[A-Za-z0-9_]+)")
_ADD_MPI_GTEST = re.compile(r"pops_add_mpi_gtest_suite\(\s*(test_[A-Za-z0-9_]+)")
_ADD_TEST = re.compile(r"pops_add_test\(\s*(test_[A-Za-z0-9_]+)\s*\)")


def _cpp_suite_names():
    """The ordered ``[[cpp.suite]]`` names declared in the manifest."""
    data = tomllib.loads(MANIFEST.read_text())
    return [row["name"] for row in data.get("cpp", {}).get("suite", []) if row.get("name")]


def _registered_names():
    """Every C++ test name registered in ``tests/CMakeLists.txt`` (set blocks + explicit calls)."""
    text = CMAKELISTS.read_text()
    registered = set()
    for _var, body in _SET_BLOCK.findall(text):
        registered.update(_TEST_WORD.findall(body))
    registered.update(_ADD_GTEST.findall(text))
    registered.update(_ADD_MPI_GTEST.findall(text))
    registered.update(_ADD_TEST.findall(text))
    return registered


def _manifest_mpi_variants():
    rows = tomllib.loads(MANIFEST.read_text()).get("cpp", {}).get("suite", [])
    return {
        row["name"]: tuple(row["mpi_variants"])
        for row in rows
        if row.get("mpi_variants")
    }


def _cmake_mpi_variants():
    text = CMAKELISTS.read_text()
    match = _MPI_VARIANT_BLOCK.search(text)
    assert match is not None, "tests/CMakeLists.txt must declare POPS_CPP_MPI_VARIANT_TESTS"
    names = _TEST_WORD.findall(match.group(1))
    rank_rows = {
        name: tuple(int(rank) for rank in ranks.split())
        for name, ranks in _MPI_RANKS.findall(text)
    }
    missing_ranks = [name for name in names if name not in rank_rows]
    assert not missing_ranks, "MPI variant suites lack POPS_MPI_RANKS rows: %r" % missing_ranks
    return {name: rank_rows[name] for name in names}


def _source_of(name):
    """The source path recorded for ``name`` in test_sources.cmake (for an actionable message)."""
    m = re.search(r"POPS_CPP_TEST_SOURCE_%s\s+\"([^\"]+)\"" % re.escape(name),
                  SOURCES_CMAKE.read_text())
    return m.group(1) if m else "(no source row)"


def test_manifest_declares_cpp_suites():
    names = _cpp_suite_names()
    assert names, "tests/test_manifest.toml must declare [[cpp.suite]] rows"


def test_every_cpp_suite_is_registered_in_cmake():
    registered = _registered_names()
    manifest = set(_cpp_suite_names())
    missing = sorted(manifest - registered)
    unmanifested = sorted(registered - manifest)
    assert not missing, (
        "every [[cpp.suite]] in tests/test_manifest.toml must be registered in "
        "tests/CMakeLists.txt (STANDARD/MPI list or an explicit pops_add_gtest_suite call), "
        "else it never builds and never runs:\n  "
        + "\n  ".join("%s  (%s)" % (name, _source_of(name)) for name in missing))
    assert not unmanifested, (
        "every CMake-registered C++ test must have a [[cpp.suite]] row, else the "
        "manifest-driven shards silently omit it:\n  " + "\n  ".join(unmanifested)
    )


def test_manifest_mpi_variants_match_cmake_registrations_exactly():
    assert _cmake_mpi_variants() == _manifest_mpi_variants(), (
        "tests/test_manifest.toml mpi_variants is the CI build authority and must exactly match "
        "the CTest MPI launch registrations in tests/CMakeLists.txt"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
