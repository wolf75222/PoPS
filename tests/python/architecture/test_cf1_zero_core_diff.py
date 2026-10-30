"""CF1 (spec 0.3, section 16.2; FC6b, FC8; Linear N2): adding a case changes no path of the frozen
engine.

For every case of ``frozen_engine/cases.toml`` (tried first on M01), the test builds the tree that
lacks the case (the integration head without the case's files), adds the case's files on top, and
requires the tree hashes of the core paths and of the native component tree (``cf0_paths.toml``)
to be those of the integration head: ``git diff <base>..<case> -- <core paths> <native tree>`` is
empty. The controls show that a case that brings a core file, a native file, a change to a core
file or the deletion of one fails, and that a case that brings files of the library, of
``examples/`` or of ``tests/`` passes. A case branch is checked the same way with
``python -m tests.python.architecture.frozen_engine.cf1 --ref <commit>``.

Trees are written to a temporary object directory that borrows the repository's objects (git
plumbing only), so the test writes nothing into the repository. It does not import ``pops``.
"""
import pytest

from tests.python.architecture.frozen_engine import cf1, cf8, scan

# The cases that CF1 is tried on in Phase 0 (Linear N2). Cases are added (N6, the case issues);
# never removed while the case exists.
B6_CASES = {"M01"}
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


@pytest.fixture(scope="module")
def store():
    with cf1.ObjectStore() as objects:
        yield objects


@pytest.fixture(scope="module")
def head_hashes(store):
    return cf1.frozen_hashes(store, store.entries(store.tree_of("HEAD")))


def _cases():
    return {case["id"]: case for case in cf8.load_cases()}


# ---------------------------------------------------------------------------------------------
# The cases.
# ---------------------------------------------------------------------------------------------
def test_manifest_names_the_phase_0_cases():
    cases = _cases()
    assert B6_CASES <= set(cases), "CF1 cases left cases.toml: %s" % sorted(B6_CASES - set(cases))
    for case in cases.values():
        assert case.get("issue") and case.get("what") and case.get("files"), case


@pytest.mark.parametrize("case_id", sorted(_cases()))
def test_case_files_exist_outside_the_frozen_engine(case_id):
    engine = scan.load_cf0()
    tracked = set(scan.tracked_files())
    files = _cases()[case_id]["files"]
    assert len(files) == len(set(files)), "%s lists a file twice" % case_id
    missing = [rel for rel in files if rel not in tracked]
    frozen = [rel for rel in files if engine.is_frozen(rel)]
    assert not missing, "%s names files that are not tracked: %s" % (case_id, missing)
    assert not frozen, "%s owns files of the frozen engine (CF1, FC6b): %s" % (case_id, frozen)


@pytest.mark.parametrize("case_id", sorted(_cases()))
def test_adding_the_case_leaves_both_trees_unchanged(store, head_hashes, case_id):
    check = cf1.check_case(_cases()[case_id], store=store)
    assert check.unchanged, check.describe(case_id)
    assert check.with_case == check.without_case == head_hashes, check.describe(case_id)


def test_native_tree_is_empty_at_b6_and_hashed_as_the_empty_tree(head_hashes):
    engine = scan.load_cf0()
    if not any(engine.is_native(rel) for rel in scan.tracked_files()):
        assert head_hashes.native == EMPTY_TREE
    assert head_hashes.core != EMPTY_TREE


def test_core_hash_is_the_tree_of_exactly_the_core_files(store, head_hashes):
    engine = scan.load_cf0()
    head = store.entries(store.tree_of("HEAD"))
    core = store.entries(head_hashes.core)
    assert set(core) == {rel for rel in head if engine.is_core(rel)}
    assert all(core[rel] == head[rel] for rel in core)


# ---------------------------------------------------------------------------------------------
# Controls: a case that touches the frozen engine fails; one that stays outside passes.
# ---------------------------------------------------------------------------------------------
def _probe(store, changes):
    blob = store.blob(b"// a file written by the CF1 control\n")
    return cf1.apply_case(store, "HEAD", {rel: (None if action == "delete" else ("100644", blob))
                                          for rel, action in changes.items()}, lacking=True)


@pytest.mark.parametrize("changes, frozen_part", [
    ({"examples/cf1/case.py": "add", "include/pops/cf1_probe.hpp": "add"}, "core"),
    ({"examples/cf1/case.py": "add", "python/pops/__init__.py": "change"}, "core"),
    ({"examples/cf1/case.py": "add", "pyproject.toml": "change"}, "core"),
    ({"examples/cf1/case.py": "add", "native/riemann/flux.hpp": "add"}, "native"),
])
def test_a_case_that_touches_the_frozen_engine_fails(store, head_hashes, changes, frozen_part):
    check = _probe(store, changes)
    assert not check.unchanged
    assert getattr(check.with_case, frozen_part) != getattr(head_hashes, frozen_part)
    frozen = sorted(rel for rel in changes if rel != "examples/cf1/case.py")
    assert check.frozen_paths == tuple(frozen)


def test_a_case_that_deletes_a_core_file_fails(store):
    check = _probe(store, {"examples/cf1/case.py": "add", "src/CMakeLists.txt": "delete"})
    assert not check.unchanged and check.frozen_paths == ("src/CMakeLists.txt",)


def test_a_case_outside_the_frozen_engine_passes(store, head_hashes):
    check = _probe(store, {"examples/cf1/case.py": "add", "python/pops/lib/cf1_probe.py": "add",
                           "tests/python/cf1_probe_test.py": "add", "docs/cf1.md": "add"})
    assert check.unchanged, check.describe("outside")
    assert check.with_case == head_hashes


def test_a_case_branch_is_checked_with_the_files_it_changes(store, head_hashes):
    head = store.git("rev-parse", "HEAD").decode().strip()
    entries = store.entries(store.tree_of("HEAD"))
    blob = store.blob(b"# a case written by the CF1 control\n")
    clean = dict(entries, **{"examples/cf1/case.py": ("100644", blob)})
    clean_commit = store.commit(store.write_tree(clean), head)
    check = cf1.check_ref(clean_commit, base=head, store=store)
    assert check.unchanged and check.case_paths == ("examples/cf1/case.py",), check.describe("ref")

    touching = dict(clean, **{"include/pops/cf1_probe.hpp": ("100644", blob)})
    touching_commit = store.commit(store.write_tree(touching), clean_commit)
    check = cf1.check_ref(touching_commit, base=head, store=store)
    assert not check.unchanged and check.frozen_paths == ("include/pops/cf1_probe.hpp",)
    assert check.with_case.core != head_hashes.core


def test_the_check_writes_nothing_into_the_repository(store):
    before = scan.tracked_files()
    _probe(store, {"examples/cf1/case.py": "add"})
    assert scan.tracked_files() == before
    status = store.git("status", "--porcelain", "--untracked-files=no", own=False).decode()
    assert "examples/cf1" not in status


def test_command_line_reports_every_case(capsys):
    assert cf1.main(["--all"]) == 0
    out = capsys.readouterr().out
    for case_id in _cases():
        assert "CF1 %s: frozen engine unchanged" % case_id in out
