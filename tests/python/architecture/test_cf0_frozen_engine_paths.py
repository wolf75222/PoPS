"""CF0 and FC6b (spec 0.3, section 16.1; Linear N2): the core paths, the native component tree and
the CF0 placement list.

``frozen_engine/cf0_paths.toml`` is the one CF0 list that every engine guard reads (CF0b, CF1, CF2,
CF7, CF8). ``frozen_engine/cf0_native_placement.toml`` names each native of IR24 that still sits in
a core path, with its issue, phase and interface; it only shrinks (MG6), keeps only Phase 2 entries
at the exit of Phase 1 and is empty at the freeze (FC6, FC8).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import tomllib

from tests.python.architecture.frozen_engine import scan

PLACEMENT = scan.HERE / "cf0_native_placement.toml"

# CF0 at b6e64c1d, verbatim: include/, src/, python/bindings/, python/pops/ except python/pops/lib/,
# cmake/, schemas/, the root CMakeLists.txt, python/CMakeLists.txt, src/CMakeLists.txt and
# pyproject.toml. Changing the core paths is a change of spec 0.3 and of the frozen engine.
CF0_CORE = ("include/", "src/", "python/bindings/", "python/pops/", "cmake/", "schemas/",
            "CMakeLists.txt", "python/CMakeLists.txt", "src/CMakeLists.txt", "pyproject.toml")
CF0_EXCLUDED = ("python/pops/lib/",)

# The ids of the b6 placement list. The list only shrinks (MG6); never add an id here.
B6_PLACEMENT_IDS = frozenset({
    "face-loop-stencil-flux", "face-loop-generated-system-block",
    "face-loop-generated-amr-system-block", "solver-krylov", "solver-newton-krylov",
    "solver-multigrid", "solver-preconditioner", "solver-fft", "solver-polar-poisson",
    "solver-cartesian-poisson", "solver-dense-residual-basis", "solver-amr-tensor",
    "direct-sum-spatial-interaction", "path-conservative-rusanov", "level-transfer-reflux-ledger",
    "level-transfer-restriction",
})
# IR24: the issue and phase of each class of native.
IR24_ISSUE_PHASE = {"N3": 1, "N9": 2, "G05": 2, "G06": 2}


def _natives():
    return tomllib.loads(PLACEMENT.read_text(encoding="utf-8"))["native"]


def _line_numbers(spec):
    for item in spec:
        if isinstance(item, int):
            yield item
        else:
            first, last = (int(part) for part in item.split("-"))
            assert first <= last, "line range %r is reversed" % item
            yield from range(first, last + 1)


def test_cf0_lists_the_core_paths_of_the_spec():
    engine = scan.load_cf0()
    assert engine.core_include == CF0_CORE, "cf0_paths.toml [core] include differs from CF0"
    assert engine.core_exclude == CF0_EXCLUDED, "cf0_paths.toml [core] exclude differs from CF0"


def test_cf0_entries_exist_in_the_tree():
    engine = scan.load_cf0()
    tracked = scan.tracked_files()
    missing = [entry for entry in engine.core_include + engine.core_exclude
               if not any(scan.covered(rel, (entry,)) for rel in tracked)]
    assert not missing, "CF0 entries that match no tracked file: %s" % missing


def test_native_tree_is_outside_the_core_paths():
    engine = scan.load_cf0()
    assert engine.native_tree.endswith("/"), "the native component tree is a directory"
    probe = engine.native_tree + "probe.hpp"
    assert not engine.is_core(probe), (
        "FC6b: the native component tree is outside the core paths (judgment D1)")
    assert engine.is_frozen(probe), "FC6b: the native component tree is inside the frozen engine"
    assert not any(entry.startswith(engine.native_tree) for entry in engine.core_include), (
        "a core path lies under the native component tree")


def test_frozen_engine_classification():
    engine = scan.load_cf0()
    assert engine.is_core("python/pops/codegen/module_emit_brick.py")
    assert not engine.is_frozen("python/pops/lib/time/ssprk.py")
    assert not engine.is_frozen("tests/python/architecture/test_cf2_architecture.py")
    assert not engine.is_frozen("examples/api_v02/models_M01_M14.py")
    assert engine.is_core("src/CMakeLists.txt") and engine.is_core("CMakeLists.txt")
    assert not engine.is_frozen("tests/CMakeLists.txt")


def test_placement_list_only_shrinks():
    ids = [row["id"] for row in _natives()]
    assert len(ids) == len(set(ids)), "duplicate ids in cf0_native_placement.toml"
    assert set(ids) <= B6_PLACEMENT_IDS, (
        "MG6: the CF0 placement list only shrinks; new entries are refused: %s"
        % sorted(set(ids) - B6_PLACEMENT_IDS))


def test_placement_entries_name_files_still_in_core_paths():
    engine = scan.load_cf0()
    tracked = set(scan.tracked_files())
    stale, outside = [], []
    for row in _natives():
        for rel in row["files"]:
            if rel not in tracked:
                stale.append("%s: %s is not in the tree" % (row["id"], rel))
            elif not engine.is_core(rel):
                outside.append("%s: %s is no longer in a core path" % (row["id"], rel))
    assert not stale and not outside, (
        "MG6: a native that moved to the native component tree, or left, removes its entry from "
        "cf0_native_placement.toml in the same PR:\n  " + "\n  ".join(stale + outside))


def test_placement_entries_name_issue_phase_and_existing_lines():
    wrong = []
    for row in _natives():
        if IR24_ISSUE_PHASE.get(row["issue"]) != row["phase"]:
            wrong.append("%s: issue %s and phase %s do not match IR24"
                         % (row["id"], row["issue"], row["phase"]))
        if not row.get("what"):
            wrong.append("%s: says nothing of what the native is" % row["id"])
        lines = list(_line_numbers(row.get("lines", ())))
        if lines:
            length = len((scan.REPO_ROOT / row["files"][0]).read_text(encoding="utf-8").splitlines())
            wrong.extend("%s: line %d is past the end of %s" % (row["id"], line, row["files"][0])
                         for line in lines if not 1 <= line <= length)
    assert not wrong, "\n".join(wrong)
