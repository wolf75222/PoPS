"""CF0 and FC6b (spec 0.3, section 16.1; Linear N2): the core paths, the native component tree and
the CF0 placement list.

``frozen_engine/cf0_paths.toml`` is the one CF0 list that every engine guard reads (CF0b, CF1, CF2,
CF7, CF8). ``frozen_engine/cf0_native_placement.toml`` names each native of IR24 that still sits in
a core path, with its issue, phase and interface; it only shrinks (MG6), keeps only Phase 2 entries
at the exit of Phase 1 and is empty at the freeze (FC6, FC8). Each entry keeps its b6 issue and
phase, and a native leaves the list only when it leaves the core paths. Every guard reads it through
``scan.load_placement``: this test, CF1 (``cf1.placed_natives``) and CF2 (the N9 path rule of
``cf2.py`` and the closed interface list).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
from tests.python.architecture.frozen_engine import scan

# CF0 at b6e64c1d, verbatim: include/, src/, python/bindings/, python/pops/ except python/pops/lib/,
# cmake/, schemas/, the root CMakeLists.txt, python/CMakeLists.txt, src/CMakeLists.txt and
# pyproject.toml. Changing the core paths is a change of spec 0.3 and of the frozen engine.
CF0_CORE = ("include/", "src/", "python/bindings/", "python/pops/", "cmake/", "schemas/",
            "CMakeLists.txt", "python/CMakeLists.txt", "src/CMakeLists.txt", "pyproject.toml")
CF0_EXCLUDED = ("python/pops/lib/",)

# The b6 placement list (IR24, adjudication section 17), verbatim: id -> (issue, phase, files).
# The list only shrinks (MG6): never add an id or a file here; an issue or a phase changes only
# with spec 0.3.
B6_PLACEMENT = {
    "face-loop-stencil-flux": ("N3", 1, ("include/pops/numerics/fv/stencil_flux.hpp",)),
    "face-loop-generated-system-block": ("N3", 1, (
        "include/pops/runtime/builders/compiled/generated_system_block.hpp",
    )),
    "face-loop-generated-amr-system-block": ("N3", 1, (
        "include/pops/runtime/builders/compiled/generated_amr_system_block.hpp",
    )),
    "solver-krylov": ("N9", 2, (
        "include/pops/numerics/elliptic/linear/generic_krylov.hpp",
        "include/pops/numerics/elliptic/linear/krylov_method_provider.hpp",
        "include/pops/numerics/elliptic/linear/krylov_workspace.hpp",
        "include/pops/runtime/amr/tensor_coarse_gmres.hpp",
    )),
    "solver-newton-krylov": ("N9", 2, (
        "include/pops/numerics/elliptic/interface/field_newton_krylov.hpp",
        "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp",
    )),
    "solver-multigrid": ("N9", 2, (
        "include/pops/numerics/elliptic/mg/geometric_mg.hpp",
        "include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp",
        "include/pops/numerics/elliptic/mg/composite_fac_poisson.hpp",
        "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp",
    )),
    "solver-preconditioner": ("N9", 2, (
        "include/pops/numerics/elliptic/mg/tensor_diagonal_preconditioner.hpp",
    )),
    "solver-fft": ("N9", 2, (
        "include/pops/numerics/elliptic/poisson/poisson_fft.hpp",
        "include/pops/numerics/elliptic/poisson/poisson_fft_1d_internal.hpp",
        "include/pops/numerics/elliptic/poisson/poisson_fft_device_kernels.hpp",
        "include/pops/numerics/elliptic/poisson/poisson_fft_multifab.hpp",
        "include/pops/numerics/elliptic/poisson/poisson_fft_solver.hpp",
    )),
    "solver-polar-poisson": ("N9", 2, (
        "include/pops/numerics/elliptic/polar/polar_poisson_solver.hpp",
        "include/pops/numerics/elliptic/polar/prepared_polar_poisson_inverse.hpp",
    )),
    "solver-cartesian-poisson": ("N9", 2, (
        "include/pops/numerics/elliptic/nd/cartesian_poisson.hpp",
        "include/pops/numerics/elliptic/poisson/poisson_operator.hpp",
    )),
    "solver-dense-residual-basis": ("N9", 2, (
        "include/pops/numerics/elliptic/interface/full_residual_basis_lu.hpp",
    )),
    "solver-amr-tensor": ("N9", 2, (
        "include/pops/runtime/amr/amr_tensor_elliptic.hpp",
        "include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp",
        "include/pops/runtime/amr/tensor_composite_fac.hpp",
    )),
    "direct-sum-spatial-interaction": ("G05", 2, (
        "include/pops/runtime/program/spatial_direct_interaction.hpp",
    )),
    "path-conservative-rusanov": ("G06", 2, ("include/pops/numerics/fv/path_flux.hpp",)),
    "level-transfer-reflux-ledger": ("G06", 2, (
        "include/pops/amr/reflux/face_flux_ledger.hpp",
        "include/pops/amr/reflux/metric_reflux.hpp",
        "include/pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp",
        "include/pops/numerics/time/amr/reflux/amr_reflux_mf.hpp",
    )),
    "level-transfer-restriction": ("G06", 2, ("include/pops/amr/transfer/transfer_provider.hpp",)),
}
# IR24: the issue and phase of each class of native.
IR24_ISSUE_PHASE = {"N3": 1, "N9": 2, "G05": 2, "G06": 2}


def _natives():
    return scan.load_placement()


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
    assert set(ids) <= set(B6_PLACEMENT), (
        "MG6: the CF0 placement list only shrinks; new entries are refused: %s"
        % sorted(set(ids) - set(B6_PLACEMENT)))


def test_placement_entries_keep_their_b6_issue_phase_and_files():
    wrong = []
    for row in _natives():
        if row["id"] not in B6_PLACEMENT:
            continue  # test_placement_list_only_shrinks
        issue, phase, files = B6_PLACEMENT[row["id"]]
        if (row["issue"], row["phase"]) != (issue, phase):
            wrong.append("%s: %s, Phase %s instead of %s, Phase %d (IR24)"
                         % (row["id"], row["issue"], row["phase"], issue, phase))
        wrong.extend("%s: %s is not a file of this native at b6" % (row["id"], rel)
                     for rel in row["files"] if rel not in files)
    assert not wrong, (
        "MG6: a CF0 placement entry keeps its b6 issue and phase and only loses files:\n  "
        + "\n  ".join(wrong))


def test_a_native_leaves_the_list_only_when_it_leaves_the_core_paths():
    """FC6: until a native moves to the native component tree or, for a named method, is removed,
    the placement list names it."""
    engine = scan.load_cf0()
    tracked = set(scan.tracked_files())
    listed = {(row["id"], rel) for row in _natives() for rel in row["files"]}
    dropped = ["%s: %s" % (native, rel) for native, (_, _, files) in sorted(B6_PLACEMENT.items())
               for rel in files
               if (native, rel) not in listed and rel in tracked and engine.is_core(rel)]
    assert not dropped, (
        "FC6: these natives left cf0_native_placement.toml but still sit in core paths:\n  "
        + "\n  ".join(dropped))


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
