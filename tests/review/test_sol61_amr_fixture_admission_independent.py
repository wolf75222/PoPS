"""Actual public authoring/admission plus math controls; no native or JIT execution."""
import ast
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime import test_public_amr_original_field as public


ROOT = Path(__file__).resolve().parents[2]
FILE = "tests/python/integration/runtime/test_public_amr_original_field.py"
FROZEN = "b943dd6627550364337d41743cb4efe2f7b33480"
PRIOR = "a2f7a2a9902de5ec21571efac7de401ea5f13167"


def source(commit):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + FILE], text=True)


def builder(commit):
    tree = ast.parse(source(commit))
    function = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == "build")
    env = public.__dict__.copy()
    exec(compile(ast.Module(body=[function], type_ignores=[]), "exact frozen public builder", "exec"), env)
    return env["build"]


def assert_source_only():
    assert Path(pops.__file__).resolve() == ROOT / "python/pops/__init__.py"
    assert not any(name == "pops._bootstrap" or name == "pops._pops" or name.endswith("._pops")
                   for name in sys.modules)


def test_layout_only_delta_does_not_change_original_body_controls_or_controller():
    assert (ROOT / FILE).read_text() == source(FROZEN)
    old, new = ast.parse(source(PRIOR)), ast.parse(source(FROZEN))
    old_build = next(row for row in old.body if isinstance(row, ast.FunctionDef) and row.name == "build")
    new_build = copy.deepcopy(next(row for row in new.body if isinstance(row, ast.FunctionDef) and row.name == "build"))
    for index, row in enumerate(new_build.body):
        if isinstance(row, ast.Assign) and isinstance(row.targets[0], ast.Name) and row.targets[0].id in ("threshold", "layout"):
            name = row.targets[0].id
            new_build.body[index] = copy.deepcopy(next(item for item in old_build.body
                                                       if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                                                       and item.targets[0].id == name))
    assert ast.dump(old_build, include_attributes=False) == ast.dump(new_build, include_attributes=False)
    assert public.DT == .01 and public.TOL == 3e-8
    assert np.array_equal(public.TARGET, (.15, .25, .18))
    assert np.array_equal(public.DIFFUSION, ((.04, .006, 0.), (-.003, .05, 0.), (0., 0., .03)))
    for name in ("capture", "check_original_saved", "test_public_original_amr_published_fields_parent_rollback_and_retry"):
        prior = next(row for row in old.body if isinstance(row, ast.FunctionDef) and row.name == name)
        current = next(row for row in new.body if isinstance(row, ast.FunctionDef) and row.name == name)
        assert ast.dump(prior, include_attributes=False) == ast.dump(current, include_attributes=False)


@pytest.mark.parametrize("cells,order,seed,guarded", [
    (16, (0,), False, False), (16, (0, 1, 2), False, False),
    (32, (2, 0, 1), True, False), (16, (2, 0, 1), False, True),
])
def test_exact_public_jacobi_case_is_really_validated_resolved_and_emitted(cells, order, seed, guarded):
    assert_source_only()
    case, layout = public.build(cells, order, seed=seed, guarded=guarded,
                                right_preconditioner="SpatialBasisJacobi@1")
    frozen = pops.validate(case)
    resolved = pops.resolve(frozen, layout=layout)
    resolved.verify()
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert cpp.count(", pops::runtime::program::AmrFieldRightPreconditioner::kSpatialBasisJacobi") == 1
    assert cpp.count("_core->solve(") == 1
    assert "stage_original_field_candidate_collectively" in cpp
    assert "publish_staged_field_components" in cpp
    assert "original_hierarchy_field_authority" in cpp
    assert "block_inverse" not in cpp and "condensed" not in cpp
    assert resolved.resolved_dimension == 2
    assert resolved.time._serialize()["version"] == 9
    assert layout.clustering.maximum_box_size == 8
    if seed:
        assert "auto& seed = ctx.hierarchy_field_scratch(" in cpp
    if guarded:
        assert "response_limit" in cpp and "StepAttemptRejected" in cpp
    assert_source_only()


def test_historical_lt_builder_is_refused_at_public_admission_before_emission():
    assert_source_only()
    case, layout = builder(PRIOR)(16, (0,), right_preconditioner="SpatialBasisJacobi@1")
    with pytest.raises(ValueError, match=r"^AMR refine value rule requires strict > threshold$"):
        pops.resolve(pops.validate(case), layout=layout)
    assert_source_only()


@pytest.mark.parametrize("cells", [16, 32])
def test_exact_average_extrema_bands_leave_untagged_interior_but_clustering_is_unreceived(cells):
    x = (np.arange(cells) + .5) / cells
    average = 1 + .04*np.sinc(1/cells)*np.cos(2*np.pi*x)
    tagged = average > 1.035
    buffered = tagged | np.roll(tagged, 1) | np.roll(tagged, -1)
    assert np.any(tagged) and not np.all(buffered)
    assert tagged[0] and tagged[-1] and buffered[0] and buffered[-1]
    assert np.array_equal(tagged, tagged[::-1]) and np.array_equal(buffered, buffered[::-1])
    assert not buffered[cells//2]
    assert np.min(np.abs(average-1.035)) > 1e-4
    # Bands meet the periodic seam. Actual max-box8 clustering must still be
    # observed natively; this arithmetic is not a manufactured active mask.


def test_public_default_keeps_legacy_realization_without_selected_jacobi():
    assert_source_only()
    case, layout = public.build(16, (0,))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert "AmrFieldRightPreconditioner::kSpatialBasisJacobi" not in cpp
    assert resolved.time._serialize()["version"] == 8
    assert_source_only()
