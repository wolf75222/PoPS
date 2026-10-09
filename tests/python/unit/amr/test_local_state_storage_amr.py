"""AMR resolution of a Program-owned state with no physical face flux."""

from dataclasses import replace

import pytest
import pops

from pops.amr._resolution import ResolvedAMRStateStorage
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.mesh._amr._transfer_contracts import COARSE_FINE_FILL
from pops.numerics import StateStorage
from tests.python.support.affine_push_forward_amr_case import (
    CELLS, REFINE_DENSITY_LOWER, REFINE_DENSITY_UPPER, affine_amr_case,
    expected_level_cell_averages, mapped_particle_moments, raw_particle_moments,
)


@pytest.mark.parametrize("levels", (1, 2))
def test_local_affine_product_resolves_full_amr_authorities(levels):
    case, layout, subject = affine_amr_case(levels=levels)
    plan = pops.resolve(pops.validate(case), layout=layout)

    assert len(plan.blocks) == 1
    block, = plan.blocks
    assert block.numerics is None
    assert type(block.spatial) is StateStorage
    assert block.state_identities == (case.resolve(subject).qualified_id,)
    assert plan.resolved_hierarchy.plan.level_count == levels
    nesting = plan.resolved_hierarchy.plan.nesting
    assert nesting.stencil.minimum_buffer == (block.spatial.ghost_depth,) * 2
    assert nesting.reflux.minimum_buffer == (0, 0)
    assert nesting.reflux.minimum_lookahead == 0
    assert plan.initial_condition_plan is not None
    assert plan.bootstrap_plan is not None
    if levels == 2:
        fill = plan.amr_transfer.for_subject(case.resolve(subject), COARSE_FINE_FILL)
        assert fill.requirements
        assert all(row.accuracy.ghost_depth == (block.spatial.ghost_depth,) * 2
                   for row in fill.requirements)

    source = emit_cpp_program(
        plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks),
        target="amr_system",
    )
    assert "prepare_local_nonlinear_problem" in source
    assert "solve" in source


def test_local_storage_authority_rejects_another_resolved_state():
    case, layout, subject = affine_amr_case(levels=1)
    plan = pops.resolve(pops.validate(case), layout=layout)
    other_case, _, other_subject = affine_amr_case(
        levels=1, case_name="another_affine_amr_public_body")
    other_case.validate()
    with pytest.raises(ValueError, match="exact resolved block state"):
        ResolvedAMRStateStorage(plan.blocks[0], other_case.resolve(other_subject))


def test_impossible_original_residual_still_resolves_and_emits():
    case, layout, _ = affine_amr_case(levels=2, impossible_residual=True)
    plan = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(
        plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks),
        target="amr_system",
    )
    assert "prepare_local_nonlinear_problem" in source


def test_independent_particle_oracle_has_exact_amr_cell_axes():
    mapped = mapped_particle_moments()
    for level in (0, 1):
        width = CELLS * 2**level
        expected = expected_level_cell_averages(mapped, level)
        assert expected.shape == (len(mapped), width, width)
        assert expected[0, 0, 0] == pytest.approx(1.0 + .6 * .5 / width)
        assert expected[0, -1, -1] == pytest.approx(1.0 + .6 * (width - .5) / width)


def test_authored_density_band_selects_interior_coarse_columns():
    initial = expected_level_cell_averages(raw_particle_moments(), 0)
    tagged_columns = tuple(i for i, value in enumerate(initial[0, 0])
                           if REFINE_DENSITY_LOWER < value <= REFINE_DENSITY_UPPER)
    assert tagged_columns == (7, 8)
    assert min(tagged_columns) > 0 and max(tagged_columns) < CELLS - 1


def test_transfer_does_not_reduce_a_deeper_exact_storage_halo():
    case, layout, subject = affine_amr_case(levels=2)
    plan = pops.resolve(pops.validate(case), layout=layout)
    qualified = case.resolve(subject)
    deeper = ResolvedAMRStateStorage(
        replace(plan.blocks[0], spatial=StateStorage(ghost_depth=3)), qualified)
    projection = plan.layout_plan.project(plan.layout_plan.layouts[0].handle)
    resolved = layout.transfer.resolve_references(case.resolve).resolve(
        projection, (), state_storage=(deeper,))
    fill = resolved.for_subject(qualified, COARSE_FINE_FILL)
    assert resolved.nesting_requirement.minimum_buffer == (3, 3)
    assert all(row.accuracy.ghost_depth == (3, 3) for row in fill.requirements)
