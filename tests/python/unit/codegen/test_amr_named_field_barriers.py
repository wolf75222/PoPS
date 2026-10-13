"""Common-time named field stages must enter the composite continuation scheduler."""
from types import SimpleNamespace

import pytest

from pops.codegen.program_emit_hierarchy_regions import (
    has_hierarchy_continuations, hierarchy_named_field_barriers, hierarchy_region_solves,
)


def node(identity, operation, *, region=0, **attrs):
    return SimpleNamespace(id=identity, op=operation, region=region, attrs=attrs)


def test_three_named_stages_and_warm_seed_are_ordered_hierarchy_barriers():
    gauss0 = node(0, "solve_fields")
    h1 = node(1, "linear_combine")
    gauss1 = node(2, "solve_fields_from_blocks")
    source = node(3, "solve_linear", scope="hierarchy")
    warm = node(4, "field_warm_start")
    gauss2 = node(5, "solve_fields")
    program = SimpleNamespace(_values=[gauss0, h1, gauss1, source, warm, gauss2])
    assert hierarchy_named_field_barriers(program) == (gauss0, gauss1, warm, gauss2)
    assert hierarchy_region_solves(program) == (source,)
    assert has_hierarchy_continuations(program)


@pytest.mark.parametrize("operation", ["solve_fields", "solve_fields_from_blocks", "field_warm_start"])
def test_named_field_stage_cannot_hide_inside_a_level_local_branch(operation):
    inner = node(1, operation, region=1)
    program = SimpleNamespace(_values=[node(0, "branch", true_block=[inner])])
    with pytest.raises(NotImplementedError, match="unscheduled top-level"):
        hierarchy_named_field_barriers(program)


def test_scheduled_named_solve_cannot_leave_hierarchy_levels_at_different_barriers():
    program = SimpleNamespace(_values=[node(0, "solve_fields", schedule={"every": 2})])
    with pytest.raises(NotImplementedError, match="unscheduled top-level"):
        hierarchy_named_field_barriers(program)


def test_multiple_legacy_singleton_providers_cannot_share_named_field_barriers():
    program = SimpleNamespace(_values=[node(0, "solve_fields"),
        node(1, "solve_linear", scope="hierarchy"), node(2, "solve_linear", scope="hierarchy")])
    with pytest.raises(NotImplementedError, match="one top-level singleton"):
        hierarchy_region_solves(program)
