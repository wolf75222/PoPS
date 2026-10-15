"""The archive resource authority covers the complete native rank table exactly."""

import sys

import pytest

from pops.runtime._checkpoint_resource_budget import (
    _accepted_program_archive_capacity,
    _checkpoint_member_names,
)


@pytest.mark.parametrize("ranks", (1, 2, 3, 17))
def test_ranked_program_archive_members_cover_every_native_rank(ranks):
    names = _checkpoint_member_names(
        runtime_kind="amr", block_names=("q",), field_names=(), history_names=(),
        cache_names=(), levels=2, rank_capacity=ranks,
    )
    assert len(names) == len(set(names))
    assert {name for name in names if name.startswith("program_accepted_state_rank_")} == {
        "program_accepted_state_rank_count",
        *("program_accepted_state_rank_%d" % rank for rank in range(ranks)),
    }
    assert "program_accepted_state" in names
    assert "program_accepted_state_source_authority" in names
    assert _accepted_program_archive_capacity(1234, ranks, ranked=True) == (ranks + 1) * 1234 + 8


def test_ranked_program_capacity_keeps_empty_image_count_and_uniform_schema():
    assert _accepted_program_archive_capacity(0, 3, ranked=True) == 8
    assert _accepted_program_archive_capacity(1234, 3, ranked=False) == 1234
    names = _checkpoint_member_names(
        runtime_kind="uniform", block_names=("q",), field_names=(), history_names=(),
        cache_names=(), levels=1, rank_capacity=3,
    )
    assert not any(name.startswith("program_accepted_state_rank_") for name in names)


def test_ranked_program_capacity_checks_native_address_space_before_multiplication():
    with pytest.raises(OverflowError):
        _accepted_program_archive_capacity(sys.maxsize, 2, ranked=True)
    with pytest.raises((TypeError, ValueError)):
        _accepted_program_archive_capacity(1024, True, ranked=True)
