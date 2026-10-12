"""Mandatory initial Field/Ghost positive and fault proofs on real MPI1 and MPI2.

Each rank must run both public fixtures successfully without a skipped outcome.
The same fixtures run in the ordinary Serial suite using no MPI communicator.
"""
from pathlib import Path

import pytest

from tests.python.support.mpi_fixture_driver import (
    require_rank_results as _require_rank_results, run_fixture_world,
)


pytestmark = pytest.mark.mpi
TESTS = (
    "tests/python/integration/amr/test_public_initial_field_ghost.py",
    "tests/python/integration/amr/test_public_initial_field_ghost_failure.py",
)
EXPECTED_CASES = {
    "test_public_initial_field_fresh_before_ghost_and_positive_point": 1,
    "test_public_initial_ghost_rank_fault_keeps_prepublication_owner": 1,
}


def require_rank_results(directory: Path, ranks: int) -> None:
    _require_rank_results(directory, ranks, EXPECTED_CASES, purpose="initial Field/Ghost")


@pytest.mark.parametrize("ranks", (1, 2))
def test_initial_field_and_ghost_fault_on_authenticated_mpi_worlds(tmp_path, ranks):
    run_fixture_world(ranks=ranks, directory=tmp_path / ("mpi%d" % ranks), tests=TESTS,
                      expected_cases=EXPECTED_CASES,
                      expected_rank_env="POPS_TEST_INITIAL_GHOST_EXPECT_RANKS",
                      purpose="initial Field/Ghost")
