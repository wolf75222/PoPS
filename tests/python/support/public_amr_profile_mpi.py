"""Complete public AMR fixture inventory shared by independent MPI1/MPI2 drivers.

The ordinary native suites own genuine Serial execution of every parameter case.
These children retain authenticated MPI coverage and refuse any absent rank/case.
"""
from tests.python.support.mpi_fixture_driver import run_fixture_world


CHILD_TIMEOUT_SECONDS = 14 * 60
TESTS = (
    "tests/python/integration/amr/test_public_tag_selection_native.py",
    "tests/python/integration/amr/test_public_evolving_accepted_halo.py",
    "tests/python/integration/runtime/test_public_evolved_stage_amr_spatial.py",
    "tests/python/integration/runtime/test_public_evolved_stage_amr_spatial_v4.py",
)
EXPECTED_CASES = {
    "test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts": 8,
    "test_stationary_regrids_do_not_redilate_retained_tag_buffer": 5,
    "test_public_evolving_accepted_halo_restart_and_refusal": 2,
    "test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux": 1,
    "test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4": 2,
}


def run_public_amr_profile_world(*, ranks, directory):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public AMR profile requires an explicit MPI1 or MPI2 world")
    run_fixture_world(ranks=ranks, directory=directory, tests=TESTS,
                      expected_cases=EXPECTED_CASES,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose="public AMR profile",
                      timeout_seconds=CHILD_TIMEOUT_SECONDS)
