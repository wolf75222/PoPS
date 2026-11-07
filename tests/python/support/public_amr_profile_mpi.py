"""Complete public AMR fixture inventory shared by independent MPI1/MPI2 drivers.

The ordinary native suites own genuine Serial execution of every parameter case.
These children retain authenticated MPI coverage and refuse any absent rank/case.
"""
from tests.python.support.mpi_fixture_driver import run_fixture_world
from tests.python.support.mpi_partition_units import parameter_partitions


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

# CI run 41037321460: the MPI2 launch of the whole tag_selection group took 828 s and that of
# stationary_regrids 761 s, against the 840-second child watchdog; each now runs as two
# parameter partitions.
_TAG = TESTS[0] + "::test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts"
_REGRID = TESTS[0] + "::test_stationary_regrids_do_not_redilate_retained_tag_buffer"
PARTITIONS = {
    **parameter_partitions({
        "tag_selection_linear": tuple(_TAG + "[linear-%d-shape%d]" % (buffer, shape)
                                      for buffer in (0, 1) for shape in (0, 1)),
        "tag_selection_injection": tuple(_TAG + "[injection-%d-shape%d]" % (buffer, shape)
                                         for buffer in (0, 1) for shape in (0, 1)),
    }),
    **parameter_partitions({
        "stationary_regrids_buffer": (_REGRID + "[0-False-1-2]", _REGRID + "[1-False-1-2]",
                                      _REGRID + "[1-True-1-2]"),
        "stationary_regrids_cadence_levels": (_REGRID + "[1-False-2-2]", _REGRID + "[1-False-1-3]"),
    }),
    "evolving_halo": (
        (TESTS[1] + "::test_public_evolving_accepted_halo_restart_and_refusal",),
        {"test_public_evolving_accepted_halo_restart_and_refusal": 2},
        None,
    ),
    "evolved_spatial": (
        (TESTS[2] + "::test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux",),
        {"test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux": 1},
        None,
    ),
    "evolved_spatial_v4": (
        (TESTS[3] + "::test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4",),
        {"test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux_v4": 2},
        None,
    ),
}


def run_public_amr_profile_world(*, ranks, directory, partition):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public AMR profile requires an explicit MPI1 or MPI2 world")
    if type(partition) is not str or partition not in PARTITIONS:
        raise ValueError("public AMR profile requires an exact declared partition")
    tests, expected_cases, units = PARTITIONS[partition]
    run_fixture_world(ranks=ranks, directory=directory, tests=tests,
                      expected_cases=expected_cases,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose="public AMR " + partition + " profile",
                      timeout_seconds=CHILD_TIMEOUT_SECONDS, partition_units=units)
