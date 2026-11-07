"""Exact frozen Field fixture groups for independently bounded MPI1/MPI2 worlds.

``amr_original`` and ``spatial_interaction`` run as declared partitions sized by measured cost,
so one 14-minute child watchdog covers one partition (CI run 41037321460: each whole group
stopped at that watchdog).
"""
from tests.python.support.mpi_fixture_driver import run_fixture_world
from tests.python.support.mpi_partition_units import parameter_partitions


CHILD_TIMEOUT_SECONDS = 14 * 60
GROUPS = {
    "spatial_interaction": (
        (
            "tests/python/integration/runtime/test_public_spatial_interaction.py::test_public_spatial_interaction_saved_history_and_exact_replay",
            "tests/python/integration/runtime/test_public_spatial_interaction.py::test_public_spatial_interaction_refuses_without_publication",
            "tests/python/integration/runtime/test_public_spatial_interaction.py::test_public_spatial_interaction_ignores_inactive_source_pole",
            "tests/python/integration/runtime/test_public_spatial_interaction.py::test_public_spatial_interaction_nonperiodic_amr_refuses_at_bind",
        ),
        {"test_public_spatial_interaction_saved_history_and_exact_replay": 3,
         "test_public_spatial_interaction_refuses_without_publication": 3,
         "test_public_spatial_interaction_ignores_inactive_source_pole": 1,
         "test_public_spatial_interaction_nonperiodic_amr_refuses_at_bind": 1},
    ),
    "completed_original": (
        (
            "tests/python/integration/runtime/test_public_completed_original_interaction.py::test_public_completed_original_interaction_exact_restart_and_saved_reductions",
        ),
        {"test_public_completed_original_interaction_exact_restart_and_saved_reductions":2},
    ),
    "amr_original": (
        (
            "tests/python/integration/runtime/test_public_amr_original_field.py::test_public_original_amr_saved_reaction_and_exact_restart",
            "tests/python/integration/runtime/test_public_amr_original_field.py::test_public_original_amr_published_fields_parent_rollback_and_retry",
        ),
        {"test_public_original_amr_saved_reaction_and_exact_restart":3,"test_public_original_amr_published_fields_parent_rollback_and_retry":1},
    ),
    "imex_time": (
        (
            "tests/python/integration/runtime/test_imex_nonautonomous_field.py::test_public_nonautonomous_imex_field_reads_explicit_time",
        ),
        {"test_public_nonautonomous_imex_field_reads_explicit_time":1},
    ),
    "nonlinear_mixed": (
        (
            "tests/python/integration/runtime/test_nonlinear_mixed_field_runtime.py::test_original_nonlinear_product_from_saved_native_observations",
            "tests/python/integration/runtime/test_nonlinear_mixed_field_runtime.py::test_original_nonlinear_failure_has_no_accepted_publication",
        ),
        {"test_original_nonlinear_product_from_saved_native_observations":3,"test_original_nonlinear_failure_has_no_accepted_publication":2},
    ),
}

_SAVED_RESTART = GROUPS["amr_original"][0][0]
AMR_ORIGINAL_PARTITIONS = {
    **parameter_partitions({
        "saved_restart_16_implicit_zero": (_SAVED_RESTART + "[16-order0-implicit_zero]",),
        "saved_restart_16_declared_zero": (_SAVED_RESTART + "[16-order1-declared_zero]",),
        "saved_restart_32_declared_zero": (_SAVED_RESTART + "[32-order2-declared_zero]",),
    }),
    "parent_rollback_retry": (
        (GROUPS["amr_original"][0][1],),
        {"test_public_original_amr_published_fields_parent_rollback_and_retry": 1},
        None,
    ),
}
_SPATIAL = GROUPS["spatial_interaction"]
SPATIAL_INTERACTION_PARTITIONS = {
    "saved_history": (
        (_SPATIAL[0][0],), {"test_public_spatial_interaction_saved_history_and_exact_replay": 3}, None,
    ),
    "refusals": (
        (_SPATIAL[0][1],), {"test_public_spatial_interaction_refuses_without_publication": 3}, None,
    ),
    "pole_and_bind": (
        (_SPATIAL[0][2], _SPATIAL[0][3]),
        {"test_public_spatial_interaction_ignores_inactive_source_pole": 1,
         "test_public_spatial_interaction_nonperiodic_amr_refuses_at_bind": 1},
        None,
    ),
}
PARTITIONED_GROUPS = {"amr_original": AMR_ORIGINAL_PARTITIONS,
                      "spatial_interaction": SPATIAL_INTERACTION_PARTITIONS}


def run_public_field_profile_world(*, group, ranks, directory, partition=None):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public Field profile requires an explicit MPI1 or MPI2 world")
    units = None
    if group in PARTITIONED_GROUPS:
        partitions = PARTITIONED_GROUPS[group]
        if type(partition) is not str or partition not in partitions:
            raise ValueError("public Field " + group + " profile requires an exact declared partition")
        tests, expected_cases, units = partitions[partition]
    else:
        if partition is not None:
            raise ValueError("only the original AMR and spatial interaction Field profiles declare partitions")
        tests, expected_cases = GROUPS[group]
    run_fixture_world(ranks=ranks, directory=directory, tests=tests,
                      expected_cases=expected_cases,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose="public Field " + group + (" " + partition if partition is not None else "") + " profile",
                      timeout_seconds=CHILD_TIMEOUT_SECONDS, partition_units=units)
