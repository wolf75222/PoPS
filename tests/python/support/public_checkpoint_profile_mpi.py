"""Complete checkpoint/public capture groups in separate mandatory MPI worlds."""
from tests.python.support.mpi_fixture_driver import run_fixture_world

CHILD_TIMEOUT_SECONDS = 14 * 60
GROUPS = {
    'accepted_halo': (
        (
            'tests/python/integration/amr/test_public_accepted_halo_stage_failure.py::test_public_accepted_halo_rank_local_failure_restores_before_publication',
            'tests/python/integration/amr/test_public_accepted_halo_substep_growth.py::test_public_accepted_halo_five_halves_substep_growth',
        ),
        {
            'test_public_accepted_halo_rank_local_failure_restores_before_publication': 1,
            'test_public_accepted_halo_five_halves_substep_growth': 1,
        },
    ),
    'scalar_bound_capture': (
        (
            'tests/python/integration/runtime/test_public_scalar_field_sample.py::test_public_auxiliary_sample_and_projected_poisson',
            'tests/python/integration/runtime/test_dt_bound_capture_independent_runtime.py::test_native_public_bound_capture_route_and_permutation',
            'tests/python/integration/runtime/test_dt_bound_capture_independent_runtime.py::test_native_dot_all_invalid_bound_input_does_not_publish',
        ),
        {
            'test_public_auxiliary_sample_and_projected_poisson': 1,
            'test_native_public_bound_capture_route_and_permutation': 2,
            'test_native_dot_all_invalid_bound_input_does_not_publish': 1,
        },
    ),
    'imex_checkpoint': (
        (
            'tests/python/integration/runtime/test_imex_checkpoint_publication_rollback.py::test_original_imex_final_checkpoint_failure_restores_science_and_outputs',
            'tests/python/integration/runtime/test_imex_checkpoint_sealed_retry.py::test_original_imex_checkpoint_retry_replays_one_completed_seal',
        ),
        {
            'test_original_imex_final_checkpoint_failure_restores_science_and_outputs': 1,
            'test_original_imex_checkpoint_retry_replays_one_completed_seal': 1,
        },
    ),
    'amr12_carrier': (
        (
            'tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py::test_native_amr12_state_carrier_refusals_and_exact_rollback',
        ),
        {
            'test_native_amr12_state_carrier_refusals_and_exact_rollback': 1,
        },
    ),
    'typed_newton': (
        (
            'tests/python/integration/runtime/test_public_newton_typed_original.py::test_public_typed_original_policy',
        ),
        {
            'test_public_typed_original_policy': 8,
        },
    ),
}


def run_public_checkpoint_profile_world(*, group, ranks, directory):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public checkpoint profile requires an explicit MPI1 or MPI2 world")
    tests, expected_cases = GROUPS[group]
    # pytest owns and has already created tmp_path; each group/world owns a new
    # child namespace whose strict creation rejects reuse or stale receipts.
    run_fixture_world(ranks=ranks, directory=directory / (group + "-mpi%d" % ranks), tests=tests,
                      expected_cases=expected_cases,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose="public checkpoint " + group + " profile",
                      timeout_seconds=CHILD_TIMEOUT_SECONDS)
