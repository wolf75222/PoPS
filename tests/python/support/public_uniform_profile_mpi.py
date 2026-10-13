"""Exact native Uniform fixture groups for independent mandatory MPI worlds."""
from tests.python.support.mpi_fixture_driver import run_fixture_world


CHILD_TIMEOUT_SECONDS = 14 * 60
EVOLVED_ORIGINAL_TESTS = (
    "tests/python/integration/runtime/test_public_evolved_original_stage.py::"
    "test_public_evolved_original_stage_saved_and_exact_replay",
)
EVOLVED_ORIGINAL_CASES = {
    "test_public_evolved_original_stage_saved_and_exact_replay": 8,
}
CAPTURED_DIFFUSION_TESTS = (
    "tests/python/integration/runtime/test_public_captured_diffusion.py::"
    "test_public_captured_diffusion_nonconstant_saved_and_exact_replay",
    "tests/python/integration/runtime/test_public_captured_diffusion.py::"
    "test_public_candidate_diffusion_nonconstant_saved_and_exact_replay",
)
CAPTURED_DIFFUSION_CASES = {
    "test_public_captured_diffusion_nonconstant_saved_and_exact_replay": 2,
    "test_public_candidate_diffusion_nonconstant_saved_and_exact_replay": 2,
}


def _run_world(ranks, directory, tests, expected_cases, purpose):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public Uniform profile requires an explicit MPI1 or MPI2 world")
    run_fixture_world(ranks=ranks, directory=directory, tests=tests,
                      expected_cases=expected_cases,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose=purpose, timeout_seconds=CHILD_TIMEOUT_SECONDS)


def run_evolved_original_world(*, ranks, directory):
    _run_world(ranks, directory, EVOLVED_ORIGINAL_TESTS, EVOLVED_ORIGINAL_CASES,
               "public evolved-original profile")


def run_captured_diffusion_world(*, ranks, directory):
    _run_world(ranks, directory, CAPTURED_DIFFUSION_TESTS, CAPTURED_DIFFUSION_CASES,
               "public captured/candidate diffusion profile")
