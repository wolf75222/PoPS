"""Exact native Uniform fixture groups for independent mandatory MPI worlds.

Each group runs as declared parameter partitions sized by measured cost, so one 14-minute child
watchdog covers one partition (CI run 41037321460: each whole group stopped at that watchdog).
"""
from tests.python.support.mpi_fixture_driver import run_fixture_world
from tests.python.support.mpi_partition_units import parameter_partitions


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
_EVOLVED = EVOLVED_ORIGINAL_TESTS[0]
EVOLVED_ORIGINAL_PARTITIONS = parameter_partitions({
    "width1_dt001": (_EVOLVED + "[1-False-0.01-8]", _EVOLVED + "[1-False-0.01-16]"),
    "width1_dt002": (_EVOLVED + "[1-False-0.02-8]", _EVOLVED + "[1-False-0.02-16]"),
    "width2_dt001": (_EVOLVED + "[2-True-0.01-8]", _EVOLVED + "[2-True-0.01-16]"),
    "width2_dt002": (_EVOLVED + "[2-True-0.02-8]", _EVOLVED + "[2-True-0.02-16]"),
})
_CAPTURED, _CANDIDATE = CAPTURED_DIFFUSION_TESTS
CAPTURED_DIFFUSION_PARTITIONS = parameter_partitions({
    "captured_width1": (_CAPTURED + "[1-order0]",),
    "captured_width3": (_CAPTURED + "[3-order1]",),
    "candidate_width1": (_CANDIDATE + "[1-order0]",),
    "candidate_width3": (_CANDIDATE + "[3-order1]",),
})


def _run_world(ranks, directory, partitions, partition, purpose):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("public Uniform profile requires an explicit MPI1 or MPI2 world")
    if type(partition) is not str or partition not in partitions:
        raise ValueError("public Uniform profile requires an exact declared partition")
    tests, expected_cases, units = partitions[partition]
    run_fixture_world(ranks=ranks, directory=directory, tests=tests,
                      expected_cases=expected_cases,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose=purpose + " " + partition, timeout_seconds=CHILD_TIMEOUT_SECONDS,
                      partition_units=units)


def run_evolved_original_world(*, ranks, directory, partition):
    _run_world(ranks, directory, EVOLVED_ORIGINAL_PARTITIONS, partition,
               "public evolved-original profile")


def run_captured_diffusion_world(*, ranks, directory, partition):
    _run_world(ranks, directory, CAPTURED_DIFFUSION_PARTITIONS, partition,
               "public captured/candidate diffusion profile")
