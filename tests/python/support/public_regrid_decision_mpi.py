"""``Regrid(criterion, when)`` written in the Program, in mandatory MPI1 and MPI2 worlds.

Review 5815613688 of #743 (N12 lot 2, ROM-95), point P3: the generated decision reduces its
indicators over the patches and the ranks (``all_reduce_max``) and checks that every rank decides
the same; the native file runs here on every rank of a genuine MPI world, with the same exact
comparisons as in its Serial run (layout cadence parity, a predicate on the state on both sides of
a runtime threshold, the checkpoint refusal).
"""
from tests.python.support.mpi_fixture_driver import run_fixture_world

CHILD_TIMEOUT_SECONDS = 14 * 60
TESTS = ("tests/python/integration/amr/test_regrid_decision_native.py",)
EXPECTED_CASES = {
    "test_every_written_in_the_program_regrids_as_the_layout_cadence": 1,
    "test_a_predicate_on_the_state_regrids_after_the_steps_where_it_holds": 1,
    "test_a_checkpoint_of_a_generated_decision_is_refused": 1,
}


def run_public_regrid_decision_world(*, ranks, directory):
    if type(ranks) is not int or ranks not in (1, 2):
        raise ValueError("regrid decision profile requires an explicit MPI1 or MPI2 world")
    run_fixture_world(ranks=ranks, directory=directory / ("regrid_decision-mpi%d" % ranks),
                      tests=TESTS, expected_cases=EXPECTED_CASES,
                      expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS",
                      purpose="regrid decision", timeout_seconds=CHILD_TIMEOUT_SECONDS)


__all__ = ["EXPECTED_CASES", "TESTS", "run_public_regrid_decision_world"]
