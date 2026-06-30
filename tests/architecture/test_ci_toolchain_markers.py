"""TASK-076: CI must exercise the final toolchain-marked route with pytest.

The historical sharded Python gate still runs many old script-style tests directly.
That is not enough for the Spec corrective route: the final examples are pytest
tests guarded by ``requires_toolchain``. This source-only guard makes sure CI keeps
an explicit pytest smoke for those marked integration tests.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_runs_spec_final_requires_toolchain_tests_with_pytest():
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "Spec final toolchain smoke (pytest markers)",
        "python3 -m pytest -q",
        "-m 'requires_toolchain'",
        "python/tests/integration/test_compile_problem_system_install.py",
        "python/tests/integration/test_manual_board_predictor_corrector.py",
        "python/tests/integration/test_lib_time_predictor_corrector.py",
        "python/tests/integration/test_matrix_free_bicgstab.py",
        "python/tests/integration/test_moments_final.py",
        "python/tests/integration/test_amr_final_route.py",
        "python/tests/test_output_policy_run.py",
        "python/tests/test_hdf5_parallel.py",
    )
    missing = [token for token in required if token not in text]
    assert not missing, (
        "CI must keep a pytest-based requires_toolchain smoke for the final public route; "
        "missing tokens:\n%s" % "\n".join(missing)
    )
