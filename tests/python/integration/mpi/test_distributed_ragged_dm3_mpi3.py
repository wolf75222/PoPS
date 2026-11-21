"""MPI3 DM3 free-placement entrypoint with per-rank pytest receipts."""

from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run("tests/python/integration/runtime/test_distributed_ragged_dm3_native.py",
                         expected_cases=1, ranks=3))
