"""MPI2 execution of every declared receiver, with per-rank receipts."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_api_v02_nonlinear_representation_history_native.py', expected_cases=2, ranks=2))
