"""Declared MPI2 entrypoint; pytest owns the single receiver module (N10 lot 4)."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_api_v02_unconstrained_distributed_native.py', expected_cases=2, ranks=2))
