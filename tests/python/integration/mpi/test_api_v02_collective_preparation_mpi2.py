"""Declared MPI2 entrypoint; pytest owns the single receiver module."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_collective_control_preparation_v10.py', expected_cases=15, ranks=2))
