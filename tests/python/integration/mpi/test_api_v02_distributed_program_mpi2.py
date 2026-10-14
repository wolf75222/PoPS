"""Declared MPI2 entrypoint; pytest owns the single receiver module."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_api_v02_distributed_program_native.py', expected_cases=4, ranks=2))
