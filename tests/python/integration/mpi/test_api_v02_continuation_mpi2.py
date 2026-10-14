"""Declared MPI2 entrypoint; pytest owns the single receiver module."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_continuation_v10_native.py', expected_cases=9, ranks=2))
