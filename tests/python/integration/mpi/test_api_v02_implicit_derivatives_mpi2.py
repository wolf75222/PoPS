"""Declared MPI2 entrypoint; pytest owns the single receiver module."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_api_v02_implicit_derivatives_native.py', expected_cases=10, ranks=2))
