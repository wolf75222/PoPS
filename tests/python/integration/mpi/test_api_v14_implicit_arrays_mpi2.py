"""Declared MPI2 entrypoint; exact mandatory receiver cardinality."""
from tests.python.support.mpi_pytest_entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_implicit_arrays_v14_native.py', expected_cases=9, ranks=2))
