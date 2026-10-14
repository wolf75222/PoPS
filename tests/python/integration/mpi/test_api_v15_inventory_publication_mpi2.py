"""Declared MPI2 entrypoint; exact mandatory inventory receiver cardinality."""
from tests.python.support.mpi_pytest_entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_inventory_publication_v15_native.py', expected_cases=7, ranks=2))
