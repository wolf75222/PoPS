"""MPI2 execution of every declared receiver, with per-rank receipts."""
from tests.python.support.mpi_pytest_entrypoint import run


if __name__ == "__main__":
    raise SystemExit(run('tests/python/integration/runtime/test_api_v02_temporal_field_native.py', expected_cases=3, ranks=2))
