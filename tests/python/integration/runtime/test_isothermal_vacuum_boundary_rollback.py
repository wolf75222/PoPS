"""Missing nonperiodic transport boundary refuses and rolls back one complete step."""
from tests.python.integration.runtime.test_isothermal_vacuum_floor_system import run_missing_boundary_rollback

# Independent native fixture phase; retain the existing process budget.
POPS_PROCESS_TIMEOUT = 900


def main():
    run_missing_boundary_rollback()


if __name__ == "__main__":
    main()
