"""Normal-density five-step isothermal floor parity in its own cold-compile process."""
from tests.python.integration.runtime.test_isothermal_vacuum_floor_system import run_inactive_floor

# Independent native fixture phase; retain the existing process budget.
POPS_PROCESS_TIMEOUT = 900


def main():
    run_inactive_floor()


if __name__ == "__main__":
    main()
