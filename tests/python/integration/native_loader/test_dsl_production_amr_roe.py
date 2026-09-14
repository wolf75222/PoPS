"""Both exact n=48 Roe reconstruction parities, sharing one native AMR package."""
import tempfile

from tests.python.integration.native_loader.test_dsl_production_amr import (
    run_roe_parity,
    require_toolchain,
)

# Each file owns an independent cold-compile phase; retain the existing native process budget.
POPS_PROCESS_TIMEOUT = 900


def main():
    require_toolchain()
    with tempfile.TemporaryDirectory() as tmp:
        run_roe_parity(tmp)


if __name__ == "__main__":
    main()
