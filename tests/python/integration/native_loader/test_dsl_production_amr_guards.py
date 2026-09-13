"""AMR compiled-package pressure, target and ABI refusals with authentic binary identity."""
import tempfile

from tests.python.integration.native_loader.test_dsl_production_amr import (
    run_guards,
    require_toolchain,
)

# Each file owns an independent cold-compile phase; retain the existing native process budget.
POPS_PROCESS_TIMEOUT = 900


def main():
    require_toolchain()
    with tempfile.TemporaryDirectory() as tmp:
        run_guards(tmp)


if __name__ == "__main__":
    main()
