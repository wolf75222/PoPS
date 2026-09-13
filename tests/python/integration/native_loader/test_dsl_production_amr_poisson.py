"""Euler-Poisson AMR public/native parity, isolated from transport package compiles."""
from tests.python.integration.native_loader.test_dsl_production_amr import (
    _euler_poisson_public_parity,
    require_toolchain,
)

# Each file owns an independent cold-compile phase; retain the existing native process budget.
POPS_PROCESS_TIMEOUT = 900


def main():
    require_toolchain()
    _euler_poisson_public_parity(48, 2e-4)


if __name__ == "__main__":
    main()
