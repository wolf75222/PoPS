#!/usr/bin/env python3
"""M09 routes to the two *distinct* public Hoffart witnesses.

``finite`` evaluates the complete, published 12-unknown midpoint source
equations with an independent NumPy oracle; it is not a PoPS native result.
``disk`` executes the existing full finite-volume Euler--Poisson tutorial,
whose model and time program use public PoPS authoring.  Its physical
discretisation is distinct from the finite random G-matrix witness.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("finite", "disk"))
    args = parser.parse_args()
    here = Path(__file__).resolve()
    if args.case == "finite":
        runpy.run_path(str(here.with_name("api040_m09_hoffart_oracle.py")), run_name="__main__")
        return
    root = here.parents[3]
    disk = root / "docs/tutorials/diocotron/01_mpi_kokkos_hoffart_euler.py"
    if not disk.is_file():
        raise FileNotFoundError(f"full Hoffart disk example is required: {disk}")
    runpy.run_path(str(disk), run_name="__main__")


if __name__ == "__main__":
    main()
