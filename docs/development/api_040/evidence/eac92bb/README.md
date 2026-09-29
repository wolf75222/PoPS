# AMR reception at production source eac92bb

These are retained execution receipts, including failures. They do not qualify
the entire migration or any unexecuted backend. `manifest.json` records exact
native/SDK identities and the C++ executable hashes. Validate the copied bytes
from this directory with `shasum -a 256 -c SHA256SUMS`.

| Reception | Actual backend | Result |
|---|---|---|
| Installed standalone User and principal AMR | Dim2 OpenMP, one thread/rank | 7 passed |
| Rank-one-owned invalid User body, initial oracle | Same native image, MPICH two ranks | One failure per rank; expected wrong route diagnostic |
| Same test with exceptions displayed | Same native image, MPICH two ranks | One failure per rank; actual flux materialization rejection established |
| Same test with reviewed route diagnostic | Same native image, MPICH two ranks | One test passed on both ranks; clocks, both states and ledger rollback checked |
| Three rebuilt C++ targets | Dim2 OpenMP; serial rows and three MPI2 aggregates | 73 rows: 69 passed, 2 failed, 2 single-rank guards skipped |

The C++ failure is one three-level restart scenario, reproduced inside the MPI2
aggregate: after fixing the owners array, `regrid_on_restart()` now reaches and
rejects an invalid pending history remap. Its correction and new reception are
still required. MPI ranks are not counted as independent test cases.

Use the commands embedded in each `result.json` with the authenticated installed
environment, `env -u PYTHONPATH`, `POPS_NATIVE_DIM=2`, and the PoPS/Kokkos include
roots from that environment. Select new empty output directories. The repository
entry points are `run_installed_checks.py` and `run_installed_mpi_checks.py` in
`docs/development/api_040`; they authenticate imports rather than importing the
handoff prototype. The C++ command selects the exact three targets via
`ctest -L 'cpp-target:(test_generated_amr_system_block|test_amr_synthetic_program_loader_transaction|test_amr_history_ring)$'`.

Raw large artifacts remain in the parent workspace; no native binary, environment
or scientific state array is included in this evidence bundle.
