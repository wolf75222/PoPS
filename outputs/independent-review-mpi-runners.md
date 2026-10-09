# Independent review: installed MPI and scientific runners

Source examined: MAIN `docs/development/api_040/run_installed_mpi_checks.py`,
`run_scientific_checks.py`, `run_installed_checks.py`, and the installed-runtime
User body witness. No native test or build was run for this review.

## What the MPI runner actually authenticates

`run_installed_mpi_checks.py` executes `mpiexec` from the active environment,
requires `world.size == --ranks` on every worker, allgathers each rank's native
module path and SHA-256, and compares against the installed-package identity
receipt. It rejects a skipped or failing pytest node on either rank and checks
identical selected node lists, stable tracked sources and native hash before and
after. This establishes real multi-process test execution. It does **not**
establish that each rank owns mesh cells. The existing Uniform rank-1 witness
skips when its `local_boxes` is empty; the runner then correctly fails because
it rejects skips. No positive rank-1-owned result should be claimed from that
node until an actual distributed layout produces ownership.

The runner hashes selected test files, but not imported repository helper files.
Its installed-identity script hashes tracked shipped Python/header files and
does not inventory untracked new production Python modules. Before release, a
receipt should hash the test dependency closure (at least `tests/python/support`
and test module imports) and record any untracked shipped production files or
refuse them. This is a provenance gap, not evidence that the current native
library was substituted: each MPI worker's loaded native SHA is checked.

## Scientific runner

`run_scientific_checks.py` authenticates the installed native package and
shipped sources before and after, hashes every `api040_*.py` example/helper,
uses an empty output directory, and requires a passing receipt with matching
native SHA, requested rank count in each record, and requested thread count.
For M17, it runs canonical then reverse; the reverse example independently
checks the canonical peer receipt, state hash, package path, native hash, and
permutation before reporting success. The outer runner still inspects only the
reverse receipt; it should independently hash/check both phase receipts and
their saved-state files if the runner is to certify the pair without relying on
the example's self-check. It also does not independently compare the example
receipt's package path/ABI or saved NPZ hashes against the installed identity.
Those checks would strengthen independent provenance. It has no execution
timeout, so a hung MPI scientific run requires external intervention.

Neither runner should describe rank-1-local physics without a positive ownership
measurement. `CartesianGrid.native_spatial_data()` declares a `single_box`
decomposition for Uniform. AMR `PatchLayout(distribute_coarse=True)` distributes
level-0 boxes, but the prior public report gave only local/total counts. The
separate diagnostic `coarse_local_box_bounds()` now exposes current native
level-0 local bounds; the new MPI test asserts rank-1 ownership and exact
coverage before injecting its invalid cell. This proposed extension and witness
remain **source-reviewed only** until root rebuilds and runs them against the
installed package under two ranks.
