# M4 native runtime and scientific I/O conformance gate

The evidence ledger is **SOURCE-CLOSED AND REQUIRED BY CI**. The ledger in
`tests/gates/m4_runtime_io.toml` records exact executable evidence for
ADC-679 through ADC-687. It contains exactly 53 executable checks and
`deferred = []`. Milestone closure is accepted only for a commit whose required MPI job
successfully executes the complete installed gate; source audit alone is not
the acceptance evidence.

The source audit already authenticates real proofs for:

- external flux, boundary, tagger, transfer, solver, and writer components
  that are compiled, loaded, and executed;
- canonical component manifests, generated registries, exact interface
  tables, and platform launch checks;
- source, manifest, and installed-binary tamper refusals, provider absence,
  native parameter capacity overflow, and a genuine wrong-ABI DSO refusal;
- an unknown device capability refused by the runtime launch validator before
  the candidate kernel is invoked;
- Program-only Uniform/AMR temporal facades, retired native source-stage
  headers and schedulers, typed component dispatch, and fail-closed unbound
  native interfaces;
- real Uniform and AMR writer transactions, a real multi-layout transfer,
  a real two-writer collision with complete ConsumerGraph compensation, and a
  positive multi-layout checkpoint/restart;
- one complete RuntimeInstance contract proof across compiled Uniform, AMR,
  and multi-layout execution, including RunReport and Program/inspection
  parity;
- a prepared native FieldSolver whose invalid first result is refused through
  RuntimeInstance with exact accepted-state rollback and a successful retry;
- accepted scientific publication, diagnostics including qualified native
  projection/reflux term selection, two-rank collective HDF5,
  and a two-rank PVD/PVTU/rank-VTU hierarchy reopened by native VTK readers.

The required Ubuntu 24.04 MPI lane installs Open MPI, parallel HDF5, NumPy,
h5py, pytest, and the native VTK Python readers. It builds the MPI-enabled
extension plus every exact CTest target selected by the ledger, then runs the
complete gate. The global required-check aggregator rejects a skipped, failed,
cancelled, or timed-out MPI lane whenever the M4 runner, ledger, source fence,
or CI workflow changes.

## Exact output evidence

There are four serial proofs that are real and remain selected:

1. the final IMEX/AMR example publishes and reopens its serial scientific
   formats through the public PoPS readers;
2. NPZ is independently reopened with NumPy and its arrays and physical clock
   are checked;
3. HDF5 is independently reopened with h5py and its dataset and physical clock
   are checked;
4. a serial VTU is independently reopened with VTK and its mesh, public field
   name, AMR level array, and `TimeValue` are checked.

An additional HDF5 refusal mutates a dataset with h5py and proves that the
authenticated PoPS reader rejects it. These tests contain no optional import
or skip. That makes their dependencies mandatory wherever the executable gate
runs; the required MPI lane provisions and imports those readers before
launching the matrix.

The selected two-rank ParaView entrypoint starts from the standard `.pvd`
catalogue, preserves its exact temporal ordering, and requires the native VTK
parallel reader to assemble every referenced `.pvtu`. It also reopens every
rank-local `.vtu` directly with VTK and checks its geometry, public arrays,
component name, and `TimeValue`. VTK imports are unconditional in the required
MPI lane: `POPS_REQUIRE_MPI_TESTS=1` turns an absent reader into a test failure.
The selected `gate_execution` proof authenticates that exact CI route, and the
same required job executes the entrypoint rather than auditing only its source.

The strict-checkpoint refusal is also provider-backed. A correctly sealed AMR
checkpoint with an inconsistent dynamic accepted-ledger claim passes the real
`RestartV3` file reopen and static preflight, then fails only when the native
AMR provider validates the restored Program image. The test proves that the
active restart transaction restores fields, hierarchy, histories, clocks,
counters, run identity, and consumer cursors, and that the same provider can
successfully retry the unmodified checkpoint.

The ConsumerGraph refusal is likewise provider-backed. Two separately
qualified native Writer components are compiled and staged in one transaction.
After the first Writer publishes, a pre-existing user-owned target makes the
second Writer fail at the runtime's atomic publication link. The transaction
must compensate the first artifact, preserve the colliding file byte-for-byte,
remove every private staging path, retain the exact accepted numerical state
and consumer cursors, and then publish both Writers on a clean retry. The
selected proof contains no handwritten publisher or prepared-publication fake.

The RuntimeInstance refusal no longer relies on `FailFirstStep`. A qualified
native FieldTopology/FieldSolver pair is packaged, compiled, resolved, bound,
and prepared through the production component ABI. An authenticated external
fault marker makes its solve report convergence while returning non-finite
values, so the production field validation fails inside the native Program
step. RuntimeInstance must restore
the conservative state, field potential, accepted clock, macro-step, temporal
authority, consumer cursors, reports, and provider evidence exactly. The
component's prepared state is not mutated by this failure. After the external
fault is removed, the same prepared component returns a finite result and the
unchanged RuntimeInstance accepts the retry. The selected test defines no step
wrapper and never replaces a native engine or step target.

That installed proof uses the MPI-enabled module with a one-rank
`MPI_COMM_WORLD`. The System adapter authenticates and accepts this singleton
communicator explicitly; it still refuses multi-rank external FieldSolver
execution until a collective distributed solve contract is proved.

The positive RuntimeInstance proof is also a compiled route. It builds and
executes one Uniform artifact, one AMR artifact, and one two-layout artifact
with a native conservative Transfer. Every execution returns the exact public
`RuntimeInstance` and `RunReport` types with aligned artifact, bind, execution,
run, clock, step, and transaction evidence. The multi-layout executor
authenticates each installed child Program, creates one domain-separated hash
for the ordered Program set, and projects local block/parameter/cache metadata
into deterministic layout-qualified report rows. The block bijection and
parameter occupancy come from the installed native `program_block_map()` and
`program_param_count()` accessors; the report does not infer an identity map
from missing bindings. Runtime inspection consumes that same complete
`ProgramRuntimeReport`; the selected test proves direct and inspection parity
without a wrapper, fake engine, replaced step target, or monkeypatch.

## Gate modes

The source-only architecture CI checks that the ledger is closed:

```bash
python scripts/run_m4_gate.py --check-only
```

`--check-only` verifies the exact nodeids, CTest selectors, manifest ownership,
empty deferred-gap ledger, and source-level anti-skip rules without launching
a compiler, test, MPI process, or native reader. `--audit-only` performs the
same structural audit and reports `AUDITED CLOSED`.

The installed MPI lane asks the same closed manifest for its exact native build
targets:

```bash
python scripts/run_m4_gate.py --list-ctest-targets
```

It then invokes the complete executable gate, with no audit-only or
Python-only reduction:

```bash
/usr/bin/python3 scripts/run_m4_gate.py \
  --build-dir build-mpi \
  --mpi-exec mpiexec
```

The full command requires the MPI-enabled extension, every selected CTest
target, NumPy, h5py, and VTK. Every selected pytest and CTest execution emits a
JUnit report and fails on any skipped or xfailed proof. A future limitation
must be restored as an explicit `[[deferred]]` row; the validator rejects
malformed or duplicate gaps, wildcard selectors, missing manifest ownership,
optional pytest imports, skip/xfail markers, mock fixtures/imports, and
disabled CTests.
