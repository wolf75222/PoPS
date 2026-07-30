# M4 native runtime and scientific I/O conformance gate

The current status is **AUDITED OPEN**. The ledger in
`tests/gates/m4_runtime_io.toml` records exact executable evidence for
ADC-679 through ADC-687 and exact deferred gaps. It deliberately does not
claim M4 closure while any `[[deferred]]` row remains.

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
- accepted scientific publication, diagnostics, two-rank collective HDF5,
  and a two-rank PVD/PVTU/rank-VTU hierarchy reopened by native VTK readers.

This evidence is intentionally narrower than the final ADC-687 acceptance
contract. The deferred rows name the missing polarity and the nearby source
that must not be mistaken for closure. They currently cover:

- a CI lane that installs every mandatory dependency, including VTK, and
  executes every selected pytest and CTest proof with zero skips;
- composite runtime rollback without an injected failure wrapper;
- complete Uniform, AMR, and multi-layout public-contract/report parity.

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
runs; it does not prove that CI currently provisions those dependencies.

The selected two-rank ParaView entrypoint starts from the standard `.pvd`
catalogue, preserves its exact temporal ordering, and requires the native VTK
parallel reader to assemble every referenced `.pvtu`. It also reopens every
rank-local `.vtu` directly with VTK and checks its geometry, public arrays,
component name, and `TimeValue`. VTK imports are unconditional in the required
MPI lane: `POPS_REQUIRE_MPI_TESTS=1` turns an absent reader into a test failure.
The separate `gate_execution` gap remains open until CI provisions VTK and
executes this selected entrypoint rather than auditing only its source.

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

## Gate modes

The architecture CI runs:

```bash
python scripts/run_m4_gate.py --audit-only
```

`--audit-only` verifies the exact nodeids, CTest selectors, manifest ownership,
deferred-gap schema, and source-level anti-skip rules. It prints
`AUDITED OPEN` and launches no compiler, test, MPI process, or native reader.

The closure check is intentionally red while the ledger is open:

```bash
python scripts/run_m4_gate.py --check-only
```

`--check-only` rejects every remaining deferred row and exits nonzero before
launching anything. Running the script without either audit flag, or with
`--python-only`, is also fail-closed until all deferred gaps are replaced by
real selected proofs.

Once `deferred = []` is honestly restored, the full command requires an
MPI-enabled build containing every selected CTest and environments with NumPy,
h5py, and VTK. Every pytest and CTest execution must emit a JUnit report with
zero skipped or xfailed proofs.

Each deferred row contains `issue`, `requirement`, `polarity`, a precise
`reason`, and existing `evidence_paths`. The validator rejects malformed or
duplicate gaps, wildcard selectors, missing manifest ownership, optional
pytest imports, skip/xfail markers, mock fixtures/imports, and disabled CTests.
