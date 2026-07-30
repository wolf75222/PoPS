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
- Program-only Uniform/AMR temporal facades, retired native source-stage
  headers and schedulers, typed component dispatch, and fail-closed unbound
  native interfaces;
- real Uniform and AMR writer transactions, a real multi-layout transfer,
  and a positive multi-layout checkpoint/restart;
- accepted scientific publication, diagnostics, and two-rank collective
  HDF5.

This evidence is intentionally narrower than the final ADC-687 acceptance
contract. The deferred rows name the missing polarity and the nearby source
that must not be mistaken for closure. They currently cover:

- a CI lane that installs every mandatory dependency, including VTK, and
  executes every selected pytest and CTest proof with zero skips;
- an unknown capability refused by a real runtime/backend before execution;
- ConsumerGraph and composite runtime rollback without handwritten publishers
  or injected failure wrappers;
- checkpoint restore rollback caused by a real provider failure;
- complete Uniform, AMR, and multi-layout public-contract/report parity;
- mandatory native VTK reopen of the MPI PVD to PVTU to rank-VTU hierarchy;

## Serial output evidence

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

The ParaView proof is limited to one serial `.vtu`. The MPI test authenticates
the `.pvd`, `.pvtu`, and rank-local `.vtu` hierarchy with PoPS and XML, but its
independent VTK reader is optional today. Consequently the standard parallel
ParaView hierarchy is useful existing evidence, not a closed native-reader
proof.

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
