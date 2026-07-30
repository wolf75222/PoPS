# M4 native runtime and scientific I/O conformance gate

`python scripts/run_m4_gate.py` is the reviewed executable acceptance matrix for
ADC-679 through ADC-687. It pins exact source-registered proofs instead of broad
test directories or nearby suites. The architecture CI runs `--check-only`, so
renaming, deleting, making optional, or removing a selected proof from
`tests/test_manifest.toml` fails before a build is launched.

The matrix covers:

- canonical component manifests, generated registries, and external AOT
  packages;
- external flux, boundary, tagger, transfer, solver, and writer components;
- native interface, provider-pack, platform, capability, and ABI refusals;
- one `RuntimeInstance` across Uniform, AMR, and multiple mapped layouts;
- transactional `ConsumerGraph` publication and rollback;
- direct format-native reopen of NPZ with NumPy, HDF5 with h5py, and ParaView
  VTU with VTK;
- real two-rank collective HDF5 and its exact CTest selector;
- strict multi-layout checkpoint/restart, including atomic restore refusal;
- exact diagnostics and the source-level retirement fence for the old Schur
  source steppers.

`deferred = []` is normative for closure. Every issue needs positive and refusal
coverage and at least one native positive proof. Each scientific family has its
own required polarity. The validator rejects duplicate or wildcard selectors,
missing manifest ownership, pytest skip/xfail and optional imports, mock-based
proofs, disabled CTests, and any explicit deferred gap.

Use:

```bash
python scripts/run_m4_gate.py --audit-only
python scripts/run_m4_gate.py --check-only
python scripts/run_m4_gate.py --python-only
python scripts/run_m4_gate.py --build-dir build-mpi
```

`--audit-only` validates the ledger while deliberately making no closure claim,
even when there are no deferred rows. `--check-only` additionally requires the
ledger to be closed, but still launches no test, compiler, MPI process, or
native reader. `--python-only` executes every selected Python proof with native
requirements forced on and omits CTest. The last command is the full gate and
requires an MPI-enabled build containing every selected CTest, plus real NumPy,
h5py, and VTK installations. Both pytest and CTest must produce JUnit reports
with zero skipped or xfailed proofs.

The native-reader tests intentionally use no PoPS reader to interpret the
written payload. PoPS is used only to produce and authenticate the output;
NumPy, h5py, and VTK independently prove that the published formats are usable.
Their proof module is deliberately named `m4_native_reopen_proof.py`, so normal
`test_*.py` shard discovery does not silently turn VTK into a dependency of
every Python shard. The exact nodeids remain mandatory in the explicit M4 gate.
