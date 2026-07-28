# M3 AMR and multi-layout conformance gate

`python scripts/run_m3_gate.py` is the reviewed executable acceptance matrix for
the M3 field, boundary, AMR, and multi-layout contracts. The architecture CI
runs `--check-only`: it validates that every exact proof still exists and is
mandatory, without running the numerical validation battery on every change.

The full gate covers:

- complete ghost producers and shared-interface planning;
- owner-qualified layout assignments, mappings, and real MPI synchronization;
- exact typed AMR tagging and native tagger execution;
- three-level hierarchy resolution, regrid, and atomic refusal paths;
- per-space transfer registries and recursive deterministic bootstrap;
- exact level clocks, flux ledgers, reflux, rollback, and real MPI execution;
- strict accepted-state restart and topology/history/ledger rollback;
- two-rank AMR restart after an accepted regrid and continuation across the next
  regrid, for replicated and distributed coarse layouts;
- checkpoint/restart of every state and mapping counter in a real two-layout
  runtime;
- two-rank composite AMR reductions checked against an independent
  covered-cell oracle for replicated and distributed coarse layouts;
- distinct strict/regrid-on-restart identities, guarantees, and fail-closed capability refusal;
- total `LoweringCoverageReport` success and structured rejection.

`deferred = []` is normative. Every ADC-672 through ADC-678 issue must have a
positive proof, a refusal or rollback proof, and at least one native positive
proof. The validator also rejects pytest skip/xfail markers, disabled or
skipping CTest sources, missing manifest targets, and duplicate selectors. An
`mpi_python` proof must be an exact `file::test` nodeid whose file is registered
with the same rank count in `tests/test_manifest.toml`. The gate executes that
manifest-owned script once with `mpiexec -n 2` and forces
`POPS_REQUIRE_MPI_TESTS=1`; a missing launcher, MPI runtime, or native
capability is therefore a failure, never an optional skip.

The multi-layout checkpoint proof currently uses two independent `Uniform`
layouts. It proves restoration of every layout state and mapping counter, but
it is not a substitute for the separate AMR hierarchy/regrid restart proofs.

Use:

```bash
python scripts/run_m3_gate.py --check-only
python scripts/run_m3_gate.py --python-only
python scripts/run_m3_gate.py --build-dir build-mpi
```

`--check-only` performs source/manifest validation and does not require an MPI
launcher. `--python-only` still runs the real Python MPI entrypoint; it only
omits CTest. The last command is the full validation battery. It expects a
complete MPI test build so the Python and C++ `MPI_COMM_WORLD` proofs are both
executable.
