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
- distinct strict/regrid-on-restart identities and guarantees, plus one real transactional
  restart/regrid with non-empty persistent tagging hysteresis;
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

The rank-change restart proof is a serial pytest orchestrator registered in the same manifest. It
launches an independent two-rank capture and one-rank restore, so the gate proves persisted
rematerialization across MPI worlds rather than rebuilding ownership inside one communicator. Its
native tagger uses non-zero hysteresis; both source ranks must publish the same non-empty tagging
payload, and the one-rank checkpoint after restore must retain those bytes exactly. A separate
two-rank process injects a byte-level producer disagreement and proves collective refusal leaves no
published or temporary checkpoint.
The serial and two-rank RegridOnRestart proofs restore that accepted hysteresis image, execute
exactly one scientific regrid, and require its cycle to advance exactly once. At unchanged MPI
cardinality, exact-`MPI_COMM_WORLD` shared-interface registries are preflighted before the transform,
rematerialized against the detached candidate hierarchy, and authenticated again before commit. A
fault or rank-divergent identity injected after the native topology/tagging mutation must roll back
the complete pre-restart Program image; a second successful attempt must reproduce the same
transformed tagging bytes before continuation.
The source validator requires that exact pytest path to remain in the manifest's
`mpi_orchestrators` category; removing or reclassifying it invalidates `--check-only`.
All Python checks run with native and MPI requirements forced on; a missing capability cannot turn
this proof into an optional skip. Pytest also emits a mandatory JUnit report with strict xfail
semantics; any skipped or xfailed proof fails the M3 gate.

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
