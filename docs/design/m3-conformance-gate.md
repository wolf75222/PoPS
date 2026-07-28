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
- total `LoweringCoverageReport` success and structured rejection.

`deferred = []` is normative. Every ADC-672 through ADC-678 issue must have a
positive proof, a refusal or rollback proof, and at least one native positive
proof. The validator also rejects pytest skip/xfail markers, disabled or
skipping CTest sources, missing manifest targets, and duplicate selectors.

Use:

```bash
python scripts/run_m3_gate.py --check-only
python scripts/run_m3_gate.py --python-only
python scripts/run_m3_gate.py --build-dir build-mpi
```

The last command is the full validation battery. It expects a complete MPI test
build so the serial native and `MPI_COMM_WORLD` proofs are both executable.
