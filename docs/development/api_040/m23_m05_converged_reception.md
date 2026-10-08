# M23 / M05 - archived SDK623 reception

This bundle preserves local CPU Kokkos/OpenMP reception, including the earlier failures. It is independent of the older SDK396 `finite-amr-converged` bundle. The XML testcase elements, rather than a moving checkout HEAD or a combined pass label, determine the counts below.

The payload is [evidence/m23-m05-converged](evidence/m23-m05-converged/manifest.json): 206 manifested files, 26 NPZ states and 10 raw ledgers. Four failed testcases remain archived. No native binary, shared library, or JIT cache is included. Raw files retain their original bytes and whitespace.

## Receipts and source windows

| Receipt directory | Exact source prefix | Recorded result and scope |
| --- | --- | --- |
| `installed-m23-source` | `22ae8449` | 74/74 source/math/complete-Program syntax checks |
| `installed-m23-native-serial` | `22ae8449` | 3/3 failed at bind: `InitialConditionPlan` conflicted with legacy `initial_state`; no evolution |
| `installed-m23-native-bind-repaired` | `a52e5809` | 3/3 native Dim1: five Hall trajectories and two matrix variants |
| `installed-m23-native-mpi2` | `15a776c6` | 3/3 on each of two ranks; strengthened stage oracle and empty-rank checks |
| `installed-m23-diffusion-nonregression` | `a52e5809` | Dim2: 4 pass / 1 implicit State-read identity failure |
| `installed-m05-periodic-shear` | `15a776c6` | Historical 14/14: 13 source/math checks and one native three-grid test |
| `installed-m05-periodic-shear-mpi2` | `15a776c6` | Historical 1/1 on each rank; no archived raw ledger |
| `installed-m05-reviewed-serial` | `f0047269` | 19/19: 18 source/math checks and one strengthened native three-grid test |
| `installed-m05-reviewed-mpi2` | `f0047269` | Strengthened 1/1 on each rank, with all three raw ledgers |
| `installed-implicit-state-reviewed-unit` | `0cd16d01` | Separate Python refresh: 33/33 source checks |
| `installed-implicit-state-native-repaired` | `0cd16d01` | Separate Python refresh: all five previous Dim2 cases rerun, 5/5 serial; no new MPI claim |

Full commit IDs are in `source/run-source-commits.json`. Exact Git snapshots of examples, tests, oracles and helpers are listed in `source/snapshots.json`, with SHA-256 and Git blob identities. The source proofs compare the 1,089 production files to the actual installed-source manifests. Across the initial four source commits production is identical; the later refresh changes only `python/pops/codegen/inspect_compiled.py`. The earlier M23/M05 results are not claimed as reruns after that refresh.

All these windows use SDK/header signature
`62398f3c13c193eb48db07518735fe755e6d108821adffc37acccd8fcea290eb`.
Installed native hashes are:

- Dim1: `42436a0ccef7c5504273c9e459f4bd946652597aa6ee025bc484ac595ad7ca02`.
- Dim2: `1afb920397c7fca0d683444c8c21b3fbc908847bdd7e8906457e6dd384c44c27`.

The two M23 build logs and the Python-only inspection refresh log are retained. Serial receipts preserve doctor, import/native identity and an empty production-diff proof. MPI receipts preserve before/after authentication logs, rank identities, test source hashes, rank XML/log hashes and testcase parity. Hall and shear receipts retain their individual nested `run_report.artifact_identity`, `run_identity`, `bind_identity` and `execution_identity`; the checker authenticates their representation and compares artifact identities across matching serial/MPI cases. Eight available matrix component-cache JSON sidecars are retained separately. They are not a process-wide dyld inventory and are not substituted for the composite run artifact identity. Loaded-library attestation beyond the recorded native hashes is not reconstructed retrospectively.

## Reopened states and independent computations

The preserved scripts `raw/recheck_m23_saved_states.py` and `raw/recheck_m05_saved_states.py` use NumPy and the saved arrays/ledgers directly. Neither imports PoPS nor calls the author's oracle. Their original root-generated JSON reports, script hashes and all referenced state/ledger hashes are included.

M23 contributes 14 states (seven per backend) and four 384-record ledgers. The Hall Fourier check distinguishes the signed phase and SSPRK2 discrete amplification, including component permutation and zero Hall coefficient. Its maximum complex error is `4.121872359214323e-15`. At N32/N64 the observed phases are approximately −0.236931781 / −0.239230156 against the continuous reference −0.24. The approximately `3.94e-10` / `4.09e-10` norm defects agree with the finite SSPRK2 amplification; they are not evidence of general Hall stability or monotone energy decay. The three-component matrix case uses rank-one symmetric D, skew R, two weighted occurrences, exact stage-qualified ledger interpretation, and permutation. Its maximum ledger/update discrepancy is `1.2751233987966237e-17`.

Strengthened M05 contributes six states and six ledgers: N32, N64 and N128 on each backend, respectively 64, 128 and 256 oriented face records. It is the closed periodic shear/diffusion equation with ν=.03, initial cell averages of sin(2πx), T=.1 and N forward-Euler steps. The recomputation separately checks the continuous Fourier solution, discrete FE amplification, semi-discrete dissipation, FE quadratic correction, and every oriented flux and cell increment. Maximum error against the discrete solution is `1.2212453270876722e-15`; maximum ledger/update discrepancy is `1.6805133673525319e-18`. Continuous errors are not monotone for this coupled space/time sequence (cancellation), so no fitted convergence order is asserted. Six additional historical M05 states are preserved without retroactively assigning them the strengthened raw-ledger qualification.

## Offline verification

From the repository root, using Python with NumPy for the optional recomputation:

```bash
python docs/development/api_040/check_m23_m05_converged.py
python docs/development/api_040/check_m23_m05_converged.py --recompute
python -m pytest -q -o pythonpath= tests/python/unit/runtime/test_m23_m05_converged_checker.py
```

The checker requires an exact payload inventory and byte hashes, consistent XML testcase counts, source windows, installed identities, nested run identities, state/ledger receipts and finite JSON values. `--recompute` runs both preserved independent scripts and compares their complete per-state metrics with the archived reports (reproduced locally with NumPy 2.5.1). It runs no compiler, PoPS import, bind or simulation. The counter-tests reject state corruption, a missing MPI rank, erased historical failures even after rehashing, NaN in a ledger, mixed Python windows, changed nested artifact identity and a binary cache payload.

These checks establish the integrity and reproducibility of the bounded archived results. They do not qualify a full Navier–Stokes/thermal model, AMR, GPU, HPC, other dimensions for the scientific examples, or a general stability theorem. The refreshed implicit test group has serial Dim2 evidence only. No new native campaign was executed while assembling this archive.
