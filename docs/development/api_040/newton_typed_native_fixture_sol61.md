# Public Original typed Newton fixture (prepared, not received)

Parent source is 93b57249 (all-kind exact convergence consensus), independently
reviewed by Banach 5cb8ff12. This separate fixture adds no production behavior.
Existing numeric tests retain their default solver and historical assertions.

ROOT owns SDK8 build and actual execution. Exact node:
`tests/python/integration/runtime/test_public_newton_typed_original.py::test_public_typed_original_policy`
Eight cases: Uniform/partial two-level AMR × Relative, Relative with AbsoluteFloor,
Absolute, overflow. Use the ordinary repository installed-package Native test
command in Serial and MPI2. The fixture asserts installed package location,
actual native dimension2 and ABI8, then validate→resolve→compile→bind through
existing authentic supports. It does not substitute a Native module.

Scientific equations, FD steps, Krylov settings, grids, active-mask assertions,
state/residual bounds remain those of the existing Original fixtures. Uniform
retains both solves and its nonzero second seed; AMR success retains zero seed,
reordered three-component product and full original saved-state oracle.
Only typed convergence policy changes. Overflow AMR uses public seed_product at
the current c0 point, constant2 in the original physical FieldSpace, witnessed by
the unchanged coefficient state. No equation, forcing or coefficient changes.
The first reaction at seed2 is at least10, its prescribed load below.2, on the
unit-volume periodic box: Original volume-L2 reference exceeds1. Uniform checks
its complete zero-seed residual norm independently from captured loads exceeds1.
Thus finite Real-max Relative coefficient produces nonrepresentable cutoff.

Overflow must fail collectively before accepted publication. Exact state arrays,
clocks/history/diagnostics and AMR full carrier bytes/registry must equal the
pre-attempt snapshot. Rank-local before/after NPZ and AMR POPSCAR1 bytes plus hash
receipt are persisted before assertions; no approval or scientific receipt is
manufactured. Ordinary provider plan consensus authenticates policies before
workspace admission. Generic Engine MPI drift/zero-JVP coverage belongs to the
separate real C++ test ConvergenceDriftAndOverflowVoteBeforeKrylov.

Source-only evidence: all8 genuine public validate/resolve cases passed30.40s,
Native collection exposes8nodes. `_pops` absent, actual private Source import;
no Native execution, ENV writes, JIT, new physics or qualification claim.
Reproduction: `env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python`
with private WT/python explicitly inserted, then pytest
`tests/review/test_sol61_newton_native_fixture_source.py`.
