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


Follow-up to independent review fdf: each actual native Program diagnostic group
must contain the complete finite nonnegative float triplet norm/reference/rawratio.
The rawratio equals norm/(reference>0?reference:1) exactly. Complete emitted solve
groups are required (two Uniform, one AMR, ordered by SSA node id). These names
come from the real emitters program_emit_nonlinear_field.py and
program_emit_amr_original_field.py; no diagnostic is inferred from a text log.
Admission independently computes Relative coefficient*reference, the explicit
floor maximum, or Absolute coefficient; no legacy max(1,reference), no infinity
allowance. Original physical error guards remain unchanged.

Uniform reopens actual saved NPZ solutions and captured coefficient/load arrays
and recomputes full Original residual including periodic diffusion using the
existing independent NumPy original_lhs oracle. Zero and .8*first-solution seeds
provide independent complete reference norms. References compare within2e-14;
norm comparison has2e-13 absolute allowance for arithmetic ordering only.
Both actual native norm and independently recomputed norm must satisfy the
unmodified typed cutoff. AMR physical checks remain genuine saved active-state
checks; this fixture has no independent arbitrary composite coarse/fine flux-L2
oracle, so its triplet admission is actual native diagnostic evidence, not an
invented independently reconstructed AMR norm. ROOT authenticates artifact/native
identity for both backends during reception.

Rollback compares dtype, shape and C-order bytes, including signedzero; reopens
persisted NPZ and compares both saved images to their actual captured snapshots.
Uniform histories, accepted diagnostic/clock representation before and after are
persisted in the receipt. AMR fullcarrier byte equality remains stronger coverage.
No nonexistent Original component-provider field snapshot is synthesized.

Follow-up Source suite:21PASS30.17s, zero skip; includes8real public authoring
cases and13pure diagnostic/bit/original-oracle checks. Native collect-only8nodes
passes0.67s. No Native execution or acceptance is claimed.
