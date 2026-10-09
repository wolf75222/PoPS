# Independent M23 complete Program compilation

Reviewed source: `97f4c77c` + `3fc51ec8` + `7ab9f572` in
`PoPS-degenerate-diffusion`, branch `codex/api040-hall-coupled`, clean when
checked. The independent test lives in
`tests/python/unit/codegen/test_m23_complete_program_syntax.py`.

**Result: 5/5 complete translation units pass syntax compilation.** No
production change was needed. This is source/real-header evidence, not an
installed, linked, executed, numerical, MPI execution, or GPU receipt.

Each isolated `python -I` worker inserts only the selected checkout's Python
source and asserts the exact `pops.__file__`. It builds a public physical law,
`CoupledGradient`, `SSPRK2`, `FixedDt`, periodic Dim1 layout, then runs
`validate -> resolve -> ProgramModelGraph -> emit_cpp_program`. The complete
result is compiled without extracting or substituting cell lambdas. PoPS
headers are taken from that same selected source checkout, not the installed
SDK's older PoPS headers. Kokkos/MPI headers come from the real `pops-api040`
environment; its enabled OpenMP configuration is respected.

The cases are:

- Two components, exactly zero D and nonzero skew R, canonical and reversed
  component order. Both matrices and labels are permuted together.
- Three components, exact rank-one D = vv^T with v=(1,2,-1), and nonzero
  three-component skew R, canonical and reversed order. D's nullspace is
  preserved; no positive-definite replacement is introduced.
- The reversed three-component case with `div(flux) + 2*div(flux)`, retaining
  the distinct accepted occurrences of the same exact physical flux.

Each complete Program must contain the selected `PreparedCoupledGradient`
and signed `DiffusiveLawResult<..., true>` specializations. Both SSPRK2
evaluations and their two accepted ledger entries must be present; the
repeated case requires four entries, including two occurrence-1 identities.
The entire provider, scratch/status assembly, Program body, and ledger call
signatures are checked by the C++ compiler. This does not establish their
runtime effects or numerical validity.

Validation on 2026-09-29: Apple clang 21.0.0, C++20, `-fsyntax-only`,
`-fno-fast-math`, `POPS_NATIVE_DIM=1`, `POPS_HAS_MPI`, `POPS_HAS_KOKKOS`,
`KOKKOS_DEPENDENCE`, shared exception ABI and the real OpenMP headers. Final
pytest result: **5 passed in 25.57 s**. Ruff passed. Each temporary directory
retains the full emitted source, exact compiler arguments and stderr.

Reproduction (run from the checkout containing the independent test):

```sh
env -u PYTHONPATH \
  POPS_M23_SOURCE_ROOT=/absolute/path/to/PoPS-degenerate-diffusion \
  POPS_NATIVE_VARIANTS_ROOT=/Users/romaindespoulain/miniforge3/envs/pops-api040/lib/python3.12/site-packages/pops/_native \
  POPS_NATIVE_DIM=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  -o pythonpath=python tests/python/unit/codegen/test_m23_complete_program_syntax.py \
  --basetemp=outputs/m23-full-program-final \
  --junitxml=outputs/m23-full-program-final.xml
```

`POPS_M23_SOURCE_ROOT` defaults to the test's checkout after integration.
Native modules imported for authoring are not claimed to implement this
increment. No compile/cache/JIT API is called, no package or environment is
changed, and no numerical run is performed.

Source fingerprints at review:

- `include/pops/numerics/diffusion/prepared_diffusion.hpp`:
  `712a36865bb8b6bab3660ba482a53c9e35a618fe405d03d57d6ba9066cede385`.
- `python/pops/codegen/program_emit_diffusion.py`:
  `3c2627b5e75010ee1e10e70c396984d11908e3a91bc8d6acb57336c5e9efd983`.

The next required evidence remains actual installed Dim1 evolution and
saved-state comparison, plus MPI ownership/rejection/ledger behavior. This
review neither broadens M23 beyond its declared periodic Uniform constant
component law nor qualifies AMR, variable matrices, other dimensions, or an
implicit Hall integrator.
