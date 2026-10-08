# M26 independent source and host review - 2026-09-30

Reviewed candidate: `ccfc656e7b4d8f4f2e765872a4ef0e5e989bb710`.
Checkout: `PoPS-resource-lifetime`, branch `codex/api040-m26-frozen-review`.
Reviewer: GPT-6.1 Sol. This review changes tests and this report only. Existing
untracked evidence and iCloud duplicate files were preserved.

## Decision

Source approval for integration of the finite measured interaction candidate.
No candidate defect was demonstrated. The final focused run passed **38 tests,
zero failures, zero skips**, in 23.01 s: 28 independent algebra/input tests,
four independently inspected complete Program translation units compiled with
Clang, and six existing candidate tests.

This is source/IR and host **syntax compilation** evidence. No shared library
was linked, no JIT or native Program was executed, and no installed extension,
MPI execution, GPU execution, or GitHub CI qualification is claimed.

## Independent checks

- The four-DOF counterexample uses weights `(0.125, 0.5, 2, 1.25)` and a signed
  symmetric kernel with independent NumPy expectations. It distinguishes
  `A = W diag(m)` from row weighting and the Euclidean transpose. It checks
  action, measured adjoint, pairing, energy, directional derivative, central
  energy variation, and the exact quadratic increment for two density pairs
  and joint support permutations, including inverse permutation.
- Zero, negative, NaN, either infinity, and Boolean measure weights are
  rejected. Malformed, nonsymmetric, NaN, infinite, Boolean, and string kernel
  entries are rejected before `FiniteLinearMap.apply` is reached. A zero
  kernel is admissible. Individually finite `W` and `m` whose product overflows
  are rejected when the map is constructed, before application IR is made.
  Kernel symmetry uses exact represented equality; no tolerance or silent
  symmetrization is introduced.
- Input lists are copied into immutable tuple authorities. A differently
  ordered support is rejected. The example's density inputs are checked
  against independent trigonometric formulas for both variations and both
  orders, including inverse permutation, contiguous `(12, 1)` shape, and
  separate storage. Zero, NaN, and infinite witness densities are refused
  before binding.
- The complete Program is emitted through `pops.validate`, `pops.resolve`,
  `ProgramModelGraph`, and `emit_cpp_program`. The worker authenticates its
  `pops` import to this checkout's `python/pops/__init__.py`.
- Both emitted 144-coefficient C++ arrays are parsed and compared against the
  independent rank-two Fourier basis formula. Both finite calls load the
  twelve components from the same spatial index, in component order. The
  emitted source contains three nonfinite diagnostic guards before the
  accepted-state `ctx.commit_many` boundary. These are static checks of the
  generated program, not a runtime rejection/rollback qualification.
- Four complete translation units cover canonical/permuted order and the
  public uniform measure/an injected nonuniform measure `(i+1)/16`. The latter
  only modifies the isolated emission worker; the public witness remains its
  original uniform measure. These weights are attached to DOF labels before
  permutation, so transposition or row weighting is detectable in emitted
  coefficients.

The maximum coefficient difference from the independent Fourier basis is
`7.353810258116794e-17` for the uniform case and `6.618429232305115e-16` for
the nonuniform case, in both orders. The measured matrix's symmetry defect is
zero for the uniform case and at most `5.551115123125783e-17` for the nonuniform
case. The latter comes from the second floating multiplication of already
rounded coefficients; exact kernel symmetry remains enforced.

## Reproduction and evidence

From this checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/python/unit/numerics/test_m26_measured_independent_review.py \
  tests/python/unit/codegen/test_m26_complete_program_review.py \
  tests/python/unit/numerics/test_api040_m26_finite_interaction.py \
  -q --junitxml=outputs/m26-sol61-reception-20260930.xml \
  --basetemp=outputs/m26-sol61-reception-20260930
```

The source-only `PYTHONPATH` is intentional for these tests and does not
qualify the installed PoPS package. The compiler workers run with Python `-I`,
authenticate the source import, and compile with `-std=c++20 -fsyntax-only
-fno-fast-math`, `POPS_NATIVE_DIM=1`, and the source SDK headers. Actual Kokkos,
MPI, and OpenMP headers come from the existing interpreter prefix and libomp;
the environment was not modified. The four compiler processes ran sequentially.

Local evidence retained outside the commit:

- `outputs/m26-sol61-reception-20260930.xml`: final 38-test JUnit result.
- `outputs/m26-sol61-reception-20260930/test_m26_full_program_syntax_F0/`:
  uniform canonical complete C++, compiler command, compiler stderr.
- Corresponding `F1`, `T0`, `T1` directories: uniform permuted, nonuniform
  canonical, nonuniform permuted complete C++ and compiler records.

The previous `outputs/m26-independent-review*` evidence is preserved and is
not used as a current native qualification.

## Native and scientific boundary

The public witness uses twelve finite quadrature DOFs as components at one
genuine native cell. The runtime source has two density inputs, two native
finite actions, measured diagnostics, a same-artifact rebind for the second
density pair, a separate permuted artifact, inverse permutation of saved
arrays, three NPZ archives, and source/oracle/native/artifact identity fields
in its receipt. The integration test checks the common artifact identity for
the two rebinds. These paths were read, but their execution and receipts were
not produced by this review.

Native reception must still run
`tests/python/integration/runtime/test_api040_m26_finite_interaction_runtime.py`
against the authenticated installed candidate and inspect all three actual
saved states. A source test or host syntax compile cannot replace that result.
The finite witness does not qualify aggregation-diffusion PDE evolution,
gradient/diffusion fluxes, or spatial convergence at N=32/64/128. Those remain
outside this candidate's closed finite interaction claim.
