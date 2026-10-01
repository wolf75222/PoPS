# Independent reception of captured diffusion

Date: 2026-10-01. Production examined: `4ae5c4de2c73b88e1ed1644293df1c3ad0c66b9a`, with author tests/docs `3ceab3b719ad3ea316ddcf6b34b73396fce0e6d1`. This review changes tests and documentation only. No installed package, SDK, native build, JIT, MPI run or saved scientific state was modified or executed.

## Result and authority

No blocking production defect was demonstrated in this bounded source reception. The independent construction inserts spatial diffusion expressions using **two distinct, authentic State captures**, then exercises the actual physical binder, request validator, registered-method validator, resolver and C++ emitter. It uses the existing public original-field fixture for physical construction only; the coefficient evaluator and finite-volume mathematics below do not import the author's captured-diffusion oracle.

`Arithmetic@1` is explicit for this extension. The registered method, source contract `pops.spatial-field-residual@2`, coefficient face-policy URI, SolveRequest schema 2 and conditional ProgramIR 10 agree. Captured diffusion combined with the selected spatial Jacobi realization retains both policies. Null/unknown policies, reversed or cloned inputs, missing/reordered dependency records, altered input ids and stale points are refused. Re-digesting a forged source contract or registered field identity does not bypass those checks. The tests restore injected metadata before continuing and verify the authored Program remains unchanged after request refusal.

Uniform emission prepares persistent matrix coefficients before the original-residual callback, uses the explicit arithmetic template parameter, and evaluates candidate-dependent reaction terms against owned captures. AMR emission assembles captured coefficients before preparing the original operator and solving. The existing original AMR route already selects arithmetic scalar faces through its guarded original application; its preparation generation is retained and checked by the prepared residual. It does not require a second policy implementation. These are source/control-flow observations, not executed Kokkos/MPI results.

Diffusion depending on a candidate unknown is explicitly refused: the implementation does not provide the full spatial nonlinear JVP of `D(q)`. For accepted State-dependent diffusion, D stays fixed during the central finite-difference JVP of the **original** residual. Signed, nonsymmetric and singular finite coefficient matrices are admitted without a symmetry, positivity or SPD certificate. That admission does not establish nonlinear convergence. The existing Jacobi realization's explicit stored-DOF setup cost and limitations remain separate from coefficient admissibility.

## Independent mathematical witnesses

An autonomous Fraction stencil builds periodic conservative arithmetic fluxes in 1D, 2D and 3D with component widths 1, 2 and 5, different shapes and physical lengths. It checks exact conservation and component permutation for spatially varying signed matrix coefficients. Its original nonlinear residual adds coupled quadratic reactions and cubic terms. The central difference matches the independently derived directional derivative plus its explicit cubic truncation term while D is fixed. A transposed matrix countermodel changes the original operator. A scalar witness separates arithmetic and harmonic face laws without changing a tolerance.

These synthetic mathematical witnesses are not native saved-state receptions and do not qualify a PDE, boundary generality, GPU execution or AMR coarse/fine convergence.

## Executed checks

The coherent final selection passed **48 tests in 379.93 seconds**: 24 independent cases and 24 author regression cases. The independent cases include source emission through both providers and their selected-policy combination; they do not instantiate a native solver. Ruff passed for the two new Python files. The fresh legacy comparison below also passed. The unusually long final selection coincided with filesystem waits; it is not an MPI timing.

## Legacy comparison

The receipt `tests/review/sol61_captured_diffusion_independent_receipt.json` compares fresh interpreters loading parent Python `c28a8f4370bbf2e9d781ddf52ddcc59597ce2e65` and candidate Python `4ae5c4de2c73b88e1ed1644293df1c3ad0c66b9a`. Both invoke the same physical fixture file through the same helper and callsite. All three profiles agree exactly: AMR width 2 identity (IR8/request1), Uniform width 3 identity (IR8/request1), AMR width 5 selected Jacobi (IR9/request2).

The comparison includes authored/resolved IR hashes, C++ bytes, all three Module hashes, complete canonical Module manifest digests, complete SolveRequest digests, equation and solver identities. No provenance path or field is removed or normalized. The receipt records actual source/package paths and the helper digest. It is historical source evidence, not a portable golden for arbitrary fixture locations.

Reproduce with the source Python explicitly selected:

```sh
env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_captured_diffusion_independent.py tests/review/test_sol61_original_captured_diffusion.py --tb=short
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_captured_diffusion_parent_parity.py /absolute/parent/source/checkout
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_captured_diffusion_parent_parity.py /absolute/candidate/source/checkout
```

Use the same helper and sibling public fixture for both parity runs. Compare their entire `profiles` arrays, rather than updating a static plan identity tied to an absolute path. Native compile/bind/run, observed captured coefficients, checkpoints/replay and serial/MPI agreement remain the root owner's separate reception. The author's later public fixture `b39f4a9` is not received as native by this report.
