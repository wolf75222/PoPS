# M22: bounded original-spatial capability diagnosis

2026-10-01; source examined: `bfae73f34174079bc6f2d8f4d50aa11d08eca5b5`. This note and its test are the only changes. No production/header, installed package, SDK, build, JIT, native run or saved-state evidence was changed. Mathematical arrays below are explicitly synthetic countermodels, never native M22 observations.

## Authority and received scope

The supplied normative handoff is at `PoPS_Codex_handoff_0.4.0` in the preserved primary checkout. Its baseline `reference/PoPS_API_v0.4.0/baseline/test_registry.json:329-342` and `results/corpus_matrix.json:450-465` prescribe only the normalized H05 two-reservoir exchange and scalar closure bounds. `context/CORPUS_ORIGINAL.md:223` proposes homogeneous equilibration, Marshak and discontinuous-opacity tests. None closes the physical Marshak initial/boundary problem.

The current `docs/development/api_040/corpus.json:488-506` receives four H05 cases serial/MPI2, including independent nonuniform cells, rate rebind and rollback. This is a registry assertion about its linked historical receipts, not a native reception repeated here. H05 satisfies

`E1-E0 + dt*k*(E1-R1)=0`, `R1-R0 - dt*k*(E1-R1)=0`.

At `(E0,R0,dt,k)=(2,1/2,2/5,4/5)`, the exact result is `(70/41,65/82)`. R is the second **normalized energy reservoir**. Calling it a temperature, substituting a T⁴ law, or attaching a spatial Marshak interpretation would change the problem. Independent closure algebra establishes χ(0)=1/3, χ(1)=1 and its realizability domain; it does not establish the full pressure tensor or transport equation.

PoPS already provides nonlinear Model flux/source authoring and finite-volume evolution. This diagnosis is about a **joint original spatial implicit residual**, not a blanket absence of nonlinear transport. T3 also provides coupled field products, nonlinear local reactions, central full-callback finite differences, actual Newton/GMRES, original recheck and transactional publication. The captured-D extension adds arbitrary matrix expressions of owned frozen State captures with explicit Arithmetic@1. These are useful capabilities; neither makes a capture into a Newton candidate.

## Two distinct remaining gaps

**Candidate-dependent diffusion:** `_program_nonlinear_problem.py:39-85` encodes D without supplying the unknown tuple. Its public refusal states `unknown-dependent D has no full spatial nonlinear JVP realization`. `_program_expression.py:46-83` otherwise distinguishes exact captures from candidate unknowns. Uniform emission prepares D before its candidate callback (`program_emit_nonlinear_field.py:73-84`). AMR evaluates its already prepared operator, then adds local nonlinear terms (`prepared_amr_field_residual.hpp:164-175`); the preparation generation is authenticated. D therefore remains fixed through Newton and its JVP. A physical diffusion limit with D depending on a solved material state would need a new capability, once that law is actually specified. This note does not assume that every opacity or Marshak variant has such a law.

**Candidate-dependent pressure flux:** χ depends on the radiation candidate. Already at constant E=13, the scalar χ law gives Eχ=13/3 for f=0, but Eχ=7 for f=8/13 (the square root is exactly 22/13). Freezing the closure changes it. A joint implicit M1 transport residual must evaluate its declared pressure/flux from each candidate and apply the spatial divergence with its declared numerical face law. The nonlinear FieldProblem compiler currently admits Laplacian/DivCoeffGrad plus local Reaction terms, not an arbitrary candidate-valued first-order face flux. Existing linear gradient couplings or explicit Model fluxes do not establish this new joint implicit realization. This second gap remains outside the single production tranche proposed below.

## Concrete countermodel

The independent test uses the generic periodic equation

`F(q)=q+q³−div((1+q²) grad(q))−f`,

with rational spacing, nonconstant rational states, signed directions, and 5/7/11 cells. This is **not** a claimed radiation constitutive law. Arithmetic face means are chosen explicitly for this mathematical witness.

For a distinct seed s, manufacture f from `F_frozen(q;s)` using D(s). Its substituted residual is exactly zero and the diffusion is exactly conservative, but `F_original(q)` is nonzero. The result depends on s even though the original equation has no seed parameter. Conservation and positivity alone cannot distinguish the equations. Component/local inversion or a dense prototype solve cannot repair this substitution.

The exact original directional derivative includes

`h + 3q²h − div(D(q) grad(h)) − div(2qh grad(q))`.

The frozen-D derivative omits the last term. The tests derive the central-FD cubic remainder independently and check it as an exact Fraction identity, with no tolerance adjustment. They also test conservation and cyclic relabeling. Public construction through the genuine FieldProblem/binder confirms the present fail-closed refusal for widths 1/2/4 and permutations, before any native solve. The helper's intended Uniform/AMR layout flags do not qualify those providers: this binder refusal occurs before layout/provider selection.

## One feasible production tranche

Implement **candidate-valued diffusion in the original Uniform residual**, initially through the already prepared periodic/homogeneous-Neumann field boundaries, with arbitrary supported dimensions and component widths. Use the existing distributed cell/patch storage and Kokkos kernels. Do not introduce a dense global matrix, host field gather, fixed cell count, model-name branch, automatic EOS, or inverted local material equation.

1. In `fields/_program_nonlinear_problem.py` and `_program_expression.py`, authenticate D expressions against both exact physical unknown slots and immutable captures. Keep seeds separate. Introduce an explicit new source/realization contract, proposed `pops.spatial-field-residual@3`, selected only by the new method option. Retain @1/@2 bytes and their refusal semantics. Include candidate coefficient role, face policy, physical equations, points, capture identities and method in request/cache authority. `fields/methods.py`, `codegen/program_field_plan.py` and `time/_program/serialization.py` own method selection, registered-physics recompile and conditional IR promotion.
2. In `codegen/program_emit_nonlinear_field.py`, move only **candidate** coefficient evaluation into the original F callback. For every q: authenticate layout/point/attempt/captures, prepare candidate halos, assemble D(q) into owned private scratch, converge local allocation/finite errors on the prepared lane, prepare D halos, then evaluate the same conservative spatial operator plus original local terms. Frozen capture-D remains prepared once. Retain finite signed/general matrices without adding an SPD assumption. No scratch becomes a published material state.
3. Reuse `runtime/program/prepared_spatial_residual.hpp`'s existing q±h·v callback evaluations. They must re-evaluate D on both candidates, so the existing central FD now differentiates the **whole original** operator. Line search and the final original recheck call the same evaluator. A coefficient refresh outside this callback or only at the Newton iterate is insufficient. Keep tolerances, method, restart and acceptance controls explicit and unchanged.
4. Retain request/outcome authority and all-rank refusal before publication. Admit this new capability only where implemented. In particular, keep AMR candidate-D and selected SpatialBasisJacobi routes refused until they have the correct candidate linearization and coarse/fine flux authority; the frozen AMR generation contract cannot silently be reused for a mutable D. Legacy frozen-D Uniform/AMR and existing preconditioners remain unchanged.

This is one coherent generic mechanism, not two proposed implementations. Its first native acceptance should use a manufactured original equation with observed coefficients/residuals, distinct seeds, permutations, multipatch/MPI ownership, nonfinite failure and unchanged state/time on rejection. Root owns that future build/reception. AMR candidate-D, candidate pressure-flux divergence, full M1/Marshak physics and scalable nonlinear preconditioning are not received by this tranche.

## Physical data still required

The corpus does not supply the material energy/temperature EOS and heat capacity, radiation constant and units/normalization, complete T⁴ emission/absorption coupling, absorption versus scattering opacity laws and their discontinuities, full moment pressure/flux/tensor convention, transport versus diffusion limiting equation, radiation/material initial states, Marshak illumination/boundary treatment, spatial/time resolutions, or reference solution/acceptance criteria. Each must be declared by the physical problem authority. H05 and χ endpoint tests cannot fill these choices. An unspecified law is not a solver defect and must not be invented to turn this capability witness into M22 completion.

## Reproduction

```sh
# Autonomous mathematical countermodels: no PoPS import or installed native package.
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath='' tests/review/test_sol61_m22_original_spatial_gap.py -k 'not public' --tb=short
# Actual public binder refusal, explicitly using this source tree; no compilation.
env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_m22_original_spatial_gap.py -k public --tb=short
```

Executed: **11 autonomous mathematical tests PASS in 0.09s**, and **3 actual public source refusal tests PASS in 1.72s**. Ruff and the staged whitespace check also pass. Results and limitations are source/math evidence only. No real Marshak, M1 pressure transport, Newton solution, native HPC scaling or native rollback is asserted here.
