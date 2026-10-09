# Independent review of M27 mixed linear fields

Reviewed commit: `d86a0d854ab5f478789a9ffce5ff796ad2a59c61`, based on `0cd16d01`. Verdict is favorable for the bounded source and complete-Program syntax evidence below. No blocking defect was established. No package was installed and no native simulation, MPI execution or JIT was run by this reviewer.

## Original equations and method separation

The witness keeps both unknowns and both equations:

```
c_next - dt Delta_h(mu_next) = c_old
mu_next - c_next + epsilon² Delta_h(c_next) = 0
```

For the actual native convention `A(U) = reaction*U - div(diffusion grad(U))`, the authoring IR carries diffusion `[[0,.01],[-.08²,0]]` and reaction `[[1,0],[-1,1]]`. Independent tests decode the real generated coefficient literals and verify both matrices, including row/column permutation. A separate dense periodic assembly for 5, 8 and 13 cells and random nonuniform loads checks the two original residuals, mass preservation and quadratic energy decay. It agrees with the author's FFT 2×2 oracle in both unknown orders. These non-corpus sizes and loads are oracle counter-checks, not additional native qualification.

`CellCenteredGeneralCoupled` explicitly selects `finite_general`, and the lowered operator advertises general linear properties. CG cannot claim SPD for this method. The native coefficient preparation uses the added `RequireSPD=false` template argument only on that path; the default remains `true` with the existing finite/symmetric/strict-positive test. The finite admission does not promise that an arbitrary general operator is invertible or that GMRES converges. The tested M27 operator is invertible mode by mode: with `lambda_h <= 0`, the determinant is `1 - dt*lambda_h + dt*epsilon²*lambda_h² > 0`.

The existing Krylov implementation remains the authority for numerical success. In `generic_krylov.hpp`, the GMRES Arnoldi estimate only requests confirmation; `physical_true_residual_measurement` recomputes the physical `b-A(u)` before success or iteration-limit classification. The generated one-iteration complete Program contains the actual GMRES provider and cap, consumes failure before throwing, and consumes success before storing `accepted_mu` or calling `commit_many`. A dense counterexample computes the optimal first unpreconditioned Krylov candidate in span(rhs); its original residual is greater than .1, so the one-iteration negative witness is not relying on an arbitrary expectation of solver difficulty. A candidate satisfying only the mass equation also demonstrably violates the chemical equation.

This proves mathematical discrimination and the generated control ordering. The author's installed negative test still must demonstrate collective `iteration_limit`, unchanged State, history, diagnostics, step and time in the real runtime. Syntax is not a rollback execution receipt.

## Cell-mean projection authority

`FieldSolution.cell_mean_state` is an explicit projection to one conservative, cell-centered, cell-volume-average State endpoint on declared support. Authoring and lowering recheck the consumed solve component, exact unknown, field-problem identity, target State and block, load StateSpace, point, and constant-coefficient condition. The accepted route aliases the already solved component; it does not silently redistribute, interpolate or reconstruct a chemical field on the host.

Independent public counter-tests reject a wrong temporal endpoint, another State with matching names/support labels, and a homonymous foreign unknown without adding Program nodes. Positive tests preserve exact endpoint space/State/point, measure and selected component in both orders. The original source tests also reject unspecified sampling. The implementation deliberately requires every load input to witness the same exact target State and refuses nonconstant coefficients or AMR projection. Broader projection of co-located but distinct State supports is not inferred from this narrow map.

The constant-coefficient inference is valid only because the admitted coefficient AST is closed: literals, qualified State reads, binary arithmetic/power and negation/sqrt/abs. `encode_field_expression` admits no coordinate, runtime parameter, named variable or other non-State capture. Independent tests explicitly refuse the real analytic coordinate object, a RuntimeParamRef and a coordinate Var. `field_expression_dependencies` computes reads by walking that AST; the emitter re-derives them and rejects forged empty dependency metadata. That detached-payload counter-test also passes. Thus absence of State reads implies spatial constancy in this bounded grammar; it must not be generalized if coordinate or other input node support is later added. This does not certify constant expressions as finite: the native finite-coefficient guard remains necessary.

The native witness reads chemical potential from accepted history slot 1 and saves it independently of concentration. Both saved fields are reopened for the original-equation checks. The example converges local construction errors before bind, gathers accepted data and status before root-side assessment, and broadcasts root failures before another collective. The negative fixture catches every exception, converges refusal messages, checks the specific iteration-limit diagnostic, and then compares the unchanged public state/report under a collective check. These are source-reviewed guards; their MPI behavior remains to be executed centrally.

## Executed evidence

- **11/11 independent public projection/IR-matrix/closed-grammar checks**.
- **6/6 independent dense mathematical checks**, including comparison to the author's FFT oracle.
- **2/2 complete C++ Program translation units**: canonical order with the normal GMRES cap, and permuted order with cap 1. `/usr/bin/clang++ -std=c++20 -fsyntax-only -fno-fast-math` uses the exact reviewed PoPS headers and real Kokkos/OpenMP/MPI headers for Dim1. The provider, scratch allocation, coefficient preparation, GMRES workspace, consumption, history and State publication are compiled together.
- The combined pure/source rerun, including the author's existing nine tests, reports **26/26**. These counts exclude the two syntax checks and do not mean 26 native runs.

Small command/XML/hash receipts are under `evidence/m27-independent-review/`. Reproduce from the reviewed source checkout:

```bash
env -u PYTHONPATH python -m pytest -q -o pythonpath=python \
  tests/python/unit/fields/test_m27_projection_independent_review.py \
  tests/python/unit/physics/test_m27_independent_mixed_equations.py \
  tests/python/unit/codegen/test_m27_complete_program_review.py
```

The scope is the stable linear periodic mixed subcase at epsilon=.08 and dt=.01. Ten steps, the two-mode initial profile and its amplitudes are explicit witness choices. Neither this review nor the proposed native tests qualify nonlinear double-well dynamics, phase separation, AMR, GPU or HPC behavior. No change to physical equations, numerical criteria or production files was requested by this review.
