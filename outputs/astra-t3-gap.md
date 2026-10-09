# T3 residual numeric-domain gap — reproduced and corrected

Red reproduction base: 226520b. Correction follows C36 commit 1c2d4ec in isolated PoPS-resource-lifetime.

Public LocalResidual with explicit frozen old state and a named source S(u)=minimum(sqrt(-u),0*u), implicit residual R=u-u_old-dt*S(u), seed=old=1. IEEE fmin(NaN,0)=0 masks an evaluated domain error; R then equals zero. The inline Program version has intermediate finite guards; the source/apply residual lowering does not.

Actual generated source saved in outputs/astra-t3-red-generated.cpp:
`residual_value_4[0] = Kokkos::fmin(std::sqrt((-quantity)), (pops::Real(0) * quantity));`

Exact source test: tests/python/unit/api040_t3_residual_domain.py. Executed with env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath=python tests/python/unit/api040_t3_residual_domain.py --junitxml=outputs/astra-t3-red.xml. Result: **2 passed, 1 failed** (named source lacks quiet_NaN/intermediate finite observation). The lifetime env contains no pytest; no package was installed. No PoPS native solve executed in this probe.

Root cause: program_emit_model_kernels.py::_emit_residual_eval source/apply branch routes every root through _checked_inline_expr. That helper opts into checked CSE only when where/rounded occurs; ordinary compound expressions are inlined. The common pointwise-expression residual branch instead calls _cse_emit(materialize_all=True, return_names=True) and propagates an invalid intermediate to the actual native solver via NaN residual.

Generic correction now implemented in program_emit_model_kernels.py: expand owned primitive recipes inside each source/apply call (not eagerly for the whole residual), bind the exact actual argument/capture, use existing common checked CSE with leaf bindings, return NaN in every residual component upon an evaluated invalid intermediate. Preserve lazy branches and rounded barriers via that same common emitter. Remove eager primitive preamble that evaluates recipes on the iterate even when the actual argument is frozen data. Source/apply, finite/invalid, captured argument, inactive/active where, and rounded tests are implemented. No model-name cases or alternate host solver.

This supports C15-C19 original-domain/fallible-result correctness needed by constitutive implicit science M05+; it does not qualify all global residuals, mixed spaces, Schur reconstruction, gauges, or heterogeneous solves. Current contracts registry still labels C17/C18 partial_gap. Existing top-level Program finite-guard claims must not be extrapolated to named residual sources.


## Validation of the correction

`tests/python/unit/api040_t3_residual_domain.py` compiles the exact lambda extracted from public-API generated code into a small CPU shared library, alongside the real `prepared_local_nonlinear.hpp` provider. It uses the actual prepare/solve entrypoints, without replacing the Newton solver or changing its numerical controls. Namespace aliases supply only scalar std::fmin/isfinite; no Kokkos/device behavior is claimed. `FENV_ACCESS ON` in this diagnostic harness makes hardware exception flags observable when asserting that an inactive branch does not evaluate an invalid recipe; production correctness is also checked through the returned finite/invalid residual.

The four invalid source/primitive/active-where/apply equations return actual LocalNonlinearStatus::kInvalidEvaluation rather than accepting the masked zero; the returned candidate remains unchanged. Six valid equations converge using the actual provider, and their original emitted residual is re-evaluated against the unchanged 1e-12 tolerance. Distinct unknown, seed and frozen-capture values discriminate wrong bindings. A captured primitive whose domain is invalid on the iterate but valid on its actual frozen argument verifies absence of eager wrong-argument evaluation. A captured NaN hidden by min has no other route to the final residual and is still refused.

Targeted command (pops-api040 environment, env -u PYTHONPATH, -o pythonpath=python): this new file plus test_local_auxiliary_outcome.py and test_program_expressions.py: **51 passed in 4.15 s**. One existing textual test now expects Kokkos::isfinite for LocalResidual, retaining std::isfinite for the separate physical emitter. No production file beyond the local model-kernel lowering changed.

An attempted legacy test_time_local_newton.py could not start its process-isolated installed runtime: no native dimension selected. This is not counted as solver runtime success. Mesh/distributed Program transaction, MPI, GPU and science M05 remain unqualified by this cell-level CPU test. No build/install/configuration was performed.

Common conditional/rounding codegen, generated Cartesian guards, and adversarial Program-expression suites: **32 passed in 2.46 s**. Together with the targeted run, **83 passed** across the selected source and CPU-cell checks. `git diff --check` passed. New file uses the requested `api040_t3_*` name and must be explicitly selected by pytest (it does not begin with the default `test_` prefix).
