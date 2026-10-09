# Native local-product layout vote: exact integral overload

The installed Dim2 reception at MAIN `cfef849` reached the generated Program
compilation after the separate storage-carrier and sparse-SSA detachment fixes.
All eight local-product/H05 cases failed at the same emitted call:
`all_reduce_max(product_layout_error_, ctx.prepared_execution_lane())`.
The preserved receipt is
`outputs/installed-cfef849-local-and-frontier/pytest.log` in the workspace:
8 failures, 11 other cases passed, 260.39 s.

The generated variable was `int`; the real `ExecutionLane` API has `Real` and
`long` overloads. Both implicit conversions are viable and neither wins. A
standalone syntax-only TU including the actual header reproduces the error,
independently of JIT caches or installed runtime execution.

The correction emits `long`. It leaves every layout, distribution and rank
comparison, the bitwise accumulation, the prepared lane and the all-rank refusal
unchanged. It neither removes the co-location requirement nor converts the vote
to floating point. Both direct original residuals (v1) and source/apply residual
subgraphs (v2) use this shared emitter.

`test_local_product_layout_vote.py` extracts the declaration and collective call
from complete emitted public Programs, then compiles them unchanged against the
real lane header. Both variants failed before the change with the same ambiguity
as the installed receipt. These tests do not execute MPI or the simulation;
centralized native re-reception is still required after integration.

Post-fix source/host validation: **19 passed in 7.00 s** using
`env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python
-m pytest -q --tb=short -o pythonpath=python` with
`tests/python/unit/codegen/test_local_product_layout_vote.py`,
`test_local_product_operator_host.py` and `test_local_residual_product.py`.
The pre-fix two-case vote test failed in 4.23 s; the subsequent host Newton
and source packing checks passed without changing their equations or tolerances.
