# Local runtime fixtures: converged checks

Base: `d913c865`. This change touches test support and seven Uniform fixtures;
it changes no production code, equation, initial value, tolerance or rollback
criterion. M11, M18 and the historical affine AMR fixture are excluded.

The previous local-product and H05 fixtures could fail a rank-local clock
assertion before entering a root-check broadcast. Operators v2, read-only
captures, affine bodies and conditional consumers had equivalent gaps involving
reports, temporal envelopes, unexpected exceptions, or local array conversion
between two state gathers. Such a fixture can strand a healthy peer after the
productive operation has already returned.

`tests/python/support/collective_checks.py` now gathers exception diagnostics
after a local-only check block or one operation. Bind and run exceptions retain
their actual class and message; a separately serialized `isinstance(error,
RuntimeError)` preserves subclass acceptance. Comparing the exact class name
was incorrect for `StepAttemptRejected`. Unexpected exception families remain
failures; they are not silently accepted as expected runtime refusals.

Global states are gathered one at a time. Each gather returns on all ranks,
then its result and local conversion are checked before the next gather. Root
array/oracle checks continue to use the existing root broadcast. Assertions on
already gathered diagnostic tuples are identical on every rank. Local clocks,
step reports and temporal envelopes use the new all-rank check.

Validation in the isolated checkout, without JIT or installation:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python \
  -m pytest -q -o pythonpath=python \
  tests/python/unit/runtime/test_collective_fixture_checks.py
```

Result: **6 passed**. The threaded two-participant seam proves that a rank-one
clock assertion, unexpected native return exception, or array-conversion error
stops both participants before the next operation. It also checks runtime-error
subclass diagnostics, rejection of another exception family, local result
ownership, and serial behavior.

Follow-up lint review: explicit default arguments bind every loop-dependent
lambda/closure. The intentional assertion failure uses `raise AssertionError`.
Ruff passes on all nine Python files in this ten-file change (the tenth is this
Markdown receipt), with no rule suppression; the six helper tests pass again.

All **20 cases collected** from these seven files:

- `test_local_residual_product_runtime.py`
- `test_api040_m22_h05_runtime.py`
- `test_api040_m22_h05_adversarial_runtime.py`
- `test_local_product_operators_runtime.py`
- `test_local_product_readonly_runtime.py`
- `test_affine_push_forward_runtime.py`
- `test_conditional_consumers_runtime.py`

These are source/test-harness checks. Installed MPI reception remains the
parent's responsibility. The helper does not repair divergence or a hang inside
a native collective operation, nor does it qualify compilation/setup failure
paths. Every participant must enter test boundaries in the same order.
