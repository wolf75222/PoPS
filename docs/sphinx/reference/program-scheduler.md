# Program scheduler

Spec 3 unifies cadence into a single Program scheduler: any Program operation can carry a
schedule, replacing the scattered block stride / substeps / source frequency knobs.

```{admonition} Status
:class: note
The unified Program scheduler RUNTIME (per-node schedules with checkpointed caches) is tracked
by ADC-458. Today the compiled time Program runs every node every step; per-block cadence is
expressed by the existing `substeps` / `stride` step policy ({doc}`time-program`). This page
documents the target design and the current mechanism.
```

## Schedules (design)

A schedule decides when a node is due:

| Schedule | Meaning |
| --- | --- |
| `always()` | due every step (the default) |
| `every(N)` | due every N macro-steps |
| `when(cond)` | due when a runtime condition holds |
| `on_start()` / `on_end()` | due at the first / last step |
| `subcycle(count, dt)` | structured sub-cycling of a block |

and a policy decides what happens when it is NOT due:

| Policy | Behaviour |
| --- | --- |
| `recompute` | always recompute (default) |
| `hold` | reuse the last cached value |
| `skip` | produce nothing (diagnostics only) |
| `zero` | return zero (optional contributions) |
| `accumulate_dt` | accumulate the skipped dt and apply with `eff_dt = sum(dt_skipped)` |
| `error` | error if not due |

With a variable `step_cfl`, `accumulate_dt` must use the real sum of the skipped dt, not
`N * dt_current`.

## Cacheable capabilities

`hold` is only valid on a cacheable operator. An operator declares this:

```python
m.operator_capabilities("fields_from_state", cacheable=True, stale_allowed=True)
m.operator_capabilities("explicit_rate", cacheable=False, requires_fresh_inputs=True)
```

so requesting `every(10).hold()` on a non-cacheable operator is an error:
`operator 'explicit_rate' is not cacheable; cannot use schedule hold`.

## Cache and checkpoint

A `hold` cache stores the cached value, the last update step/time, the input versions, the
accumulated dt and a validity flag. It must be typed, C++-allocated, Kokkos/MPI-safe,
checkpointed and restored, and invalidated on a Program-hash mismatch. This runtime is the
core of ADC-458; the checkpoint format and the restart validation extend the existing
checkpoint path.

## Current mechanism

Until ADC-458 lands, use the step policy: a block advances `substeps` sub-steps of `dt/substeps`,
or `1` macro-step out of `stride`. See {doc}`time-program`.
