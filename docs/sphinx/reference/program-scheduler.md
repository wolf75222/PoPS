# Program scheduler

Spec 3 unifies cadence into a single Program scheduler: any Program operation can carry a
schedule, replacing the scattered block stride / substeps / source frequency knobs.

```{admonition} Status
:class: note
The schedule AUTHORING surface (the vocabulary below, policy chaining, recording a schedule on
a Program node, and the cacheable-capability validation) is available in `adc.time`. The
RUNTIME that honors a non-trivial schedule (the typed cache, `accumulate_dt`, the checkpoint) is
tracked by ADC-458, so a node carrying a non-`always` schedule is recorded and inspectable but
refuses to lower (it is never silently ignored). Per-block cadence that runs today is the
existing `substeps` / `stride` step policy ({doc}`time-program`).
```

## Authoring

```python
import adc.time as adctime
P = adctime.Program("step").bind_operators(mod)
U = P.state("plasma")
fields = P.call("fields_from_state", U, schedule=adctime.every(10).hold())  # refresh every 10
transport = P.call("flux", U, schedule=adctime.subcycle(4))                 # 4 inner steps
```

`P.dump_operator_ir()` shows the schedule on each node. A caching policy (`hold` /
`accumulate_dt`) requires the operator to be cacheable, declared on the model:

```python
mod.operator_capabilities("fields_from_state", cacheable=True)
```

otherwise the call is rejected: `operator 'flux' is not cacheable; cannot use schedule hold`.
See `examples/spec3/scheduled_fields_subcycled_transport.py`.

## Schedules

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
