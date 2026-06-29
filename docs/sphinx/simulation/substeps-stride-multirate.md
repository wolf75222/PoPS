# Substeps, stride, and multirate programs

Substeps and multirate behavior belong in the time program or in typed cadence
policies. They should not be hidden in runtime-only setters.

## Program-level structure

Use the `Program` language to express repeated stages, histories, schedules, or
operator calls. Ready macros in `pops.lib.time` can build common patterns.

```python
from pops.time import Program, every

T = Program("scheduled")
U = T.state("U", block="plasma")

# Schedules are metadata on IR operations. They are lowered to C++ orchestration.
schedule = every(4)
```

## Cadence at install time

When a compiled problem exposes a cadence descriptor, pass it to `sim.install`.

```python
sim.install(compiled, instances={"plasma": {"model": module, "initial": state}},
            cadence=cadence)
```

The installed runtime executes the cadence in C++.

## AMR

AMR compatibility follows the same rule as other public features: if a cadence
policy is public on the compiled problem route, it must either lower to AMR or declare a precise
descriptor incompatibility before runtime.
