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

## Cadence at bind time

When a compiled program exposes a cadence descriptor, pass it to `pops.bind`.

```python
from pops.time import CompiledTime

sim = pops.bind(compiled, state=state, cadence=CompiledTime(substeps=2, stride=1))
```

The bound runtime executes the cadence in C++.

## AMR

AMR compatibility follows the same rule as other public features: if a cadence
policy is public on `Case`, it must either lower to AMR or declare a precise
descriptor incompatibility before runtime.
