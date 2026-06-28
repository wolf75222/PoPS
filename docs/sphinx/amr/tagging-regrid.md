# Tagging and regrid

Refinement is described with typed AMR policies and executed by the C++ runtime.

```python
from pops.mesh.amr import PatchLayout, Refine, RegridEvery, TagUnion
from pops.mesh.layouts import AMR

layout = AMR(
    mesh,
    max_levels=2,
    ratio=2,
    regrid=RegridEvery(8),
    patches=PatchLayout(coarse_max_grid=32),
    refine=TagUnion(
        Refine.on("density").above(0.05),
        Refine.on("phi").gradient_above(0.5),
    ),
)
```

The subject passed to `Refine.on(...)` is a user/model name or handle. The
criterion object chooses the behavior.

## Regrid cadence

`RegridEvery(n)` asks the runtime to rebuild the hierarchy every `n`
macro-steps. `FrozenRegrid()` requests a hierarchy built once and then kept
fixed.

## Multi-block union of tags

For multi-block cases, the hierarchy is shared. The runtime forms a union of
all declared tag criteria, then clusters one hierarchy and applies prolongation,
restriction, reflux, and field updates per block.

The user does not script this loop in Python.

## Validation

Validation must catch:

- incomplete criteria;
- unknown model subjects when the model advertises its subjects;
- unsupported level count or refinement ratio;
- incompatible output/checkpoint policies;
- backend/platform routes that cannot serve the requested AMR feature.
