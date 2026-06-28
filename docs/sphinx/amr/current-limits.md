# AMR compatibility contract

This page replaces the old "current limits" framing. Public AMR behavior is a
contract, not a backlog list.

## Public contract

The public assembly surface is `Case`. A case that is valid on a uniform layout
should be valid on an AMR layout unless a descriptor declares a precise
mathematical incompatibility.

Examples of legitimate incompatibilities:

- a spectral FFT field solver that requires one periodic uniform box;
- a geometry descriptor that declares no AMR lowering;
- an output format that explicitly supports only single-level fields.

Examples that are not legitimate public limitations:

- a missing Python binding;
- a missing codegen branch;
- an unimplemented install path;
- an AMR runtime route that exists in C++ but is not reached from `pops.compile`
  or `pops.bind`.

Those must be implemented or the public API must not expose the feature.

## Validation

AMR descriptors must declare:

- requirements;
- capabilities;
- options;
- availability;
- lowering metadata.

The route should fail before runtime when a combination is invalid.

```python
layout = AMR(mesh, max_levels=2, ratio=2)
layout.validate()

case = pops.Case(layout=layout).block("plasma", physics=model)
case.validate()
```

## Execution

AMR execution is C++/Kokkos/MPI:

- tag cells;
- cluster patches;
- exchange halos;
- prolong and restrict;
- reflux flux corrections;
- solve fields;
- run sources and time stages;
- write output and checkpoints.

Python must not run per-cell AMR logic.
