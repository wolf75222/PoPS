# PoPS model DSL design

This root design note is intentionally short. The current normative references
are the Sphinx pages under `docs/sphinx/reference/`.

## Design invariant

`pops.physics` is a writing facade for physical models. It lowers to the typed
`pops.model` representation. It does not compile, run, own runtime data, or
execute numerical loops.

```text
pops.physics  -> authoring facade
pops.model    -> typed operator IR
pops.time     -> Program IR
pops.codegen  -> C++ generation and native route selection
pops.runtime  -> C++/Kokkos/MPI execution facade
```

Python creates descriptors, handles, expressions, and cases. C++ performs the
calculation.

## Public model flow

```python
from pops.physics import Model

m = Model("euler_poisson")
U = m.state("U", components=["rho", "mx", "my"])

# Declare fluxes, sources, fields, capabilities, and diagnostics here.
module = m.lower()
```

The lowered module is inserted into a `pops.Case`:

```python
case = (
    pops.Case(layout=layout)
    .block("plasma", physics=module, spatial=spatial)
    .field(poisson)
    .time(program)
)

compiled = pops.compile(case, backend=Production())
sim = pops.bind(compiled, state={"plasma": U0}, params=params)
sim.run(t_end=1.0, cfl=0.5)
```

The older pattern where a model compiles itself or where a user constructs a
runtime system directly is not the documentation front door.

## Descriptor rule

Any Python object that selects a compiled behavior is a typed descriptor:

- mesh layouts: `Uniform(...)`, `AMR(...)`;
- Riemann solvers: `Rusanov()`, `HLL()`, `HLLC()`, `Roe()`;
- reconstruction and limiters: `MUSCL(...)`, `WENO5Z()`, `Minmod()`;
- field problems and solvers: `PoissonProblem(...)`, `GeometricMG()`, `FFT()`;
- codegen backend and platform: `Production(...)`, `AOT(...)`, `JIT(...)`;
- output, checkpoint, profiling, and optimization policies.

Descriptors declare requirements, capabilities, options, availability, and a
native ID. Validation happens before runtime execution.

## Handles and expressions

State handles, field handles, expressions, equations, stage handles, and bound
runtime arrays are not descriptors. They carry references or data; they do not
choose a route.

## Package ownership

- `pops.physics`: author physical formulas and lower to `pops.model`.
- `pops.model`: typed operator-first IR.
- `pops.time`: language for programs and stages.
- `pops.lib.time`: ready-made time schemes.
- `pops.numerics`: finite-volume, Riemann, reconstruction, limiters.
- `pops.fields`: field and elliptic problems.
- `pops.linalg`: matrix-free operators and algebraic problem objects.
- `pops.solvers`: compiled solver descriptors.
- `pops.mesh`: meshes, layouts, AMR policies, boundaries, geometry.
- `pops.lib`: ready-made models and presets, not core authoring primitives.

## No Python numerical backend

Debug helpers may exist under `pops.experimental` or `pops.debug`, but the public
production docs must not present NumPy or Python flux loops as solver backends.
They are not the PoPS runtime model.
