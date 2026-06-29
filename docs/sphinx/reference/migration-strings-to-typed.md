# Strings to typed objects

The public API no longer documents string selectors for behavior. Strings name
objects chosen by the user. Typed descriptors choose algorithms, layouts,
solvers, policies, and backends.

## Conversion table

| Old selector style | Public typed style | Package |
| --- | --- | --- |
| Riemann flux token | `Rusanov()`, `HLL()`, `HLLC()`, `Roe()` | `pops.numerics.riemann` |
| Reconstruction token | `FirstOrder()`, `MUSCL(...)`, `WENO5Z()` | `pops.numerics.reconstruction` |
| Limiter token | `Minmod()`, `VanLeer()`, `Superbee()` | `pops.numerics.reconstruction.limiters` |
| Time scheme token | `ssprk2(P, block)`, `ssprk3(P, block)`, `rk4(P, block)` | `pops.lib.time` |
| Backend token | `Production()`, `AOT()`, `JIT()` | `pops.codegen` |
| Uniform target | `Uniform(mesh)` | `pops.mesh.layouts` |
| AMR target | `AMR(mesh, ...)` | `pops.mesh.layouts` |
| Regrid cadence | `RegridEvery(n)`, `FrozenRegrid()` | `pops.mesh.amr` |
| Elliptic solver token | `GeometricMG()`, `FFT()` | `pops.solvers.elliptic` |
| Runtime param kind | `RuntimeParam(...)`, `ConstParam(...)` | `pops.params` |
| Output format token | `OutputPolicy(format=HDF5(...))` | `pops.output` |
| External brick id alone | `CompiledBrickRef(manifest, native_id, expect_category=...)` | `pops.external` |

## What stays a string

Strings remain valid when they are names:

```python
Model("euler")
PoissonProblem(name="phi", unknown="phi")
RuntimeParam("nu", default=0.1)
T.state("U", block="ions")
sim.install(compiled, instances={"ions": {"model": ions, "initial": U0, "spatial": spatial}})
```

Later references should use handles returned by the API when handles exist.

## Documentation rule

Do not write examples that present a string as a behavior choice. If a page
must mention a native token, call it an internal lowered id and show the typed
descriptor users should write.
