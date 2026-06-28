# Add a numerical flux

This how-to selects the Riemann numerical flux that a block uses to combine the
reconstructed interface states: `rusanov`, `hll`, `hllc`, or `roe`. The numerical flux is
carried by the block (the spatial scheme), not by the model, so the same model runs with
different fluxes. For the math behind these solvers, see
[fluxes, sources and eigenvalues](../concepts/fluxes-sources-eigenvalues.md).

This page assumes you already have a model and a `Case`. If not, start with the
[tutorial](../getting-started/tutorial.md).

## Choose a flux

Pass the flux through `pops.FiniteVolume`, where the numerical flux is named `riemann`. Every
selector is a TYPED `pops.numerics` descriptor (Spec 5 sec.7 rejects a bare string):

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod

spatial = pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov())
```

Replace `riemann` with one of these typed `pops.numerics.riemann` descriptors, matched to your model:

- `Rusanov()`: the generic minimal flux. It needs only `max_wave_speed`, so it works with any
  model. Use it as the default.
- `HLL()`: a generic flux with signed waves. It requires `model.wave_speeds` (a native
  isothermal or compressible model, or a DSL model that declares the primitive `p`). It is the
  path for a non-Euler model with signed waves; pair it with `Minmod()`.
- `HLLC()` and `Roe()`: contact-resolving (HLLC) and Roe-linearized solvers. They run on the canonical
  2D Euler layout (4 variables and perfect-gas pressure: a compressible transport), and also
  generically on any model that supplies the capability hooks -- `contact_speed` plus
  `hllc_star_state` for HLLC (`HasHLLCStructure`), or `roe_dissipation` for Roe
  (`HasRoeDissipation`), including some 3-variable non-Euler models. In the DSL, emit the hooks with
  `m.enable_hllc()` / `m.enable_roe()`. Both read a pressure, so declare the primitive `p`; without
  it (and without the capability) the wiring raises a `ValueError`.

## Wire the flux to a block

Pass the spatial scheme as the `spatial=` argument of `case.block(...)` (the public path), which
flows through to the runtime install:

```python
from pops.numerics.riemann import HLL
from pops.numerics.reconstruction.limiters import Minmod

case.block("gas", physics=m, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLL()))
```

For the full list of limiters, fluxes and reconstruction variables, see the
[native bricks reference](../reference/native-bricks.md).

## Declare a pressure for hllc or roe

`HLLC()` and `Roe()` read a pressure. A compressible ready-made model carries it. A DSL
model declares the primitive `p` and provides eigenvalues, which makes the generated route expose
`pressure` and `wave_speeds`. A DSL model can also become a generic `hllc`/`roe` model by emitting
the capability hooks: `m.enable_hllc()` / `m.enable_roe()` generate them from the declared roles
(including some 3-variable, non-Euler systems), or provide `m.roe_dissipation()` for a user-supplied
eigenstructure. See `pops.capabilities()["riemann"]` for the exact gates and
[write a model with the DSL](../tutorials/write-a-model-with-dsl.md).

## Check backend support

The descriptor validates the route before execution. A wrong choice fails at compile/bind time,
never silently.
For the per-backend matrix, see the [backend matrix](../reference/backend-matrix.md).

## Where to go next

To add a flux that no native brick provides, write it as a hyperbolic brick in the DSL and
compile it: see [write a model with the DSL](../tutorials/write-a-model-with-dsl.md). To make `HLLC()`
or `Roe()` work on a non-Euler model, supply the Riemann capability hooks (`HasHLLCStructure` or
`HasRoeDissipation`) from the DSL with `m.enable_hllc()` / `m.enable_roe()` rather than treating the
flux as Euler-only. Python host flux prototypes belong under experimental/debug routes, not the
production public API.
