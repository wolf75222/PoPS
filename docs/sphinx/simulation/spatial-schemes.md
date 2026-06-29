# Spatial schemes


The spatial scheme = reconstruction (limiter) + numerical Riemann flux + reconstructed
variables. Two equivalent facades describe it.

Every scheme is chosen with a TYPED `pops.numerics` descriptor; a bare string or boolean shortcut
raises a `TypeError` that names the typed alternative (Spec 5 sec.7).
`pops.numerics.spatial.Spatial(limiter=, flux=, recon=)` is the direct facade:

```python
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction import WENO5
from pops.numerics.reconstruction.limiters import Minmod, VanLeer
from pops.numerics.spatial import Spatial
from pops.numerics.variables import Primitive

Spatial(limiter=Minmod())                      # MUSCL minmod, Rusanov, conservative variables
Spatial(limiter=VanLeer(), flux=HLLC())        # MUSCL Van Leer, HLLC
Spatial(limiter=WENO5(), recon=Primitive())    # WENO5-Z, primitive reconstruction
```

`pops.numerics.spatial.FiniteVolume(limiter=, riemann=, variables=)` is the same thing, but the numerical
Riemann flux is called `riemann` (and not `flux`, reserved for the physical flux of a DSL model).
The reconstruction slot also accepts the Spec 5 sec.14.1 alias `reconstruction=`:

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.numerics.variables import Conservative

FiniteVolume(limiter=Minmod(), riemann=Rusanov(), variables=Conservative())
```

The typed descriptors:

- limiter / reconstruction (`pops.numerics.reconstruction`): `FirstOrder()` (first-order Godunov),
  `.limiters.Minmod()`, `.limiters.VanLeer()` (second-order MUSCL, 2 ghosts),
  `WENO5()` / `WENO5Z()` (WENO5-Z, order 5 in a smooth zone, 5-point stencil / 3 ghosts,
  oscillation-free capture near a front). WENO5 is exposed by the native block route and
  the compiled `aot`/`production` backends (the `prototype` JIT path rejects it);
- Riemann flux (`pops.numerics.riemann`): `Rusanov()` (the most stable, default for scalar
  transport), `HLL()` (generic signed-wave, requires `model.wave_speeds`), `HLLC()`, `Roe()`. HLLC
  and Roe run on the canonical Euler 2D layout (4 variables + pressure), or generically on any model
  that supplies the capability hooks (`HasHLLCStructure` / `HasRoeDissipation`, emitted in the DSL
  with `m.enable_hllc()` / `m.enable_roe()`);
- variables (`pops.numerics.variables`): `Conservative()` or `Primitive()`. The primitive set is
  more stable for Euler (positivity of `rho` and `p`).

On the C++ side, the limiters are policies in `numerics/reconstruction.hpp` (`NoSlope`,
`Minmod`, `VanLeer`, `Weno5`), the fluxes in `numerics/numerical_flux.hpp` (`RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux`). Detail and formulas: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
sections 2 and 3.
