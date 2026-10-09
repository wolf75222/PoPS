# M07: frozen lake-at-rest reception and historical coordinated-face gap

The counterexample below is preserved at commit `68c6958`. The subsequent
[coordinated-face version-one implementation](coordinated_face_v1.md) adds a
distinct public construction and installed reception script. The old User/path
refusals remain tested, but are no longer evidence that the new construction is
inexpressible. No native M07 result is inferred from source implementation.

Base examined: `3bd6ea9`. This delivery is a physical declaration, public API
counterexample, and independent NumPy preflight. **No native M07 trajectory is
claimed**, and no missing capability is reclassified as out of scope.

## Fixed physical and numerical definition

The corpus requires Saint-Venant with topography, not homogeneous transport:

```
h_t + q_x = 0
q_t + (q²/h + g h²/2)_x = -g h z_x
z_t = 0
```

The auxiliary stationary bed coordinate retains the product `B[q,z]=g*h`.
It must remain unchanged in every cell; preserving only `h+z` is insufficient.
The full signed balance contains both the conservative flux and the topographic
product exactly once. There is no post-step state correction.

Use units depth H=1, horizontal length L=1, velocity sqrt(g_physical H), and time
L/sqrt(g_physical H); hence dimensionless g=1. On [-1,1], set
`z=.2 exp(-50x²)`, `h+z=1`, `q=0`. Fix N=40,80,160, T=1, first-order hydrostatic
reconstruction, HLL, Forward Euler dt=.1 dx =1/(5N), giving 200,400,800 steps.
The normalized error is `max(||h+z-1||inf, ||q||inf, ||z-z_initial||inf)` with
threshold **1e-12**, fixed before any native run. H=1 and velocity unit 1 make
each quantity dimensionless. The reference checks a face Courant number <=.1
up to two machine-rounding ulps; no measured result selects the criterion.

Initial bed values are exact finite-volume means from the Gaussian erf integral;
h means are `1-zbar`. Cell-center samples are explicitly rejected by a test.
Nonperiodic boundary ghost cells use the analytic continuation of these same
equilibrium means on their adjacent intervals. No periodic wrapping occurs.
The fully wet lake is the current reception; perturbation/front wet-dry cases
from the original corpus remain separate open obligations, not accepted claims.

The hydrostatic reconstruction and pressure corrections follow equations
(2.12)–(2.16) of [Audusse et al., SIAM J. Sci. Comput. 25(6), 2004](https://publications.imp.fu-berlin.de/478/1/file_2004_siam.pdf).
At a face, z*=max(zL,zR), h*K=max(0,hK+zK-z*), q*K=h*K qK/hK. An HLL flux
on these reconstructed states receives separate incident-cell momentum
corrections `.5g(hL²-h*L²)` and `.5g(hR²-h*R²)`. This pressure difference is
part of the discrete source realization, not an additional physical source.
The max in reconstructed depth is the prescribed hydrostatic method, not a
floor/projection of the evolved state. Cell depths outside the fully wet
reference domain are refused.

## Actual public counterexamples

`api040_m07_saint_venant.py` declares the entire physical balance using public
Model/state/flux/nonconservative-product APIs. It also expresses both required
faces in the common scalar IR, for canonical and permuted component orders.

1. `riemann.User` accepts only one width-N tuple. Returning the required two
   width-N incident-cell contributions raises `TypeError: riemann.User body must
   return a tuple of one Expr per state component`. **EXPR:** no typed public
   coordinated-face output contract is available on this route.
2. Keeping just one side authors a User descriptor, but ordinary FiniteVolume
   rejects the retained topographic product. That refusal is correct; dropping
   the product would silently change the PDE.
3. PathConservativeFiniteVolume represents the physical product and has native
   side contributions, but its public construction requires FirstOrder/Rusanov
   and refuses the authored face descriptor. **IMPL:** that route does not realize
   this hydrostatic construction. The supplied straight path is solely a typed
   carrier of the physical product, not a substitute for hydrostatic numerics.

The independent reference discriminates an omitted correction and a doubled
correction by a momentum change >1e-3 after one step for every required grid.
A single lake face with (hL,zL)=(.9,.1), (hR,zR)=(.8,.2) needs momentum outputs
.405 and .32: one shared flux cannot satisfy both. A perturbation test requires
a nonzero response, so an identity evolution would not satisfy this reference.

## Reproduction and next mechanism

From the checkout, source tests (explicit source import, not installed proof):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python -m pytest -q -o pythonpath=python tests/python/unit/numerics/test_m07_hydrostatic_public_gap.py
```

Against the historical installed API, call the preserved minimal failing
authoring function; it does not invoke a compiler or runtime:

```sh
rtk proxy env -u PYTHONPATH python -c 'import runpy; runpy.run_path("examples/migration/scientific/api040_m07_saint_venant.py")["unavailable_hydrostatic_method"]()'
```

Observed receipt on 2026-09-29: source suite **10 passed in 0.44 s**. The actual
installed interpreter `/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`
also exited 1 on this example, with the TypeError above from its
`lib/python3.12/site-packages/pops/numerics/riemann/user.py:117`. This confirms the
authoring refusal in the installed package; no native dimension was selected,
and there was no compile or simulation. The NumPy trajectories gave normalized
errors 3.33e-17 for all three grids, with maximum face Courant .1. Their results
do not qualify the PoPS PDE runtime. This old authoring function remains red;
the standalone example now uses the separately versioned public construction.

Independent GPT-6 Sol review reran these 10 tests and its separately implemented
7-test oracle suite. The two reference trajectories agree within 2.22e-16 on
all three grids; the reviewer checked the two pressure corrections against the
primary paper and confirmed each distinct public refusal. This is additional
mathematical/source evidence, not a native acceptance receipt.

The next generic capability must bind a numerical construction to the complete
physical flux/product occurrences and return one shared conservative face plus
two signed side contributions, with explicit stability/status and immutable
captures. Both sides must publish atomically and use exact stage/halo authority;
conservative components, boundary contributions, and eventual AMR reflux need
their existing conservation contracts. The physical source must then be marked
covered once. This is a proposal for core work, not an implementation in this
science-only delivery. No private descriptor, compiled callback, hard-coded
Saint-Venant runtime, artificial stencil ceiling, or alternative PDE is used.
