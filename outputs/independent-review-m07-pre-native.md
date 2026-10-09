# M07 independent pre-native oracle and public-route boundary

The corpus calls for the Saint-Venant lake at rest on `[-1,1]`,
`z(x)=0.2 exp(-50x²)`, `h+z=1`, `q=hu=0`, `N=40,80,160`, `T=1`, and a
hydrostatic reconstruction error no larger than `1e-12`. This bounded wet
variant uses `g=1`, first-order HLL/Forward Euler, `dt=0.1 dx` (200/400/800
steps), and nonperiodic ghost cells set to the same analytic lake equilibrium.
The wet/dry extension is outside this witness.

The independent NumPy oracle in
`tests/python/support/api040_m07_lake_oracle_independent.py` integrates each
cell's bottom exactly with the error function, including the two ghost cells;
`hbar=1-zbar`. Forty-point Gauss–Legendre quadrature independently verifies
those means. At a face, `z*=max(zL,zR)` and reconstructed depths are
`hL*=max(0,hL+zL-z*)`, `hR*=max(0,hR+zR-z*)`; reconstructed momenta preserve
their original velocities. A common signed HLL flux is evaluated on those
states. The left and right cells then require **different** momentum face
contributions, respectively adding `g(hL²-hL*²)/2` and
`g(hR²-hR*²)/2`. In each equilibrium cell this makes its right and left
pressure contributions equal, while the mass flux stays shared.

Seven pure oracle/adversarial tests pass. The complete 200/400/800-step oracle
changes its initial lake by only `3.33e-17` in max norm on each grid. If the
same HLL face flux is used for both cells without the distinct corrections,
the momentum changes after **one** step by `5.114e-3`, `2.637e-3`, and
`1.332e-3` for `N=40,80,160`; this is a discriminating counterexample, not a
roundoff effect. At a face, the left/right pressure contributions differ by at
least `5.114e-2`, `2.637e-2`, and `1.332e-2` respectively on those grids.

The current public `User` finite-volume body returns one shared flux, whereas
the `PathConservative` route does not accept that authored HLL/User pairing.
An example using only the shared flux therefore cannot close M07, even if a
separate unqualified source is added. The frozen public red witness below
isolates this missing pairing. No native state, installed artifact,
MPI result, or full wet/dry claim is asserted by this oracle.

## Independent review of the frozen author witness

Author commit `68c6958` (base `3bd6ea9`) contains only the four science,
oracle, test and documentation files. Its public declaration retains
`h_t+q_x=0`, `q_t+(q²/h+gh²/2)_x=-gh z_x`, and `z_t=0` under component
permutation. The symbolic hydrostatic face agrees with its separate NumPy
oracle at an off-equilibrium face, and the NumPy HLL/FE trajectory agrees with
this precommitted independent oracle within `2.22e-16` for h/q and z over all
three prescribed grids. I reran the author's ten pure tests: 10/10 pass.

The [primary Audusse et al. paper](https://publications.imp.fu-berlin.de/478/1/file_2004_siam.pdf)
confirms the reconstruction and distinct incident-cell fluxes in equations
(2.12)–(2.16). The author demonstrates three separate public constraints:
`User` refuses a two-sided return, ordinary `FiniteVolume` refuses omission
of the physical product, and `PathConservativeFiniteVolume` refuses the
authored User face. This is a valid open EXPR/IMPL witness, not a failed native
simulation. No production change or installed numerical evidence is included.
