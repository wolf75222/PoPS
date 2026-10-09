# M02 Burgers / authored Godunov preparation

Native scientific status: **not_executed**. No native kernel compilation, build, or time evolution was run by this worker. The installed `pops-api040` exposes the old `riemann.User(brick_id)` C++ selector, which is not the requested body/state API. `outputs/astra-m02-burgers-preparation.json` records the exact installed path/signature, planned criteria, source hashes, and empty native run list. The example explicitly rejects the old signature and never substitutes Rusanov.

Files added:

- `examples/migration/scientific/api040_m02_burgers.py`: linear public authoring and reception script; conserved u, flux `(u²/2,0)`, exact characteristic speeds `(u,0)`, `riemann.User(body=godunov_face,state=U)`, `FirstOrder`, `ForwardEuler`, adaptive CFL .4.
- `examples/migration/scientific/api040_m02_burgers_oracle.py`: independent analytic Riemann cell integrals, convex-flux extremum characterization, and weak shock speed. No PoPS import, native selector, quadrature, or point-sampling substitution.
- `tests/python/unit/numerics/test_api040_m02_burgers_oracle.py`: 33 pure mathematical cases. These execute the example's actual face function by extracting its AST and supplying ordinary scalar conditional evaluation; they do not imitate a native execution result.

The full authored face distinguishes shocks by `u_left>u_right`, then selects upstream physical flux from the Rankine–Hugoniot speed `(u_left+u_right)/2`. Rarefactions select the appropriate physical flux when both characteristic directions have one sign and zero for a fan crossing zero. Every branch is expressed using callable bodies of the common symbolic `where`. State/flux arguments are indexable vectors even for N=1; output is a one-component tuple. Using the supplied physical fluxes ensures the y-axis face stays zero. Recreating u²/2 without regard to the physical axis would not have that property.

**Tests: 33 passed in 0.19 s**, JUnit `outputs/astra-m02-burgers-oracle.xml`. Coverage includes every sign/entropy region, scalar consistency, reflection, canonical left/right ordering, zero transverse physical flux, exact cell integrals for odd and even meshes, integral boundary balances, and the nonlinear-coordinate weak-law counterexample. For w=u²/2 on u>=0, the correct conservation law retains accumulation sqrt(2w). Replacing it by conservation of w is equivalent only for smooth solutions and gives shock speed 2/3 instead of the correct 1/2. The tests also demonstrate that averaging and this nonlinear coordinate transformation do not commute.

The native reception includes a discriminating first step, before the main evolution. At `dt=.01*dx` on the even, face-aligned meshes, a single first-order Godunov/ForwardEuler step equals the exact entropy cell averages for both 1->0 and 0->1. The test establishes that a Rusanov substitution differs by more than .002, while the predeclared native tolerance is 2e-13. The actual first-step state will be saved and reopened, separately from the final state.

The prescribed native campaign is shock 1->0 and rarefaction 0->1 on x=[-1,1], y=[0,1], full N x N uniform Dim=2 meshes for N=100/200/400, x Outflow, periodic y, t=.2. All thresholds are fixed in `CRITERIA` before compilation or measurement: integral boundary-mass defect <=1e-11; integrated L1 <=.08/.06/.04 with strict decrease; bounds, spatial monotonicity, transverse invariance, initial/first-step accuracy, time, and a shock half-height location within 1.5 dx of x=.1. The half-height location measures speed independently of total mass. No unqualified first-order convergence rate is asserted across discontinuities. No native Dim=1 support is claimed.

For this domain the total conserved mass is `2*mean(u)`, and its expected change is `t*(u_left²-u_right²)/2`: +.1 for the shock and -.1 for the rarefaction. Diagnostics use reopened full-domain states and no clipping. Initial and exact arrays use `(component,y,x)` with the last dimension varying in x. Receipts use the shared `api040_receipts.receipt_json`, preserving byte identities without `default=str`.

The actual main example's authoring AST, through `pops.validate`, was additionally executed against Sol6_native's isolated source API at `work/PoPS-numerical-bodies` with the `pops-api040-bodies` interpreter. **Both shock and rarefaction validate and resolve successfully** with the real FiniteVolume/riemann.User route and boundary setup. Receipt: `outputs/astra-m02-source-authoring.json`, including exact source package path and source hashes. This deliberately bypassed the installed-package preamble and run loop to inspect source authoring; it is not an authenticated native run. The first source probe exposed my outdated where call syntax; changing the example to callable branches fixed it without any production changes.

After root merges and installs the native body implementation, launch each variant under the same authenticated SDK/compiler/runtime environment as other API040 cases, with PYTHONPATH unset and `POPS_NATIVE_DIM=2`:

```sh
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_THREADS=1 POPS_API040_M02_CASE=shock python examples/migration/scientific/api040_m02_burgers.py
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_THREADS=1 POPS_API040_M02_CASE=rarefaction python examples/migration/scientific/api040_m02_burgers.py
```

Successful runs produce first-step and final NPZ files plus `outputs/api040_m02/{shock,rarefaction}/receipt.json`. Until those genuine saved states and receipts exist, mathematical/source checks do not qualify native Godunov lowering, flux sharing, scientific convergence, MPI, GPU, or AMR.
