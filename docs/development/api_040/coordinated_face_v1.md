# Coordinated conservative/nonconservative faces, version 1

`CoordinatedFace(flux=F, product=B, frame=frame, body=...)` binds one numerical
construction to the exact declared physical flux and nonconservative product.
The expression-building callable receives `(left, right, axis)` and returns
`FaceBalance(flux=..., left=..., right=..., stability=...)`. Its three vectors
have the complete declared state width; width is not limited to a demonstration
size. `CoordinatedFiniteVolume(face=...)` owns the complete physical balance
`ddt(U) == -div(F) - B`. Partial coverage, omission and duplicate occurrences are
rejected. There is no additional source term for the same product.

For a face oriented in the positive coordinate direction, `left` and `right`
are **already signed cell RHS contributions**, in flux-density units:

```
dU_i/dt = -(shared_upper - shared_lower)/dx
          +(right_lower + left_upper)/dx.
```

Audusse's incident pressure corrections therefore use `left=-C_L`, `right=+C_R`.
The two sources need not be equal. Declared conservative components require
literal zero in both source vectors. Mathematical consistency on the diagonal
(`shared(U,U)=F(U)`, sides zero), and consistency with the physical product off
the diagonal, remain author obligations tested by independent witnesses. The
compiler does not claim to prove arbitrary symbolic identities.

`stability` is a nonnegative finite **speed** bounding the whole numerical
update, including the chosen split sources and numerical dissipation. The
existing face-frequency reduction and explicit-consumer coefficient budget
apply. Merely bounding the spectrum of `DF+B` is insufficient: `F=-U`, `B=1`
has zero physical speed while an asymmetric split can still amplify a
checkerboard. No universal TVD, positivity, entropy or wet/dry claim follows
from a finite nonnegative authored number. The M07 fully wet witness fixes
its own physical/numerical assumptions and step before reception.

The callable runs once per axis during authoring. Its immutable expressions
use the ordinary PoPS IR, CSE, lazy `where`, rounding and finite-intermediate
checks. Exact RuntimeParam references retain their owning model/block and
stable runtime slots; values are not baked into the face code. Foreign free
variables, captured quantities without an explicit trace realization and
foreign parameters are refused. Version 1 receives two adjacent stored cell
averages. It does not silently attach higher-order reconstruction without a
compatible within-cell balance. This is a concrete versioned realization;
general stencil extensions remain possible and are not assigned an arbitrary
maximum radius.

The generated model publishes `coordinated_face_contract_version=1`; the
distinct native policy is `CoordinatedFaceFlux<N>`. Its complete route is
`coordinated_face:v1:<operator identity>`. The identity covers the resolved
physical law and selected method, expression body, captures, signs and version.
Typed Uniform and AMR installers compare the complete identity against the
compiled model before publishing a block. A syntax-only route preflight is
not authentication. Existing `riemann.User` scalar/v1 and joint-stencil/v2
contracts remain separate. Legacy path composites publish version zero and
select `PathRusanovFlux`; unknown nonzero versions fail compilation.

One `PathInterfaceResult` carries shared flux, both signed sides, bound and
status. A failed input/domain/body/bound publishes none of its tuple. Native
path workspaces retain all four successful components together and materialize
the residual only after collective success. Existing face geometry owns face
measures exactly once. AMR reuses path stage/halo and reflux authority, with
the existing synchronous/injection conditions; this source extension is not a
new native AMR acceptance receipt. Nonconservative sides are not relabelled as
conservative reflux fluxes.

## Evidence and reception boundary

`test_coordinated_face.py` exercises real public resolution, complete occurrence
identity, omission/double-count rejection, foreign capture rejection, generated
C++ finite/lazy checks and atomic failure. Two host executables compile real
generated `CompositeModel` bricks through the selected policy, including the
legacy SymbolicPath branch; the M07 host checks `.405/.32` incident momentum
outputs and a negative-depth domain refusal. These are source/host checks,
not PoPS spatial integration or MPI acceptance.

Sol's independent three-component witness uses `F=(c,p/2,-r)`,
`B[p,c]=beta*r`, `B[r,p]=gamma*c`, with an exact straight-path integral split
30/70. Its independently authored oracle and native reception test are separate
from the hydrostatic example. Installed Uniform, AMR and MPI runs remain the
central integrator's responsibility after a matched rebuild.

The M07 example now provides a genuine Dim=1 installed reception for both
component orders and N=40/80/160, T=1, FE/HLL, `dt=1/(5N)`. Its unchanged NumPy
oracle and `1e-12` criterion predate this implementation. Native public Inflow
uses reflected ghosts `2*face_value-interior`; each face value is fixed to the
mean of the initial interior and analytic equilibrium ghost averages. This
reproduces the required equilibrium ghosts. Away from equilibrium it is a
Dirichlet mirror closure, so this reception does not qualify perturbations
against the frozen-ghost oracle. Both boundaries are tested independently.

```
env -u PYTHONPATH POPS_NATIVE_DIM=1 python examples/migration/scientific/api040_m07_saint_venant.py
```

The script authenticates installed Dim=1, checks gathered initial averages,
reopens saved NPZ states, records package/native hashes, execution context and
actual time/step counts, and converges root-only validation/I/O failures through
the communicator. No installed M07 result is asserted in this source delivery.
