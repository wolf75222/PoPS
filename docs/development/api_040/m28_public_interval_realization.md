# Coupled public ALE realization and accepted geometry

This tranche connects `MovingControlVolumes` and Program `geometry_state` /
`reynolds_update` to the owned native `project_moving_interval` producer and
`prepare_moving_interval_update` / `commit_moving_interval` publication. The
carrier and POPSEX03 checkpoint contracts remain those of the separately frozen
native tranche. Native execution and scientific reception are required after a
rebuild; source tests and syntax checks establish no runtime qualification.

The realization reads the original identity-accumulation `-div(flux)` Equation,
its authored component bodies and **signed** characteristic bounds, its actual
FiniteVolume numerical declaration, and its original named source occurrences.
The face model evaluates `F(U)-wg*U`, and the Rusanov dissipation uses
`max(abs(lambda(U)-wg))` on both sides. This permits reversal of the physical
upwind direction when a face moves faster than the fluid. Explicit left/right
density weights and old/new cell-measure source weights belong to the Program
projection. The native receipt retains physical amounts, density traces,
endpoint displacements and source amounts on the same authenticated interval.
The numerical CFL check uses the actual adjacent moving cell lengths.

The selected native Rusanov formula averages the two physical trace fluxes.
Applied to `F-wg*U`, its mesh-density trace is therefore `(UL+UR)/2`. Preparation
requires explicit `face_weights=(1/2,1/2)` for this realization. Other density
policies remain valid authoring/carrier contracts but require a numerical-face
provider that actually consumes them. They cannot merely alter a receipt
decomposition while cancelling from the numerical update. For example,
`a=.3`, `wg=.7`, `UL=1`, `UR=4` gives the centered relative upwind flux `-1.6`;
substituting a declared left trace into the same physical Rusanov construction
would give a different relative flux. Source old/new measure weights remain
general and affect the actual source amount, including `(1,0)`, `(0,1)`,
centered and other exact affine quadratures. No native carrier/header restriction
on quadrature is added by this provider-specific prepare gate.

Preparation currently selects the periodic Uniform1D, FirstOrder/Rusanov,
single complete state realization with a 1D analytic coordinate law. It proves
the initial map is identity at time zero and validates its actual finite
initialized endpoints before use. Nonidentity initial projection, moving
nonperiodic boundaries, other reconstructions/fluxes, joint-state conversions,
discrete-input coordinate fields, higher dimensions, stages and AMR geometric
transfer/reflux need additional prepared providers. These are realization
limits, not restrictions on the Reynolds/GCL mathematics or the authoring
contract. An unsupported realization fails at resolution before JIT. Ordinary
static-dx metric diagnostics on moving states also refuse until an accepted
measure reduction provider exists; saved fields expose actual measures for
independent inventory calculations.

The read-only native `_output_moving_geometry_snapshot` authenticates the exact
Program state route and the real accepted receipt, then broadcasts endpoints
and volumes from each native owner. Replicated storage uses rank zero as its
single authoritative image owner; it is never summed or divided by MPI size.
Python authenticates identical geometry requests and local preparation failures
before entering ordered native collectives. `LevelGeometry.node_coordinates`
owns an immutable detached physical node array, and requires exact positive
endpoint-derived volumes. Observer archives, NPZ and HDF5 retain the node array;
ParaView and Catalyst use these physical nodes rather than reference spacing.
Static geometry serialization remains unchanged when nodes are absent.

`tests/python/integration/runtime/test_public_moving_interval.py` supplies the
real public Case/resolve/compile/bind/run/output/checkpoint/restart chain. It
declares arbitrary scalar or three-component transport bodies, permutations,
optional original sources, and two distinct configurable resolutions via
`POPS_ALE_RESOLUTIONS=16,32`. It recomputes inventory and endpoint lengths from
authenticated saved NPZ arrays and checks exact native restart continuation.
It must run against a freshly rebuilt native Dim1 SDK. A fixed physical-domain
coordinate contract is not a globally translating-domain realization; Galilean
reception must additionally test the relative face producer directly.

No public AMR ALE claim follows from Uniform1D acceptance, no new IntegralState
trace/reduction provider is supplied here, and rank-local POPSEX03 receipts
remain required on every rank. A later ownership-aware accepted-budget protocol
must preserve these local restart obligations while counting physical coverage
once.
