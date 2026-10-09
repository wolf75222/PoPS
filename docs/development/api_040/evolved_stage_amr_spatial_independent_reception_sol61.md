# Independent nonconstant AMR Stage reception, 2026-10-01

This review receives author fixture commit
`f4152ab94a98cf0bc58e05cb6301af545528e8a2`, parent `d3f79a5935cdf6f8654d8daa5d758f5fc6e97f50`.
It adds only an independent reader, SOURCE_ONLY tests, and this note. It does not
modify the fixture, production, installed environment, numerical controls, or
the homogeneous reader delivered in `6108fbf0`.

No concrete P1/P2 defect was found in the source stencil, sign, initialization,
or quotient during this bounded review. Native execution and convergence are
pending ROOT. Neither a successful synthetic solve nor reading the native
source is a Native reception.

## Original equations and distinct references

For a two-component temperature, the physical accumulations are

```
H0(a,b) = a + a² + b²/10
H1(a,b) = b + b² + ab/5
z = a/4 + b/2
D = [[.012, .002], [-.001, .014]]
F = H(T) - Qprevious - .01*(div(D grad T) + f)
```

The load is the original opposite transfer of the homogeneous fixture; its
marker component is the exact cell mean of `1+.04*cos(2π(x-.5))`. The initial
temperature profiles are `.15+.02*cos(2πx)` and `.25+.015*sin(2πx)`. Initial Q is
the exact **cell mean of H**, not H at the centre or H of a mean temperature.
The independent reader expands trigonometric first and second moments and the
cross moment analytically. A test evaluates only the actual declared
CellBounds primitive from its AST and compares N=1/8/16/23 to those moments;
no author builder, accumulation function, spatial oracle, or PoPS is imported.

The spatial reference uses a graph of finest leaf control volumes. Each face
contributes once, with opposite signs to its adjacent leaves and their own
inverse cell widths. At a fine/coarse interface, the missing fine-centre value
comes from the polynomial through three restricted coarse centres. Its weights
are solved from a Vandermonde system, rather than copying the author's
all-faces-array implementation. A fine face replaces the coarse face, conserving
each signed row of D. Full-y extrusion makes the transverse face average equal
to that single x-face flux. This premise is checked against actual global boxes
and both valid/finest masks; it is not a production restriction.

Covered T must equal the average of fine T. Covered Q must equal the average of
fine H(T). Their nonlinear difference must remain nonzero. Active cells must
satisfy the original F, the H projection and the z constraint, with the unchanged
`3e-8` scientific guard. Flux/divergence activity is required. Composite amounts
use the finest mask and the explicitly derived Cartesian unit-square measures;
both individual opposite loads and total Q conservation are checked. There is
no assumption that signed/nonsymmetric D is SPD.

## Actual source audit

The relevant implementation paths were read on the exact private fixture base:

- `include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp`:
  `synchronize_original_field_candidate` copies every component through the
  scalar provider and calls `synchronize_linear_solution`; `apply_impl_` loops
  full row/column entries without SPD authority for original apply-only use.
- `include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp`,
  `synchronize_linear_solution` / `apply_linear_composite`: fine-to-coarse
  restriction precedes ghosts, arithmetic coefficient faces and `image -=`
  implement the spatial `-div` image, then `apply_flux_mismatch` replaces the
  coarse interface flux. Periodic parent staging is used by the same quadratic
  interpolation as local/remote paths.
- `include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp`:
  `QuadraticInterpolationTransfer` uses `s=±1/4`, weights
  `(s²-s)/2, 1-s², (s²+s)/2`; this agrees with the independent polynomial.
- `src/runtime/amr/amr_system.cpp`, `materialize_bootstrap_state`: a staged
  analytic source is rematerialized on the actual target-level Geometry with
  `exact_initial_integral`; no centre-sample initialization was assumed.
- `python/pops/codegen/program_emit_evolved_field.py`: the Q publication copies
  the previous state and evaluates H only on active cells. The accepted average
  down route must supply the covered Q from fine Q; the future saved-state
  discriminator explicitly checks this, rather than inferring it from a local
  H kernel or from native convergence.

The reviewed public `build` AST digest is
`1cd73f8bfcdcccb3f676e7ee11f276a5fcc9a47305847825f2719232597e6aca`.
Original imports and single-builder/no-global-rebinding admission are explicit.
The earlier independent reader also checks the original homogeneous build,
equations and seven controls without executing those source files. The selected
fixture source is externally pinned, not regenerated.

## Future real archive contract

The new reader is `tests/review/sol61_evolved_stage_amr_spatial_reception.py`.

```
python tests/review/sol61_evolved_stage_amr_spatial_reception.py contract
python tests/review/sol61_evolved_stage_amr_spatial_reception.py receive \
  --pins OWNER.json --pins-sha256 EXTERNAL_SHA \
  --approval APPROVAL.json --approval-sha256 EXTERNAL_SHA
```

It requires `sol61.evolved-stage-amr-spatial.owner-pins@1` and a separate
`sol61.evolved-stage-amr-spatial.root-approval@1`, both externally hashed by ROOT.
The qualification is `nonconstant-original-composite-Q-periodic-strip@1`;
homogeneous seals/receipts are not accepted or upgraded. One actual case,
cells8/width2, is counted once, even with two rank XMLs.

Pins retain the earlier independent owner schema's exact source/build commits,
ABI, actual Program IR16/17 choice, source/native/SDK/package paths and SHA256,
RO evidence roots, closed full-batch XML names and all-rank properties. Source
files are now `spatial`, `amr`, `equations`, `controls`, `fixture`. The actual
receipt must be `pops.evolved-stage-amr-spatial-native-fixture@1`.

Each case must pin the actual receipt, native module, component DSOs/sidecars,
same-handle Program IR/CPP, the three distinct checkpoints, ten phase/level NPZs,
eight author-reference NPZs, and three fixture-source copies. The latter
references are hashed as inventory only; they are never science inputs. The
compiler IR is matched to its declared Program hash and the CPP export, without
forging an aggregate artifact or a CPP-to-DSO build graph.

The unchanged independent durable helpers validate AMR checkpoint11/envelope1,
physical Q/forcing, spatial geometry identity/owner maps, native level/logical
clocks, FixedDt temporal cursors, POPSHID1 raw slot1 latest/slot0 previous,
POPSDIA1 opaque diagnostic bits for every real rank, empty exchange wire,
accepted report and rank offsets. Exact replay compares arrays and images;
run/continuation/restart lineage is reconstructed independently. Full raw XML
must have no failures/errors/skips and exactly the externally pinned batch;
the selected spatial test and rank/artifact/receipt properties must be present
once per rank. No filtered or synthesized XML is admitted.

## Checks and boundaries

The dedicated suite contains 39 SOURCE_ONLY tests. Synthetic algebra uses a
small NumPy central-FD Newton system only to construct countermodels on this
independent leaf graph. It is never presented as native output, a production
solver, or HPC. These tests distinguish transposed/signed/diagonal D, missing
reflux, constant ghosts, absent restriction, fine-index reversal, stale T,
incorrect z, seed in RHS, shifted/wrong initial Q, changed readonly load/marker,
H of restricted T on covered cells, history inversion, negative branch,
one-ULP replay mutation, a y mode, mask and measure corruption. Interface cases
at both periodic ends conserve also signed and singular finite matrices.

No fabricated ROOT approval/native receipt is produced. The complete new
`receive` path has not yet been positively exercised on actual native archives;
ROOT must supply those files and two seals. Protocol helper tests from the
homogeneous reader remain separate evidence.

Remaining limits are explicit: full-y periodic strip, two ratio-two levels,
constant signed D and declared no-EB Cartesian geometry. This is not arbitrary
2D AMR, candidate-dependent constitutive coefficients, full M06/M13, or GPU.
ROOT must authenticate installed source entries against the exact Git/build
archive; a package-manifest SHA alone does not perform that reconciliation.
CPP-to-DSO cryptographic linkage remains false. Native accepted-program/auxiliary
bodies remain opaque, hashed and replay-compared, not completely decoded;
private live leases/attempt authority are not persisted and cannot be inferred
from these observations. No Native reception is claimed in this commit.
