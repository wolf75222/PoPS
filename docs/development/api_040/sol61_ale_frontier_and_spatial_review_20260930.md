# Independent ALE/frontier review and spatial interaction design - 2026-09-30

This is a GPT-6.1 Sol review of `6c51d6f` and `f6aad44`, read by Git object
without merging them into the frozen M26 review checkout. No environment was
modified. The M26 source review is separately frozen at `ae131467`.

## ALE declaration and primitive: source/host approval

`6c51d6f` introduces an inert moving-control-volume requirement and a host/device
arithmetic primitive. Its native resolve gate rejects the immutable Cartesian
carrier before JIT, so the declaration cannot silently execute as static geometry.
The obligation inventory explicitly requires trial/accepted coordinates and
measures, same-quadrature swept exchanges, relative flux, atomic publication and
rollback, duration/input-qualified reuse, and saved geometry/inventory receipts.
Those are obligations, not provided ALE execution capabilities.

The real candidate `SweptInterval` header was copied with `git show`, compiled
with Clang C++20 and `-fno-fast-math`, and executed as a small host assertion
program. It passes:

- expansion/contraction and translations in both directions;
- constant-density GCL, with amounts divided by the new measure;
- three-cell global inventory cancellation at shared faces;
- the positive-x Reynolds balance, with physical flux and source integrated
  once, and correctly oriented mesh sweeps;
- stale sweep rejection, collapsed cell rejection, measure overflow rejection,
  and NaN/either-infinity rejection in coordinates, sweeps, and tolerance.

`updated_amount` is deliberately an unchecked `noexcept` arithmetic formula;
it propagates nonfinite scientific inputs and can overflow. The probe confirms
this rather than claiming a scientific acceptance guard. Neither the primitive
nor this commit certifies an actual interval duration, a mesh cache, native
geometry mutation, MPI rollback, or device execution. A future carrier must
validate those inputs and the resulting candidate amount before publication.
No blocker was demonstrated for this declared scope.

Probe: `tests/review/sol61_swept_interval_6c51d6f.cpp`.

```sh
rtk proxy git show 6c51d6f:include/pops/mesh/geometry/swept_interval.hpp > outputs/swept_interval-6c51d6f.hpp
rtk proxy clang++ -std=c++20 -fno-fast-math -Ioutputs -Iinclude \
  -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
  tests/review/sol61_swept_interval_6c51d6f.cpp -o outputs/swept_interval-sol61-review
rtk proxy outputs/swept_interval-sol61-review
```

## Computed frontier: corrections required before approval

`f6aad44` preserves the exact legacy policy, declares its computed policy as a
frozen version-2 descriptor, and checks the actual native landing without
relabeling the clock. A source-only controller probe passes the honest one-ULP
landings `-0.1 -> 0.2` and `-0.3 -> 0.4`, then checkpoint restore and progression
to `0.6`. MPI proposal equality includes both the policy descriptor and reached
coordinate; the native cadence entry contract also adds effective duration.
These observations are source/control-flow evidence, not native MPI execution.

The following defects are reproduced against the exact `f6aad44` Python snapshot:

1. **Historical interval authority is incomplete.** On grid
   `(-0.125, 0.125, 0.6)`, the first honest interval is `start=-0.125`,
   `duration=0.25`. A checkpoint with `start=0`, `duration=0.125`, and
   `last_accepted_dt=0.125` is accepted at the same correct reached/requested
   point and index. Algebraic self-consistency is insufficient: start must be
   attached to the preceding grid coordinate under the declared computed
   frontier policy.
2. **The last accepted duration is detached from its receipt.** Changing only
   `last_accepted_dt` to `42` is accepted by restore. Bind it to the actual
   represented elapsed interval `(reached-start).hex()`; retain the separately
   authored native input duration in the receipt.
3. **A live corrupted index is silently repaired after advancing.** After the
   honest first step, replacing `external_frontier.index` with `99` still
   allows the next step to `0.6`; it is overwritten with `2`. Restart correctly
   rejects that index, but live preparation validates only requested/reached.
   Validate the entire previous receipt before preparing the next candidate.

Deleting the entire computed receipt is also accepted at an exactly reached
grid point. A stable computed controller should authenticate its accepted
receipt. If newly attaching/changing controllers at an accepted boundary is
allowed, give that initial state explicit provenance rather than treating
missing accepted metadata as an indistinguishable initial entry.

The probe uses a scalar clock fixture and an exception shim only. It loads no
native extension and does not implement a scientific execution backend. Its
`ACCEPTED` corruption output records candidate defects; exit zero is not a
statement that all rejection requirements passed.

Probe: `tests/review/sol61_frontier_f6aad44.py`. Prepare the pinned source with
`git archive f6aad44 python` extracted under
`outputs/frontier-sol61-f6aad44-source/`, then run:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I \
  tests/review/sol61_frontier_f6aad44.py
```

## Concrete generic spatial separable interaction route

The frozen source already provides `Program.sum`, `sum_component`, and `dot`
reductions. `program_emit_ops.py` routes them through owner/block/lane-qualified
`ProgramContext` calls, including empty ranks. They deliberately exclude
inactive cells but have **no cell volume/kappa weighting**. The separate
`RuntimeInstance.integral` is an after-execution diagnostic, not an intra-stage
Program functional. Two other missing links are explicit: reduced runtime
scalars cannot be multiplied into pointwise field expressions, and the current
pointwise encoder does not accept coordinate/trigonometric analytic bodies.

A minimal generic capability should therefore be a **prepared separable
spatial interaction**, not an N-by-N finite DOF map:

1. Author an immutable separable kernel `K(x,y)=b(x)^T C b(y)` with an ordered
   analytic basis, exact coefficient/parameter identities, and a finite exactly
   symmetric rank-by-rank matrix `C`. For the cosine kernel the basis is
   `(cos(2*pi*x), sin(2*pi*x))` and `C=I`; the provider contains no M26 names.
2. On a genuine one-component distributed density `MultiFab`, accumulate
   `mu_k=sum_owned_cells volume_i*b_k(x_i)*rho_i`. For the initial uniform
   Cartesian scope, physical weights come from authenticated geometry spacing,
   not a hardcoded N or raw unweighted `Program.sum`. Reuse the execution lane,
   owner checks, empty-rank behavior, and collective failure consensus of the
   existing reductions. Pack the two scientific sums into one fixed-length
   all-reduce, with a converged invalid-value status; no cell arrays are gathered.
3. Reconstruct `Phi(x_i)=b(x_i)^T C mu` into a trial scalar field on the same
   native layout. Rank, not cell count, controls generated code size. N=32/64/128
   changes layout/storage while preserving this source algorithm.
4. Expose the result through an authenticated stage field context or a generic
   scalar-to-field/materialized-expression bridge. The existing field context
   records the consumed state/stage identity and is the appropriate link to
   drift/diffusion rate calls. `Program.gradient` already exists for scalar
   fields, but a compatible face flux is a separate numerical choice. Returning
   only a cell gradient and applying an unrelated centered divergence does not
   certify the requested discrete energy law.
5. Keep immutable basis samples/geometry/masks and reduction buffers in a
   `prepared_resource_lease`, keyed by node, owner, exact layout/distribution,
   lane, basis/parameter identity, and geometry epoch. Recompute moments for
   every consumed stage/attempt. Never reuse a moment or potential merely
   because its buffer pointer is unchanged. Rejected attempts discard trial
   scalar/field data; accepted density/clock/state only publish at the existing
   transaction boundary. Rebinds, parameter changes, regrids, and peer-local
   nonfinite failures must invalidate/reject collectively.

The existing `PreparedFieldLoweringProvider` registry can carry generic field
method identities and opaque native contracts, but attaching this operation
must not falsely label an integral evaluation as an elliptic solve. A dedicated
Program operation and C++ `PreparedSeparableInteraction<Dim, Rank>` is the
smaller first carrier, with a proper stage-field observation/result binding.
The uniform Cartesian case can be closed first; AMR requires composite
finest-owner coverage, physical kappa/volume, and its own geometry epoch.

For positive constant diffusivity, the existing `ScharfetterGummel` realization
already couples one exact drift and diffusion occurrence at faces and consumes
potential differences. Reusing that lane is preferable to inventing an unrelated
cell-gradient flux, after checking its energy and balance receipt against the
chosen interaction functional. **The M26 PDE diffusion coefficient has no
current authority and must not be invented.** Declaring the separable interaction
capability alone does not close the PDE witness or its energy law.

Required qualification for the next concrete tranche: source/code-size
independence from N, independent Fourier moment oracle, serial/MPI split and
empty-rank equivalence, no mesh-to-Python gather, conserved mass from face
exchanges, actual saved states, a compatible discrete free-energy/interaction
identity, NaN rejection without accepted mutation, and exact parameter/cache
invalidation. The final scientific criteria depend on the authorized diffusion
and flux choices.
