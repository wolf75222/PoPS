# Exact zero coefficients in ordinary diagonal diffusion

This contract extends only the conservative two-point `Diffusion` realization.
For a Cartesian diagonal constitutive law `F_i=D_i grad_i W`, each `D_i` may
equal zero, including all axes at once. Every coefficient must remain finite
and nonnegative. Off-diagonal entries are still exact zero for this method;
cross-gradient tensors require `TensorDiffusion` and its separate energy
contract. No epsilon, projection, or coefficient clipping is introduced.
This is the second admissibility version of the method contract; the existing
`to_data()` `schema_version=1` describes the unchanged descriptor *shape*,
while a freshly rebuilt native header/ABI and artifact identity distinguish
the changed execution semantics. Reusing a pre-change native module would
invalidate this qualification.

For an interior face, the existing arithmetic average of endpoint
coefficients is retained. A zero face coefficient gives exactly zero flux and
zero conductance; it bypasses the variable secant and boundary difference, so
`0 × overflow` cannot create a spurious NaN. The measured explicit frequency is the sum of realized
nonnegative face losses, so a null axis contributes zero and an all-zero law
has zero diffusive frequency. Transport, if selected in the same rate, still
contributes its own frequency to the combined bound.

At a nonperiodic face, the existing linear coefficient extrapolation
`D_face=1.5 D_center-0.5 D_inside` is retained. A **value** boundary with
`D_face=0` contributes zero flux and conductance, even if its prescribed
value differs from the cell state. The inactive value trace need not be
evaluated; a nonfinite trace on a contributing face is rejected by the flux
and frequency checks. A **conormal** boundary with `D_face=0`
accepts only an exactly zero prescribed normal flux; nonzero conormal data
fail during face preparation, before an accepted face ledger can be
published. A negative or nonfinite extrapolated face coefficient fails,
including overflow to infinity from finite cell coefficients;
the implementation never clips it to zero. The public `DiffusiveBoundary`
API currently exposes `periodic`, `value`, and `conormal`, not Robin.

`ScharfetterGummel` fitted drift remains strictly positive in its diffusion
coefficient: its mobility/diffusion ratio and Bernoulli fitted face have a
different contract. `TensorDiffusion` retains its SPD checks, including the
strict native Cholesky; rank-deficient full tensors are a separate extension
and still cannot be selected via this change.

Source checks: `test_nonnegative_diagonal_diffusion.py` covers exact x-only,
all-zero, and y-only laws, negative/static nonfinite refusal, dynamic
coefficient authoring, and the absence of Robin. Native tests are prepared in
`test_nonnegative_diagonal_diffusion_runtime.py`: Uniform x-only stencil and
reduced frequency, all-zero law, compatible value and zero-conormal faces,
negative/nonfinite dynamic coefficient rollback, incompatible conormal
rollback, and periodic AMR composite conservation. They require rebuilding
and authenticating the isolated installed native artifact before execution.


Constitutive cell support and provider transport @1
-------------------------------------------------

A pointwise coefficient expression can require neighboring cell values when its
selected numerical operation constructs face fluxes. Provider halos therefore
come from the typed resolved operation's stencil radius and its exact qualified
component reads, alongside explicit boundary widths and derived prerequisite
closure. A field declaration's pointwise shape alone is insufficient. The same
resolved width must enter provider materialization and the Program consumer plan.
This transport requirement changes no physical coefficient or boundary formula.
This internal support contract uses the exact resolved sampling variants:
`face_trace`, `left_face_trace` and `right_face_trace` read adjacent cells,
independently of the wider State reconstruction stencil. Cell reads use the
selected operation radius. It adds no physical model selector or public math
syntax, and leaves Native ABI13 and checkpoint schemas unchanged. The changed
header content changes the SDK signature, requiring rebuilt consumers.

For AMR diagonal/fitted diffusion, the native constitutive law is evaluated on
`valid_box.grow(1)`; the adjoint tensor realization needs `valid_box.grow(2)`.
Nonperiodic exterior cells retain the existing clipping to the physical domain.
The tensor gradient is then evaluated on the existing inner box, one layer inside
that constitutive support. State allocations may retain larger transport halos;
those extra cells are not additional constitutive inputs. Minimal one/two-cell
State allocations preserve their previous evaluation boxes exactly. Every status,
finite-coefficient and admissibility check within the mathematical support remains
mandatory, and invalid required ghost data still fail before RHS/face publication.

The durable `test_prepared_diffusion.cpp` additions compare a three-halo State
against minimal diagonal/tensor references using real one/two-halo coefficient
storage, with a refusal before any out-of-storage dereference. The diagonal case
fails against the prior header and both pass with the bounded evaluation box.
Existing sparse periodic, internal-ghost failure and mixed physical-boundary
checks cover the retained guard and clipping behavior. These are Source/Host
checks; actual matching installed Native reception requires its rebuilt SDK.
