# Pre-implementation review: generic coordinated face

This review concerns the proposed versioned public `CoordinatedFace` /
`CoordinatedFiniteVolume`, before native execution or author implementation.
The complete physical rate is `U_t = -div F - B(U) grad U`. The numerical face
returns a **shared conservative flux** `G` and two separately signed source
vectors `left`, `right`. Existing `MaterializePathResidual` adds `left` at the
cell's upper face and `right` at its lower face. Thus in one dimension:

`R_i = -(G_(i+1/2)-G_(i-1/2))/dx + (left_(i+1/2)+right_(i-1/2))/dx`.

For the Audusse lake, the physical incident-cell fluxes are `G+C_L` and
`G+C_R`, so the sources must be `left=-C_L`, `right=+C_R`. This orientation is
not interchangeable. The side workspaces, AMR face canonicalization and reflux
must carry all three vectors under the same stage/layout authority and reject
the entire proposal before RHS or ledger publication if any active component
or speed is invalid.

The proposed physical ownership is one exact flux occurrence and one exact
nonconservative-product occurrence, on the same complete state. Authoring must
reject omission, duplication, foreign owner and mismatched rate signs rather
than covering a product twice. A component declared conservative must have
zero side source. On equal endpoint states, consistency demands `G(U,U)=F(U)`
and both sides zero; an arbitrary authored body cannot be considered proved
consistent merely because it has the right tuple shape. A finite nonnegative
runtime speed is necessary, but the author must additionally justify that it
majorizes the **composed** `DF+B` on the path and any numerical split effect.
The composed spectral radius alone is insufficient: the scalar physical
operator `F(u)=-u`, `B=+1` has `DF+B=0`, but a 70/30 side split with zero
numerical dissipation grows a checkerboard by 8% in one step (`dt=0.1 dx`).
This is a structural warning about the authored stability certificate, not a
claim that arbitrary coordinated faces can be mechanically certified.
Runtime parameters/captures must be bound by exact owner identity, included in
the versioned descriptor/ABI, and read at the active stage. Lazy branch
evaluation and finite checks must include every active intermediate in flux,
left, right and speed without hoisting inactive invalid branches.

The current `PathRusanovFlux` calls `path_rusanov_interface`, which fixes
`left_ncp == right_ncp == -0.5*integral`. Reusing its workspaces is appropriate;
reusing that formula for a coordinated body would silently destroy asymmetric
faces. A dedicated typed `model.path_interface` dispatch must precede the
legacy Rusanov formula while retaining the old path behavior for existing
models. All native `path_conservative_model` branches, including AMR, need an
exact descriptor/manifest check for the new kind; no edits to the separate AMR
installer work owned by another agent are implied here.

## Independent structurally different witness

The new pure oracle and tests under `tests/python/support/` and
`tests/python/unit/numerics/` define a nonlinear three-state product unrelated
to topography:

`F(c,p,r)=(c, p/2, -r)`, `B[p,c]=beta*r`, `B[r,p]=gamma*c`.

Its straight-path integral is exactly
`I=(0, beta*(rL+rR)*Delta(c)/2, gamma*(cL+cR)*Delta(p)/2)`. A closed rectangle
in `(c,r)` has a nonzero p-integral, so this B cannot be replaced globally by
a conservative flux gradient. The oracle uses a common Rusanov flux with
`||DF||_inf + max_path ||B||_inf` as a conservative bound on `DF+B`, and
**unequal** sources `left=-0.3 I`,
`right=-0.7 I`. Thus `left+right=-I`, the first component stays conservative,
and changing the signs of beta/gamma at fixed magnitudes changes the coupled
states while their first-step common-flux speed and conservative update remain
identical. Later feedback can change the bound, while the conservative
integral remains fixed.
It also tests nontrivial component permutation, equal-state diagonal behavior,
finite inputs, exact trigonometric cell means, and eight FE steps on a periodic
mesh. Six pure tests pass. This is a math/source contract witness only; it
must later be authored through the new public API and run on an authenticated
native artifact before claiming generic runtime support.
