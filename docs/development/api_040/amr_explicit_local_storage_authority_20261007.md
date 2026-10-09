# AMR authority from an explicit local StateStorage plan

An AMR block can select a flux-free pointwise rate through an explicit
`DiscretizationPlan.rates.add(rate, StateStorage(...))`. Its resolved storage authority comes
from that numerical plan and its exact block-state subjects; having a numerical plan does not
remove this local storage authority.

Admission requires an exact resolved block whose spatial descriptor is exactly StateStorage.
For an explicit numerical plan, every selected rate method must be exactly StateStorage, the
plan's canonical block-instance owner must equal the resolved block owner, and the spatial
descriptor (including ghost depth) must equal the plan's `primary_spatial()` projection. Each
rate must join the same resolved State owner and complete Rate(State) signature, the registered
operator, and one exact registered flux-free physical rate contract. A foreign block/rate, an
omitted subject, a conflicting spatial authority, or a substituted halo cannot supply authority.

Diffusion and other spatial methods can share a private runtime storage adapter. That sharing
does not make their numerical method StateStorage and does not grant local AMR authority. Their
existing spatial accuracy and transfer paths remain authoritative. Pointwise StateStorage also
does not invent a gradient stencil for AMR tagging.

The complete numerical plan remains available for stencil and boundary requirements. Only plans
joined to exact local storage authorities are excluded from spatial reflux requirements: a
flux-free pointwise rate has no face flux to reflux. Mixed or spatial plans keep their existing
reflux projection.

The existing case of a resolved local block without an explicit numerical plan is retained:
it still requires its previously authenticated StateStorage descriptor and exact resolved State.
The absence of numerics alone grants no storage authority. Per-layout and per-subject selection,
transfer-provider compatibility, nesting and reflux requirements remain unchanged.

This completes the existing local-storage contract. It changes no canonical payload shape,
wire format, Native API/ABI, equation, threshold, or runtime storage descriptor version. Plans
previously refused now carry the already existing `ResolvedAMRStateStorage` authority alongside
their unchanged resolved numerical plan. Numerical-plan identities and ghost requirements remain
part of the resolved simulation evidence.

Validation is Source-only in the IR17 environment with the selected checkout's Python directory
explicitly inserted before importing PoPS. The evidence folder is
`/Users/romaindespoulain/dev/tmp/sol61-amr-explicit-local-storage-source-20261007`.
The independent nonautonomous witness, its physical solve, Native installation and subsequent
numerical execution are separate obligations. No PDE result is synthesized by these tests.
