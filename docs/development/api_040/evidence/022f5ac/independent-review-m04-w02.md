# M04/W02 advection–diffusion — pre-native source receipt

Date: 2026-09-29. GPT-6 Sol authored this independent migration example and
oracle; no heavy native build or run was performed here.

## Physical and installed scope

Corpus `docs/development/api_040/corpus.json` M04/W02 prescribes
`u_t+a u_x=D u_xx`, `a=1`, `D=.01`, periodic cosine, N=32/64/128 and t=.1.
`examples/migration/scientific/api040_m04_advection_diffusion.py` declares the
real installed Dim=2 PDE `u_t+a u_x=D(u_xx+u_yy)` using public
`Model.flux`, `Model.diffusive_flux`, `Diffusion(flux=...,transport=FiniteVolume(...))`,
`DiscretizationPlan`, and `ForwardEuler(FixedDt)`. For the y-invariant initial
and exact states, this PDE reduces *exactly* to the specified 1D equation.
The result qualifies only this 2D invariant trajectory, not Dim=1 or general
anisotropic diffusion.

The physical 1D frequency is `|a|/h+2D/h²`. The native isotropic 2D method has
two real diffusive contributions and uses `|a|/h+4D/h²`. Fixed steps are
preselected as `.9/(|a|/h+4D/h²)` for each N, satisfying the requested 1D
bound more strictly. The selected `Diffusion` method owns the exact transport
flux and both diffusion axes in **one** physical rate. This follows the
composition tested by `test_public_drift_diffusion_matrix.py` and
`test_amr_transport_diffusion_qualification.py`, rather than scheduling
independently stable updates. For N=32/64/128 the actual selected steps are
`0.012335526315789472`, `0.0039501404494382015`, and
`0.0011488970588235295`.

An exactly x-only tensor `diag(.01,0)` is not currently realizable by the
two-point monotone `Diffusion` method: `python/pops/numerics/diffusion.py`
rejects a zero diagonal coefficient as not strictly positive. The focused
`test_api040_m04_degenerate_diffusion_gap.py` exercises that explicit rejection.
No tiny positive surrogate coefficient is used to claim an x-only tensor.

## Oracle and acceptance fixed before native measurements

`api040_m04_oracle.py` calculates actual periodic cell means of
`1+.2 exp(-4π²Dt) cos(2π(x-at))` using the sinc factor. The saved native
initial/final/exact full grids are reopened before errors and invariants are
calculated; a center sample is rejected. The predeclared final criteria are
density L1 errors at N=32/64/128 at most `.009/.0048/.0024`, observed order at
least `.7`, mass defect at most `2e-11`, initial bind error at most `1e-12`,
time error at most `1e-12`, y variation at most `2e-11`, and state within
`[.7999999999,1.2000000001]`. These are proposed acceptance limits, **not**
observed native results. The receipt records saved-state hashes, native file
hash/ABI, package identity, run reports, and the reduced-equation scope.
It uses the shared strict `api040_receipts.receipt_json` serializer, which
tags CBOR digest bytes as hex and refuses other opaque or nonfinite values.

The W02 mathematical counterexample takes N=32 and `dt=.75/32`, so
`c=a dt/h=.75` and `r=D dt/h²=.24`. Individual transport (`c<1`) and diffusion
(`2r<1` in 1D; `4r<1` in 2D) bounds pass. Their sum fails:
`c+2r=1.23` and `c+4r=1.71`. A nonnegative unit impulse produces central
Forward-Euler weights `-0.23` in 1D and `-0.71` in 2D. These weights follow
directly from upwind advection plus centered diffusion and sum to one. The
prepared native test `test_api040_m04_combined_bound_runtime.py` requires a
rejection and unchanged state/time for the actual Dim=2 combined method; it
has not been executed in this source-only pass.

## Checks completed and pending

- Pure oracle and explicit x-only gap checks: **8 passed**.
- Public installed `pops.validate` + `pops.resolve` pass for N=32,64,128 with
  the composed method, selected steps, and periodic Dim=2 layout. This does
  not establish generated C++ compilation or native evolution.
- Native M04 convergence/balance and W02 refusal remain for the root-owned
  installed artifact reception. No thresholds should be relaxed after seeing
  its saved states.
