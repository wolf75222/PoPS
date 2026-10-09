# ADC-944: bounded physical support mapping

The public `Model.state` and `Model.species` declarations accept `PhysicalSupport`,
`PhysicalDimension` units and explicit sampling metadata. Physical population,
quantity, support and storage identities remain separate. `PhysicalSupportMap`
represents two explicit directional operations: a tensor-product
cell-average reduction with exact authored per-axis weights, and a
constant-extension field pullback. It has no inverse
moment closure.

## Native realization and timing

One native Dim=2 artifact/process stores the distribution on `(Nx,Nv)` and the
mathematically 1x field on `(Nx,1)`. The field hidden axis is explicitly periodic
with interval `[0,1]`, one cell and measure one. Its differential contribution is
zero. Physical x geometry and topology must match exactly. The velocity domain
and cell count must authenticate the declared exact rational quadrature weights.
Velocity integration multiplies each cell average by `dv`; it does not compute
an average over the velocity interval.

The existing aggregate step transaction executes this exact chain:

1. Native moment transfer overwrites the explicitly declared density state.
2. The field-layout Program solves its generic physical FieldProblem at accepted
   time c=0, consumes the outcome and commits the field observation state.
3. Native pullback captures that fresh observation and extends it along velocity.
4. The phase-layout Program consumes the explicitly declared observation through
   the existing coupled-rate provider and evolves the distribution.

All cell work uses compiled C++/Kokkos. Python schedules existing native sessions;
it never transfers numerical field arrays to implement a map. The native System
transaction owns state rollback. Field solve, state evolution and mapping receipt
counters publish only through the existing acceptance authority.

Resolve verifies the field equation reads its declared mapped moment, its storage
belongs to the field layout, and the pullback source depends on the consumed
fresh field solve. Local coupled-rate and matrix-free regions are partitioned by
actual dependency closure; a captured coefficient from another layout fails.
Legacy multi-layout FieldOperator installation remains unavailable.

## ABI and capability boundary

Transfer ABI-v1 struct layouts and operation 1 are unchanged. Operation values 2
(velocity moment) and 3 (physical pullback) are additive. The catalog generator
owns their declarations. Provider source and the exact physical map are both
included in the authenticated component package; ordinary header/ABI signature
checks reject incompatible stale packages. Older native runtimes reject unknown
operation values. No incompatible Dim=1 library is loaded alongside Dim=2.

The source realization now supports generic strict support subsets, explicit axis
permutations and one or more eliminated coordinates within the native Dim1/2/3
storage contract. `AxisQuadrature` accepts signed, nonuniform complete weights;
the uniform velocity measure remains a convenience declaration. Components stay
separate from spatial coordinates. Product cells are never packed into components.
The singleton embedding of a lower-dimensional field has zero differential
contribution; it does not turn its mathematical support into phase space.

The prepared Uniform consumer authenticates exact shared geometry/topology,
rank space, locally owned patch distributions and exact domain tiling. It gathers
complete reduction fibres across patches and ranks into target-owned carriers,
then invokes the loaded Transfer provider. An extension may repeat source
carriers per target patch; its receipt counts those carriers explicitly. MPI
replicated fields are refused by this consumer because its contribution protocol
requires unique ownership. This is a realization boundary, not a restriction on
physical product supports. Host float64 remains the realized mapping execution
space. Native device execution is pending.

AMR has a separate prepared composite route (`amr_layout_transfer.cpp`) using
finest-owned coverage, measures and negotiated `TransferApiV2` intersection
integrals. Inter-stage maps use the existing Program continuation rather than
relabeling authored stage points. The source contracts and native fixtures for
these routes exist; their reception must be pinned to matching installed native
artifacts. This document no longer states the superseded one-rank/one-patch bound.

The finite M19 witness and exact current pending/received boundary are tracked in
[`api_040/m19_product_support_reception.md`](api_040/m19_product_support_reception.md).

## Qualification ledger

- Representable: explicit population/quantity/support/unit/quadrature identities.
- Validated: incompatible support, sampling, units, missing map and inverse
  reconstruction fail before native provider binding.
- Resolved: geometry, field ownership, exact moment dependency and fresh
  field-observation dependency are checked against the actual Program.
- Emitted: independent field and phase Program sources and C++/Kokkos map
  provider sources are checked in the unit source/emission suite.
- Executed: pending the integration-owned matching immutable Dim=2 package build.
- Numerically checked: pending that native build. The declared integration suite
  contains `(Nx,Nv)=(16,12),(32,24),(64,48)` with 20 coupled steps each, exact
  polynomial cell-average velocity integrals, discrete Fourier evolution/field
  oracles, total inventory, c=0 timing, prescribed pullback, native solve
  rejection rollback and checkpoint/restart continuation.
- Performance-characterized: unqualified; no timing claim is made by source or
  structural checks.

Run `tests/python/integration/runtime/test_physical_support_mapping.py` in the
matching installed package environment for the full declared numerical workload.
The source-only evidence is not a claim of native integration completion.
