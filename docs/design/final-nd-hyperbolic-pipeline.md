# Final compile-time ND hyperbolic pipeline

The spatial rank is selected once when the native Python extension is built. The selected
`kNativeDimension` instantiates `System<Dim>` and its complete numerical graph. Python may inspect
input rank and choose an already-built 1D, 2D or 3D artifact, but a native artifact never discovers
or switches rank while executing a kernel.

The canonical numerical path is:

1. `Geometry<Dim>` prepares one `PreparedMetricProvider<Dim, Metric>`.
2. `Fab<Dim>` or `MultiFab<Dim>` carries conservative state and an exact `Extent<Dim>` ghost
   envelope.
3. `reconstruct_face_pair<Axis, Variables>` reconstructs both traces with static axis,
   orientation, variables and policy.
4. the selected Riemann policy writes integrated fluxes to one `FaceField<Dim>`;
5. `PreparedHyperbolicBoundary<Dim>::apply_physical_flux_conditions` applies qualified
   post-Riemann face laws such as `no_flux` to that candidate field;
6. `assemble_residual_from_face_fluxes` / `conservative_residual` accumulates every axis of that
   same field.

`PreparedCartesianOperator<Dim, Model, Metric, Reconstruction, NumericalFlux, Variables>` is the
sole Cartesian preparation. Face and residual candidates are validated before publication. There
is no `Fx`/`Fy` pair, contiguous-span adapter, `Box2D` endpoint or run-time dimension cascade.

`PreparedHyperbolicBoundary<Dim>` similarly consumes `MultiFab<Dim>` and fills physical halos with
axis-static `FieldView<Real, Dim>` kernels. `periodic_axes()` is the topology projection used by a
generic system; no `BCRec` or two-dimensional `Periodicity` projection belongs to this contract.
`no_flux` is not inferred from an extrapolated halo: it explicitly zeroes the matching boundary
plane after Riemann evaluation and before the transactional residual publication.

## Specialized providers still requiring qualification

The following providers are retained as transitive implementation surfaces, not included by the
canonical spatial umbrella and never selected as a two-dimensional fallback:

- polar coordinates must compose through a qualified `PreparedMetricProvider` and one
  `FaceField<Dim>` without importing the legacy Cartesian operator;
- embedded-boundary and mask operators must accept `Index<Dim>`, `FieldView`, axis-static face
  iteration and the same fail-before-publication status channel; their admissible topology may be
  narrower than all three ranks, but that limitation must be an explicit capability;
- analytic physical boundaries need a device evaluator accepting the complete physical
  `RealVector<Dim>`, generic tangent wrapping and collective finite-value preflight;
- characteristic no-inflow needs a model-qualified axis-static provider over the canonical state
  schema;
- the legacy prepared boundary-plan orchestration must be rebuilt over generic halo schedules and
  `FaceField<Dim>` before it can become a standalone API again.

Until those contracts are supplied, analytic and characteristic physical laws fail closed and the
specialized spatial headers remain outside the public standalone authority.
