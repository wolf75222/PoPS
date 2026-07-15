# Final executable acceptance targets

[`EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py`](EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py)
is the final public target, not a migration example. It deliberately contains one authority per
concern and no fallback to an older or lower-level API.

[`EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py`](EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py)
extends the same public lifecycle with an explicit additive IMEX tableau, typed field solves,
two-level subcycled AMR, conservative transfers and accepted-state consumers. Its matching
contract note is
[`docs/design/final-advection-imex-amr.md`](../../docs/design/final-advection-imex-amr.md).

[`EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py`](EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py)
composes the provided plain `HyQMOM15` model with a periodic Poisson field, conservative HLL
transport, an inspectable IMEX Program, scientific artifacts and exact restart continuation. Its
matching contract note is
[`docs/design/hyqmom15-final-contract.md`](../../docs/design/hyqmom15-final-contract.md).

[`EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py`](EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py)
selects two state spaces of one model into two owner-qualified blocks, couples them through a typed
elliptic field on the same periodic layout, and proves scientific outputs plus bit-identical restart
continuation through the public lifecycle.

## Public contract

- immutable typed rectangle, Cartesian frame, boundary handles, grid and periodic-axis partition;
- frame-aware conservative state placement;
- strict Handle/Expr separation for every runtime parameter read;
- typed vector and axis-keyed physical flux `F`;
- explicit rate equation `A: ddt(U) == -div(F)`;
- family-organized `DiscretizationPlan` with `FiniteVolume(F, MUSCL(VanLeer), ScalarUpwind)`;
- owner-qualified block state in the sole explicit SSPRK2 `Program`;
- exact `StagePoint`/`TimePoint`, named SSA values and explicit commit;
- Case-owned AMR threshold parameters and a separate run-control mapping.

## Resolution obligations

These are requirements of the final public protocols, not invitations to use private adapters:

1. `pops.boundary.TransportBoundarySet` accepts domain boundary handles, qualified state
   handles and parameter-dependent data, then resolves to the boundary-port/ghost plan.
   The resulting set is registered only in `DiscretizationPlan.boundaries`.
2. One public AMR aggregate owns hierarchy, tagging, regrid, transfer and execution. Resolution
   derives the discrete indicator context from the qualified state plus its numerical plan and
   lowers the aggregate to the hierarchy/transfer/initial/bootstrap authorities. Transfer
   order and halo depth come from `pops.lib.amr.StateTransfer`; users do not repeat them.
3. `InitialCondition` and its projection enter the Case through `case.initials.add`, which accepts the
   qualified state once. `Constant`/`Gaussian` lower analytic data without a Python callback;
   `BindArray()` instead declares a complete conservative value supplied exactly once through
   `pops.bind(initial_values={block[U]: array})` and prolonged by the resolved AMR transfer.
4. `ConsumerGraph.from_consumers` lowers direct `ScientificOutput`, `Checkpoint`, and diagnostic
   descriptors to the exact layout-qualified runtime graph. Parallel I/O is derived from the format
   (`HDF5(parallel=True)`), so no second switch can disagree with it. `case.consumers(graph)` is the
   only Case attachment. The shipped final provider is serial, so the normative scripts use
   `HDF5(parallel=False)`; a collective descriptor requires a proved non-serial `ExecutionContext`
   and is otherwise refused during planning.
5. Named component handles select multi-component boundary, initial-data, AMR and
   diagnostic selection. A symbolic component expression cannot be used as a dictionary key because
   `Expr` is intentionally non-hashable; this scalar target safely addresses the whole
   one-component qualified state.
6. Analytic coordinate expressions and ready initial profiles use a public protocol. The target
   uses a pre-implemented `pops.lib.initial.Gaussian`; the frame axes themselves remain immutable
   identity descriptors rather than pretending to be symbolic coordinates.

Every join preserves the public lifecycle shown in `main()`:

```text
Model + Program + Case -> validate -> resolve(layout=...) -> compile -> bind -> run
```

`resolve()` is strict by definition. There is no `strict=True`, old-manifest escape or silent
substitution flag in the target.
