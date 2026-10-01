# Spatial direct interaction, source realization @1 (IR17)

This extension realizes a nonlocal spatial map, not the complete M26 aggregation–diffusion PDE. No flux, diffusion, energy-gradient identity, Poisson surrogate, model-specific matrix, or native scientific reception is supplied by this receipt.

For each explicitly selected component c, the physical declaration is

    I_c(x) = integral W(x,y) rho_c(y) dmu(y).

`CellVolumeMeasure` selects Cartesian physical cell volume multiplied by the actual prepared EB volume fraction kappa. `CellMidpoint` selects piecewise constant cell source values and kernel evaluation at physical cell centers. `DirectSpatialInteraction(max_workspace_bytes=...)` selects the numerical direct realization. These declarations remain distinct in the inspectable request.

```python
from pops.fields import (SpatialInteractionKernel, CellVolumeMeasure,
                         CellMidpoint, DirectSpatialInteraction)
from pops.model.spaces import FieldSpace

interaction = P.spatial_interaction(
    rho.n, SpatialInteractionKernel(2, lambda x, y: 1 + x[0]*y[1]),
    output_space=FieldSpace("interaction", components=("c", "a"),
        frame=rho.n.space.frame, support=rho.n.space.support,
        sampling="cell_center"), components=(2, 0),
    measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
    realization=DirectSpatialInteraction(64*1024*1024), source_scope="accepted")
```

W is a scalar applied independently to the selected ordered components; signed and nonsymmetric kernels are permitted. The closed coordinate Expr grammar is finite lossless binary64 literals, x/y coordinates, arithmetic/power, negation, abs, sqrt and exp. Opaque runtime Python callbacks and foreign physical variables are rejected. This grammar is an inspectable realization profile, not a restriction on future physical kernel extensions. Self interaction evaluates W(x,x); no diagonal omission, regularization, minimum periodic image, symmetrization, positivity or SPD assumption is introduced. Singular/nonfinite evaluated kernels/products/sums refuse collectively. Periodic geometry does not alter the declared W.

The provisional result is an owner-qualified FieldSpace with cell_center sampling, ordered selected components, the source point/clock/frame/support/layout, and explicit physical units. Known output units must equal W units + selected rho units + every coordinate-measure unit. Unknown inputs cannot manufacture known units. The operator never commits a State; guarded acceptance, observation, physical conversion and publication remain explicit Program operations.

## Source profiles and authority

* Uniform `source_scope="issued"` reads the issued State/candidate. `accepted` requires the actual State.n carrier. A candidate cannot be relabeled accepted. A State.n read must retain exact TimePoint(clock,0), including every original/canonical dependency leaf under a calculated candidate, with a visited-object cycle guard, and detached-contract/IR serialization; a calculated candidate at a later point remains admissible under issued scope.
* Composite AMR `accepted` requires SSA State.n and resolves `prepared_amr_block_state` on every level, excluding live attempt states. This is an explicit accepted hierarchy read, not an implicit conversion of candidate data.
* Typed state history uses its exact HistoryContract, lag and TimePoint. AMR requires the same authenticated retained sample and registered descriptor on every level, and rejects pending remaps. The consumer's current runtime frame is authenticated separately from the retained source sample.
* AMR issued candidates/direct fields with more than one level refuse: the existing level-local execution has no simultaneous composite candidate barrier. Implementing that barrier is an open extension, not a permanent limit on PoPS. Mono-level issued fields remain available.
* Moving maps refuse this Cartesian provider. A moving physical midpoint/measure provider is a separate extension; static Cartesian positions are not substituted for moving coordinates.

The complete request, clock and current native frame, geometry/domain/boxes/rank space/distribution, actual active/coverage/kappa measures and selected source bits are checked on the prepared execution lane. AMR topology epoch/materialization generation and live resource attempt are captured and checked again before consumption. There is no invented independent data-epoch counter: selected data are freshly snapshotted from the authenticated carriers in that exact attempt, not cached across calls. History supplies its actual sample identity. Source family/history descriptor and attempt ordinal qualify the AMR request.

Every stored source cell must have exactly one physical contributor. Replicated fields contribute on lane rank zero, while all replicas validate relevant masks and selected source values and compare their exact binary64 snapshot against the owner. Active, coverage and kappa bits are also compared independently on replicas, including excluded cells: equal product measure is insufficient to authenticate geometry. This physical rank is unrelated to the rank publishing evidence files. Distributed and empty ranks follow every vote and scalar transport phase. Invalid masks on a noncontributing replica cannot be skipped. Covered coarse cells and zero-measure EB cells do not enter the source quotient; target values are not multiplied by a target measure. Covered/inactive targets are zero.

The actual PreparedMultiBlockAmrSubcyclingEngine copies a detached candidate tower in prepare_attempt_, completes every recursive/synchronous callback, then calls publish_attempt_. AmrSystem's prepared state getter addresses that hierarchy's accepted carriers. The actual-engine host probe moves the coarse candidate (the active commit_many behavior), then before fine verifies accepted coarse/fine pointer/value/revision/epoch/generation identity and matching attempt/windows. The real direct-header convolution remains 4; a candidate-coarse substitution would give 22. Only terminal publication changes accepted revision/state. This is an actual header-class host proof with explicitly synthetic unit-test fields, not a native facade/DSO reception. The true emitted Uniform/AMR calls are separately syntax received; Native ROOT must still exercise the facade and multi-rank lifecycle.

AMR results enter the existing scratch registry keyed by owner, active level and exact SSA node ID only after calculation/votes. A detached map node is allocated and voted first; the final transfer allocates no node, and replacement uses statically checked noexcept field move assignment. Subsequent reduction/history does not infer an ambiguous level solely from identical box layouts.

## Work and workspace

The immutable compact source snapshot has N rows of coordinates, physical measure and selected component values. There is no N-by-N matrix. Work is direct O(Nsource*Ntarget*nselected), with additional canonical topology checks; communication uses owner scalar word reductions rather than an MPI int-count-sized buffer. No model-size/DOF cap or INT_MAX total-source cap is imposed. This first implementation prioritizes exact ownership and bounded workspace over communication throughput.

The voted bound uses checked size_t sums/products and includes both explicit host/device snapshots, returned local target values, copied target layout/distribution/owner and Fab/index vectors, level descriptors and selected-component storage. AMR checks level-descriptor storage before its allocation. Every rank votes budget failure before target/snapshot allocation. The API budget is not an RSS cap: borrowed source/mask carriers, pre-existing runtime caches, exact-contract strings/control metadata, registry nodes, allocator overhead and internal Kokkos/MPI implementation storage are outside that bound. uint64 budgets use canonical 16-character lowercase hex in IR17; shared legacy CBOR and existing IR schemas are unchanged.

## Source receipt and pending native obligations

Production gel dd46ea495f01f0e8e59e0caa3f9b8f697483f9b2 plus correction 61c78b9b430f3101c49f1a9ec306e2936edf78fc. The independent review found State.n point relabeling and malformed-kernel IR serialization missing refusals on dd46; 61c seals both before code emission/identity and stages scratch allocation before publication. The subsequent correction closes the independently demonstrated encapsulated linear-combination leaf as well; historical dd46 or 61c alone is not this final receipt. Private base bf1b2cfb34bf047f2c6b5afb599dd2187235a58d plus exact HistoryStorageOwner IR16 production 0bc86f3e91d54137decbf9524ab07ffac8a90012 (private cherry-pick 189e500b). Existing graphs without the interaction keep their previous schema/IR/semantic/codegen behavior; IR17 is conditional on a reachable spatial_interaction node.

Command from the exclusive worktree:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q --tb=short tests/review/test_sol61_spatial_interaction.py
```

The suite uses actual public Case/validate/resolve/ProgramModelGraph/emit_cpp_program for Uniform and synchronous AMR, syntax-checks both genuine emitted Dim2 MPI branches, and compiles/runs the actual direct header with Kokkos CPU and two threads. The host quadrature checks real MultiFab dimensions 1/2/3, signed nonsymmetric component permutation, EB fractional measure, covered coarse exclusion, genuine partial fine source coverage, target nonweighting, finite/overflow/budget refusals and signed-zero cell extraction. The final bounded suite receives 33 checks and 59 actual-header host assertions. Four existing public profiles (mixed linear, its component permutation, implicit stage, nonlinear-map implicit stage) preserve exact IR and emitted C++ hashes against the git archive of bf1b2cfb via tests/review/sol61_spatial_field_legacy_parity.py. These are source/host checks, not native DSO execution, AMR runtime, MPI execution, GPU, or scientific PDE qualification.

ROOT must rebuild the new System/AmrSystem volume-fraction exports and receive real native Uniform/AMR and MPI2: owned/distributed/replicated/empty-rank snapshots, one-rank invalid kappa/source/budget, replica divergence, authority mismatches, nonfinite self kernel, exact refusal before physical publication, saved output quadrature/selected-component permutations, history/regrid/restart provenance and retry rollback. Full candidate AMR needs a separately reviewed composite barrier. A scientific M26 acceptance still requires its original flux/diffusion and equation-specific oracle.
