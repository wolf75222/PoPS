# M19 product support: existing mechanism, pending native reception

Baseline: `9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`. The mechanism already
exists; this patch changes no production header, ABI, IR or support identity.
The corpus remains `not_executed_for_this_mapping` until actual installed
artifacts and saved states are independently received.

Authority: handoff abstraction notes, “Kinetic moments and fluid closures”,
lines 867–890. Distribution support is `Omega_x × Omega_v`, potential support
`Omega_x`; integration and field pullback are physical maps. A kinetic closure
is not an inverse moment map. No BGK closure, charge subtraction or kinetic
boundary condition is invented here.

## Generic public and native seam

`PhysicalSupport` gives exact coordinate/domain pairs, independent of shape,
components, mesh and communicator. `PhysicalSupportMap` reduction/extension@2
joins strict subsets with explicit storage embeddings, cell-average sampling
and units. `AxisQuadrature` gives exact complete weights per eliminated-axis
cell, including measure and authored moment factor. Signed moments do not assert
a positive quadrature norm. Per component, `R f(x)=sum_j w_j f(x,j)` and
`L E(x,j)=E(x)`; `R L=(sum_j w_j) I`, with no hidden normalization or inverse.
Multiple eliminated axes use tensor-product factors.

The actual route is `LayoutPlanBuilder.require_mapping` or `Program.map`, then
`mesh/native_physical_mapping.py` authenticated provider, then common Kokkos
`dynamic/physical_support_transfer.hpp`. `system_layout_transfer.cpp` validates
exact rank space, distribution, retained geometry/topology and domain tiling,
transports complete fibres across patches/ranks, invokes the provider on
scratch candidates and publishes through native transactions. Extension receipt
source counts include repeated carriers per target patch. MPI replicas currently
refuse: the Uniform consumer requires unique ownership. That boundary does not
restrict mathematical supports. Dim1/2/3 and host float64 are realized; device
mapping is pending. Missing storage axes are explicitly authenticated periodic
unit-measure singletons, not extra physical coordinates.

`amr_layout_transfer.cpp` is the separate composite route with coverage,
measures and negotiated TransferApiV2 intersection integrals. Program continuations
preserve inter-stage points. `_multi_layout_executor.py` owns aggregate rollback,
receipts/counters and complete checkpoint restoration. These source routes and
fixtures require matching native reception; no new qualification follows from
this private checkout.

## Prepared reception

`test_m19_product_support_runtime.py` uses public validate→resolve→compile→bind→run
for `(Nx,Nv,width)=(4,3,3),(2,5,1),(7,3,5)` and both map declaration orders.
Velocity is native axis zero, position axis one; the reduced field's position is
axis zero. A nonseparable affine distribution is reduced with uniform measure
and with signed nonuniform weights, then extended. Zero flux preserves the
values through genuine authored internal stages. Four actual phases (initial,
accepted, restored, replayed) save NPZ, complete checkpoints, actual mapping
counters/clock/ownership, native SHA/ABI/capabilities, compiled components/plan
and source hashes. JUnit records dimension/rank/size and receipt paths.

This is a finite product-map witness. It is not a Poisson, transport, BGK or
Landau simulation. The future owner must externally pin the closed inventory,
authenticate Python/SDK/System-package/generated-C++ provenance, and run an
independent physical oracle. Fixture-produced hashes are not an external seal.
No native result or future NPZ has been fabricated.

`test_m19_product_support_contracts.py` receives actual public declarations and
resolved geometry without compilation, and refuses homonymous domains, wrong
units and changed retained geometry. Its NumPy premise check is source/math only.
The C++ unit `ProductReduceThenLiftVariesFibresAndComponentWidths` calls the real
Kokkos kernel and checks every component and `R L`. A second new kernel test
refuses active NaN/Inf even at zero moment weight and receives a subsequent
finite evaluation; raw-kernel failure does not itself claim transaction rollback. The existing MPI integration
DSO now also calls that kernel for op2/3 instead of a private scalar loop; its
real PreparedSystemLayoutTransfer still covers split patches/ranks, invalid axes,
divergent metadata, receipts and parent rollback/retry. Its scalar configuration
is distinct from the new public Python witness.

## Exact source command and future native commands

Run from the checkout root, without changing the shared environment:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python - <<'PY'
import sys
sys.path.insert(0, "python")
import pytest
raise SystemExit(pytest.main(["-q",
 "tests/python/unit/codegen/test_m19_product_support_contracts.py",
 "tests/python/unit/codegen/test_m19_product_mpi_publication.py",
 "tests/python/unit/codegen/test_mpi_compile_once.py",
 "tests/python/unit/codegen/test_generic_physical_map_contracts.py",
 "tests/python/unit/codegen/test_physical_support_mapping.py",
 "tests/python/unit/runtime/test_physical_mapping_rejection.py",
 "tests/python/integration/runtime/test_generic_physical_maps.py",
 "tests/python/integration/runtime/test_interstage_physical_maps.py",
 "tests/python/integration/runtime/test_physical_mapping_transfer_properties.py",
 "tests/python/integration/runtime/test_m19_product_support_runtime.py",
 "-m", "not compiler and not native_loader"]))
PY
```

After ROOT's matching Dim2 package build, authentic installed Python:

```sh
env -u PYTHONPATH POPS_KEEP_GENERATED=1 python -m pytest -q tests/python/integration/runtime/test_m19_product_support_runtime.py --junitxml=serial.xml
mpiexec -n 2 env -u PYTHONPATH POPS_KEEP_GENERATED=1 sh -c 'python -m pytest -q tests/python/integration/runtime/test_m19_product_support_runtime.py --junitxml=rank-${PMI_RANK}.xml'
```

The ROOT runner must authenticate its actual launcher rank variable and distinct
rank XML paths. The C++ owner runs `PhysicalSupportTransfer.*` and the existing
MPI integration target `test_mpi_system_layout_transfer` in the rebuilt tree.
Local source reception after the MPI fixture correction: **55 PASS**, fourteen native cases deselected; Ruff and
diff checks pass. No C++ compilation/native/JIT/install occurred here. GPU, complete kinetic PDE,
velocity-boundary budgets, field compatibility/gauge and continuum refinement
remain unreceived.

## MPI fixture publication correction

The first fixture gel `357e2c2` incorrectly used rank-local `tmp_path` for both
provider resolution/compilation and checkpoints. The real composite checkpoint
validator correctly refuses different targets across ranks. No production fix or
native tolerance change is needed.

The corrected fixture selects the real Dim2 module and communicator before
resolution. `collective_directory` publishes rank0's exact base path. Rank0 alone
emits provider sources/manifests; peers load the published source package and
authenticate the exact physical map and its common lowered source bytes without
rewriting files. `compile_resolved_plan_once` publishes rank0's isolated cache
and complete artifact identity; peers perform verified cache loading. The typed
`artifact_execution_context` authenticates the actual communicator before bind.
The receipt directory is published collectively. `_checkpoint_target` requires
exact common canonical paths on all ranks before even reading state or calling
checkpoint; an escaping phase or divergent path refuses collectively.

`test_m19_product_mpi_publication.py` exercises the true source package publication
and read-only peer resolution, true preparation/compile-once helper with only
communication/compiler substituted, root/peer rank-local caches, restored peer
cache environment, exact paths on three ranks including an empty peer, and
pre-native refusal of divergence/escape. It does not execute MPI or authenticate
a fake test artifact as a native package. Source/host results and native pending
remain separate.
