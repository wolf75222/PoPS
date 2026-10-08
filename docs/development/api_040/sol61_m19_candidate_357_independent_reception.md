# Independent reception of M19 candidate 357 - 1 October 2026

Candidate: `357e2c2bf5dee3487218bfec65f65132195db5a0`, exact parent
`9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6`. Private checkout
`PoPS-sol61-m19-candidate-review`. The earlier independent oracle commit
`c7196e8534ddadc1e87d590916bedb5bc4c16ac3` is available in this review checkout.
No production, MAIN, environment, root evidence or native artifacts were edited.
No build, JIT, install, Kokkos or MPI execution occurred.

## Finding: rank-local paths block the new MPI fixture

The initial candidate's public test resolves component packages beneath each
rank's `tmp_path`, calls `pops.compile` directly on each rank, then saves all four
phases beneath `tmp_path / "m19-product-receipts"`. `_save` passes that path into
`runtime.checkpoint`. Pytest temporary paths can differ by rank. The authentic
`_multi_layout_executor._checkpoint` compares canonical targets across ranks and
raises exactly `ValueError("multi-layout checkpoint target differs across ranks")`.

This is a fixture P1 for the planned MPI2 reception, not a production permission
to relax checkpoint authentication. The new independent test extracts and executes
that real condition: identical paths pass and distinct rank-local paths refuse.
It does not simulate a native checkpoint, MPI transport or a successful restart.
The author was notified before integration. Shared path publication must precede
provider materialization/resolution; provider source writes must have one elected
publisher; `compile_resolved_plan_once` must authenticate one complete artifact.
Then checkpoint and restart must use the same exact published path on all ranks.

## Received source routes and equations

The candidate changes eight tests/docs files and no file beneath `include`,
`python` or `schemas`. Thus the generic production kernel, Python IR and ABI are
unchanged from the exact parent. The MPI integration DSO's op2/3 branch now builds
`PhysicalSupportTransfer` and returns the authentic
`apply_physical_support_transfer(descriptor, request, status)`. The previous
scalar fixture loop has been removed from that branch. The real raw DSO literal
is written through `transfer_component_source(Dim)` and passed to `compile_shared`.
Independent source mutants replace the call by a private loop, alter retained
axis/reduction extent, disconnect weights/header, or stop publishing that literal;
all six are refused. These are source routing checks, not a DSO compilation or
execution result.

The DSO witness uses unit weights and a scalar System configuration. It does not
receive the vector/signed-weight Python witness by implication. The new C++ unit
test separately passes widths 1/3/5 and three Nx/Nv pairs through the real kernel,
checks each component, and checks `R L = (sum weights) I`. Another real-kernel test
places NaN/+Inf/-Inf at an active cell whose moment weight is zero and requires
failure before a subsequent finite call. Both are future native tests here.
Their raw-view numerical failure is correctly distinguished from publication
rollback, which belongs to the prepared System transaction.

The public Python fixture has six cases: three `(Nx,Nv,width)` tuples
`(4,3,3),(2,5,1),(7,3,5)` and both mapping declaration orders. The product has
separate v/x physical coordinates; v occupies native axis 0 and x axis 1, whereas
the reduced field uses x at native axis 0. Vector components occupy independent
product fields. There is no component packing into the velocity dimension or
production hardcoded 4×3 extent. `reverse` reverses map declaration order; it does
**not** claim component-order permutation. The axis permutation is real.

For the declared affine input
`f[c,x,j]=(c+1)(x+1)+(c+2)j+(-1)^c x j`, the review independently computes the
uniform velocity functional with rational weights `4/Nv`, the signed moment with
weights `(-1)^j(j+1)`, and the constant extension of each moment component.
It extracts only the candidate's actual input/expectation function, and compares
its output to the separate Fraction/Cartesian-coordinate oracle. It does not use
the author's affine sum formula as its reference. Three families pass; eight
artificial countermodels (unweighted/absolute/normalized weights, swapped
components, stale sentinel, wrong axis/measure, component-zero lift) refuse the
original equations. They are mathematical negatives, not resealed native receipts.

The public declarations retain exact support/domain identities, units,
cell-average representation and authenticated providers; the author's source
selection receives changed domain/unit/geometry refusals and original internal
stage points. The finite witness uses dimensionless quantities/quadratures,
while a separate unit probe exercises measure-dependent physical units. No
Poisson field equation, charge subtraction, kinetic boundary flux, Vlasov
transport or BGK closure is supplied by the map chain.

## Source evidence

Using `/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python` (`PY`):

```sh
rtk proxy env -u PYTHONPATH "$PY" -m pytest -q tests/review/test_sol61_m19_product_oracle.py tests/review/test_sol61_m19_candidate_reception.py
rtk proxy env PYTHONPATH=python "$PY" -m pytest -q tests/review/test_sol61_m19_source_contract.py
rtk proxy env PYTHONPATH=python "$PY" -m pytest -q tests/python/unit/codegen/test_m19_product_support_contracts.py tests/python/unit/codegen/test_generic_physical_map_contracts.py tests/python/unit/codegen/test_physical_support_mapping.py tests/python/unit/runtime/test_physical_mapping_rejection.py tests/python/integration/runtime/test_generic_physical_maps.py tests/python/integration/runtime/test_interstage_physical_maps.py tests/python/integration/runtime/test_physical_mapping_transfer_properties.py tests/python/integration/runtime/test_m19_product_support_runtime.py -m 'not compiler and not native_loader'
rtk proxy "$PY" -m ruff check tests/review/test_sol61_m19_candidate_reception.py
rtk git diff --check
```

The exact candidate selection produced **38 PASS / 14 native deselected**.
Independent reception produced **20 new PASS**, with **58 autonomous oracle PASS**
and **15 descriptor PASS** replayed on the candidate. Ruff/diff checks pass.
The unresolved MPI fixture path issue prevents a positive MPI reception of 357.
The source result does not overwrite that finding.

## Saved-state and scientific boundary

Planned phases are initial, accepted, restored and replayed, with NPZ,
checkpoint digest, mapping counters, clock, exact artifact/bind identities and
all-rank local-box inventories. Provenance includes dimension, native DSO hash,
ABI/capabilities, compiled plan/components and source files; JUnit supplies rank,
size and receipt paths. Documentation and corpus consistently leave native
reception pending. Fixture-produced hashes are not an external owner seal.

Root must pin the closed inventory, actual installed Python/SDK/System package,
generated binaries/source, dimension, every rank's JUnit and exact phase order.
Only authentic saved states can then qualify original reduction/lift equations,
initial sentinels, accepted counters/time, exact restart and replay. Future
multi-patch/cross-rank native reception must establish each physical owner once
and preserve rollback. Mathematical replicas are received by the independent
oracle; the current Uniform runtime explicitly rejects replicated ownership.
That runtime limitation is not a limit on physical support mathematics.
AMR, GPU, field gauge/compatibility, transport/BGK and continuum refinement remain
outside this finite witness. No native positive has been fabricated or reused.
