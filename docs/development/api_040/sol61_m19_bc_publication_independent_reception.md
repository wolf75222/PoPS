# M19 publication correction: independent source reception — 1 October 2026

Received exact author correction `bc8259c7eae2c165ea9106aa4bcfac7cde3e0915`,
parent `357e2c2bf5dee3487218bfec65f65132195db5a0`, in the private
`PoPS-sol61-m19-candidate-review` checkout. The historical review of 357 remains
unchanged: its rank-local checkpoint finding is valid for that SHA. This
correction closes that fixture issue at source/host level; native reception is
still pending. No production, MAIN, environment, SDK or root evidence was edited.
No build, JIT, install or native/MPI execution occurred.

## Corrected publication and authentication

The fixture selects the Dim2 module/communicator before resolving the plan.
`_prepare_artifact` publishes rank zero's base directory through the existing
`collective_directory`. Only the elected root materializes provider sources.
Every rank converges after that publication before peers resolve using the
read-only `_load_published_provider`. That loader requires the real
`SourceComponentPackage`, loads the declared map through the Transfer interface,
compares exact physical-map metadata and compares the published source bytes to
the common `_native_source` lowering. It does not rewrite peer source files.

MPI compilation/loading uses the existing `compile_resolved_plan_once`, which
compares exact plan identities, publishes the root cache and artifact identity,
and authenticates each peer's artifact before runtime construction. The execution
context is typed, and its communicator identity/rank/size is checked. Serial
execution uses the ordinary public compiler. No rank-count restriction is added.

The receipt directory is also published collectively. `_save` calls the new
`_checkpoint_target` **before** any state/ownership read. It checks phase basename
and directory containment, resolves the target and allgathers exact path strings
before entering the runtime. Checkpoint and restart then use that common target.
No production checkpoint guard was relaxed, and no support/unit/owner contract
was removed. The complete diff from parent 9fb7 still contains no production
file beneath `include`, `python` or `schemas`.

## Independent execution of the actual save protocol

`test_sol61_m19_save_host_protocol.py` extracts the authentic `_save` and
`_checkpoint_target`, executes them with the authentic `collective_checks.py`
functions, and substitutes only the native collective transport and runtime.
It imports no PoPS package and never calls Kokkos/MPI. Its sequential rank
substitutes do not establish true collective progress or deadlock freedom.

Twelve checks cover three ranks and all four phases (initial, accepted, restored,
replayed), including empty peers. They verify that path consensus occurs first,
that every block's state and ownership read occurs in the same declared order,
and that the exact resolved target reaches checkpoint. Each admitted trace stops
at an explicitly raised checkpoint boundary before any file write. No fake native
checkpoint, NPZ, receipt, or positive saved-state qualification is produced.

Three divergent-peer tests refuse before **any** runtime call. Seven invalid
phases refuse before path transport, state reads or files. All 22 checks pass.
The original candidate's Fraction equations and authentic DSO source route remain
received by the 20 independent checks from `d2a1b38d0184fa2fea904e2dd340421df3236085`.
The 58 autonomous product/ownership checks also pass on the corrected tree.

## Commands and observed results

From the private corrected checkout, with
`PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`:

```sh
rtk proxy env -u PYTHONPATH "$PY" -m pytest -q tests/review/test_sol61_m19_save_host_protocol.py tests/review/test_sol61_m19_candidate_reception.py tests/review/test_sol61_m19_product_oracle.py
rtk proxy env PYTHONPATH=python "$PY" -m pytest -q tests/python/unit/codegen/test_m19_product_mpi_publication.py tests/python/unit/codegen/test_m19_product_support_contracts.py tests/python/unit/codegen/test_generic_physical_map_contracts.py tests/python/unit/codegen/test_physical_support_mapping.py tests/python/unit/runtime/test_physical_mapping_rejection.py tests/python/integration/runtime/test_generic_physical_maps.py tests/python/integration/runtime/test_interstage_physical_maps.py tests/python/integration/runtime/test_physical_mapping_transfer_properties.py tests/python/integration/runtime/test_m19_product_support_runtime.py -m 'not compiler and not native_loader'
rtk proxy "$PY" -m ruff check tests/review/test_sol61_m19_save_host_protocol.py tests/review/test_sol61_m19_candidate_reception.py
rtk git diff --check
```

Observed: **100 independent offline checks PASS** (58 + 20 + 22), **47 selected
author source/host checks PASS / 14 native cases deselected**, Ruff/diff PASS.
The author's separately reported 55 checks are not reassigned to this narrower
explicit command. The selected actual publication test also verifies root/peer
plan parity, unchanged source mtime/digests, and refusal of tampered source bytes.
Its communicator/compiler substitutions remain explicit.

Source integration of the correction is supported by this reception. Only the
root's rebuilt installed artifact and authentic serial/all-rank MPI saved states
can qualify compile/load, physical transfers, checkpoint/restart, rollback and
the original equations. The six public tests remain a finite product map witness,
with separate supports, widths 1/3/5, axis and map-declaration permutations and
signed/nonuniform moments. They do not solve Vlasov, Poisson or BGK; AMR, GPU and
continuum scientific acceptance are unchanged open work.
