# Pybind Binding Audit

ADC-593 stops the `_pops` extension from growing by one hand-written binding file per numeric
combination (transport x Riemann x AMR mode). This note is the permanent classification of the
pybind translation units, the block-build dispatch chain, why the per-route seam TUs exist, why the
native bricks stay pre-instantiated, and the growth rule that replaces the per-combination file.

## Classification of the binding TUs

The bindings live under `python/bindings/`. Each TU falls in one of five categories.

| Category | Files | Role |
| --- | --- | --- |
| Module entry + init split | `core/bindings.cpp`, `core/init/init_core.cpp`, `core/init/init_system.cpp`, `core/init/init_amr.cpp`, `core/bindings_detail.hpp` | `PYBIND11_MODULE` and the `py::class_` / `.def` registrations. Internal seams of the bind flow, not public vocabulary. |
| Runtime-core facades | `system/base/system.cpp`, `amr/amr_system.cpp` | The `System` / `AmrSystem` facade + `Impl`. The two heavy TUs; string dispatch to the seam symbols lives here. |
| Legacy / geometry routes | `system/base/system_polar.cpp` | The verbatim polar (ring) visitor body. Unique shape; kept hand-written. |
| Riemann dispatchers | `amr/block/compressible/amr_block_compressible.cpp`, `amr/compiled/compressible/amr_compiled_compressible.cpp` | Thin `if (riemann == ...)` routers, one per transport, routing to the per-flux seam leaves. Unique control flow; kept hand-written. |
| Per-route seam leaves | generated from `bindings/seam_combinations.cmake` (system/isothermal, system/compressible, amr/block, amr/compiled) | One `pops::detail::build_*` function per `(transport, flux)`, each instantiating ONE leaf of the template product. Formerly hand-written; now generated. |

Counts after ADC-593: 9 hand-written binding TUs remain in git; 19 per-route seam leaves are
generated at configure time from a single manifest (no longer tracked source files).

## The block-build dispatch chain

A Python string selects a route; no algorithm selection lives in pybind lambdas.

1. `pops.System.add_block(...)` (Python) calls the `System.add_block` binding (`init_system.cpp`).
2. `System::add_block` (`system/base/system.cpp`) runs the shared `validate_riemann` / `validate_limiter`
   and does a small `if/else` on the transport / riemann STRING.
3. Each branch calls a `pops::detail::build_block_<transport>[_<flux>]` SEAM symbol (declared in
   `include/pops/runtime/builders/block/block_seam.hpp`, defined in a generated seam TU).
4. The seam calls `build_block_for` / `build_block_for_make` -> `make_block_<flux>` (the typed template
   route dispatch, phase-2 `route_ids` / `model_factory`). The AMR side mirrors this with
   `build_amr_block_*` / `build_amr_compiled_*` and `dispatch_amr_*`.

The declarative registry is NOT the bindings. Transports come from `brick_catalog.hpp` (Python mirror
`brick_catalog.py`), fluxes from `route_ids.hpp` (Python mirror `routes.py` `_REGISTRY["riemann"]`).
`brick_catalog.hpp` static_asserts itself against the registry and route tables. The bindings only
translate a validated string to the pre-instantiated seam symbol.

## Why the per-route seam TUs exist (build memory)

The seam symbols could be ONE template `make_block<TR, Flux, Limiter, Model, ...>` product. Emitting the
full product (~1700 leaves) in a single TU exceeds 7 GB at `-O3` under Kokkos (`cc1plus` peak). That
kills CI runners and slows local builds. ADC-335 / ADC-342 / ADC-359 split the product one TU per
`(transport, flux)`: each TU instantiates only its leaves, TUs compile in parallel, and peak memory is
bounded by ONE leaf TU. A Ninja `JOB_POOL` (`pops_heavy_module_tu` for the module,
`pops_heavy_tu` for the tests) serializes the two big facades so two multi-GB compiles never overlap.

This mitigation is correct and is UNCHANGED by ADC-593. What was wrong was the GROWTH STRATEGY: a new
Riemann or reconstruction meant a new hand-written pybind file. ADC-593 keeps the one-leaf-per-TU memory
shape but GENERATES those TUs from one declarative manifest.

## Why native bricks stay pre-instantiated (not the runtime .so loader)

The seams are pre-instantiated template leaves compiled into `_pops`. They are not moved to the runtime
`.so` loader (the DSL `production` / `native` path) because that loader honestly CANNOT express the
routes the pre-instantiated path covers: `stride > 1` sub-cycling, IMEX-RK, partial IMEX masks
(implicit vars / roles), and MPI / GPU dispatch. The pre-instantiated seam is the right OWNER of the
builtin numeric routes today; the `.so` loader owns USER-authored models compiled on the fly. This is a
deliberate boundary, not an omission.

## The growth rule (manifest row, never a new file)

Adding a Riemann or reconstruction is:

1. one row in `python/bindings/seam_combinations.cmake` (side, transport, flux, symbol, output path);
2. the `make_block_<flux>` / `dispatch_amr_*_<flux>` template in the headers -- that is NUMERICS, not
   bindings;
3. the flux row in `route_ids.hpp` / `routes.py` (the registry) if the flux is new.

No new hand-written pybind file. `tests/architecture/test_pybind_seam_manifest.py` enforces this: the
former leaf files must be absent from git, every manifest `(transport, flux)` must be a legal catalog /
registry route, and no new `.cpp` under `python/bindings/` may carry the seam-leaf signature outside the
generated dir and templates. `python/tests/test_seam_combinations.py` is the runtime sibling: it drives
a native `add_block` for every manifest combination and asserts a CFL step advances finitely.
