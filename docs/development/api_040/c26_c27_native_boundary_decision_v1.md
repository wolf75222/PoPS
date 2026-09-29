# C26/C27 — production native boundary decision v1

Evidence snapshot: MAIN `5d65191c3e70bdf6f6359694001d83ab21a203f0`
(dirty working tree; source paths checked on 2026-09-29). Reference:
`PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/document/02_contracts.tex`,
C26 lines 119–126 and C27 line 128. The reference `spec/include/pops_contract.hpp`
calls itself a compilable ABI sketch, not an upstream PoPS ABI.

## Decision

Preserve the **meaning** of C26/C27 through PoPS's authenticated, typed
compile/bind/native-execution boundary. Do not add the reference mini-runtime
exports `pops_initial`, `pops_step`, or `pops_run`: they accept flat buffers and
would create a second state layout, time controller, and publication authority
beside the production `Program`, `System`, and `AmrSystem` routes. An absent C
export with that spelling is not by itself a missing production capability.

Keep four different version domains explicit:

| Domain | Current authority | Role |
| --- | --- | --- |
| PoPS release native ABI | generated `NATIVE_ABI_VERSION = 3`, native `__abi_version__`, release-contract digest | Python extension and release compatibility. |
| Generated System package protocol | `kNativeSystemPackageAbiVersion = 5` and `pops_native_system_package_abi_version` | Produced model package to host System/AMR installer. |
| Native component interface | `POPS_COMPONENT_PROTOCOL_ABI_V1 = 1`, generated catalog/interface versions | Out-of-tree component table and batch-call shape. |
| Reference conceptual header | `pops_contract::abi_version = 2` | Mini-runtime sketch only; not a production PoPS extension version. |

## Property mapping

| Contract property | Existing production mechanism | Evidence and test state | Decision |
| --- | --- | --- | --- |
| C26 sizes, state plan, and configuration fixed before cell loops | `CompiledArtifactManifest` includes blocks, variables, roles, parameters, bind schema, ghosts, dimension, precision and native entrypoints; generated package binds parameters before native loops. | `external/artifact_manifest.py`; `test_bind_schema.py`, `test_bind_validation.py`, `test_program_emit_params_multimodel.py`. | Covered by typed manifest; do not use name lookup through per-cell pointers. |
| C26 ABI and dimension at load | `_native_selector` authenticates extension bytes, origin, version, dimension, ABI and all-rank identity; generated packages expose exact key/model identity/protocol v5; `check_compiled_matches_module` rejects known header/dimension mismatch before `dlopen`; native installers recheck exact key and protocol. | `_native_selector.py`, `codegen/abi.py`, `runtime/_system_install.py`, `runtime/_amr_system_equation.py`, `amr_system.cpp`; wrong-ABI native integration already exists. New source-only negative tests pin pre-load header and dimension refusals. | Covered, subject to actual installed-native receipt. |
| C26 artifact identity and stale cache | Semantic/artifact-spec identities and sidecar bind emitted model, toolchain, route and native ABI; cache hit rehashes binary. Installed component also authenticates exact binary bytes and exported symbols. | `codegen/_artifact_identity.py`, `compile_provenance.py`, `external/artifacts.py`; existing spec/semantic mismatch tests; new tests alter cached binary bytes and artifact token. | Covered for the tested routes. |
| C26 temporary candidate, validation, collective publication and rollback | Production Program/System/AMR owns accepted snapshots and staged candidates, with collective rejection and restoration; no flat-buffer `pops_run` controller is needed. | `program_runtime_state.hpp`, `system_program.cpp`, `amr_system.cpp`; native transaction tests exist. | Mechanism exists; exact reference status integers 1–5/20–23 are not a production ABI promise. |
| C26 diagnostic guard identity | Typed native rejection/action and per-route diagnostics supersede the sketch's coarse status 2; completeness of every scientific guard identifier still needs per-route evidence. | `StepAttemptRejected`, `PopsComponentStatusV1`, route-specific tests. | Partially covered; do not claim a universal host guard-id mapping. |
| C27 borrowed local views and typed resources | Generated `PopsConstFieldViewV1`/`PopsFieldViewV1` include rank, shape, strides, memory space, patch/layout identity and ownership; platform execution context proves communicator/device/datatype. | `generated_component_abi.hpp`, `_platform_contracts.py`, `test_platform_manifest.py`. | Covered for declared component interfaces; no universal `std::span` C++ plugin ABI. |
| C27 registration, no dynamic lookup in hot loops | Component loader validates table version, catalog digest, semantic/manifest identity, exact interface set/size/operation completeness; preparation resolves state once before batch calls. | `component_loader.hpp`, `external/artifacts.py`, generated catalog tests and component integration. | Covered for generated SDK interfaces. |
| C27 async completion and leases | Prepared resource tickets/leases retain owners and buffers until query/drain/cancel acknowledgement; collective step rejection coordinates completion. | `prepared_resource_lifetime.hpp`, `prepared_resource_cache.hpp`, `collective_step_rejection.hpp`, C36 tests. | Implemented as internal prepared-resource route, **not** a universal external component completion-token ABI. GPU overlap is not qualified here. |
| C27 visible allocations, collectives and transfers | Manifest effects, platform capabilities, prepared execution context and route-specific schedules expose these requirements before realization. | `external/artifacts.py`, `_platform_contracts.py`, generated component tables. | Partial by route; this review does not certify every arbitrary extension or backend. |

## New negative evidence and remaining checks

`tests/python/unit/codegen/test_api040_c26_c27_authority.py` adds four pure
source tests: reject foreign header signature and dimension before dynamic
load, reject changed cached binary bytes, and reject a forged artifact identity
with unchanged bytes. All four pass against the isolated source checkout;
they are **not** a native ABI test. The prior suite already tests bind ABI,
OpenMPI/MPICH mismatch, wrong native loader ABI, component table/manifest
signatures, source package digest, and missing/stale sidecars, so duplicating
those fixtures would add no independent evidence.

For a release claim, run the existing installed-native and component loader
checks on the exact final source/native hash. A separate design decision is
needed before promising a general asynchronous *external* component ABI:
its completion result must own buffers, propagate cancellation acknowledgement,
state supported devices, and define collectives in its declared execution
context. Merely copying `pops_contract.hpp` would not provide those guarantees.
