# M8 replacement inventory : retention decision

**Snapshot: 2026-09-13.** The semantic deletion set is **empty**. This page records the current
decision and consumer inventory; the detailed historical inventory remains available at the same
path in [Git history at source `3eb2722`](https://github.com/wolf75222/PoPS/blob/3eb2722e4618d25f1586632b592ee20b4dc505fa/docs/development/migration_m8_replacement_inventory.md).
The final report is [`migration_m3_m8_results.md`](migration_m3_m8_results.md); commands,
provenance, receipts, and hashes are indexed in the
[`migration_evidence/representative_20260913/index.json`](migration_evidence/representative_20260913/index.json).

Qualification follows the representative/reuse policy: an unaffected successful receipt may be
reused with source-impact evidence, while a changed route needs its own terminal witness. Repeating
an entire matrix at one identical source SHA is no longer required. Source presence or static checks alone do not qualify an M8 replacement. No new generic feature falls back to a
lossy legacy model; unsupported combinations keep explicit refusals.

## Retained consumers and assets

| Item | Current reason for retention |
| --- | --- |
| `native_scalar_flux.cpp.in` | The legacy conservative shared-interface `NumericalFlux` remains used by `examples/migration/scientific/legacy_interface_flux.py` and `examples/migration/r1_workflows.py`. The new periodic imported `NativeFunction` example has a different equation, boundary, and discretization, so it is not an equivalence result. |
| `FieldOperator` / `_b_field_operator` | Active compatibility and installation paths remain in `python/pops/fields/operator.py`, `python/pops/codegen/field_install.py`, `python/pops/codegen/program_field_plan.py`, and `python/pops/codegen/module_lowering.py`. |
| `PhysicalModelFor` / `PhysicalModel` | Public aggregate concepts remain in `include/pops/core/model/physical_model.hpp` and are required by the Dim1/Dim2/Dim3 narrow-operation tests. Narrow concepts are additive. |
| `HierarchyTensorSolverProvider` | Active production registration and API remain in `include/pops/runtime/amr_system.hpp` and `src/runtime/amr/amr_system.cpp`; typed component solvers are not blanket replacements. |
| Gaussian preset and schema | The public Gaussian preset is already generic supported authoring and remains available. Native API/storage/route-selection overloads and Python schema validation remain because active consumers use them, after the recorded exact-integral, tail, restart, and regrid qualification. |
| Physical-support map/provider | Generic map carriers and active consumers remain in `include/pops/runtime/dynamic/physical_support_transfer.hpp`, `python/pops/mesh/native_physical_mapping.py`, `python/pops/runtime/_multi_layout_executor.py`, and `python/pops/runtime/_amr_physical_mapping.py`. |
| Seven verification tests | `test_run_verification.py`, `test_reference_errors.py`, `test_verification_report_schema.py`, `test_verification_provenance_schema.py`, `test_verification_metrics_schema.py`, `test_verification_manifest_schema.py`, and `test_check_verification_manifest.py` are present in both baseline and current trees. Their historical campaign assets were already absent at baseline; that absence is not replacement support and must not be reconstructed as one. |

Accepted representative M8 witnesses include Gaussian uniform+AMR2, signed tails, the repaired
Dim3 mixture fixture, and separately authored transport/tensor composition. The original failures
remain in the evidence history. No listed consumer is obsolete under its qualified replacement
comparison, so the reviewed semantic deletion set remains empty. Active consumers, ABI/header
routes, public presets, and retained verification evidence remain supported.

The new optional grouped face capture supports prepared Cartesian physical or periodic closures.
Shared-interface, active embedded-boundary, and unsupported legacy capture requests are refused
before publication; their existing calls without capture remain available. The changed Uniform and
AMR capture contracts pass their own MPI2 failure/retry tests at source
`175b0883f9f0752f5710ef3db2c7d7d6802e400a`. This limitation does not remove an existing route.
