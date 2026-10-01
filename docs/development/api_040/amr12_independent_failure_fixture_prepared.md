# AMR12 independent real-native failure fixture — prepared source only

Base `f1edec55`; private worktree `PoPS-sol61-amr12-failure-fixture`. This change edits tests/documentation only. No environment installation, native compilation, JIT, numerical execution or MPI launch was performed. Python AST parsing and in-memory bytecode compilation succeeded; the AST contains one selected native test. This is source inventory, not successful pytest collection or a native test result.

The distinct [fixture](../../../tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py) reuses `build(8,2)` from the existing evolved-stage AMR support and `capture/same_images` from the original public runtime test. Constants, physical equations, controls and guards are unchanged. Actual installed Dim2 PoPS must validate→resolve→compile→bind→run two steps before this campaign can inject anything. Source-only execution cannot count as reception; a missing binding/import/assertion cannot count as a successful refusal.

Prepared real operations:

- Capture the canonical accepted POPSCAR1 archive and POPSDIA1 diagnostics from the native owner; require canonical archive digest agreement across real ranks.
- Validate the authentic image without mutation; reject truncated image, bytearray and multidimensional ndarray on rank0 while peers provide the genuine bytes.
- At the real Python checkpoint transport boundary, reject int16 dtype and multidimensional uint8 shape, with all-rank consensus and unchanged live state.
- On MPI2 only, give rank0 a same-shape codec image with one final IEEE-word byte changed. Native exact archive consensus must reject the disagreement on both ranks. Serial does not receive this MPI refusal.
- Reject restoration outside a native restart transaction; compare exact accepted state/diagnostic images and actual arrays/history/clock/carrier manifests.
- Begin the real native restart transaction. Native capture must reject an active transaction. Set actual coarse Q0 valid cells one ULP upward on rank0 through `set_block_level_state`; require changed native rank0 carrier manifest and changed global scientific state, not a fixture marker. Restoring the genuine accepted archive must reject specifically `valid cells contradict scientific state projection` and leave the contradicted live state unchanged.
- Roll back through the real native method; compare the accepted image, diagnostics, complete carrier hashes, arrays, histories and temporal/clock metadata exactly.
- A second real transaction restores the genuine unchanged image successfully, then rolls back. Restoration after rollback must again refuse the absent transaction.

Native `checkpoint_state_carriers` and `_checkpoint_program_diagnostics` intentionally refuse active restart transactions. Their guards are respected: active snapshots compare actual full-grown native carrier manifests, projected arrays/history, and diagnostic IEEE double bits from the readonly native map. Accepted snapshots before/after rollback compare the raw POPSCAR1/POPSDIA1 bytes too. No Python cell reconstruction, ghost fill or fabricated accepted codec image is introduced. The diagnostic bits are serialized observations of native doubles, not substitute diagnostics.

The direct codec fixture does not claim commit/finalize coverage: those phases require full history/auxiliary restart authority and belong to the existing public `test_public_evolved_stage_amr_checkpoint_and_composite_Q`. Native transaction bindings are required to exist. No fake GPU/OOM, allocation failure, regrid, ownership redistribution or crash durability injection is included.

At successful future runtime completion, each real rank writes `receipt-rankN.json`, native and fixture digests, accepted/rolled-back raw images, projected NPZ arrays, full carrier metadata and the exact per-rank refusal records. These files do not exist as results from this preparation. ROOT must authenticate the installed source/header/DSO, runner result/XML and source/test inventories independently. Refusal records are nine named operations in Serial and ten in MPI2; counts are not multiplied into independent experiments by rank count.

Prepared ROOT execution commands, from the integrated test checkout and a freshly rebuilt authentic v12 installation. Set POPS_RECEPTION_ENV to ROOT's actual rebuilt environment and POPS_RECEPTION_EVIDENCE to a new retained output directory; do not select an old SDK by default.

```sh
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 \
  OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  "$POPS_RECEPTION_ENV/bin/python" docs/development/api_040/run_installed_checks.py \
  --output "$POPS_RECEPTION_EVIDENCE/amr12-independent-refusals-serial-FRESH" \
  --test tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py::test_native_amr12_state_carrier_refusals_and_exact_rollback

env -u PYTHONPATH POPS_REQUIRE_NATIVE_TESTS=1 \
  OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  "$POPS_RECEPTION_ENV/bin/python" docs/development/api_040/run_installed_mpi_checks.py \
  --output "$POPS_RECEPTION_EVIDENCE/amr12-independent-refusals-mpi2-FRESH" \
  --ranks 2 --dimension 2 --threads 1 --timeout 1800 \
  --test tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py::test_native_amr12_state_carrier_refusals_and_exact_rollback
```

The earlier independent docs-only worktree/commit `0f19e70` remains untouched. The separate capture/save-only commit `b4ced74` extends the existing public AMR fixture with a phase-live all-rank `sol61.amr.carrier-registry@1` JSON and `receipt.carrier_registry` absolute path/digest; ROOT must add the actual generated file to its owner-pinned archive. It retains native row order and actual rank order. This preparation contains no new backend proof.
