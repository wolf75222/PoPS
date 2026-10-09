# Independent review: qualified parameter inspection

Reviewed the root's uncommitted patch against MAIN `005d287419e5d73bc0646620e74ac7da9eeac35e`.
Production file SHA-256 values at review:

* `python/pops/codegen/inspect_compiled.py`:
  `50f252d26b2ebe2d08e9e2b0e8a86085f43d81e840cc599cc59d624e1210229a`.
* `python/pops/codegen/_artifact_models.py`:
  `3ff0009bc53655494101bd5d8871e70602cdb699817289dfd3f712925646a4d7`.

Verdict: favorable within source/reporting scope. No blocking defect was found.
No production file was edited by this review.

The installed red receipt `outputs/qualified-parameters-installed-red-repaired.log`
contains two actual rejections of different `gain` defaults owned by distinct
blocks. The old merge compared those local names *before* entering
`build_parameter_arguments`. That function already disregarded local metadata
whenever the artifact carried a BindSchema. Removing that preliminary merge
therefore repairs a false rejection; it does not replace the bind authority.

`BindSchema` validates unique qualified identities and contiguous ordinals,
stores immutable slots, and returns detached declaration dictionaries. Reports
continue to derive kind, dtype, default, domain, unit, provenance and canonical
handle from these slots. Manifest classification still separates const/runtime
slots by qualified identity. Both single-layout and multilayout paths reach this
same function. Per-layout argument views retain the artifact-wide parameter table
(existing behavior); their State/provider partitions are separately projected.
The memory formula receives aggregate metadata but does not use the parameter
value objects; state counts and byte formulas are unchanged.

The changed historical runtime-planning test used a schema with **no parameter
slots** and injected synthetic local metadata. Its old rejection did not prove
parameter consistency at an executable boundary. Accepting an empty inspection
table in that fixture is consistent with the existing authority rule. Crucially,
inspection is not permission to install a missing native parameter: the existing
`_slot_for_block` checks still require exactly one matching runtime/bind-derived
slot. Independent tests explicitly retain rejection of a missing slot in both
block and Program installation routes, and of a same-name **const** slot where
the native model requires a runtime slot. No extra metadata-merge guard is needed
to preserve that refusal. Low-level handles without any schema retain their local
report and conflict refusal.

## Independent evidence

`tests/python/unit/codegen/test_qualified_parameter_inspection_review.py`:
**7 passed** against the reviewed MAIN Python sources. The tests use real
BindSchema construction from two authored block instances and source-only compiled
artifact fixtures. They cover:

* same local name with runtime/positive-real versus const/integer declarations,
  both block orders and single/multiple layouts;
* local metadata objects whose equality operation throws, proving no hidden merge;
* argument fields, const/runtime manifest partition, aggregate metadata and the
  actual memory formula's state-byte result;
* mutation of returned nested default/domain/handle dictionaries followed by
  unchanged schema, schema serialization roundtrip and fresh inspection;
* continued low-level fallback/refusal and native-install routing rejection for
  missing or wrong-kind slots, plus a positive qualified value route.

The command selected the existing Dim2 native variant only to obtain required
platform/precision facts for typed artifacts and the memory formula. It used
`env -u PYTHONPATH`, `POPS_NATIVE_DIM=2`, the installed `_native` directory via
`POPS_NATIVE_VARIANTS_ROOT`, and pytest `-o pythonpath=<MAIN>/python` with this
review test file in the isolated worktree. This is **source inspection evidence
with native platform facts**, not an installed-only product solve, a JIT run,
MPI qualification or rebuilt-package reception. No build or environment mutation
was performed. The central product/rebind reception remains necessary.

Minor documentation follow-up: `build_arguments`'s introductory parameter bullet
still refers to `model.params`; it should name the artifact BindSchema for public
artifacts, with the low-level fallback described separately.
