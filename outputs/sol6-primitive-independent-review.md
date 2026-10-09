# Independent review: joint principal primitive coordinates

Source state reviewed: `f81be74` plus Astra's uncommitted Primitive patch in
`PoPS-principal-group` on 2026-09-29. This review did not edit production files or
build/install native artifacts.

The authored map retains the complete selected state group and its explicitly
authored inverse; the carrier lowers both through row-specific parameter sets.
The generated `recover` and `make_conservative` methods initialize a failed
`StateConversion`, refuse nonfinite inputs/intermediates, and set `Success` only
after conversion and domain checks. Native face reconstruction consumes those
statuses and the group residual rejects an invalid face or frequency before
marking its candidate evaluated. This is a source-path review; the six installed
Primitive tests were not executed here, so inversion, rollback and inactive
`where` branches remain runtime qualification obligations.

Three new independent source tests in
`tests/python/unit/codegen/test_principal_primitive_independent.py` pass (3/3):
changing only the inverse or only the admissible domain changes both Module
content and manifest hashes; schema-9 payloads cannot be rehydrated by the
schema-10 parser, while schema-10 roundtrips exactly. A reversed two-row joint
map with seven components emits the full checked conversion and 3+4 storage
contract; this is source coverage, not a seven-component native run. These
checks support cache/SDK
invalidation at the model-manifest boundary. It does not authenticate an
already-installed native SDK; that requires the central rebuild/doctor receipt.

One source risk was found by Astra during the parallel review: a joint recovery
reads every component at each reached stencil offset, so differing per-row halo
depths require a common maximum halo. Astra owns this correction. I found no
separate, reproducible defect in the conversion/status/manifest paths in this
bounded review. The existing Primitive installed tests cover (1,1), (2,3),
(1,4), reversal, parameter rebind and a domain refusal, but are still marked
not executed here.
