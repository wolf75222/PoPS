# Lossless typed integral unit authoring

Date: 30 September 2026. Base: `45cb04ea53745f24b5df51cd60454657d59d78fc`.
Exclusive worktree `PoPS-sol61-integral-units-authoring`. This Python correction
is separate from the native canonical-unit guard `7745cfe` and its independent
reception `7c65b951`, and from the MPI fixture correction `5358f693`.

Python JSON encoding can map two surrogate code units to the same image as a
single non-BMP codepoint. For example, the distinct strings `\ud800\udc00` and
U+10000 both encode as `"\ud800\udc00"`. Decoding merges the pair. A dimension
containing both names can produce duplicate decoded bases; the pair alone
changes its base name, and mixing it with U+E000 can change the decoded order.
Silently sorting or merging these images would alter the declared type.

`integral_units_bytes` now authenticates the roundtrip through
`PhysicalDimension.from_data(json.loads(encoded))` and requires equality with
the exact original PhysicalDimension. Duplicate or changed dimensions raise
ValueError with the precise message:

`integral units require a lossless canonical PhysicalDimension JSON roundtrip`

Typed `Program.integral_state` calls the helper before writing either
`_integral_states` or `_integral_units`. Failed declarations preserve both
containers, their contents and the serialized Program, and the same name can be
used by a subsequent valid declaration. The general PhysicalDimension
constructor, untyped `units=None`, canonical JSON spelling, unit bytes and
successful integral identities remain unchanged. No units are normalized or
converted. No C++ header, SDK, installed package or MAIN checkout is modified.

Independent source regressions cover three ambiguous paired-surrogate images
with atomicity and retry; dimensionless/true Unicode/non-BMP/60-bit rational
images with exact bytes and digest; untyped v1 declaration and typed-capture
refusal; a 2049-digit fraction with exact JSON roundtrip; and lone high/low
surrogates retaining their existing public Case validation refusal.

The large-fraction JSON test does not claim a successful full Program identity:
the historical canonical CBOR identity codec rejects integers outside signed
int64. That pre-existing, distinct limit is explicitly received by the test and
is unchanged here. Similarly, lone surrogates roundtrip through JSON, but public
Case validation already rejects them as `not valid Unicode` in canonical CBOR.
The new check closes lossless typed-integral encoding; it does not change the
general identity contract or Unicode policy of other authoring APIs.

Source-only command:

```sh
rtk proxy env -u PYTHONPATH PYTHONPATH=python \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
  tests/review/test_sol61_integral_units_authoring_roundtrip.py \
  tests/python/unit/codegen/test_integral_candidate_capture.py \
  tests/python/unit/codegen/test_integral_state_independent.py \
  tests/python/unit/codegen/test_integral_state_names.py \
  tests/python/unit/codegen/test_program_v8_contract.py
```

Result: **31 PASS**, including 11 new independent source tests. Ruff and diff
whitespace checks pass. No native System step, Kokkos, MPI, installed-package,
checkpoint or scientific feedback reception is claimed by this Python lot.
