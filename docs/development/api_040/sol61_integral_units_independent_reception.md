# Independent canonical integral units reception

Pinned candidate `7745cfecc4d4da92377fc9d71f0db8b05ff928d6`, exact parent
`bb87174d82d95422c3086eea7b0c28cf8390fb15`. Private review checkout only;
MAIN, native SDK/environment, JIT and native builds untouched.

The actual C++ reader receives **76 independent stdlib-generated cases**:
57 accepted, 19 refused. Cases include BMP/nonBMP codepoint ordering, lowercase
Unicode escapes, control short escapes, quote/backslash/slash, escaped DEL,
individual lone surrogate escapes, 40 independently reduced random rationals,
a 2049-digit coprime pair, and a giant unreduced pair. Refusals cover envelopes,
key order/whitespace/extra keys, zero/unreduced/negative-denominator/leading-zero
rationals, duplicate/unsorted/empty bases, uppercase hex, printable ASCII escaped
with Unicode, escaped slash, literal DEL, and raw nonASCII. The driver imports
only stdlib, not PoPS or an author oracle, and fails unexpected results.

The true PreparedIntegralCapture, PreparedResourceCache, AcceptedExchangeLedger
and serial ExecutionLane pass **12 additional independent host checks**. Four
malformations have matching redigested typed v2 identities: not-json, reordered
JSON, unreduced rational, leading zero. Both prepare and consume refuse with
the canonical-unit diagnostic before calling attempt/point/ledger callbacks.
Ledger bytes remain exact; the already valid capture remains usable after each
refusal. The same test compiled against an archived exact parent include tree
fails its first refusal assertion (exit SIGABRT): the old guard accepts not-json.
The shim exposes private capture entry points after dependencies are included;
it does not instantiate a substitute System or qualify native consumers.

The independent public System/provider counterprobe from review186 is included
as `tests/review/sol61_units_public_native.inc`. Full true ProgramRuntime TU +
that fixture passes Dim1/Kokkos/MPI syntax (one existing gtest char8_t warning).
Actual public native execution, rank-local failures, Kokkos POD consumption and
rollback/retry remain reserved for central reconstruction/reception. The syntax
result does not replace that pending proof.

Source inspection confirms the guard is called before digest construction.
Both prepare and consume call require_units_ inside their existing try, then
catch and collectively_rethrow_exception before exact contract agreement/POD
return. The production guard diff is only include + call; ownership, point,
attempt, exact current scalar/ledger, and publication remain unchanged.

Affected authentic Python source suite passes **13 tests**:
tests/python/unit/codegen/test_integral_candidate_capture.py and
tests/python/unit/codegen/test_program_v8_contract.py. This tests source
authoring/emission/versioning, not the installed native package.

## Unicode roundtrip boundary

Correct accepted ordering is U+E000 then U+10000. A lone escaped high or low
surrogate is accepted by this JSON reader, consistent with Python json.dumps.
That does **not** prove such a base is accepted by a full Program: genuine
validate(case) refuses surrogate-containing names in canonical CBOR as invalid
Unicode before native emission.

There is a second ambiguity: Python strings consisting of two distinct
surrogate codeunits U+D800/U+DC00 and one true U+10000 codepoint are distinct in
PhysicalDimension but serialize to identical JSON name bytes. The native reader
correctly merges valid surrogate pairs and refuses duplicate decoded bases.
Two-codeunit names sorted before U+E000 also lose their authored ordering after
JSON decoding. These two vectors are treated as **invalid v2 roundtrip images**,
not silently normalized or announced as supported names. They are not a new
demonstrated public Program native failure: full CBOR validation already refuses.

Recommended targeted authoring follow-up: integral_units_bytes must require
PhysicalDimension.from_data(json.loads(bytes)) == original and refuse ambiguity.
Perform it before publishing a typed integral declaration when possible. Keep
PhysicalDimension legacy behavior unchanged, and retain strict native duplicate
refusal. This review does not modify production or declare that Python follow-up
already implemented in candidate7745.

## Reproduction

clang++ -std=c++20 -O0 -Iinclude tests/review/sol61_units_reader.cpp
-o /tmp/pops-sol61-units-reader, then run sol61_units_cases.py with that path.
Compile sol61_units_capture.cpp similarly and execute it. For the parent,
git archive bb87174 include into an isolated output tree and compile the same
capture probe against only that include directory. No new canonical header is
used in the parent probe.

Python commands use env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 and the existing
read-only pops-api040 interpreter. The Python source suite is loaded explicitly
from this private checkout/python. Logs for the parent refusal are preserved in
outputs/sol61-units-independent-review-20260930 outside the checkout. No M14 or
new scientific native qualification is asserted.
