# Actual compiler capture independent review

Reviewed `65a6786375ecb6cade513301e37c7015230b1638`: test-only interception
for the existing compiler driver, original command and original compiler
call. Input is retained before invocation; successful records authenticate
TU, compiler executable, header-signature argument and output bytes.
`require_binary` proves content correspondence through staging rename; it
does not prove file ownership or reject another copy of identical binary
bytes. It refuses absence/ambiguity of an observed compile and poisoned TU.
No source regeneration substitutes for a cache hit.

Five Source/host tests pass, including two independent tests of actual tiny
C++ compilation, retained-TU poison, ambiguous outputs, same original
compiler exception object and restoration of the driver after failure.
This is no Native, MPI, compiler graph, SDK or scientific qualification.
Compiler failure retains its TU but intentionally has no successful record.
One attempt per fresh capture directory is required by these fixtures.

The separate fixture integration uses profile `raw-native-fixture@2` and
`parametric-refusal@2`. The shared collective output directory is created
before compile. Only the publisher wraps the genuine `pops.compile` call;
MPI peers load the authenticated publication through the existing helper
without claiming to compile a TU. Every `artifact.blocks` model binary must
match a unique observed output by hash, after artifact verification. The
record names the actual block and published model path, and recursive
receipt pins include all retained TU/command/compiler/model-binary records.
No source is regenerated from a model descriptor. A publisher cache hit
with no observed model TU is deliberately refused; fresh ROOT per-test
caches are needed. Program source remains the separately retained genuine
Program output; this does not add a new claim that its compiler route was
observed. No per-rank model-compilation record is minted for cache loads.
