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
