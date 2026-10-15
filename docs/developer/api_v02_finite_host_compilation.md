# Finite host compilation boundaries

Finite map kernels remain the unchanged, interned C++/Kokkos kernels generated
from the Python graph. Their host staging functions now have explicit noninline
boundaries. This keeps validation, allocation, dispatch and publication in the
same lexical function and execution order, while preventing their expansion
into one giant installer/step function. Compiler optimization flags are unchanged.

Repeated Domain checks and the five typed allocation operations are ordinary
PoPS host helpers: device view, host mirror, copied mirror, vector storage and
any-valued cell. They preserve live-instance checks, exact capacity/family,
replicated size/version, exception classification and collective order. They
cache no authority and never deduplicate checks across executions. Remaining
structural allocation forms retain the existing generic allocation path.

For the complete T04 Schur/monolithic Python graph, a source-only before/after
comparison retained the exact IR hash and all 25 interned device kernels. Useful
C++ shrank from 8,203,330 to 3,656,654 bytes (83,255 to 38,407 lines), with 265
explicit host map boundaries. Source reduction and helper syntax checks do not
prove a compilation speedup or scientific execution; the native T04 receiver and
MPI failure tests remain the qualification gates, with unchanged tolerances.
