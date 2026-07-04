"""pops._native_facts -- the declared native-core facts, as a dependency-free leaf.

These constants state what the compiled engine IS (2D mesh core per ADC-294, refinement
ratio 2, double precision, world communicator); they are not runtime lifecycle state.
They live in this leaf so EVERY layer -- including ``pops.time``, which is fenced from
``pops.runtime*`` imports (test_no_legacy_runtime_routes) -- can validate against them
without touching the runtime layer. ``pops.runtime_environment`` re-exports them and
stays the public spelling; import from there unless you are inside the fence.
"""
from __future__ import annotations

NATIVE_DIMENSION = 2
NATIVE_AMR_REFINEMENT_RATIO = 2
NATIVE_PRECISION = "double"
NATIVE_REAL_BYTES = 8
NATIVE_COMMUNICATOR = "MPI_COMM_WORLD"
