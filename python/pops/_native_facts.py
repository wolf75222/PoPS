"""Dimension-neutral native facts shared by pure authoring layers.

The active spatial dimension is deliberately absent from this module: it belongs to the
compiled ``pops._pops`` artifact and is queried only at the native boundary.  Pure Python
authoring therefore never freezes a process-wide 2D assumption into a Program.
"""
from __future__ import annotations

NATIVE_SUPPORTED_DIMENSIONS = (1, 2, 3)
NATIVE_AMR_REFINEMENT_RATIO = 2
NATIVE_PRECISION = "double"
NATIVE_REAL_BYTES = 8
NATIVE_COMMUNICATOR = "MPI_COMM_WORLD"
NATIVE_MAX_RUNTIME_PARAMS = 32
