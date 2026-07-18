"""Exact native bounds owned by the builtin prepared Krylov providers."""

from pops.identity import CPP_INT_MAX


# The native GMRES exceptional path flattens two robust 67-double payloads per Arnoldi
# projection into one MPI_Allreduce.  This mirrors max_krylov_batched_basis_extent exactly.
PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH = 67
PREPARED_GMRES_MAX_RESTART = (
    (CPP_INT_MAX - 1) // (2 * PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH) - 1
)


__all__ = [
    "PREPARED_GMRES_MAX_RESTART",
    "PREPARED_GMRES_ROBUST_DOT_PAYLOAD_WIDTH",
]
