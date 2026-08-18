"""AMR leaf-only oracle norms.

Plan §7.2: volume-weighted errors vs the in-memory oracle on leaf cells
only. Covered/parent cells are never counted. The caller supplies the same
boolean coverage mask used for conservative balances.

L1/L2/L∞ reductions stay in ``reference_errors``; this module only validates
and applies ``leaf_mask``.
"""
from __future__ import annotations

import numpy as np

from verification.pops_verify.reference_errors import ReferenceErrors, reference_errors


def leaf_reference_errors(u, u_exact, volumes, leaf_mask) -> ReferenceErrors:
    """Return §7.1 norms of ``U - U_exact`` restricted to AMR leaf cells."""
    field = np.asarray(u)
    oracle = np.asarray(u_exact)
    cell_volumes = np.asarray(volumes)
    mask = np.asarray(leaf_mask)
    if mask.dtype != np.bool_:
        raise ValueError("leaf_mask must be boolean")
    try:
        field, oracle, cell_volumes, mask = np.broadcast_arrays(
            field, oracle, cell_volumes, mask
        )
    except ValueError as exc:
        raise ValueError("shape mismatch") from exc
    if not np.any(mask):
        raise ValueError("empty leaf set")
    return reference_errors(field[mask], oracle[mask], cell_volumes[mask])
