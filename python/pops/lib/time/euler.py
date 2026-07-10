"""pops.lib.time.euler -- Forward Euler time-stepping scheme.

Builds a pops.time.Program step for the classic first-order explicit method.
The backward_euler name is not defined here (the implicit BDF1 path is accessed
via bdf(..., order=1, linear_source=...)); only forward_euler lives here.
"""
from __future__ import annotations

from typing import Any

from ._helpers import _DEFAULT_SOURCES, _commit, _stage_rhs, _time_state, program_macro


@program_macro
def forward_euler(P: Any, block: Any, state: Any = None, *,
                  sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """Forward Euler: U^{n+1} = U + dt * R(U).

    ``block, state`` are a typed ``BlockHandle`` and model state declaration. A Program-owned
    ``TimeState`` may be passed as ``block`` with ``state=None``. ``sources`` contains typed source
    ``OperatorHandle`` values or RHS term descriptors; it never contains names."""
    temporal = _time_state(P, block, state)
    U = temporal.n
    R = _stage_rhs(P, U, sources, flux)
    _commit(P, temporal, P.linear_combine("fe_step", U + P.dt * R))
