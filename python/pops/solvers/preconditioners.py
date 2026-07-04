"""pops.solvers.preconditioners -- the preconditioner brick catalog (Spec 5 sec.5.7).

Only the geometric-multigrid preconditioner has a native type (``pops::GeometricMG``);
identity / jacobi / block-jacobi have none yet (the polar solver has its own PolarPrecond
enum), so they are catalogued as PLANNED descriptors. :func:`User` surfaces a loaded external
preconditioner brick. This is the ONE public home of the catalog formerly parked under
``pops.lib.solvers.preconditioners`` (that re-export shim is removed; no second public path).

ADC-502 RATIFIES ``pops.solvers.preconditioners`` as that single home: a preconditioner configures
a solver, so it lives with the solver descriptors (not under ``pops.linalg``); no move, no shim. The
invariant is pinned by ``tests/python/architecture/test_spec5_public_api.py`` (``pops.linalg`` has
NO ``preconditioners`` submodule).
"""
from __future__ import annotations

from types import SimpleNamespace

from pops.descriptors import _external_descriptor, _native, _planned

preconditioners = SimpleNamespace(
    Identity=lambda: _planned("identity", "identity", category="preconditioner"),
    Jacobi=lambda: _planned("jacobi", "jacobi", category="preconditioner"),
    BlockJacobi=lambda: _planned("block_jacobi", "block_jacobi",
                                 category="preconditioner"),
    GeometricMG=lambda **o: _native("geometric_mg", "pops::GeometricMG", "geometric_mg",
                                    category="preconditioner", **o),
    User=lambda brick_id: _external_descriptor(brick_id, expect_category="preconditioner"),
)

__all__ = ["preconditioners"]
