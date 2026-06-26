"""pops.lib.diagnostics -- the diagnostic brick catalog (Spec 3).

Scalar reductions (integral / norm / mass / momentum / energy / ...) as macro
descriptors. The conservation-invariant descriptors are catalogued separately in
:mod:`pops.lib.diagnostics.invariants` (re-exported as ``lib.invariants``).

DEFER (no catalogued source): a ``reductions.py`` impl file -- lib does no numeric
Python (see the PR-D blueprint DEFER list).
"""
from types import SimpleNamespace

from ..descriptors import BrickDescriptor
from .invariants import invariants


def _diag(_dname, **o):
    return BrickDescriptor(_dname, "macro", category="diagnostic", scheme=_dname,
                           options=o or None)


diagnostics = SimpleNamespace(
    integral=lambda expr=None, **o: _diag("integral", expr=expr, **o),
    norm=lambda kind="l2", **o: _diag("norm", kind=kind, **o),
    mass=lambda **o: _diag("mass", **o),
    momentum=lambda **o: _diag("momentum", **o),
    energy=lambda **o: _diag("energy", **o),
    invariant_error=lambda name=None, **o: _diag("invariant_error", name=name, **o),
    residual=lambda **o: _diag("residual", **o),
)

__all__ = ["diagnostics", "invariants"]
