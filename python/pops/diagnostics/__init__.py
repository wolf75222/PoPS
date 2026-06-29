"""pops.diagnostics -- the diagnostic brick catalog (Spec 3 / Spec 5).

Scalar reductions (integral / norm / mass / momentum / energy / ...) as macro
descriptors. The conservation-invariant descriptors are catalogued separately in
:mod:`pops.diagnostics.invariants`.

Spec 5 (sec.4 / sec.5.13) homes diagnostics in the top-level ``pops.diagnostics``
package (formerly ``pops.lib.diagnostics``). The reduction macros stay inert
descriptors; nothing here computes in Python.

Spec 5 sec.5.13 / 14.2.7 names norm diagnostics with a TYPED object (a
:class:`~pops.diagnostics.measures.Norm` / :class:`~pops.diagnostics.measures.Integral` /
:class:`~pops.diagnostics.measures.MinMax` / :class:`~pops.diagnostics.measures.ConservationCheck`
descriptor) rather than ``diagnostics.norm(kind="l2")``. Those typed measures live in
:mod:`pops.diagnostics.measures` and are the SINGLE SOURCE of the native reduction scheme
labels. The reductions with no typed counterpart yet (``mass`` / ``momentum`` / ``energy`` /
``invariant_error`` / ``residual``) keep their own self-named macro descriptor.
"""
from types import SimpleNamespace

from pops.descriptors import BrickDescriptor
from .invariants import invariants
from .measures import ConservationCheck, Integral, MinMax, Norm


def _diag(_dname, *, scheme=None, **o):
    """An inert ``diagnostic`` macro descriptor; @p scheme defaults to the reduction name."""
    return BrickDescriptor(_dname, "macro", category="diagnostic",
                           scheme=scheme if scheme is not None else _dname,
                           options=o or None)


diagnostics = SimpleNamespace(
    # ``integral`` borrows its scheme label from the typed measure class, so the native reduction
    # label lives in ONE place (pops.diagnostics.measures), not two. ``norm`` has no factory:
    # use Norm(L1/L2/LInf) so the norm kind is typed.
    integral=lambda expr=None, **o: _diag("integral", scheme=Integral.scheme, expr=expr, **o),
    mass=lambda **o: _diag("mass", **o),
    momentum=lambda **o: _diag("momentum", **o),
    energy=lambda **o: _diag("energy", **o),
    invariant_error=lambda name=None, **o: _diag("invariant_error", name=name, **o),
    residual=lambda **o: _diag("residual", **o),
)

# Spec 5: expose only macro reductions that do not use a string selector.
integral = diagnostics.integral
mass = diagnostics.mass
momentum = diagnostics.momentum
energy = diagnostics.energy
invariant_error = diagnostics.invariant_error
residual = diagnostics.residual

__all__ = ["diagnostics", "invariants", "integral", "mass", "momentum",
           "energy", "invariant_error", "residual",
           # Spec 5 typed measure descriptors (pops.diagnostics.measures).
           "Norm", "Integral", "MinMax", "ConservationCheck"]
