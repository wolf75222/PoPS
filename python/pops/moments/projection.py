"""pops.moments.projection -- the realizability-projection descriptor (Spec 5 sec.6).

Maps the moment-model realizability floors onto the engine's ``robust`` / ``eps_m00`` /
``eps_cov`` knobs. The engine applies a smooth floor ``max(x, eps)`` on M00 (division)
and C20/C02 (sqrt) when ``robust=True``.

It CHOOSES the realizability-floor strategy (smooth guard vs bare guard-free), so it is a
typed :class:`pops.descriptors.Descriptor` (Spec 5 sec.6): it declares its options /
capabilities and is inspectable. Inert -- it records the floor parameters; the floor
arithmetic is generated and runs in C++.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


class RealizabilityProjection(Descriptor):
    """The realizability floor a moment hierarchy applies (route-choosing descriptor).

    ``(eps_m00, eps_cov, robust)`` map to the engine's smooth ``max(x, eps)`` floors on
    M00 and the covariance C20/C02. With ``robust=False`` the bare guard-free path runs
    (faithful to the references; may NaN on a degenerate state). It records the choice;
    the floor is generated and lowers to C++.
    """

    category = "realizability"

    def __init__(self, eps_m00: Any = 1e-12, eps_cov: Any = 1e-12, robust: bool = True) -> None:
        self.eps_m00 = float(eps_m00)
        self.eps_cov = float(eps_cov)
        self.robust = bool(robust)

    @classmethod
    def none(cls) -> Any:
        """The bare, guard-free projection (``robust=False``)."""
        return cls(robust=False)

    def options(self) -> dict:
        return {"eps_m00": self.eps_m00, "eps_cov": self.eps_cov, "robust": self.robust}

    def capabilities(self) -> Any:
        return CapabilitySet({"guard_level": "smooth" if self.robust else "bare"})

    def __repr__(self) -> str:
        return ("RealizabilityProjection(eps_m00=%g, eps_cov=%g, robust=%r)"
                % (self.eps_m00, self.eps_cov, self.robust))


#: The issue vocabulary spells the projection ``MomentProjection``; it is the SAME
#: descriptor (an identity alias, so ``isinstance`` / descriptor identity are unchanged).
MomentProjection = RealizabilityProjection


class RealizableSet(Descriptor):
    """The realizable moment cone at ``order`` (an inert capability descriptor).

    Describes WHICH states a moment vector of this order may take (the realizable set the
    :class:`RealizabilityProjection` floors a state back into): a positive density
    (``M00 > 0``), a positive-semidefinite covariance (``C20`` / ``C02 >= 0``) and the Schur
    (Hankel) conditions on the higher moments. It CHOOSES no algorithm and computes nothing --
    it is the typed, inspectable record of the cone's constraints, so tooling can report what
    "realizable" means for an order without touching the runtime.
    """

    category = "realizability_set"

    def __init__(self, order):
        if order < 2:
            raise ValueError("RealizableSet: order >= 2 required (got %r)" % (order,))
        self.order = int(order)

    def options(self):
        return {"order": self.order}

    def capabilities(self):
        return CapabilitySet({"constraints": "m00_positive,cov_psd,schur"})

    def __repr__(self):
        return "RealizableSet(order=%d)" % (self.order,)


__all__ = ["RealizabilityProjection", "MomentProjection", "RealizableSet"]
