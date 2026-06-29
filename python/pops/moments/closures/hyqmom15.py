"""pops.moments.closures.hyqmom15 -- the provided HyQMOM15 (order-4) closure.

The HyQMOM15 system is the 15-variable order-4 2D moment hierarchy. Its provided
closure is the Levermore / Gaussian closure of order 4 (:func:`gaussian_closure(4)`),
which is the closure the adc_cases HyQMOM15 reference validates against.

It CHOOSES the closure variant (the only physics of a moment model), so it is a typed
:class:`pops.descriptors.Descriptor` (Spec 5 sec.6): it declares its options / capabilities
and is inspectable. The bare ``gaussian_closure`` factory stays a plain callable (a builder
that does the arithmetic, not a route chooser); this descriptor wraps it.
"""
from pops.descriptors import Descriptor

from .gaussian import gaussian_closure

_HYQMOM15_ORDER = 4


class HyQMOM15Closure(Descriptor):
    """The HyQMOM15 (order-4) moment closure (route-choosing descriptor).

    ``variant='levermore'`` delegates to :func:`gaussian_closure(4)` -- the standardized
    order-5 moments of a Gaussian. No unimplemented custom route is exposed.

    A :class:`HyQMOM15Closure` instance is itself a closure callable: ``self(S)`` returns the
    order-5 standardized moments the generator consumes.
    """

    category = "closure"

    _VARIANTS = ("levermore",)

    def __init__(self, variant="levermore"):
        if variant not in HyQMOM15Closure._VARIANTS:
            raise ValueError("HyQMOM15Closure variant %r must be one of %s"
                             % (variant, ", ".join(HyQMOM15Closure._VARIANTS)))
        self.variant = variant
        self.order = _HYQMOM15_ORDER
        self._closure = gaussian_closure(_HYQMOM15_ORDER)

    def options(self):
        return {"variant": self.variant, "order": self.order}

    def capabilities(self):
        return {"provides": "order_%d_standardized_moments" % self.order}

    def __call__(self, S):  # noqa: N803  (S mirrors the engine variable name)
        return self._closure(S)

    def __repr__(self):
        return "HyQMOM15Closure(variant=%r)" % (self.variant,)


__all__ = ["HyQMOM15Closure"]
