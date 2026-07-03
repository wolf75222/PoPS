"""pops.moments.transport -- the moment-transport (flux order-shift) handle (inert).

An inert structural handle documenting the flux rule the generator emits: the 2D
moment flux is the pure order shift ``Fx[M_pq] = M_{p+1,q}``, ``Fy[M_pq] = M_{p,q+1}``
(:func:`pops.moments.model_builder.build_moment_model`'s flux loop). The order ``N+1``
moments the shift needs are supplied by the closure; the rule itself has no free
parameter, so this is a handle, not a route-choosing :class:`pops.descriptors.Descriptor`.
"""


class MomentTransport:
    """The moment-transport flux rule at ``order`` (inert handle).

    Documents the order-shift the generator emits for the 2D moment flux:
    ``Fx[M_pq] = M_{p+1,q}`` and ``Fy[M_pq] = M_{p,q+1}``. The ``+1`` order moments are
    provided by the closure and folded into the flux AST; this handle records the rule,
    it computes nothing.
    """

    def __init__(self, order):
        if order < 2:
            raise ValueError("MomentTransport: order >= 2 required (got %r)" % (order,))
        self.order = int(order)

    def flux_shift(self):
        """The order-shift rule as ``{"x": (1, 0), "y": (0, 1)}`` (the ``(dp, dq)`` per axis)."""
        return {"x": (1, 0), "y": (0, 1)}

    def __repr__(self):
        return "MomentTransport(order=%d)" % (self.order,)


__all__ = ["MomentTransport"]
