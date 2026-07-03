"""pops.moments.space -- the velocity-space and moment-state handles (inert).

Two inert structural handles that NAME the generic construction surface of the
generator (:mod:`pops.moments.model_builder`) without computing anything:

* :class:`VelocitySpace` -- the 2D Cartesian velocity axis (``vx`` / ``vy``) the
  moments integrate over. It documents the domain of :func:`moment_indices`
  (``(p, q)`` are the ``vx^p vy^q`` powers); it holds no numeric data.
* :class:`MomentState` -- a snapshot of the conservative moment vector at a given
  order (its names, via :func:`moment_names`). It records the state layout a
  hierarchy transports; it computes nothing.

Neither is a :class:`pops.descriptors.Descriptor`: they choose no math route, so
they stay lightweight handles (Spec 5 sec.6: only route-choosers are descriptors).
"""


class VelocitySpace:
    """The 2D Cartesian velocity axis a moment hierarchy integrates over (inert handle).

    ``dim`` is the velocity dimension (2 for the Cartesian ``vx`` / ``vy`` plane the
    generator supports) and ``names`` labels the axes. It documents the domain of
    :func:`pops.moments.moment_indices`: a moment ``M_pq`` is ``E[vx^p vy^q]``, so
    ``(p, q)`` index the powers over exactly these axes. It records the axis; it
    performs no arithmetic.
    """

    def __init__(self, dim=2, names=("vx", "vy")):
        if dim != 2:
            raise ValueError("VelocitySpace: only the 2D Cartesian velocity axis is "
                             "supported (got dim=%r)" % (dim,))
        if len(names) != dim:
            raise ValueError("VelocitySpace: names must have %d entries (got %r)"
                             % (dim, names))
        self.dim = int(dim)
        self.names = tuple(names)

    def __repr__(self):
        return "VelocitySpace(dim=%d)" % (self.dim,)


class MomentState:
    """A snapshot of the conservative moment vector at ``order`` (inert handle).

    Records the moment-state layout a hierarchy transports: the order and the
    canonical variable names (``M{p}{q}``, via :func:`pops.moments.moment_names`).
    It holds NO numeric data -- it is the structural handle a caller inspects to see
    which conservative components a model of a given order carries.
    """

    def __init__(self, order):
        if order < 2:
            raise ValueError("MomentState: order >= 2 required (got %r)" % (order,))
        self.order = int(order)

    def names(self):
        """The conservative moment names of this state (``M{p}{q}`` at its order)."""
        from .model_builder import moment_names
        return moment_names(self.order)

    def __len__(self):
        return len(self.names())

    def __repr__(self):
        return "MomentState(order=%d, vars=%d)" % (self.order, len(self))


__all__ = ["VelocitySpace", "MomentState"]
