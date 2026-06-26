"""pops.lib.moments.ordering -- the moment-variable ordering descriptor (inert).

Documents the canonical ordering the generator uses (``moment_indices``): q outer,
p inner, increasing. There is no user choice in Spec 4; this is a read-only label.
"""


class MomentOrdering:
    """The canonical moment-variable ordering: ``Q_OUTER_P_INNER``.

    Matches :func:`pops.lib.moments.moment_indices` (q outer increasing, then p inner
    increasing). It records nothing the engine can vary; it documents the layout so a
    :class:`MomentHierarchy` can report it. Inert -- it computes nothing.
    """

    Q_OUTER_P_INNER = "q_outer_p_inner"

    def __init__(self, layout=Q_OUTER_P_INNER):
        if layout != MomentOrdering.Q_OUTER_P_INNER:
            raise ValueError("MomentOrdering: only %r is supported in Spec 4 (got %r)"
                             % (MomentOrdering.Q_OUTER_P_INNER, layout))
        self.layout = layout

    def __repr__(self):
        return "MomentOrdering(%r)" % (self.layout,)


__all__ = ["MomentOrdering"]
