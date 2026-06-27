"""pops.moments.projection -- the realizability-projection descriptor (inert).

Maps the moment-model realizability floors onto the engine's ``robust`` / ``eps_m00`` /
``eps_cov`` knobs. The engine applies a smooth floor ``max(x, eps)`` on M00 (division)
and C20/C02 (sqrt) when ``robust=True``. This descriptor records the floor parameters;
the floor arithmetic is generated and runs in C++.
"""


class RealizabilityProjection:
    """The realizability floor a moment hierarchy applies (inert descriptor).

    ``(eps_m00, eps_cov, robust)`` map to the engine's smooth ``max(x, eps)`` floors on
    M00 and the covariance C20/C02. With ``robust=False`` the bare guard-free path runs
    (faithful to the references; may NaN on a degenerate state). It records the choice;
    the floor is generated and lowers to C++.
    """

    def __init__(self, eps_m00=1e-12, eps_cov=1e-12, robust=True):
        self.eps_m00 = float(eps_m00)
        self.eps_cov = float(eps_cov)
        self.robust = bool(robust)

    @classmethod
    def none(cls):
        """The bare, guard-free projection (``robust=False``)."""
        return cls(robust=False)

    def __repr__(self):
        return ("RealizabilityProjection(eps_m00=%g, eps_cov=%g, robust=%r)"
                % (self.eps_m00, self.eps_cov, self.robust))


__all__ = ["RealizabilityProjection"]
