"""pops.lib.models.moments.gaussian -- a provided Gaussian-closure moment model.

A minimal provided moment model: a generic-order Cartesian-velocity hierarchy closed by
the Gaussian / Levermore closure, with optional transport-only or Vlasov-Poisson coupling.
A PURE composition of the Spec-4 moment facade -- no ``custom.py``.
"""
from __future__ import annotations

from typing import Any

from pops.moments.hierarchy import CartesianVelocityMoments
from pops.moments.closures import gaussian_closure


class Gaussian:
    """The provided Gaussian-closure moment models."""

    @staticmethod
    def transport(order: Any = 2, *, robust: Any = True, exact_speeds: Any = True,
                  roe: Any = False) -> Any:
        """A transport-only Gaussian-closure moment model (no coupling, no source).

        @p order: the moment order (order=2 -> 6 vars, order=4 -> 15). Returns a
        ``physics.PdeModel`` ready to compile.
        """
        return (CartesianVelocityMoments(order, closure=gaussian_closure(order),
                                         robust=robust, exact_speeds=exact_speeds, roe=roe)
                .add_transport()
                .build(name="gaussian"))

    @staticmethod
    def vlasov_poisson(order: Any = 2, *, robust: Any = True, exact_speeds: Any = True,
                       roe: Any = False, q_over_m: Any = "q_over_m", eps: Any = 1.0) -> Any:
        """A Vlasov-Poisson Gaussian-closure moment model (electric source, no magnetic term).

        @p order: the moment order. @p q_over_m: the charge/mass param name. @p eps: the
        Poisson coupling scale. Returns a ``physics.PdeModel`` ready to compile.
        """
        return (CartesianVelocityMoments(order, closure=gaussian_closure(order),
                                         robust=robust, exact_speeds=exact_speeds, roe=roe)
                .add_transport()
                .add_poisson_coupling(eps=eps)
                .add_vlasov_electric_source("grad_x", "grad_y", q_over_m)
                .build(name="gaussian"))


__all__ = ["Gaussian"]
