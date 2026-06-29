"""pops.lib.models.moments.hyqmom15 -- the provided HyQMOM15 moment model.

HyQMOM15 is the 15-variable order-4 2D moment hierarchy. The provided builder composes
the Spec-4 moment facade into the Vlasov-Poisson-magnetic model used by the diocotron /
column reference cases. It is a PURE composition of the facade -- no ``custom.py``.
"""
from pops.moments import (CartesianVelocityMoments, VlasovElectricSource,
                          MagneticRotationSource)
from pops.moments.closures import HyQMOM15Closure

_HYQMOM15_ORDER = 4


class HyQMOM15:
    """The provided HyQMOM15 (order-4) moment models."""

    @staticmethod
    def vlasov_poisson_magnetic(order=_HYQMOM15_ORDER, *, closure=None, robust=True,
                                exact_speeds=False, roe=False,
                                q_over_m="q_over_m", omega_c="omega_c",
                                eps=1.0):
        """The Vlasov-Poisson moment model with a magnetic (Lorentz) source.

        A pure composition of the :func:`CartesianVelocityMoments` facade: transport flux,
        a Poisson coupling (electric field gradient aux ``grad_x`` / ``grad_y``), the Vlasov
        electric source over those gradients, and the magnetic Lorentz source. The default
        closure is :class:`HyQMOM15Closure` (the Levermore order-4 closure); pass ``closure=``
        to override it.

        @p order: the moment order (HyQMOM15 is order 4 -- 15 variables); other orders build a
           hierarchy of the same family.
        @p q_over_m / @p omega_c: the param names for the charge/mass ratio and the cyclotron
           frequency. @p eps: the Poisson coupling scale on the charge density.
        Returns a public ``pops.physics.Model`` ready for ``to_module()`` / ``compile_problem``.
        """
        cl = closure if closure is not None else HyQMOM15Closure()
        return (CartesianVelocityMoments(order, closure=cl, robust=robust,
                                         exact_speeds=exact_speeds, roe=roe)
                .add_transport()
                .add_poisson_coupling(eps=eps)
                .add_source(VlasovElectricSource(
                    electric_field=("grad_x", "grad_y"),
                    charge_over_mass=q_over_m))
                .add_source(MagneticRotationSource(omega_c=omega_c))
                .build(name="hyqmom15"))


__all__ = ["HyQMOM15"]
