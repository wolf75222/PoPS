"""Selected cell-gradient / constitutive-face / divergence diffusion construction."""
from __future__ import annotations
from pops.descriptors import Descriptor
from pops.model.balance_analysis import diffusion_balance_supported


class Diffusion(Descriptor):
    category = "diffusion"
    native_id = "pops::runtime::program::PreparedDiffusion"
    ghost_depth = 1

    @property
    def formal_order(self):
        return 2 if self.transport is None else min(2,self.transport.formal_order)

    def __init__(self, *, flux, transport=None):
        from pops.physics.diffusion import DiffusiveFluxHandle

        if type(flux) is not DiffusiveFluxHandle:
            raise TypeError("Diffusion requires the exact constitutive diffusive flux declaration")
        self.flux = flux
        self.law = flux.law
        self.transport = transport
        self.validate()

    def validate(self):
        from pops._ir.expr import Const
        from pops.physics.diffusion import DiffusiveFluxLaw

        if self.transport is not None:
            from .spatial import FiniteVolume
            if type(self.transport) is not FiniteVolume:
                raise TypeError("combined diffusion transport must be an exact FiniteVolume selection")
            self.transport.validate()
            if self.transport.formal_order != 1 or self.transport.riemann.scheme not in {"rusanov", "scalar_upwind"}:
                raise ValueError("combined explicit diffusion requires a monotone first-order Rusanov or scalar-upwind transport realization")
            if not getattr(self.transport.flux, "is_default", False):
                raise ValueError("combined diffusion requires the exact default physical transport flux")
        law = self.law
        if type(law) is not DiffusiveFluxLaw or law.dimension not in (1, 2, 3):
            raise ValueError("native diffusion requires a Cartesian frame with one through three axes")
        from pops._ir.quantity import QuantityRef
        from pops._ir.expr import Var
        from pops._ir.visitors import _children
        for component, (variable, tensor) in enumerate(zip(law.variables, law.component_coefficients, strict=True)):
            # The physical declaration can express cross-gradients. This two-point
            # monotone realization requires each W_i to depend only on U_i; its
            # positive coefficients may depend on every component and field.
            pending = [variable]
            while pending:
                node = pending.pop()
                if isinstance(node, QuantityRef) and node.handle.kind == "state":
                    if node.handle != law.state or node.index != component:
                        raise ValueError("two-point monotone diffusion cannot realize cross-component gradient variables; select a coupled matrix realization")
                if len(law.variables) > 1 and isinstance(node, Var) and node.kind == "prim":
                    raise ValueError("multicomponent diffusion requires explicit gradient expressions to prove componentwise monotonicity")
                pending.extend(_children(node))
            for axis, row in enumerate(tensor):
                for column, value in enumerate(row):
                    if axis != column and not (isinstance(value, Const) and value.value == 0):
                        raise ValueError("two-point monotone diffusion requires diagonal spatial tensors; off-diagonal fluxes require a transverse-gradient realization")
                    if axis == column and isinstance(value, Const) and value.value <= 0:
                        raise ValueError("diffusion coefficients must be strictly positive")
        return True

    def validate_rate_contract(self, contract):
        if contract["state"] != self.law.state:
            raise ValueError("Diffusion must discretize its exact accumulated state")
        if self.transport is None:
            if contract.get("flux") not in (None, ()):
                raise ValueError("a combined transport/diffusion balance requires an explicit transport selection")
        else:
            self.transport.validate_rate_contract(contract)
        return True

    def validate_balance_view(self, view):
        if not diffusion_balance_supported(view):
            raise ValueError("Diffusion requires a diffusion/source physical balance")
        for occurrence in view.occurrences:
            if occurrence.kind == "diffusion":
                if occurrence.payload != self.flux or occurrence.coefficient <= 0:
                    raise ValueError("Diffusion requires positive uses of its exact constitutive flux")
        transport_rows = tuple(row for row in view.occurrences if row.kind == "flux")
        if self.transport is None:
            if transport_rows:
                raise ValueError("transport occurrence has no selected finite-volume construction")
        elif len(transport_rows) != 1 or transport_rows[0].payload != self.transport.flux or transport_rows[0].coefficient != -1:
            raise ValueError("combined diffusion requires exactly one -div of its selected transport flux")
        return self.validate()

    def resolve_references(self, resolver):
        result = object.__new__(type(self))
        result.flux = resolver(self.flux)
        result.law = self.law.resolve_references(resolver)
        result.transport = None if self.transport is None else self.transport.resolve_references(resolver)
        return result

    def to_data(self):
        return {"schema_version": 1, "method": "diffusion", "flux": self.flux.canonical_identity(),
                "law": self.law.to_data(), "gradient": "two_point_variable_difference",
                "face_coefficient": "arithmetic_interior_linear_boundary_extrapolation",
                "divergence": "oriented_face_difference", "formal_order": self.formal_order, "ghost_depth": 1,
                "transport": None if self.transport is None else self.transport.to_data(),
                "explicit_restriction": "sum_transport_frequencies_plus_diffusive_row_frequency<=1/dt"}

    def runtime_configuration(self):
        if self.transport is not None:
            return self.transport.runtime_configuration()
        from .state_storage import StateStorage
        return StateStorage().runtime_configuration()

    def runtime_spatial(self):
        if self.transport is not None:
            return self.transport.runtime_spatial()
        from pops.runtime._state_storage import StateStorageSpatial
        return StateStorageSpatial()


def explicit_diffusion_dt_bound(*, spacings, diffusivities, speeds=None, time_ratio=1):
    """Local Cartesian sufficient monotonicity bound, including a level's substep ratio.

    The returned duration is expressed in the parent clock. The selected native uniform update
    checks its measured conductances separately; this algebra does not authorize AMR execution.
    """
    import math
    if isinstance(time_ratio, bool) or not isinstance(time_ratio, int) or time_ratio < 1:
        raise ValueError("time_ratio must be a positive integer")
    h, a = tuple(spacings), tuple(diffusivities)
    c = (0.,) * len(h) if speeds is None else tuple(speeds)
    if not h or len(h) != len(a) or len(h) != len(c):
        raise ValueError("stability inputs must cover the exact spatial axes")
    if any(not math.isfinite(x) for x in (*h, *a, *c)) or any(x <= 0 for x in h) or any(x < 0 for x in a):
        raise ValueError("stability requires finite positive spacing and nonnegative diffusion")
    frequency = sum(abs(v)/dx + 2*k/(dx*dx) for dx,k,v in zip(h,a,c,strict=True))
    return math.inf if frequency == 0 else time_ratio/frequency
