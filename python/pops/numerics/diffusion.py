"""Selected cell-gradient / constitutive-face / divergence diffusion construction."""
from __future__ import annotations
from pops.descriptors import Descriptor
from pops._ir.expr import Const
from pops.physics.diffusion import DiffusiveFluxHandle, DiffusiveFluxLaw


def diffusion_balance_supported(view):
    if view is None or not view.accumulation.is_identity or not any(
            row.kind == "diffusion" for row in view.occurrences):
        return False
    return all(row.kind in {"diffusion", "source"} for row in view.occurrences)


class Diffusion(Descriptor):
    category = "diffusion"
    native_id = "pops::runtime::program::PreparedDiffusion"
    formal_order = 2
    ghost_depth = 1

    def __init__(self, *, flux):
        if type(flux) is not DiffusiveFluxHandle:
            raise TypeError("Diffusion requires the exact constitutive diffusive flux declaration")
        self.flux = flux
        self.law = flux.law
        self.validate()

    def validate(self):
        law = self.law
        if type(law) is not DiffusiveFluxLaw or law.dimension not in (1, 2):
            raise ValueError("native diffusion currently selects Cartesian Dim1/Dim2 only")
        for axis, row in enumerate(law.coefficients):
            for column, value in enumerate(row):
                if axis != column and not (isinstance(value, Const) and value.value == 0):
                    raise ValueError("native diffusion selects positive diagonal tensors; off-diagonal unavailable")
                if axis == column and isinstance(value, Const) and value.value <= 0:
                    raise ValueError("diffusion coefficients must be strictly positive")
        return True

    def validate_rate_contract(self, contract):
        if contract["state"] != self.law.state or contract.get("flux") not in (None, ()):
            raise ValueError("Diffusion must discretize its exact scalar state without a transport flux")
        return True

    def validate_balance_view(self, view):
        if not diffusion_balance_supported(view):
            raise ValueError("Diffusion requires a diffusion/source physical balance")
        for occurrence in view.occurrences:
            if occurrence.kind == "diffusion":
                if occurrence.payload != self.flux or occurrence.coefficient <= 0:
                    raise ValueError("Diffusion requires positive uses of its exact constitutive flux")
        return self.validate()

    def resolve_references(self, resolver):
        result = object.__new__(type(self))
        result.flux = resolver(self.flux)
        result.law = self.law.resolve_references(resolver)
        return result

    def to_data(self):
        return {"schema_version": 1, "method": "diffusion", "flux": self.flux.canonical_identity(),
                "law": self.law.to_data(), "gradient": "two_point_variable_difference",
                "face_coefficient": "arithmetic_interior_linear_boundary_extrapolation",
                "divergence": "oriented_face_difference", "formal_order": 2, "ghost_depth": 1}

    def runtime_configuration(self):
        from .state_storage import StateStorage
        return StateStorage().runtime_configuration()

    def runtime_spatial(self):
        from pops.runtime._state_storage import StateStorageSpatial
        return StateStorageSpatial()
