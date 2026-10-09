"""Declared raw-moment paths and first-order path-conservative finite volumes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from pops.model import Handle
from .spatial import FiniteVolume, _brick_data, _resolved_brick
from .symbolic_path import SymbolicPath
from .normalized_polynomial_path import NormalizedPolynomialPath, PathArithmeticComposition
from .coordinated_face import CoordinatedFace, FaceBalance

if TYPE_CHECKING:
    from pops.model.expression_language import Expr
    from pops.physics.nonconservative import NonconservativeProductHandle


def path_balance_supported(view: Any) -> bool:
    """Exact equation shape supported by the single-flux/product native route."""
    if view is None or not view.accumulation.is_identity:
        return False
    rows = view.occurrences
    return (len(rows) == 2
            and sum(row.kind == "flux" for row in rows) == 1
            and sum(row.kind == "nonconservative" for row in rows) == 1
            and all(row.coefficient == -1 for row in rows))




class PathConservativeFiniteVolume(FiniteVolume):
    """Shared conservative flux plus distinct signed nonconservative side terms.

    The first executable route uses complete raw states and no spatial slopes.
    Higher-order reconstruction requires an additional within-cell path integral.
    """

    category = "path_conservative_finite_volume"

    def __init__(self, *, flux: Any, path: Any, variables: Any,
                 reconstruction: Any, riemann: Any, zero_measure_faces: Any = ()) -> None:
        if not isinstance(path, (PathArithmeticComposition, SymbolicPath)):
            raise TypeError("path-conservative transport requires a typed numerical path")
        super().__init__(flux=flux, variables=variables, reconstruction=reconstruction, riemann=riemann)
        if (isinstance(flux, tuple) or flux.owner_path != path.product.owner_path
                or self.variables.options.get("state") != path.product.state):
            raise ValueError("path, physical flux and reconstruction must own the same exact state")
        if (self.formal_order != 1 or self.reconstruction.name != "firstorder"
                or self.variables.scheme != "conservative"
                or self.riemann.native_id != "pops::RusanovFlux"):
            raise ValueError("raw path-conservative transport currently requires FirstOrder and Rusanov")
        path.validate_flux(flux)
        self.path = path
        from pops.domain.rectangle import DomainBoundary
        faces = tuple(zero_measure_faces)
        expected = () if not faces else path.frame.boundaries.all
        if (any(type(face) is not DomainBoundary or face not in expected for face in faces)
                or len(set(faces)) != len(faces)):
            raise ValueError("zero_measure_faces must name distinct exact boundaries of the path frame")
        # This is an explicit physical degeneracy contract, independent of a NoFlux policy.
        # Bounded physical moments at these zero-measure faces give zero F and path terms.
        self.zero_measure_faces = faces

    def options(self) -> dict[str, Any]:
        return {**super().options(), "path": self.path, "zero_measure_faces": self.zero_measure_faces}

    def validate_rate_contract(self, contract: Any) -> bool:
        if (contract.get("flux") != self.flux
                or contract.get("state") != self.variables.options.get("state")
                or contract.get("nonconservative_products") != (self.path.product,)
                or contract.get("sources")):
            raise ValueError("path-conservative method must cover the exact flux/product/state balance")
        return True

    def validate_balance_view(self, view: Any) -> bool:
        if not path_balance_supported(view):
            raise ValueError("path-conservative transport requires exactly -div(F) - B grad(U)")
        if any((row.kind == "flux" and row.payload != self.flux)
               or (row.kind == "nonconservative" and row.payload != self.path.product)
               for row in view.occurrences):
            raise ValueError("path-conservative construction does not match the retained balance")
        return True

    def resolve_references(self, resolver: Any) -> PathConservativeFiniteVolume:
        # Resolution preserves the captured law and path; it does not rebuild the
        # physical expressions using newly rebound, potentially different symbols.
        result = object.__new__(type(self))
        for name in ("variables", "reconstruction", "riemann"):
            setattr(result, name, _resolved_brick(getattr(self, name), resolver))
        result.flux = resolver(self.flux)
        result.sampling = tuple(resolver(item) for item in self.sampling)
        result.path = self.path.resolve_references(resolver)
        result.positivity_floor = None
        result.zero_measure_faces = self.zero_measure_faces
        return result

    def to_data(self) -> dict[str, Any]:
        flux = self.flux
        if (not isinstance(flux, Handle) or not flux.is_resolved
                or not self.path.product.is_resolved):
            raise ValueError("PathConservativeFiniteVolume.to_data requires resolved physical handles")
        return {"schema_version": 1, "method": "path_conservative_finite_volume",
                "flux": flux.canonical_identity(), "path": self.path.to_data(),
                "sampling": [item.canonical_identity() for item in self.sampling],
                "variables": _brick_data(self.variables),
                "reconstruction": _brick_data(self.reconstruction),
                "riemann": _brick_data(self.riemann),
                "formal_order": self.formal_order, "ghost_depth": self.ghost_depth,
                "positivity_floor": None,
                "zero_measure_faces": [face.to_dict() for face in self.zero_measure_faces],
                "nonconservative_interfaces": "canonical_fine_subface_side_contributions"}


class CoordinatedFiniteVolume(PathConservativeFiniteVolume):
    """Complete flux/product realization with an atomic authored face tuple.

    This first-order method samples the two adjacent stored cell averages.
    It uses the existing path residual/flux-register authority, but a distinct
    numerical interface contract, body identity and native policy.
    """
    category = "coordinated_finite_volume"

    def __init__(self, *, face):
        from pops.descriptors import BrickDescriptor
        from . import reconstruction, variables
        if type(face) is not CoordinatedFace:
            raise TypeError("CoordinatedFiniteVolume requires an exact CoordinatedFace")
        numerical = BrickDescriptor("coordinated_face", "generated", category="riemann",
            native_id="pops::CoordinatedFaceFlux", scheme="coordinated_face",
            options={"interface_contract": 1},
            requirements={"capabilities": ("physical_flux", "provider_pack", "stability_bound")},
            capabilities={"shared_flux_two_signed_sources": True, "atomic_publication": True})
        FiniteVolume.__init__(self, flux=face.flux,
            variables=variables.Conservative(face.product.state),
            reconstruction=reconstruction.FirstOrder(), riemann=numerical)
        self.path = face
        self.zero_measure_faces = ()

    @property
    def face(self):
        return self.path

    def to_data(self):
        data = super().to_data()
        data.update(method="coordinated_finite_volume", interface_contract=1)
        return data

    def native_identity(self):
        from pops.identity.digest import make_identity
        from pops.identity.semantic import semantic_value
        return make_identity("numerics.coordinated-face.v1", semantic_value({
            "method": self.to_data(), "physical_law": self.path.product.law.to_data(),
        }, where="complete numerical path operator")).token

    def runtime_spatial(self):
        from copy import copy
        from pops.runtime._bricks_scheme import Spatial
        numerical = copy(self.riemann)
        object.__setattr__(numerical, "options", {
            "interface_contract": 1, "operator_identity": self.native_identity()})
        return Spatial(limiter=self.reconstruction, flux=numerical,
                       recon=self.variables, positivity_floor=None)


__all__ = ["NormalizedPolynomialPath", "SymbolicPath", "PathConservativeFiniteVolume",
           "CoordinatedFace", "FaceBalance", "CoordinatedFiniteVolume"]
