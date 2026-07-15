"""Private exact per-space AMR transfer registry and provider resolution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pops.identity import Identity, make_identity
from .._layout_plan_contracts import LayoutHandle
from ._contracts import canonical_handle
from .hierarchy import CanonicalOptions, NestingRequirementSource


def _name(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("%s must be a non-empty string" % where)
    return value.strip()


def _generic_handle(value: Any, *, where: str, kind: str | None = None) -> Any:
    projection = getattr(value, "canonical_identity", None)
    data = projection() if callable(projection) else None
    actual = data.get("kind") if isinstance(data, Mapping) else None
    if not isinstance(actual, str):
        raise TypeError("%s must be an owner-qualified Handle protocol" % where)
    return canonical_handle(value, where=where, kinds=kind or actual)


@dataclass(frozen=True, slots=True)
class BuiltinTransferAxis:
    """Open built-in axis value; extensions use the same canonical-identity protocol."""

    category: str
    name: str
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if self.category not in {"space", "centering", "representation", "storage"}:
            raise ValueError("unsupported BuiltinTransferAxis.category")
        object.__setattr__(self, "name", _name(self.name, where="BuiltinTransferAxis.name"))

    @property
    def qualified_id(self) -> str:
        return "pops.amr.%s.v1::%s" % (self.category, self.name)

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "qualified_id": self.qualified_id,
            "category": self.category,
            "name": self.name,
            "authority": "pops.builtin",
        }


CELL_SPACE = BuiltinTransferAxis("space", "cell")
FACE_SPACE = BuiltinTransferAxis("space", "face")
NODE_SPACE = BuiltinTransferAxis("space", "node")
FIELD_SPACE = BuiltinTransferAxis("space", "field")
CACHE_SPACE = BuiltinTransferAxis("space", "cache")

CELL_CENTERED = BuiltinTransferAxis("centering", "cell")
FACE_CENTERED = BuiltinTransferAxis("centering", "face")
FACE_X_CENTERED = BuiltinTransferAxis("centering", "face_x")
FACE_Y_CENTERED = BuiltinTransferAxis("centering", "face_y")
NODE_CENTERED = BuiltinTransferAxis("centering", "node")
CONSERVATIVE_REPRESENTATION = BuiltinTransferAxis("representation", "conservative")
PRIMITIVE_REPRESENTATION = BuiltinTransferAxis("representation", "primitive")
DENSE_STORAGE = BuiltinTransferAxis("storage", "dense")
SPARSE_STORAGE = BuiltinTransferAxis("storage", "sparse")


def _axis_data(value: Any, *, category: str, where: str) -> dict[str, Any]:
    if isinstance(value, BuiltinTransferAxis):
        if value.category != category:
            raise TypeError("%s requires an AMR %s identity" % (where, category))
        return value.canonical_identity()
    projection = getattr(value, "canonical_identity", None)
    data = projection() if callable(projection) else None
    if not isinstance(data, Mapping):
        raise TypeError(
            "%s requires a BuiltinTransferAxis or owner-qualified canonical identity protocol"
            % where
        )
    qualified_id = data.get("qualified_id")
    if not isinstance(qualified_id, str) or not qualified_id:
        raise TypeError("%s extension requires a non-empty qualified_id" % where)
    if getattr(value, "qualified_id", qualified_id) != qualified_id:
        raise ValueError("%s extension identity does not authenticate qualified_id" % where)
    if data.get("owner_path") is None and data.get("owner") is None:
        raise TypeError("%s extension identity must be owner-qualified" % where)
    return dict(data)


@dataclass(frozen=True, slots=True)
class TransferOperation:
    name: str
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, where="TransferOperation.name"))

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name}


PROLONGATION = TransferOperation("prolongation")
RESTRICTION = TransferOperation("restriction")
COARSE_FINE_FILL = TransferOperation("coarse_fine_fill")
TEMPORAL_INTERPOLATION = TransferOperation("temporal_interpolation")


@dataclass(frozen=True, slots=True)
class TransferKey:
    space: Any
    centering: Any
    representation: Any
    storage: Any
    operation: TransferOperation
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _axis_data(self.space, category="space", where="TransferKey.space")
        _axis_data(self.centering, category="centering", where="TransferKey.centering")
        _axis_data(
            self.representation,
            category="representation",
            where="TransferKey.representation",
        )
        _axis_data(self.storage, category="storage", where="TransferKey.storage")
        if type(self.operation) is not TransferOperation:
            raise TypeError("TransferKey.operation must be a TransferOperation")

    @property
    def identity(self) -> Identity:
        return make_identity("amr-transfer-key", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "space": _axis_data(self.space, category="space", where="TransferKey.space"),
            "centering": _axis_data(
                self.centering, category="centering", where="TransferKey.centering"
            ),
            "representation": _axis_data(
                self.representation,
                category="representation",
                where="TransferKey.representation",
            ),
            "storage": _axis_data(
                self.storage, category="storage", where="TransferKey.storage"
            ),
            "operation": self.operation.to_data(),
        }


@dataclass(frozen=True, slots=True)
class TransferCapabilities:
    order: int
    ghost_depth: tuple[int, ...]
    dimensions: tuple[int, ...] = (1, 2, 3)
    conservative: bool = False
    temporal: bool = False
    refinement_ratios: tuple[int, ...] = (2,)
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("TransferCapabilities.order must be an integer >= 1")
        ghost = tuple(self.ghost_depth)
        if not ghost or len(ghost) > 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ghost
        ):
            raise ValueError("TransferCapabilities.ghost_depth must contain 1-3 integers >= 0")
        dimensions = tuple(self.dimensions)
        if not dimensions or len(set(dimensions)) != len(dimensions) or any(
            value not in (1, 2, 3) for value in dimensions
        ):
            raise ValueError("TransferCapabilities.dimensions must be unique values from {1,2,3}")
        if type(self.conservative) is not bool or type(self.temporal) is not bool:
            raise TypeError("TransferCapabilities flags must be exact bool values")
        ratios = tuple(self.refinement_ratios)
        if not ratios or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in ratios
        ):
            raise ValueError("TransferCapabilities.refinement_ratios must be integers >= 2")
        object.__setattr__(self, "ghost_depth", ghost)
        object.__setattr__(self, "dimensions", tuple(sorted(dimensions)))
        object.__setattr__(self, "refinement_ratios", tuple(sorted(set(ratios))))

    def supports(self, requirements: tuple[TransferRequirement, ...]) -> bool:
        return all(
            requirement.accuracy.dimension in self.dimensions
            and len(self.ghost_depth) in (1, requirement.accuracy.dimension)
            and self.order >= requirement.accuracy.order
            and all(
                available >= needed
                for available, needed in zip(
                    self.ghost_depth * requirement.accuracy.dimension
                    if len(self.ghost_depth) == 1 else self.ghost_depth,
                    requirement.accuracy.ghost_depth * requirement.accuracy.dimension
                    if len(requirement.accuracy.ghost_depth) == 1
                    else requirement.accuracy.ghost_depth,
                    strict=True,
                )
            )
            and all(
                ratio in self.refinement_ratios
                for ratio in requirement.accuracy.refinement_ratio
            )
            and (not requirement.accuracy.conservative or self.conservative)
            and (
                not requirement.accuracy.temporal or self.temporal
            )
            for requirement in requirements
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "ghost_depth": list(self.ghost_depth),
            "dimensions": list(self.dimensions),
            "conservative": self.conservative,
            "temporal": self.temporal,
            "refinement_ratios": list(self.refinement_ratios),
        }


@dataclass(frozen=True, slots=True)
class TransferProviderRoute:
    key: TransferKey
    capabilities: TransferCapabilities
    options: CanonicalOptions = CanonicalOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.key) is not TransferKey:
            raise TypeError("TransferProviderRoute.key must be TransferKey")
        if type(self.capabilities) is not TransferCapabilities:
            raise TypeError("TransferProviderRoute.capabilities must be TransferCapabilities")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("TransferProviderRoute.options must be CanonicalOptions")

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key.to_data(),
            "capabilities": self.capabilities.to_data(),
            "options": self.options.to_data(),
        }


@dataclass(frozen=True, slots=True)
class TransferProvider:
    provider: Any
    routes: tuple[TransferProviderRoute, ...]
    options: CanonicalOptions = CanonicalOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _generic_handle(
            self.provider, where="TransferProvider.provider", kind="amr_transfer_provider"
        )
        routes = tuple(self.routes)
        if not routes or any(type(route) is not TransferProviderRoute for route in routes):
            raise TypeError("TransferProvider.routes must contain TransferProviderRoute values")
        route_ids = [route.key.identity.token for route in routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("TransferProvider contains a duplicate exact transfer key")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("TransferProvider.options must be CanonicalOptions")
        object.__setattr__(self, "routes", routes)

    @property
    def qualified_id(self) -> str:
        return self.provider.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "qualified_id": self.qualified_id,
            "routes": [route.to_data() for route in self.routes],
            "options": self.options.to_data(),
        }


PHYSICAL = "physical"
DERIVED_FIELD = "derived_field"
CACHE = "cache"


class NativeAMRActionKind(Enum):
    """Closed native effects which extension actions may select."""

    APPLY_TRANSFER_PROVIDER = "apply_transfer_provider"
    RECOMPUTE = "recompute"
    INVALIDATE_THEN_REBUILD = "invalidate_then_rebuild"


class NativeAMRMaterializationKind(Enum):
    """Typed projection of the three AMR materialization families."""

    PHYSICAL = PHYSICAL
    DERIVED_FIELD = DERIVED_FIELD
    CACHE = CACHE


@dataclass(frozen=True, slots=True)
class MaterializationProvider:
    """Executable provider selected for derived-field or cache materialization."""

    provider: Any
    materialization: str
    options: CanonicalOptions = CanonicalOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _generic_handle(self.provider, where="MaterializationProvider.provider")
        if self.materialization not in {DERIVED_FIELD, CACHE}:
            raise ValueError("MaterializationProvider supports only derived fields or caches")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("MaterializationProvider.options must be CanonicalOptions")

    @property
    def qualified_id(self) -> str:
        return self.provider.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "provider": self.provider.canonical_identity(),
            "qualified_id": self.qualified_id,
            "materialization": self.materialization,
            "options": self.options.to_data(),
        }


def _native_capability_ids(
    materialization: NativeAMRMaterializationKind,
    operation: TransferOperation,
) -> tuple[str, str]:
    if type(materialization) is not NativeAMRMaterializationKind:
        raise TypeError("native AMR materialization must be an exact typed kind")
    if type(operation) is not TransferOperation:
        raise TypeError("native AMR operation must be an exact TransferOperation")
    return (
        "pops.amr.materialization.%s.v1" % materialization.value,
        "pops.amr.operation.%s.v1" % operation.name,
    )


@dataclass(frozen=True, slots=True)
class NativeAMRMaterializationCapabilities:
    """Immutable capability evidence returned by an AMR action extension."""

    capability_ids: tuple[str, ...]
    transfer: TransferCapabilities | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.capability_ids) is not tuple \
                or not self.capability_ids \
                or any(not isinstance(item, str) or not item for item in self.capability_ids):
            raise TypeError(
                "NativeAMRMaterializationCapabilities.capability_ids must be a non-empty "
                "exact tuple of strings"
            )
        if len(self.capability_ids) != len(set(self.capability_ids)):
            raise ValueError("native AMR capability ids must be unique")
        if self.capability_ids != tuple(sorted(self.capability_ids)):
            raise ValueError("native AMR capability ids must be in canonical sorted order")
        if self.transfer is not None and type(self.transfer) is not TransferCapabilities:
            raise TypeError("native AMR transfer capabilities must be exact TransferCapabilities")

    @classmethod
    def for_materialization(
        cls,
        materialization: NativeAMRMaterializationKind,
        operation: TransferOperation,
        *,
        transfer: TransferCapabilities | None = None,
    ) -> NativeAMRMaterializationCapabilities:
        return cls(tuple(sorted(_native_capability_ids(materialization, operation))), transfer)

    def to_data(self) -> dict[str, Any]:
        return {
            "capability_ids": list(self.capability_ids),
            "transfer": self.transfer.to_data() if self.transfer is not None else None,
        }


@dataclass(frozen=True, slots=True)
class NativeAMRMaterializationDescriptor:
    """Versioned closed IR returned by an open AMR action protocol.

    Extension actions implement ``native_amr_materialization(key=...)`` and return this
    exact data-only value.  The action class is never inspected by validation or runtime
    preparation.
    """

    schema_version: int
    action: NativeAMRActionKind
    materialization: NativeAMRMaterializationKind
    operation: TransferOperation
    transfer_key_identity: Identity
    provider_qualified_id: str
    provider_identity: CanonicalOptions
    options: CanonicalOptions
    native_route: str
    capabilities: NativeAMRMaterializationCapabilities
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("native AMR materialization schema_version must be exactly 1")
        if type(self.action) is not NativeAMRActionKind:
            raise TypeError("native AMR materialization action must be an exact typed kind")
        if type(self.materialization) is not NativeAMRMaterializationKind:
            raise TypeError("native AMR materialization family must be an exact typed kind")
        if type(self.operation) is not TransferOperation:
            raise TypeError("native AMR materialization operation must be a TransferOperation")
        if type(self.transfer_key_identity) is not Identity \
                or self.transfer_key_identity.domain != "amr-transfer-key":
            raise TypeError(
                "native AMR materialization must authenticate one exact transfer-key identity"
            )
        if not isinstance(self.provider_qualified_id, str) or not self.provider_qualified_id:
            raise TypeError("native AMR provider_qualified_id must be non-empty")
        if type(self.provider_identity) is not CanonicalOptions:
            raise TypeError("native AMR provider_identity must be immutable canonical data")
        provider_data = self.provider_identity.to_data()
        if provider_data.get("qualified_id") != self.provider_qualified_id:
            raise ValueError("native AMR provider identity does not authenticate qualified_id")
        if type(self.options) is not CanonicalOptions:
            raise TypeError("native AMR options must be immutable canonical data")
        if not isinstance(self.native_route, str) or not self.native_route:
            raise TypeError("native AMR native_route must be non-empty")
        if self.options.to_data().get("native_route") != self.native_route:
            raise ValueError("native AMR options do not authenticate native_route")
        if type(self.capabilities) is not NativeAMRMaterializationCapabilities:
            raise TypeError("native AMR materialization requires exact capability evidence")

        expected_action = {
            NativeAMRMaterializationKind.PHYSICAL:
                NativeAMRActionKind.APPLY_TRANSFER_PROVIDER,
            NativeAMRMaterializationKind.DERIVED_FIELD: NativeAMRActionKind.RECOMPUTE,
            NativeAMRMaterializationKind.CACHE:
                NativeAMRActionKind.INVALIDATE_THEN_REBUILD,
        }[self.materialization]
        if self.action is not expected_action:
            raise ValueError("native AMR action is incompatible with its materialization family")
        required = set(_native_capability_ids(self.materialization, self.operation))
        missing = sorted(required - set(self.capabilities.capability_ids))
        if missing:
            raise ValueError("native AMR materialization is missing capabilities: %s" % missing)
        if self.materialization is NativeAMRMaterializationKind.PHYSICAL:
            if self.capabilities.transfer is None:
                raise ValueError("physical AMR materialization requires transfer capabilities")
        elif self.capabilities.transfer is not None:
            raise ValueError(
                "derived-field/cache AMR materialization cannot claim transfer capabilities"
            )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "descriptor_type": "pops.amr.native_materialization",
            "action": self.action.value,
            "materialization": self.materialization.value,
            "operation": self.operation.to_data(),
            "transfer_key_identity": self.transfer_key_identity.token,
            "provider": self.provider_identity.to_data(),
            "route": {"options": self.options.to_data()},
            "options": self.options.to_data(),
            "native_route": self.native_route,
            "capabilities": self.capabilities.to_data(),
        }


def prepare_native_amr_materialization(
    action: Any,
    *,
    key: TransferKey,
    where: str,
) -> NativeAMRMaterializationDescriptor:
    """Authenticate an open action through one deterministic exact-IR protocol."""

    if type(key) is not TransferKey:
        raise TypeError("%s requires an exact TransferKey" % where)
    protocol = getattr(action, "native_amr_materialization", None)
    if not callable(protocol):
        raise TypeError(
            "%s action must implement native_amr_materialization(key=...)" % where
        )
    first = protocol(key=key)
    second = protocol(key=key)
    if type(first) is not NativeAMRMaterializationDescriptor \
            or type(second) is not NativeAMRMaterializationDescriptor:
        raise TypeError(
            "%s action protocol must return an exact NativeAMRMaterializationDescriptor"
            % where
        )
    if first.to_data() != second.to_data():
        raise ValueError("%s action protocol is non-deterministic" % where)
    if first.transfer_key_identity != key.identity or first.operation != key.operation:
        raise ValueError("%s action descriptor authenticates another transfer key" % where)
    return first


@dataclass(frozen=True, slots=True)
class AccuracyRequirement:
    order: int
    ghost_depth: tuple[int, ...]
    dimension: int
    refinement_ratio: tuple[int, ...]
    conservative: bool = False
    temporal: bool = False
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("AccuracyRequirement.order must be an integer >= 1")
        if self.dimension not in (1, 2, 3):
            raise ValueError("AccuracyRequirement.dimension must be 1, 2, or 3")
        ghost = tuple(self.ghost_depth)
        ratio = tuple(self.refinement_ratio)
        if len(ghost) not in (1, self.dimension) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ghost
        ):
            raise ValueError("AccuracyRequirement.ghost_depth is incompatible with dimension")
        if len(ratio) not in (1, self.dimension) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in ratio
        ):
            raise ValueError("AccuracyRequirement.refinement_ratio is incompatible with dimension")
        if type(self.conservative) is not bool or type(self.temporal) is not bool:
            raise TypeError("AccuracyRequirement flags must be exact bool values")
        object.__setattr__(self, "ghost_depth", ghost)
        object.__setattr__(self, "refinement_ratio", ratio)

    def to_data(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "ghost_depth": list(self.ghost_depth),
            "dimension": self.dimension,
            "refinement_ratio": list(self.refinement_ratio),
            "conservative": self.conservative,
            "temporal": self.temporal,
        }


@dataclass(frozen=True, slots=True)
class TransferRequirement:
    subject: Any
    layout: LayoutHandle
    key: TransferKey
    materialization: str
    accuracy: AccuracyRequirement
    materializer: MaterializationProvider | None = None
    provider: TransferProvider | None = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _generic_handle(self.subject, where="TransferRequirement.subject")
        if not isinstance(self.layout, LayoutHandle):
            raise TypeError("TransferRequirement.layout must be a LayoutHandle")
        if type(self.key) is not TransferKey:
            raise TypeError("TransferRequirement.key must be TransferKey")
        if self.materialization not in {PHYSICAL, DERIVED_FIELD, CACHE}:
            raise ValueError("unsupported transfer materialization %r" % self.materialization)
        if type(self.accuracy) is not AccuracyRequirement:
            raise TypeError("TransferRequirement.accuracy must be an AccuracyRequirement")
        if self.materialization == PHYSICAL:
            if self.materializer is not None:
                raise ValueError("physical transfer requirements cannot carry a materializer")
            if self.provider is not None and type(self.provider) is not TransferProvider:
                raise TypeError("TransferRequirement.provider must be a TransferProvider or None")
        elif type(self.materializer) is not MaterializationProvider \
                or self.materializer.materialization != self.materialization:
            raise ValueError(
                "derived-field/cache requirements require an exact matching materializer"
            )
        elif self.provider is not None:
            raise ValueError("derived-field/cache requirements select their materializer directly")
        space_id = _axis_data(self.key.space, category="space", where="TransferRequirement.space")[
            "qualified_id"
        ]
        if self.materialization == DERIVED_FIELD and space_id != FIELD_SPACE.qualified_id:
            raise ValueError("derived fields must use FIELD_SPACE")
        if self.materialization == CACHE and space_id != CACHE_SPACE.qualified_id:
            raise ValueError("caches must use CACHE_SPACE")
        if self.materialization == PHYSICAL and space_id in {
            FIELD_SPACE.qualified_id,
            CACHE_SPACE.qualified_id,
        }:
            raise ValueError("field/cache values cannot request silent physical transfer")
        centering_id = _axis_data(
            self.key.centering, category="centering", where="TransferRequirement.centering"
        )["qualified_id"]
        if space_id == FACE_SPACE.qualified_id and self.accuracy.dimension > 1 \
                and centering_id == FACE_CENTERED.qualified_id:
            raise ValueError(
                "multi-dimensional face transfer requires an oriented face_x/face_y centering"
            )

    @property
    def identity(self) -> Identity:
        return make_identity("amr-transfer-requirement", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "subject": self.subject.canonical_identity(),
            "layout": self.layout.canonical_identity(),
            "key": self.key.to_data(),
            "materialization": self.materialization,
            "accuracy": self.accuracy.to_data(),
            "materializer": (
                self.materializer.canonical_identity() if self.materializer is not None else None
            ),
            "provider": self.provider.canonical_identity() if self.provider is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ApplyTransferProvider:
    provider: TransferProvider
    route: TransferProviderRoute
    capabilities: TransferCapabilities
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.provider) is not TransferProvider:
            raise TypeError("ApplyTransferProvider.provider must be TransferProvider")
        if type(self.route) is not TransferProviderRoute:
            raise TypeError("ApplyTransferProvider.route must be TransferProviderRoute")
        if self.route not in self.provider.routes:
            raise ValueError("ApplyTransferProvider.route must belong to provider")
        if self.capabilities != self.route.capabilities:
            raise ValueError("ApplyTransferProvider.capabilities must match route")

    def to_data(self) -> dict[str, Any]:
        return {
            "action": "apply_provider",
            "provider": self.provider.canonical_identity(),
            "route": self.route.to_data(),
            "derived_capabilities": self.capabilities.to_data(),
        }

    def native_amr_materialization(
        self, *, key: TransferKey,
    ) -> NativeAMRMaterializationDescriptor:
        if type(key) is not TransferKey or key != self.route.key:
            raise ValueError("ApplyTransferProvider received another transfer key")
        route_options = self.route.options
        return NativeAMRMaterializationDescriptor(
            schema_version=1,
            action=NativeAMRActionKind.APPLY_TRANSFER_PROVIDER,
            materialization=NativeAMRMaterializationKind.PHYSICAL,
            operation=key.operation,
            transfer_key_identity=key.identity,
            provider_qualified_id=self.provider.qualified_id,
            provider_identity=CanonicalOptions(self.provider.canonical_identity()),
            options=route_options,
            native_route=route_options.to_data().get("native_route"),
            capabilities=NativeAMRMaterializationCapabilities.for_materialization(
                NativeAMRMaterializationKind.PHYSICAL,
                key.operation,
                transfer=self.capabilities,
            ),
        )


@dataclass(frozen=True, slots=True)
class Recompute:
    provider: MaterializationProvider
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.provider) is not MaterializationProvider \
                or self.provider.materialization != DERIVED_FIELD:
            raise TypeError("Recompute requires a derived-field MaterializationProvider")

    def to_data(self) -> dict[str, Any]:
        return {"action": "recompute", "provider": self.provider.canonical_identity()}

    def native_amr_materialization(
        self, *, key: TransferKey,
    ) -> NativeAMRMaterializationDescriptor:
        options = self.provider.options
        return NativeAMRMaterializationDescriptor(
            schema_version=1,
            action=NativeAMRActionKind.RECOMPUTE,
            materialization=NativeAMRMaterializationKind.DERIVED_FIELD,
            operation=key.operation,
            transfer_key_identity=key.identity,
            provider_qualified_id=self.provider.qualified_id,
            provider_identity=CanonicalOptions(self.provider.canonical_identity()),
            options=options,
            native_route=options.to_data().get("native_route"),
            capabilities=NativeAMRMaterializationCapabilities.for_materialization(
                NativeAMRMaterializationKind.DERIVED_FIELD,
                key.operation,
            ),
        )


@dataclass(frozen=True, slots=True)
class InvalidateThenRebuild:
    provider: MaterializationProvider
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.provider) is not MaterializationProvider \
                or self.provider.materialization != CACHE:
            raise TypeError("InvalidateThenRebuild requires a cache MaterializationProvider")

    def to_data(self) -> dict[str, Any]:
        return {
            "action": "invalidate_then_rebuild",
            "provider": self.provider.canonical_identity(),
        }

    def native_amr_materialization(
        self, *, key: TransferKey,
    ) -> NativeAMRMaterializationDescriptor:
        options = self.provider.options
        return NativeAMRMaterializationDescriptor(
            schema_version=1,
            action=NativeAMRActionKind.INVALIDATE_THEN_REBUILD,
            materialization=NativeAMRMaterializationKind.CACHE,
            operation=key.operation,
            transfer_key_identity=key.identity,
            provider_qualified_id=self.provider.qualified_id,
            provider_identity=CanonicalOptions(self.provider.canonical_identity()),
            options=options,
            native_route=options.to_data().get("native_route"),
            capabilities=NativeAMRMaterializationCapabilities.for_materialization(
                NativeAMRMaterializationKind.CACHE,
                key.operation,
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedTransfer:
    key: TransferKey
    requirements: tuple[TransferRequirement, ...]
    action: Any
    _native_materialization: NativeAMRMaterializationDescriptor = field(
        init=False, repr=False, compare=False
    )
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.key) is not TransferKey:
            raise TypeError("ResolvedTransfer.key must be TransferKey")
        requirements = tuple(self.requirements)
        if not requirements or any(type(row) is not TransferRequirement for row in requirements):
            raise TypeError("ResolvedTransfer.requirements must contain requirements")
        if any(row.key != self.key for row in requirements):
            raise ValueError("ResolvedTransfer requirements must share the exact key")
        ids = [row.identity.token for row in requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("ResolvedTransfer contains duplicate requirements")
        native = prepare_native_amr_materialization(
            self.action,
            key=self.key,
            where="ResolvedTransfer",
        )
        materializations = {row.materialization for row in requirements}
        if materializations != {native.materialization.value}:
            raise ValueError(
                "ResolvedTransfer action descriptor disagrees with requirement materialization"
            )
        object.__setattr__(
            self, "requirements", tuple(sorted(requirements, key=lambda row: row.identity.token))
        )
        object.__setattr__(self, "_native_materialization", native)

    @property
    def native_materialization(self) -> NativeAMRMaterializationDescriptor:
        return self._native_materialization

    @property
    def identity(self) -> Identity:
        """Exact identity of this resolved route, including subjects and provider action."""
        return make_identity("amr-resolved-transfer", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "key": self.key.to_data(),
            "requirements": [requirement.to_data() for requirement in self.requirements],
            "action": self.native_materialization.to_data(),
        }


@dataclass(frozen=True, slots=True)
class ResolvedAMRTransfer:
    """The one immutable authority for all intra-hierarchy materialization actions."""

    layout_plan_id: str
    requirement_manifest: tuple[Identity, ...]
    entries: tuple[ResolvedTransfer, ...]
    nesting_requirement: NestingRequirementSource
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout_plan_id, str) or not self.layout_plan_id:
            raise TypeError("AMRTransfer.layout_plan_id must be non-empty")
        manifest = tuple(self.requirement_manifest)
        if not manifest or any(
            type(item) is not Identity or item.domain != "amr-transfer-requirement"
            for item in manifest
        ):
            raise TypeError("AMRTransfer requires a non-empty exact requirement manifest")
        manifest = tuple(sorted(manifest, key=lambda item: item.token))
        if len(manifest) != len({item.token for item in manifest}):
            raise ValueError("AMRTransfer requirement manifest contains duplicates")
        entries = tuple(self.entries)
        if not entries or any(type(entry) is not ResolvedTransfer for entry in entries):
            raise TypeError("AMRTransfer.entries must contain resolved transfers")
        entries = tuple(
            sorted(
                entries,
                key=lambda entry: entry.identity.token,
            )
        )
        covered = sorted(
            requirement.identity.token
            for entry in entries
            for requirement in entry.requirements
        )
        if covered != [item.token for item in manifest]:
            raise ValueError("AMRTransfer entries do not exactly cover requirement manifest")
        if type(self.nesting_requirement) is not NestingRequirementSource \
                or self.nesting_requirement.provider.kind != "amr_transfer_requirement":
            raise TypeError("AMRTransfer.nesting_requirement must be a transfer source")
        object.__setattr__(self, "requirement_manifest", manifest)
        object.__setattr__(self, "entries", entries)

    @property
    def identity(self) -> Identity:
        return make_identity("amr-transfer", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layout_plan_id": self.layout_plan_id,
            "requirement_manifest": [item.to_data() for item in self.requirement_manifest],
            "entries": [entry.to_data() for entry in self.entries],
            "nesting_requirement": self.nesting_requirement.canonical_identity(),
        }

    def for_subject(self, subject: Any, operation: TransferOperation) -> ResolvedTransfer:
        subject_id = _generic_handle(subject, where="ResolvedAMRTransfer.for_subject").qualified_id
        matches = [
            entry
            for entry in self.entries
            if entry.key.operation == operation
            and any(row.subject.qualified_id == subject_id for row in entry.requirements)
        ]
        if len(matches) != 1:
            raise KeyError("no exact AMR transfer entry for %s / %s" % (subject_id, operation.name))
        return matches[0]
