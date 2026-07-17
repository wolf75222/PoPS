"""Private AMR transfer registry resolution over immutable exact contracts."""
# ruff: noqa: F405
from __future__ import annotations

import json
from typing import Any

from pops.identity import make_identity
from pops.model import Handle

from .._layout_plan_contracts import LayoutHandle, LayoutPlan
from .hierarchy import NestingRequirementSource
from ._transfer_contracts import *  # noqa: F403
from ._transfer_contracts import _generic_handle


def _authoring_handle(value: Any, *, where: str, kind: str) -> Handle:
    if not isinstance(value, Handle) or value.kind != kind:
        raise TypeError("%s requires a typed Handle(kind=%r), never a name" % (where, kind))
    return value


def _policy_data(policy: Any, *, expected_kind: str, where: str) -> dict[str, Any]:
    """Authenticate an open transfer-policy implementation by value, never by class name."""
    protocol = getattr(policy, "amr_transfer_policy_data", None)
    if not callable(protocol):
        raise TypeError("%s policy must implement amr_transfer_policy_data()" % where)
    data = protocol()
    if not isinstance(data, dict) or data.get("authority_type") != "amr_transfer_policy" \
            or data.get("policy_kind") != expected_kind:
        raise TypeError("%s policy does not authenticate kind %r" % (where, expected_kind))
    try:
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s policy identity must be strict JSON data" % where) from exc
    return data


def _kernel_data(kernel: Any, *, where: str) -> dict[str, Any]:
    protocol = getattr(kernel, "amr_transfer_kernel_data", None)
    if not callable(protocol):
        raise TypeError("%s must implement amr_transfer_kernel_data()" % where)
    data = protocol()
    required = {
        "schema_version", "kernel_type", "native_route", "order", "ghost_depth",
        "dimensions", "refinement_ratios", "conservative", "temporal",
    }
    if not isinstance(data, dict) or set(data) != required \
            or data.get("kernel_type") != "amr_transfer_kernel":
        raise TypeError("%s returned an unsupported kernel identity" % where)
    for name in required - {"schema_version", "kernel_type"}:
        value = getattr(kernel, name, object())
        projected = data[name]
        if isinstance(value, tuple):
            projected = tuple(projected)
        if value != projected:
            raise ValueError("%s kernel identity disagrees with attribute %s" % (where, name))
    return data


class AMRTransfer:
    """Object-level public declaration of how AMR values are materialized.

    ``state()``, ``field()`` and ``cache()`` accept typed policies from :mod:`pops.lib.amr`.
    Capability/order/halo requirements are derived from those descriptors during ``resolve``.
    """

    def __init__(self) -> None:
        self._frozen = False
        self._states: list[tuple[Any, Any, LayoutHandle | None]] \
            | tuple[tuple[Any, Any, LayoutHandle | None], ...] = []
        self._faces: list[tuple[tuple[Any, ...], Any, LayoutHandle | None]] \
            | tuple[tuple[tuple[Any, ...], Any, LayoutHandle | None], ...] = []
        self._nodes: list[tuple[Any, Any, LayoutHandle | None]] \
            | tuple[tuple[Any, Any, LayoutHandle | None], ...] = []
        self._fields: list[tuple[Any, Any, LayoutHandle | None]] \
            | tuple[tuple[Any, Any, LayoutHandle | None], ...] = []
        self._caches: list[tuple[Any, Any, LayoutHandle]] \
            | tuple[tuple[Any, Any, LayoutHandle], ...] = []

    def state(self, subject: Any, policy: Any, *, layout: LayoutHandle | None = None) -> None:
        if self._frozen:
            raise RuntimeError("AMRTransfer is frozen")
        data = _policy_data(policy, expected_kind="state", where="AMRTransfer.state")
        for route in ("prolongation", "restriction", "coarse_fine", "time_interpolation"):
            kernel = _kernel_data(
                getattr(policy, route, None), where="AMRTransfer.state.%s" % route)
            if data.get("routes", {}).get(route) != kernel:
                raise ValueError("AMRTransfer.state identity does not authenticate %s" % route)
        if not isinstance(self._states, list):
            raise RuntimeError("AMRTransfer is frozen")
        self._states.append((
            _authoring_handle(subject, where="AMRTransfer.state", kind="state"), policy, layout
        ))

    cell = state

    def face(
        self, subjects: Any, policy: Any, *, layout: LayoutHandle | None = None
    ) -> None:
        """Declare one coupled normal-face vector (one owner-qualified subject per axis)."""
        if self._frozen:
            raise RuntimeError("AMRTransfer is frozen")
        data = _policy_data(policy, expected_kind="face", where="AMRTransfer.face")
        kernel = _kernel_data(
            getattr(policy, "prolongation", None), where="AMRTransfer.face.prolongation")
        if data.get("routes", {}).get("prolongation") != kernel:
            raise ValueError("AMRTransfer.face identity does not authenticate prolongation")
        if isinstance(subjects, (str, bytes)):
            raise TypeError("AMRTransfer.face requires an ordered vector of face subjects")
        try:
            values = tuple(subjects)
        except TypeError as exc:
            raise TypeError(
                "AMRTransfer.face requires an ordered vector of face subjects"
            ) from exc
        if not values:
            raise ValueError("AMRTransfer.face requires at least one face subject")
        if not isinstance(self._faces, list):
            raise RuntimeError("AMRTransfer is frozen")
        self._faces.append((tuple(
            _authoring_handle(value, where="AMRTransfer.face", kind="state") for value in values
        ), policy, layout))

    def node(self, subject: Any, policy: Any, *, layout: LayoutHandle | None = None) -> None:
        if self._frozen:
            raise RuntimeError("AMRTransfer is frozen")
        data = _policy_data(policy, expected_kind="node", where="AMRTransfer.node")
        kernel = _kernel_data(
            getattr(policy, "prolongation", None), where="AMRTransfer.node.prolongation")
        if data.get("routes", {}).get("prolongation") != kernel:
            raise ValueError("AMRTransfer.node identity does not authenticate prolongation")
        if not isinstance(self._nodes, list):
            raise RuntimeError("AMRTransfer is frozen")
        self._nodes.append((
            _authoring_handle(subject, where="AMRTransfer.node", kind="state"), policy, layout
        ))

    def field(self, subject: Any, policy: Any, *, layout: LayoutHandle | None = None) -> None:
        if self._frozen:
            raise RuntimeError("AMRTransfer is frozen")
        data = _policy_data(policy, expected_kind="field", where="AMRTransfer.field")
        if not isinstance(data.get("native_route"), str) or not data["native_route"]:
            raise ValueError("AMRTransfer.field policy must authenticate native_route")
        if data["native_route"] != getattr(policy, "native_route", None):
            raise ValueError("AMRTransfer.field identity disagrees with native_route")
        if not isinstance(self._fields, list):
            raise RuntimeError("AMRTransfer is frozen")
        self._fields.append((
            _authoring_handle(subject, where="AMRTransfer.field", kind="field"), policy, layout
        ))

    def cache(self, subject: Any, policy: Any, *, layout: LayoutHandle) -> None:
        if self._frozen:
            raise RuntimeError("AMRTransfer is frozen")
        data = _policy_data(policy, expected_kind="cache", where="AMRTransfer.cache")
        if data.get("native_route") != getattr(policy, "native_route", None):
            raise ValueError("AMRTransfer.cache identity disagrees with native_route")
        if not isinstance(layout, LayoutHandle):
            raise TypeError(
                "AMRTransfer.cache requires an explicit LayoutHandle"
            )
        if not isinstance(self._caches, list):
            raise RuntimeError("AMRTransfer is frozen")
        self._caches.append((
            _authoring_handle(subject, where="AMRTransfer.cache", kind="cache"), policy, layout
        ))

    def freeze(self) -> AMRTransfer:
        if self._frozen:
            return self
        self._states = tuple(self._states)
        self._faces = tuple(self._faces)
        self._nodes = tuple(self._nodes)
        self._fields = tuple(self._fields)
        self._caches = tuple(self._caches)
        self._frozen = True
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise RuntimeError("AMRTransfer is frozen")
        object.__setattr__(self, name, value)

    def inspect(self) -> dict[str, Any]:
        def handle(value: Any) -> Any:
            projection = value.canonical_identity if value.is_resolved else value.inspect
            return projection()

        return {
            "authority_type": "amr_transfer_authoring",
            "states": [{"subject": handle(subject), "policy": _policy_data(
                policy, expected_kind="state", where="AMRTransfer.state")}
                       for subject, policy, _ in self._states],
            "faces": [{"subjects": [handle(subject) for subject in subjects],
                       "policy": _policy_data(
                           policy, expected_kind="face", where="AMRTransfer.face")}
                      for subjects, policy, _ in self._faces],
            "nodes": [{"subject": handle(subject), "policy": _policy_data(
                policy, expected_kind="node", where="AMRTransfer.node")}
                      for subject, policy, _ in self._nodes],
            "fields": [{"subject": handle(subject), "policy": _policy_data(
                policy, expected_kind="field", where="AMRTransfer.field")}
                       for subject, policy, _ in self._fields],
            "caches": [{"subject": handle(subject), "policy": _policy_data(
                policy, expected_kind="cache", where="AMRTransfer.cache")}
                       for subject, policy, _ in self._caches],
        }

    def resolve_references(self, resolver: Any) -> AMRTransfer:
        """Detach the registry while canonicalizing every declaration reference exactly once."""
        if not callable(resolver):
            raise TypeError("AMRTransfer.resolve_references requires a callable resolver")
        result = type(self)()
        result._states = [(resolver(subject), policy, layout)
                          for subject, policy, layout in self._states]
        result._faces = [(tuple(resolver(subject) for subject in subjects), policy, layout)
                         for subjects, policy, layout in self._faces]
        result._nodes = [(resolver(subject), policy, layout)
                         for subject, policy, layout in self._nodes]
        result._fields = [(resolver(subject), policy, layout)
                          for subject, policy, layout in self._fields]
        result._caches = [(resolver(subject), policy, layout)
                          for subject, policy, layout in self._caches]
        for family in (
                result._states, result._faces, result._nodes, result._fields, result._caches):
            subjects = [row[0] for row in family]
            flattened = [item for subject in subjects
                         for item in (subject if isinstance(subject, tuple) else (subject,))]
            if any(not isinstance(item, Handle) or not item.is_resolved for item in flattened):
                raise TypeError("AMRTransfer resolver must return canonical Handle values")
        return result

    @staticmethod
    def _layout_contract(
        layout_plan: LayoutPlan, subject: Any, layout: LayoutHandle | None
    ) -> tuple[LayoutHandle, int, tuple[int, ...]]:
        if layout is None:
            try:
                layout = layout_plan.layout_for(subject)
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "transfer subjects outside state/field/block require an explicit plan layout"
                ) from exc
        normalized = layout_plan.normalized(layout)
        dimension = normalized.capabilities.get("dim")
        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("AMR layout manifest must authenticate dimension 1, 2, or 3")
        if not normalized.adaptive or not normalized.transition_ratios:
            raise ValueError("AMRTransfer requires an adaptive layout with level transitions")
        return layout, dimension, normalized.transition_ratios

    @staticmethod
    def _accuracy(
        policy: Any, *, dimension: int, ratio: int, temporal: bool = False
    ) -> AccuracyRequirement:
        return AccuracyRequirement(
            order=policy.order,
            ghost_depth=policy.ghost_depth,
            dimension=dimension,
            refinement_ratio=(ratio,) * dimension,
            conservative=policy.conservative,
            temporal=temporal,
        )

    @staticmethod
    def _capabilities(policy: Any) -> TransferCapabilities:
        _kernel_data(policy, where="AMR transfer kernel")
        return TransferCapabilities(
            order=policy.order,
            ghost_depth=policy.ghost_depth,
            dimensions=policy.dimensions,
            conservative=policy.conservative,
            temporal=policy.temporal,
            refinement_ratios=policy.refinement_ratios,
        )

    def resolve(self, layout_plan: LayoutPlan) -> ResolvedAMRTransfer:
        if not (self._states or self._faces or self._nodes):
            raise ValueError("AMRTransfer requires at least one typed physical policy")
        resolver = AMRTransferBuilder(layout_plan)
        owner = layout_plan.owner

        def provider_handle(subjects: tuple[Any, ...], route: str, kind: str) -> Handle:
            token = make_identity(
                "amr-authored-provider",
                {
                    "subjects": sorted(subject.qualified_id for subject in subjects),
                    "route": route,
                    "kind": kind,
                },
            ).token
            return Handle("%s_%s" % (route, token), kind=kind, owner=owner)
        operations = (
            ("prolongation", PROLONGATION),
            ("restriction", RESTRICTION),
            ("coarse_fine", COARSE_FINE_FILL),
            ("time_interpolation", TEMPORAL_INTERPOLATION),
        )
        for subject, policy, layout in self._states:
            layout, dimension, ratios = self._layout_contract(layout_plan, subject, layout)
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "the selected transfer requirement schema cannot collapse heterogeneous "
                    "transition ratios; select a per-transition transfer provider"
                )
            ratio = ratios[0]
            for attribute, operation in operations:
                kernel = getattr(policy, attribute)
                key = TransferKey(
                    CELL_SPACE,
                    CELL_CENTERED,
                    CONSERVATIVE_REPRESENTATION,
                    DENSE_STORAGE,
                    operation,
                )
                provider = TransferProvider(
                    provider_handle((subject,), attribute, "amr_transfer_provider"),
                    (
                        TransferProviderRoute(
                            key,
                            self._capabilities(kernel),
                            CanonicalOptions({"native_route": kernel.native_route}),
                        ),
                    ),
                )
                resolver.register(provider)
                resolver.require(
                    subject,
                    key,
                    accuracy=self._accuracy(
                        kernel,
                        dimension=dimension,
                        ratio=ratio,
                        temporal=operation == TEMPORAL_INTERPOLATION,
                    ),
                    layout=layout,
                    provider=provider,
                )
        face_centerings = (FACE_X_CENTERED, FACE_Y_CENTERED)
        for subjects, policy, layout in self._faces:
            resolved_layout, dimension, ratios = self._layout_contract(
                layout_plan, subjects[0], layout
            )
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "the selected face transfer provider requires homogeneous transitions"
                )
            ratio = ratios[0]
            for subject in subjects[1:]:
                try:
                    subject_layout = layout_plan.layout_for(subject)
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        "every coupled face subject must belong to the same LayoutPlan"
                    ) from exc
                if subject_layout != resolved_layout:
                    raise ValueError(
                        "coupled face subjects cannot cross layout authorities"
                    )
            if len(subjects) != dimension:
                raise ValueError(
                    "AMRTransfer.face requires exactly one normal component per layout axis"
                )
            if dimension != 2:
                raise NotImplementedError(
                    "the installed divergence-preserving face provider supports dimension 2"
                )
            kernel = policy.prolongation
            keys = tuple(
                TransferKey(
                    FACE_SPACE,
                    face_centerings[axis],
                    CONSERVATIVE_REPRESENTATION,
                    DENSE_STORAGE,
                    PROLONGATION,
                )
                for axis in range(dimension)
            )
            provider = TransferProvider(
                provider_handle(subjects, "face_pair", "amr_transfer_provider"),
                tuple(
                    TransferProviderRoute(
                        key,
                        self._capabilities(kernel),
                        CanonicalOptions({"native_route": kernel.native_route}),
                    )
                    for key in keys
                ),
                CanonicalOptions({
                    "paired_subjects": [subject.qualified_id for subject in subjects],
                }),
            )
            resolver.register(provider)
            for subject, key in zip(subjects, keys, strict=True):
                resolver.require(
                    subject,
                    key,
                    accuracy=self._accuracy(
                        kernel, dimension=dimension, ratio=ratio
                    ),
                    layout=resolved_layout,
                    provider=provider,
                )
        for subject, policy, layout in self._nodes:
            resolved_layout, dimension, ratios = self._layout_contract(
                layout_plan, subject, layout
            )
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "the selected node transfer provider requires homogeneous transitions"
                )
            ratio = ratios[0]
            kernel = policy.prolongation
            key = TransferKey(
                NODE_SPACE,
                NODE_CENTERED,
                PRIMITIVE_REPRESENTATION,
                DENSE_STORAGE,
                PROLONGATION,
            )
            provider = TransferProvider(
                provider_handle((subject,), "node", "amr_transfer_provider"),
                (TransferProviderRoute(
                    key,
                    self._capabilities(kernel),
                    CanonicalOptions({"native_route": kernel.native_route}),
                ),),
            )
            resolver.register(provider)
            resolver.require(
                subject,
                key,
                accuracy=self._accuracy(kernel, dimension=dimension, ratio=ratio),
                layout=resolved_layout,
                provider=provider,
            )
        for subject, policy, layout in self._fields:
            resolved_layout, dimension, ratios = self._layout_contract(
                layout_plan, subject, layout
            )
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "the selected field materializer requires homogeneous transitions"
                )
            ratio = ratios[0]
            resolver.require(
                subject,
                TransferKey(
                    FIELD_SPACE,
                    CELL_CENTERED,
                    PRIMITIVE_REPRESENTATION,
                    DENSE_STORAGE,
                    COARSE_FINE_FILL,
                ),
                materialization=DERIVED_FIELD,
                accuracy=AccuracyRequirement(1, (0,), dimension, (ratio,) * dimension),
                layout=resolved_layout,
                materializer=MaterializationProvider(
                    provider_handle((subject,), "field", "field_operator"),
                    DERIVED_FIELD,
                    CanonicalOptions({"native_route": policy.native_route}),
                ),
            )
        for subject, policy, layout in self._caches:
            resolved_layout, dimension, ratios = self._layout_contract(
                layout_plan, subject, layout
            )
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "the selected cache materializer requires homogeneous transitions"
                )
            ratio = ratios[0]
            resolver.require(
                subject,
                TransferKey(
                    CACHE_SPACE,
                    CELL_CENTERED,
                    PRIMITIVE_REPRESENTATION,
                    DENSE_STORAGE,
                    COARSE_FINE_FILL,
                ),
                materialization=CACHE,
                accuracy=AccuracyRequirement(1, (0,), dimension, (ratio,) * dimension),
                layout=resolved_layout,
                materializer=MaterializationProvider(
                    provider_handle((subject,), "cache", "cache_provider"),
                    CACHE,
                    CanonicalOptions({"native_route": policy.native_route}),
                ),
            )
        return resolver.resolve()


class AMRTransferBuilder:
    """Mutable local registry whose only output is one detached immutable AMRTransfer."""

    def __init__(self, layout_plan: LayoutPlan) -> None:
        if type(layout_plan) is not LayoutPlan:
            raise TypeError("AMRTransferBuilder requires an exact LayoutPlan")
        self._layout_plan = layout_plan
        self._providers: dict[str, TransferProvider] = {}
        self._requirements: dict[tuple[str, str], TransferRequirement] = {}

    def register(self, provider: TransferProvider) -> None:
        if type(provider) is not TransferProvider:
            raise TypeError("AMRTransferBuilder.register requires TransferProvider")
        if provider.qualified_id in self._providers:
            raise ValueError("duplicate AMR transfer provider registration %s" % provider.qualified_id)
        self._providers[provider.qualified_id] = provider

    def _layout(self, subject: Any, layout: LayoutHandle | None) -> LayoutHandle:
        if layout is None:
            try:
                return self._layout_plan.layout_for(subject)
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "transfer subjects outside state/field/block require an explicit plan layout"
                ) from exc
        if not isinstance(layout, LayoutHandle):
            raise TypeError("transfer layout must be a LayoutHandle")
        self._layout_plan.normalized(layout)
        return layout

    def require(
        self,
        subject: Any,
        key: TransferKey,
        *,
        materialization: str = PHYSICAL,
        accuracy: AccuracyRequirement,
        layout: LayoutHandle | None = None,
        materializer: MaterializationProvider | None = None,
        provider: TransferProvider | None = None,
    ) -> TransferRequirement:
        subject = _generic_handle(subject, where="AMRTransferBuilder.require subject")
        if type(accuracy) is not AccuracyRequirement:
            raise TypeError("transfer accuracy must be a derived AccuracyRequirement")
        if provider is not None and type(provider) is not TransferProvider:
            raise TypeError("transfer provider selection must be a TransferProvider")
        if accuracy.temporal != (key.operation == TEMPORAL_INTERPOLATION):
            raise ValueError("transfer temporal accuracy disagrees with the requested operation")
        requirement = TransferRequirement(
            subject,
            self._layout(subject, layout),
            key,
            materialization,
            accuracy,
            materializer,
            provider,
        )
        registry_key = (requirement.key.identity.token, subject.qualified_id)
        if registry_key in self._requirements:
            raise ValueError("duplicate AMR transfer requirement for exact key and subject")
        self._requirements[registry_key] = requirement
        return requirement

    def require_field_context(
        self,
        context: Any,
        *,
        centering: Any,
        operation: TransferOperation,
        accuracy: AccuracyRequirement,
        materializer: MaterializationProvider,
    ) -> TransferRequirement:
        operator = _generic_handle(
            getattr(context, "operator", None),
            where="AMRTransferBuilder.require_field_context operator",
            kind="field_operator",
        )
        binding = getattr(context, "layout", None)
        layout = getattr(binding, "layout", None)
        if not isinstance(layout, LayoutHandle):
            raise TypeError("FieldContext must carry a plan-owned LayoutHandle binding")
        return self.require(
            operator,
            TransferKey(
                FIELD_SPACE,
                centering,
                PRIMITIVE_REPRESENTATION,
                DENSE_STORAGE,
                operation,
            ),
            materialization=DERIVED_FIELD,
            accuracy=accuracy,
            layout=layout,
            materializer=materializer,
        )

    def resolve(self) -> ResolvedAMRTransfer:
        if not self._requirements:
            raise ValueError("AMRTransferBuilder requires an explicit requirement manifest")
        grouped: dict[tuple[str, str], list[TransferRequirement]] = {}
        keys: dict[str, TransferKey] = {}
        actions: dict[tuple[str, str], Any] = {}
        consumed_providers: set[str] = set()
        for requirement in self._requirements.values():
            token = requirement.key.identity.token
            keys[token] = requirement.key
            if requirement.materialization == DERIVED_FIELD:
                materializer = requirement.materializer
                if materializer is None:
                    raise RuntimeError("derived-field transfer lost its materialization provider")
                group = (token, materializer.qualified_id)
                action: Any = Recompute(materializer)
            elif requirement.materialization == CACHE:
                materializer = requirement.materializer
                if materializer is None:
                    raise RuntimeError("cache transfer lost its materialization provider")
                group = (token, materializer.qualified_id)
                action = InvalidateThenRebuild(materializer)
            else:
                candidates = []
                incompatible = []
                for provider in self._providers.values():
                    if requirement.provider is not None \
                            and provider.qualified_id != requirement.provider.qualified_id:
                        continue
                    for route in provider.routes:
                        if route.key.identity.token != token:
                            continue
                        if route.capabilities.supports((requirement,)):
                            candidates.append((provider, route))
                        else:
                            incompatible.append(provider.qualified_id)
                if len(candidates) != 1:
                    if not candidates and incompatible:
                        raise ValueError(
                            "incompatible AMR transfer provider(s) for exact requirement: %s"
                            % sorted(incompatible)
                        )
                    if not candidates:
                        raise ValueError("missing AMR transfer provider for exact key %s" % token)
                    raise ValueError(
                        "ambiguous AMR transfer providers; select provider= explicitly: %s"
                        % sorted(provider.qualified_id for provider, _ in candidates)
                    )
                provider, route = candidates[0]
                consumed_providers.add(provider.qualified_id)
                group = (token, provider.qualified_id)
                action = ApplyTransferProvider(provider, route, route.capabilities)
            grouped.setdefault(group, []).append(requirement)
            previous = actions.setdefault(group, action)
            if previous.to_data() != action.to_data():
                raise ValueError("batched transfer requirements resolved to different kernels")
        entries = []
        for group in sorted(grouped):
            token, _ = group
            requirements = tuple(
                sorted(grouped[group], key=lambda row: row.subject.qualified_id)
            )
            entries.append(ResolvedTransfer(keys[token], requirements, actions[group]))
        unused = sorted(set(self._providers) - consumed_providers)
        if unused:
            raise ValueError("unused AMR transfer provider registration(s): %s" % unused)
        physical = [
            (
                entry.native_materialization.capabilities.transfer,
                max(row.accuracy.dimension for row in entry.requirements),
            )
            for entry in entries
            if entry.native_materialization.materialization
            is NativeAMRMaterializationKind.PHYSICAL
        ]
        if any(capabilities is None for capabilities, _ in physical):
            raise ValueError("physical AMR transfer action omitted transfer capabilities")
        dimension = max(row.accuracy.dimension for row in self._requirements.values())
        buffers = []
        for capabilities, route_dimension in physical:
            ghost = capabilities.ghost_depth
            route_buffer = ghost * route_dimension if len(ghost) == 1 else ghost
            buffers.append(route_buffer + (0,) * (dimension - route_dimension))
        minimum_buffer = tuple(
            max((row[axis] for row in buffers), default=0) for axis in range(dimension)
        )
        minimum_lookahead = max((row.order - 1 for row, _ in physical), default=0)
        nesting = NestingRequirementSource(
            Handle(
                "resolved_registry_%s" % make_identity(
                    "amr-transfer-nesting-source",
                    {
                        "requirements": sorted(
                            row.identity.token for row in self._requirements.values()
                        ),
                        "minimum_buffer": list(minimum_buffer),
                        "minimum_lookahead": minimum_lookahead,
                    },
                ).token,
                kind="amr_transfer_requirement",
                owner=self._layout_plan.owner,
            ),
            minimum_buffer,
            minimum_lookahead,
        )
        manifest = tuple(sorted(
            (row.identity for row in self._requirements.values()),
            key=lambda identity: identity.token,
        ))
        return ResolvedAMRTransfer(
            self._layout_plan.qualified_id, manifest, tuple(entries), nesting
        )


__all__ = [
    "AMRTransfer",
    "ResolvedAMRTransfer",
]
