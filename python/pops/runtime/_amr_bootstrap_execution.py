"""Strict BootstrapPlan execution with one receipt required per authored action."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.identity import Identity, make_identity


@dataclass(frozen=True, slots=True)
class BootstrapReceipt:
    action_identity: Identity
    consumer_identity: Identity
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.action_identity) is not Identity \
                or self.action_identity.domain != "amr-bootstrap-action":
            raise TypeError("BootstrapReceipt.action_identity must authenticate an action")
        if type(self.consumer_identity) is not Identity:
            raise TypeError("BootstrapReceipt.consumer_identity must be an Identity")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("BootstrapReceipt.evidence must be a mapping")

    def to_data(self) -> dict[str, Any]:
        return {
            "action_identity": self.action_identity.to_data(),
            "consumer_identity": self.consumer_identity.to_data(),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class BootstrapExecution:
    plan_identity: Identity
    receipts: tuple[BootstrapReceipt, ...]

    @property
    def identity(self) -> Identity:
        return make_identity("amr-bootstrap-execution", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "plan_identity": self.plan_identity.to_data(),
            "receipts": [receipt.to_data() for receipt in self.receipts],
        }


def execute_bootstrap(plan: Any, consumer: Any) -> BootstrapExecution:
    """Execute every action in order; missing, wrong, or duplicate receipts fail immediately."""
    from pops.mesh._amr import BootstrapPlan

    if type(plan) is not BootstrapPlan:
        raise TypeError("execute_bootstrap requires an exact BootstrapPlan")
    consume = getattr(consumer, "consume_bootstrap_action", None)
    identity = getattr(consumer, "bootstrap_consumer_identity", None)
    if not callable(consume) or type(identity) is not Identity:
        raise TypeError("bootstrap consumer must expose identity and consume_bootstrap_action()")
    receipts = []
    try:
        for action in plan.actions:
            receipt = consume(action)
            if type(receipt) is not BootstrapReceipt:
                raise TypeError("bootstrap consumer must return an exact BootstrapReceipt")
            if receipt.action_identity != action.identity:
                raise ValueError("bootstrap consumer returned a receipt for another action")
            if receipt.consumer_identity != identity:
                raise ValueError("bootstrap receipt does not authenticate the active consumer")
            receipts.append(receipt)
        finalize = getattr(consumer, "finalize_bootstrap", None)
        if callable(finalize):
            finalize()
    except BaseException:
        abort = getattr(consumer, "abort_bootstrap", None)
        if callable(abort):
            abort()
        raise
    if len(receipts) != len(plan.actions):
        raise ValueError("bootstrap execution did not consume every action")
    return BootstrapExecution(plan.identity, tuple(receipts))


class NativeAMRBootstrapConsumer:
    """Consumer for the native coarse-only AmrSystem bootstrap seam."""

    def __init__(
        self, engine: Any, plan: Any, initial_values: Any, field_routes: Any = None,
    ) -> None:
        self._engine = engine
        self._plan = plan
        self._initial = {
            subject_id: (block, value, space, centering, method, source)
            for subject_id, block, value, space, centering, method, source in initial_values
        }
        self._field_routes = dict(field_routes or {})
        if any(
            not isinstance(name, str) or not name
            or not isinstance(route, str) or not route
            for name, route in self._field_routes.items()
        ):
            raise TypeError("native bootstrap field routes must map names to provider slots")
        self._tagged_level = None
        self._clustered = False
        self._pending_level: int | None = None
        self._active = True
        self._engine._s._begin_bootstrap_plan()
        self.bootstrap_consumer_identity = make_identity(
            "native-amr-bootstrap-consumer", {"plan": plan.identity.to_data()}
        )

    def _receipt(self, action: Any, **evidence: Any) -> BootstrapReceipt:
        return BootstrapReceipt(action.identity, self.bootstrap_consumer_identity, evidence)

    @staticmethod
    def _route(action: Any) -> str | None:
        route = action.evidence.get("route", {})
        options = route.get("options", {}) if isinstance(route, Mapping) else {}
        if not options:
            provider = action.evidence.get("provider", {})
            options = provider.get("options", {}) if isinstance(provider, Mapping) else {}
        if not options:
            options = action.evidence.get("options", {})
        return options.get("native_route") if isinstance(options, Mapping) else None

    @staticmethod
    def _number(value: Any) -> float:
        if isinstance(value, Mapping) and set(value) == {"binary64"}:
            return float.fromhex(value["binary64"])
        return float(value)

    def consume_bootstrap_action(self, action: Any) -> BootstrapReceipt:
        operation = action.operation
        if operation == "initialize_level_zero":
            if action.subject_id not in self._initial:
                raise ValueError("native bootstrap is missing an authenticated level-zero value")
            _, value, _, _, method, source = self._initial[action.subject_id]
            expected_source = source.get("native_route") \
                if method == "analytic" else "bound_level_zero"
            if not isinstance(expected_source, str) or not expected_source:
                raise ValueError(
                    "native analytic level-zero materialization has no provider route"
                )
            if self._route(action) != expected_source:
                raise ValueError(
                    "native level-zero initialization requires %s" % expected_source
                )
            if method == "analytic":
                materialized = self._engine._s._bootstrap_analytic_reproject(
                    action.subject_id, 0
                )
                return self._receipt(
                    action,
                    operation=operation,
                    level=0,
                    materialized_values=materialized,
                )
            return self._receipt(
                action,
                operation=operation,
                level=0,
                materialized_values=int(value.size),
            )
        elif operation == "tag_parent":
            tagging = action.evidence.get("tagging", {})
            lowerings = tagging.get("lowerings", ()) if isinstance(tagging, Mapping) else ()
            provider_ids = tuple(
                row.get("lowering", {}).get("qualified_id")
                for row in lowerings if isinstance(row, Mapping)
            )
            if not provider_ids or any(not value for value in provider_ids):
                raise ValueError(
                    "native bootstrap tagging lacks authenticated prepared lowering providers"
                )
            self._tagged_level = action.evidence["parent_level"]
            self._clustered = False
            return self._receipt(
                action,
                operation=operation,
                level=action.level,
                indicator_providers=provider_ids,
                device_host_boundary="explicit_mask_mirror_then_host_clustering",
                mirror_cost="one scalar mask value per local parent cell plus MPI tag union",
            )
        elif operation == "cluster_tags":
            if self._tagged_level != action.level - 1:
                raise ValueError("native bootstrap clustering has no matching parent tag action")
            self._clustered = True
        elif operation == "create_level":
            if not self._clustered or self._tagged_level != action.level - 1:
                raise ValueError("native bootstrap create requires tag then cluster")
            ratios = tuple(action.evidence["ratio"])
            if len(set(ratios)) != 1:
                raise NotImplementedError(
                    "native bootstrap currently requires an isotropic resolved transition"
                )
            self._engine._bootstrap_next_level(ratios[0])
            self._pending_level = action.level
            if self._engine.n_levels() != action.level + 1:
                raise ValueError("native bootstrap created an unexpected hierarchy depth")
            boxes = tuple(row for row in self._engine.patch_boxes() if row[0] == action.level)
            if not boxes:
                raise ValueError("native bootstrap created a level without tag-derived patches")
            return self._receipt(
                action, operation=operation, level=action.level, patch_boxes=boxes
            )
        elif operation == "prolong_from_parent":
            if action.subject_id not in self._initial or self._engine.n_levels() <= action.level:
                raise ValueError("native prolongation did not materialize the requested state level")
            action_route = self._route(action)
            space = self._initial[action.subject_id][2]
            expected_route = {
                "cell": "conservative_linear",
                "face": "face_divergence_preserving",
                "node": "node_bilinear",
            }[space]
            if action_route != expected_route:
                raise ValueError("native prolongation receipt does not authenticate its provider")
            materialized = self._engine._s._bootstrap_prolong_array(
                action.subject_id, action.level
            )
            return self._receipt(
                action,
                operation=operation,
                level=action.level,
                materialized_values=materialized,
                provider=action.evidence.get("provider", {}).get("qualified_id"),
            )
        elif operation == "analytic_reprojection":
            materialized = self._engine._s._bootstrap_analytic_reproject(
                action.subject_id, action.level
            )
            return self._receipt(
                action, operation=operation, level=action.level,
                materialized_values=materialized,
            )
        elif operation == "recompute":
            if self._route(action) != "elliptic_solve":
                raise ValueError("native field recompute requires elliptic_solve")
            field_name = action.evidence["field_name"]
            provider_slot = self._field_routes.get(field_name)
            if provider_slot is None:
                raise ValueError(
                    "native field recompute has no authenticated provider slot for %r"
                    % field_name
                )
            materialized = self._engine._s._recompute_bootstrap_field(
                action.subject_id, provider_slot
            )
            return self._receipt(
                action,
                operation=operation,
                level=action.level,
                materialized_cells=materialized,
                provider=action.evidence.get("qualified_id"),
            )
        elif operation == "invalidate_cache":
            if self._route(action) != "patch_topology":
                raise ValueError("native cache invalidation requires patch_topology")
            self._engine._invalidate_bootstrap_cache(action.subject_id, action.level)
        elif operation in {"rebuild_cache", "invalidate_then_rebuild"}:
            if self._route(action) != "patch_topology":
                raise ValueError("native cache rebuild requires patch_topology")
            if operation == "invalidate_then_rebuild":
                self._engine._invalidate_bootstrap_cache(action.subject_id, action.level)
            topology = tuple(
                self._engine._rebuild_bootstrap_topology_cache(
                    action.subject_id, action.level
                )
            )
            return self._receipt(
                action,
                operation=operation,
                level=action.level,
                topology=topology,
                epoch=self._engine._s._bootstrap_cache_epoch(action.subject_id),
            )
        elif operation == "apply_constraint":
            options = action.evidence.get("options", {})
            if options.get("native_route") != "component_floor":
                raise ValueError("native bootstrap constraint requires component_floor")
            changed = self._engine._apply_bootstrap_component_floor(
                action.subject_id,
                action.level,
                int(options["component"]),
                self._number(options["floor"]),
            )
            return self._receipt(
                action, operation=operation, level=action.level, changed_cells=changed
            )
        elif operation == "synchronize_covered_cells":
            route = self._route(action)
            if route != "volume_average":
                raise ValueError("native final synchronization requires volume_average")
            self._engine._s._synchronize_bootstrap_state(action.subject_id, action.level)
            return self._receipt(
                action,
                operation=operation,
                level=action.level,
                provider=action.evidence.get("provider", {}).get("qualified_id"),
            )
        elif operation not in {"tag_parent", "cluster_tags", "create_level"}:
            raise ValueError("unknown BootstrapAction operation %r" % operation)
        return self._receipt(action, operation=operation, level=action.level)

    def finalize_bootstrap(self) -> None:
        if self._active:
            self._engine._commit_bootstrap_level()
            self._pending_level = None
            self._active = False

    def abort_bootstrap(self) -> None:
        if self._active:
            self._engine._rollback_bootstrap_level()
            self._pending_level = None
            self._active = False


def execute_native_bootstrap(
    engine: Any, plan: Any, initial_values: Any, field_routes: Any = None,
) -> BootstrapExecution:
    return execute_bootstrap(
        plan, NativeAMRBootstrapConsumer(engine, plan, initial_values, field_routes)
    )


__all__ = [
    "BootstrapExecution",
    "BootstrapReceipt",
    "NativeAMRBootstrapConsumer",
    "execute_bootstrap",
    "execute_native_bootstrap",
]
