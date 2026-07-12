"""Exact owner-qualified model graph consumed by whole-Program compilation."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def _model_owner(model: Any) -> Any:
    owner = getattr(model, "owner_path", None)
    if owner is None:
        owner = getattr(getattr(model, "_m", None), "owner_path", None)
    if owner is None:
        raise TypeError(
            "ProgramModelGraph model %s has no authoritative owner_path"
            % type(model).__name__
        )
    from pops.model import OwnerKind, OwnerPath

    owner = OwnerPath.coerce(owner)
    if owner.kind is not OwnerKind.MODEL_DEFINITION:
        raise ValueError(
            "ProgramModelGraph model owner must be a model_definition OwnerPath, got %s"
            % owner
        )
    return owner


class ProgramModelGraph:
    """Immutable block/owner routing for every model used by one compiled Program.

    Owner keys are canonical model-definition identities. Distinct live authoring authorities that
    collapse to the same canonical identity are rejected rather than selected by insertion order.
    Every block name has exactly one owner and every lookup validates that complete relation.
    """

    __slots__ = (
        "_models_by_owner",
        "_source_modules_by_owner",
        "_owners_by_block",
        "_authorities_by_owner",
    )

    def __init__(
        self,
        *,
        models_by_owner: Mapping[Any, Any],
        source_modules_by_owner: Mapping[Any, Any],
        owners_by_block: Mapping[str, Any],
        authorities_by_owner: Mapping[Any, Any],
    ) -> None:
        if not models_by_owner:
            raise ValueError("ProgramModelGraph requires at least one model owner")
        keys = set(models_by_owner)
        if set(source_modules_by_owner) != keys or set(authorities_by_owner) != keys:
            raise ValueError(
                "ProgramModelGraph owner tables must have exactly the same keys"
            )
        if not owners_by_block:
            raise ValueError("ProgramModelGraph requires at least one block route")
        if any(not isinstance(name, str) or not name for name in owners_by_block):
            raise TypeError("ProgramModelGraph block names must be non-empty strings")
        unknown = set(owners_by_block.values()) - keys
        if unknown:
            raise ValueError(
                "ProgramModelGraph block routes reference unknown model owners %s"
                % sorted(str(owner) for owner in unknown)
            )
        unused = keys - set(owners_by_block.values())
        if unused:
            raise ValueError(
                "ProgramModelGraph contains model owners unused by every block %s"
                % sorted(str(owner) for owner in unused)
            )
        self._models_by_owner = MappingProxyType(dict(models_by_owner))
        self._source_modules_by_owner = MappingProxyType(dict(source_modules_by_owner))
        self._owners_by_block = MappingProxyType(dict(owners_by_block))
        self._authorities_by_owner = MappingProxyType(dict(authorities_by_owner))

    @classmethod
    def from_resolved_blocks(cls, blocks: Any) -> ProgramModelGraph:
        """Lower every distinct exact ``ResolvedBlock`` model and capture total owner routing."""
        from pops.codegen._plans import ResolvedBlock
        from pops.codegen.module_lowering import lower_and_validate

        blocks = tuple(blocks)
        if not blocks or any(type(block) is not ResolvedBlock for block in blocks):
            raise TypeError(
                "ProgramModelGraph requires one or more exact ResolvedBlock values"
            )
        models: dict[Any, Any] = {}
        modules: dict[Any, Any] = {}
        authorities: dict[Any, Any] = {}
        routes: dict[str, Any] = {}
        lowered_by_authority: dict[int, tuple[Any, Any]] = {}
        for block in blocks:
            if block.name in routes:
                raise ValueError(
                    "ProgramModelGraph contains duplicate block name %r" % block.name
                )
            owner = _model_owner(block.model)
            canonical = owner.canonical()
            previous_authority = authorities.get(canonical)
            if previous_authority is not None and previous_authority != owner:
                raise ValueError(
                    "ProgramModelGraph distinct authoring model authorities collide at canonical "
                    "owner %s" % canonical
                )
            previous_model = models.get(canonical)
            authority_key = id(block.model)
            lowered = lowered_by_authority.get(authority_key)
            if lowered is None:
                lowered = lower_and_validate(block.model, facade=block.model)
                lowered_by_authority[authority_key] = lowered
            emit_model, source_module = lowered
            emit_owner = _model_owner(emit_model).canonical()
            if emit_owner != canonical:
                raise ValueError(
                    "ProgramModelGraph lowering changed model owner %s to %s"
                    % (canonical, emit_owner)
                )
            if previous_model is not None and previous_model is not emit_model:
                raise ValueError(
                    "ProgramModelGraph canonical owner %s maps to multiple model definitions"
                    % canonical
                )
            models[canonical] = emit_model
            modules[canonical] = source_module
            authorities[canonical] = owner
            routes[block.name] = canonical
        return cls(
            models_by_owner=models,
            source_modules_by_owner=modules,
            owners_by_block=routes,
            authorities_by_owner=authorities,
        )

    @property
    def models_by_owner(self) -> Mapping[Any, Any]:
        return self._models_by_owner

    @property
    def source_modules_by_owner(self) -> Mapping[Any, Any]:
        return self._source_modules_by_owner

    @property
    def owners_by_block(self) -> Mapping[str, Any]:
        return self._owners_by_block

    def model_for_owner(self, owner: Any) -> Any:
        from pops.model import OwnerPath

        canonical = OwnerPath.coerce(owner).canonical()
        try:
            return self._models_by_owner[canonical]
        except KeyError:
            raise KeyError(
                "ProgramModelGraph has no model for owner %s; known owners: %s"
                % (canonical, sorted(str(item) for item in self._models_by_owner))
            ) from None

    def source_module_for_owner(self, owner: Any) -> Any:
        from pops.model import OwnerPath

        canonical = OwnerPath.coerce(owner).canonical()
        try:
            return self._source_modules_by_owner[canonical]
        except KeyError:
            raise KeyError("ProgramModelGraph has no source module for owner %s" % canonical) from None

    def owner_for_block(self, block: Any) -> Any:
        from pops.problem.handles import BlockHandle

        if type(block) is BlockHandle:
            name = block.local_id
            actual_owner = block.model_owner_path.canonical()
        elif isinstance(block, str) and block:
            name = block
            actual_owner = None
        else:
            raise TypeError(
                "ProgramModelGraph block lookup requires an exact BlockHandle or non-empty name"
            )
        try:
            expected = self._owners_by_block[name]
        except KeyError:
            raise KeyError(
                "ProgramModelGraph has no route for block %r; known blocks: %s"
                % (name, sorted(self._owners_by_block))
            ) from None
        if actual_owner is not None and actual_owner != expected:
            raise ValueError(
                "ProgramModelGraph block %r carries model owner %s, expected %s"
                % (name, actual_owner, expected)
            )
        return expected

    def model_for_block(self, block: Any) -> Any:
        return self.model_for_owner(self.owner_for_block(block))


def model_for_node(authority: Any, node: Any) -> Any:
    """Resolve one Program node to its exact emit model.

    A single-model authority remains valid only for the explicit low-level compiler route.  Whole
    Program compilation passes :class:`ProgramModelGraph`; a node without a block must retain an
    owner-qualified ``operator_handle`` so it can never fall back to a representative model.
    """
    if type(authority) is not ProgramModelGraph:
        return authority
    block = getattr(node, "block", None)
    if block is not None:
        return authority.model_for_block(block)
    attrs = getattr(node, "attrs", {})
    operator = attrs.get("operator_handle") if hasattr(attrs, "get") else None
    owner = getattr(operator, "owner_path", None)
    if owner is None:
        raise ValueError(
            "Program node %r has no block or owner-qualified operator_handle"
            % getattr(node, "name", node))
    return authority.model_for_owner(owner)


def prepare_program_authority(model: Any, model_graph: Any) -> tuple[Any, Any, Any, Any]:
    """Validate/lower exactly one compile authority and return its detached compile facts."""
    if model is not None and model_graph is not None:
        raise TypeError("compile_problem received competing model and model_graph authorities")
    if model_graph is not None:
        if type(model_graph) is not ProgramModelGraph:
            raise TypeError("compile_problem model_graph must be an exact ProgramModelGraph")
        return model, None, None, model_graph
    from pops.codegen.module_lowering import lower_and_validate

    emit_model, source_module = lower_and_validate(model, facade=model)
    coverage = getattr(emit_model, "_lowering_coverage_report", None)
    return emit_model, source_module, coverage, emit_model


__all__ = [
    "ProgramModelGraph", "model_for_node", "prepare_program_authority",
]
