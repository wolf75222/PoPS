"""Block declarations and model-to-instance qualification for a Case."""
from __future__ import annotations

from typing import Any

from pops.model import DeclarationIndex, Handle
from pops.model.ownership import (
    AmbiguousReferenceError,
    DoubleOwnershipError,
    IdentityCollisionError,
    MissingOwnershipError,
    OwnerKind,
    OwnerPath,
)
from pops.problem._registry_freeze import (
    FreezableRegistry as _FreezableRegistry,
    flatten_freeze_members,
)
from pops.problem._registry_support import strict_name
from pops.problem.handles import BlockHandle
from pops._report import ReportTree


class BlockRegistry(_FreezableRegistry):
    """Physics blocks and the sole authority for qualifying model declarations."""

    family = "block"

    def __init__(self, owner: Any) -> None:
        self._owner_path = OwnerPath.coerce(owner).require_authoring_root(
            OwnerKind.CASE, where="BlockRegistry owner"
        )
        self._blocks = {}
        self._handles = {}
        self._model_owners = {}
        self._declaration_indexes = {}
        self._instances = {}

    @property
    def owner_path(self) -> Any:
        return self._owner_path

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(
            *(
                value
                for spec in self._blocks.values()
                for value in (
                    spec["model"],
                    spec["spatial"],
                    spec["time"],
                    spec["diagnostics"],
                )
            )
        )

    def _prepare_freeze(self) -> None:
        """Materialize every declaration instance before storage becomes immutable.

        Qualification is deterministic derived state, but caching it after freeze would mutate a
        mapping proxy.  Eagerly issuing the complete finite set here keeps ``qualify`` and schema
        extraction read-only on a frozen Case.  The enclosing freeze transaction rolls this
        cache back if a later member fails to freeze.
        """
        for block in self._handles.values():
            index = self._index_for(block)
            for declaration in index.records():
                if (declaration.kind == "state"
                        and declaration not in self._blocks[block.local_id]["states"]):
                    continue
                self._qualify_new(declaration, block)

    def add(
        self,
        name: Any,
        model: Any,
        *,
        states: Any = None,
        spatial: Any = None,
        time: Any = None,
        diagnostics: Any = None,
    ) -> Any:
        self._guard_frozen("add a block")
        key = strict_name(name, "block name")
        if model is None:
            raise ValueError("block(%r): a physics model is required" % key)
        if key in self._blocks:
            raise ValueError("block(%r): a block of that name already exists" % key)
        model_owner = getattr(model, "owner_path", None)
        if model_owner is None:
            raise MissingOwnershipError(
                "block %r model must expose its authoritative OwnerPath owner_path; model names "
                "are display labels and are never promoted into owners" % key
            )
        model_name = getattr(model, "name", None)
        if not isinstance(model_name, str) or not model_name:
            raise MissingOwnershipError(
                "block %r model must expose a non-empty name matching its authoring owner" % key
            )
        model_owner = OwnerPath.coerce(model_owner).require_authoring_root(
            OwnerKind.MODEL_DEFINITION,
            name=model_name,
            where="block %r model owner" % key,
        )
        # Building the authoritative declaration index attaches the richest available stable
        # content fingerprint (Module hash, operator-registry hash, or declaration fallback) before
        # the owner is ever projected into canonical data.
        declaration_index = self._provided_index(model, model_owner)
        declared_states = tuple(
            handle for handle in declaration_index.records() if handle.kind == "state")
        if states is None:
            if len(declared_states) > 1:
                raise ValueError(
                    "block %r uses a multi-state Model; pass states=(state_handle, ...) explicitly"
                    % key)
            selected_states = declared_states
        else:
            if isinstance(states, (str, bytes, Handle)):
                raise TypeError("block states= must be a non-empty sequence of typed StateHandles")
            try:
                selected_states = tuple(states)
            except TypeError:
                raise TypeError(
                    "block states= must be a non-empty sequence of typed StateHandles") from None
            if not selected_states:
                raise ValueError("block states= must select at least one declared state")
            authenticated = []
            for state in selected_states:
                if not isinstance(state, Handle) or state.kind != "state":
                    raise TypeError("block states= entries must be typed StateHandles")
                authenticated.append(declaration_index.authenticate(state))
            selected_states = tuple(authenticated)
            if len(set(selected_states)) != len(selected_states):
                raise ValueError("block states= contains a duplicate StateHandle")
        canonical_model_owner = model_owner.canonical()
        existing = self._model_owners.get(canonical_model_owner)
        if existing is not None and existing is not model:
            raise IdentityCollisionError(
                "models %r and %r claim the same owner %s; rename one model before registering "
                "it in this case"
                % (type(existing).__name__, type(model).__name__, canonical_model_owner)
            )
        self._model_owners[canonical_model_owner] = model
        handle = BlockHandle(
            key,
            owner=self.owner_path,
            model_owner=model_owner,
            instance_registry=self,
        )
        self._blocks[key] = {
            "model": model,
            "states": selected_states,
            "spatial": spatial,
            "time": time,
            "diagnostics": diagnostics,
        }
        self._handles[key] = handle
        self._declaration_indexes[model_owner] = declaration_index
        return handle

    def handle(self, name: Any) -> BlockHandle:
        key = strict_name(name, "block name")
        try:
            return self._handles[key]
        except KeyError:
            raise KeyError(
                "unknown block %r (known: %s)"
                % (key, ", ".join(self._blocks) or "<none>")
            ) from None

    def handles(self) -> Any:
        return dict(self._handles)

    @staticmethod
    def _provided_index(model: Any, model_owner: OwnerPath) -> DeclarationIndex:
        provider = getattr(model, "declaration_index", None)
        if callable(provider):
            index = provider()
        else:
            operator_registry = getattr(model, "operator_registry", None)
            if callable(operator_registry):
                registry = operator_registry()
                index_provider = getattr(registry, "declaration_index", None)
                index = index_provider() if callable(index_provider) else None
            else:
                index = None
        if index is None:
            index = DeclarationIndex(owner=model_owner, handles=())
        if not isinstance(index, DeclarationIndex):
            raise TypeError(
                "model %r declaration_index() must return pops.model.DeclarationIndex"
                % getattr(model, "name", type(model).__name__)
            )
        if index.owner_path != model_owner:
            raise MissingOwnershipError(
                "model declaration index owner %s does not match block model owner %s"
                % (index.owner_path, model_owner)
            )
        return index

    def _index_for(self, block: BlockHandle) -> DeclarationIndex:
        cached = self._declaration_indexes.get(block.model_owner_path)
        if cached is not None:
            return cached
        model = self._blocks[block.local_id]["model"]
        index = self._provided_index(model, block.model_owner_path)
        self._declaration_indexes[block.model_owner_path] = index
        return index

    def _canonical_block(self, block: BlockHandle) -> BlockHandle:
        return block._resolved(self.owner_path.canonical())

    def canonical_block(self, block: Any) -> BlockHandle:
        return self._canonical_block(self._registered_block(block))

    def _registered_block(self, block: Any) -> BlockHandle:
        if not isinstance(block, BlockHandle):
            raise TypeError("block= must be a BlockHandle, not a name string")
        registered = self._handles.get(block.local_id)
        if registered is None:
            raise MissingOwnershipError(
                "block handle %s is not registered by this case" % block.qualified_id
            )
        if block.is_resolved:
            expected = self._canonical_block(registered)
            if (
                block.owner_path != expected.owner_path
                or block.canonical_identity() != expected.canonical_identity()
                or block.model_owner_path != expected.model_owner_path
            ):
                raise MissingOwnershipError(
                    "block handle %s is not registered by this case" % block.qualified_id
                )
            return registered
        if (
            not isinstance(block, BlockHandle)
            or registered != block
            or registered.model_owner_path != block.model_owner_path
        ):
            raise MissingOwnershipError(
                "block handle %s is not registered by this case" % block.qualified_id
            )
        return registered

    def _authenticate_existing_instance(self, declaration: Handle) -> Handle:
        """Map one canonical/live qualified identity back to the registry-issued instance."""
        block_ref = declaration.block_ref
        declaration_ref = declaration.declaration_ref
        if block_ref is None or declaration_ref is None:
            raise MissingOwnershipError(
                "qualified declaration %s is missing its block/declaration provenance"
                % declaration.qualified_id
            )
        registered_block = self._registered_block(block_ref)
        registered_declaration = self._index_for(registered_block).authenticate(declaration_ref)
        expected = self._qualify_new(registered_declaration, registered_block)
        if declaration.is_resolved:
            expected_canonical = self._canonicalize_instance(expected, registered_block)
            if declaration.canonical_identity() != expected_canonical.canonical_identity():
                raise MissingOwnershipError(
                    "qualified declaration %s was not issued by this case registry"
                    % declaration.qualified_id
                )
        elif declaration != expected:
            raise MissingOwnershipError(
                "qualified declaration %s was not issued by this case registry"
                % declaration.qualified_id
            )
        return expected

    def _canonicalize_instance(
        self,
        qualified: Handle,
        live_block: BlockHandle,
    ) -> Handle:
        canonical_block = self._canonical_block(live_block)
        declaration_ref = qualified.declaration_ref
        if declaration_ref is None:
            raise MissingOwnershipError(
                "qualified declaration %s is missing declaration provenance"
                % qualified.qualified_id
            )
        canonical_declaration = declaration_ref._resolved(
            live_block.model_owner_path.canonical()
        )
        return qualified._with_owner(
            live_block.instance_owner_path.canonical(),
            declaration_ref=canonical_declaration,
            block_ref=canonical_block,
        )

    def _qualify_new(self, declaration: Handle, block: BlockHandle) -> Handle:
        registered = self._index_for(block).authenticate(declaration)
        if (registered.kind == "state"
                and registered not in self._blocks[block.local_id]["states"]):
            raise MissingOwnershipError(
                "state %s is not selected by block %r; selected states are %s"
                % (registered.qualified_id, block.local_id,
                   tuple(state.local_id for state in self._blocks[block.local_id]["states"])))
        key = (block.local_id, registered)
        cached = self._instances.get(key)
        if cached is not None:
            return cached
        qualified = registered._with_owner(
            block.instance_owner_path,
            declaration_ref=registered,
            block_ref=block,
        )
        self._instances[key] = qualified
        return qualified

    def qualify(
        self,
        declaration: Any,
        *,
        block: Any = None,
        allow_existing: bool = True,
    ) -> Any:
        if not isinstance(declaration, Handle):
            raise TypeError("block declaration resolution requires a Handle")
        if not isinstance(allow_existing, bool):
            raise TypeError("allow_existing must be a bool")
        if declaration.owner_path.nodes[0].kind is OwnerKind.SHARED:
            if declaration.is_instance:
                raise DoubleOwnershipError(
                    "shared declaration %s cannot carry a block instance owner"
                    % declaration.qualified_id
                )
            return declaration
        if block is not None:
            registered_block = self._registered_block(block)
            if declaration.owner_path.contains(OwnerKind.BLOCK):
                if not allow_existing:
                    raise DoubleOwnershipError(
                        "declaration %s is already block-qualified and cannot be qualified by %s"
                        % (declaration.qualified_id, registered_block.qualified_id)
                    )
                existing = self._authenticate_existing_instance(declaration)
                if existing.block_ref == registered_block:
                    return existing
                raise MissingOwnershipError(
                    "qualified declaration %s belongs to a different registered block"
                    % declaration.qualified_id)
            return self._qualify_new(declaration, registered_block)
        if declaration.owner_path.contains(OwnerKind.BLOCK):
            if allow_existing:
                return self._authenticate_existing_instance(declaration)
            raise MissingOwnershipError(
                "qualified declaration %s was not issued by this case registry"
                % declaration.qualified_id
            )
        owner_candidates = [
            handle for handle in self._handles.values() if handle.accepts(declaration)
        ]
        candidates = []
        for handle in owner_candidates:
            index = self._index_for(handle)
            if not index.contains(declaration):
                continue
            authenticated = index.authenticate(declaration)
            if (authenticated.kind == "state"
                    and authenticated not in self._blocks[handle.local_id]["states"]):
                continue
            candidates.append(handle)
        if len(candidates) == 1:
            return self._qualify_new(declaration, candidates[0])
        if len(candidates) > 1:
            owners = ", ".join(str(handle.instance_owner_path) for handle in candidates)
            raise AmbiguousReferenceError(
                "declaration %s is unqualified and matches %d block instances; candidates: %s. "
                "Select one explicitly as block[declaration]."
                % (declaration.qualified_id, len(candidates), owners)
            )
        if owner_candidates:
            self._index_for(owner_candidates[0]).authenticate(declaration)
        known = ", ".join(str(handle.model_owner_path) for handle in self._handles.values())
        raise MissingOwnershipError(
            "declaration %s has owner %s, which no block in this case instantiates (known model "
            "owners: %s)"
            % (declaration.qualified_id, declaration.owner_path, known or "<none>")
        )

    def canonicalize(self, declaration: Any, *, block: Any = None) -> Handle:
        qualified = self.qualify(declaration, block=block, allow_existing=True)
        if qualified.owner_path.nodes[0].kind is OwnerKind.SHARED:
            return qualified._resolved()
        if not qualified.is_instance:
            raise MissingOwnershipError(
                "only shared or block-qualified declarations can be canonicalized by a case"
            )
        live_block = self._registered_block(qualified.block_ref)
        return self._canonicalize_instance(qualified, live_block)

    def get(self, name: Any) -> Any:
        return self._blocks.get(strict_name(name, "block name"))

    def names(self) -> Any:
        return list(self._blocks)

    def spec(self, name: Any) -> Any:
        return self._blocks.get(strict_name(name, "block name"))

    def items(self) -> Any:
        return self._blocks.items()

    def __iter__(self) -> Any:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._blocks

    def validate(self, context: Any = None) -> Any:
        report = ReportTree(
            phase="validation", severity="info", code="validation.block.root",
            source=self.family)
        if not self._blocks:
            report = report.error(
                self.family,
                "no_block",
                "no block declared; add one with block(name, model)",
                alternatives=["block(name, model)"],
            )
            return report
        for name, spec in self._blocks.items():
            if spec.get("model") is None:
                report = report.error(
                    self.family,
                    "no_model",
                    "block %r has no physics model" % name,
                    context={"block": name},
                )
        return report

    def inspect(self) -> Any:
        return {
            name: {
                "model": getattr(spec["model"], "name", repr(spec["model"])),
                "states": tuple(state.local_id for state in spec["states"]),
                "spatial": getattr(spec["spatial"], "name", spec["spatial"]),
                "time": getattr(spec["time"], "name", None),
                "diagnostics": getattr(spec["diagnostics"], "name", None),
            }
            for name, spec in self._blocks.items()
        }


__all__ = ["BlockRegistry"]
