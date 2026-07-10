"""The ordered, name-keyed :class:`OperatorRegistry` (Spec 2).

Insertion order fixes each operator's integer ``OperatorId`` so the C++ codegen
can dispatch by integer in hot kernels while strings stay for debug / validation.
"""
from __future__ import annotations

from typing import Any

from .operators import Operator, validate_operator_signature


class OperatorRegistry:
    """An ordered, name-keyed registry of :class:`Operator` with stable integer ids.

    Insertion order fixes the ``OperatorId`` (``id_of`` / ``by_id``) so the C++
    codegen (S2-6) can dispatch by integer in hot kernels while strings stay for
    debug / validation only. Re-registering an existing name raises.
    """

    def __init__(self, owner: Any = None) -> None:
        from .handles import OwnerPath
        self._owner_path = OwnerPath.coerce(owner) if owner is not None else None
        self._by_name = {}
        self._order = []
        self._aliases = {}

    @property
    def owner_path(self) -> Any:
        """Read-only declaration owner shared by every handle in the registry."""
        return self._owner_path

    def register(self, operator: Any) -> Any:
        """Register ``operator`` and return it; its id is its insertion index."""
        if not isinstance(operator, Operator):
            raise TypeError("register expects an Operator, got %r" % (operator,))
        # Registry is a trust boundary too: Operator remains an internal mutable
        # record for codegen, so revalidate in case a record was modified between
        # construction and registration.
        validate_operator_signature(
            operator.kind, operator.signature, operator_name=operator.name)
        if operator.name in self._aliases:
            raise ValueError(
                "operator %r collides with registered alias targeting %r"
                % (operator.name, self._aliases[operator.name]))
        if operator.name in self._by_name:
            raise ValueError("operator %r already registered" % (operator.name,))
        self._by_name[operator.name] = operator
        self._order.append(operator.name)
        return operator

    def register_alias(self, alias: Any, target: Any) -> str:
        """Declare one immutable public alias for an existing registry operator.

        Alias resolution is registry-authenticated: carrying a different target
        inside an :class:`OperatorHandle` never grants access by itself. Compatible
        repeats are idempotent; a collision or retarget attempt fails loudly.
        """
        if not isinstance(alias, str) or not alias:
            raise ValueError("operator alias must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise ValueError("operator alias target must be a non-empty string")
        if target not in self._by_name:
            known = ", ".join(self._order) or "<none>"
            raise ValueError(
                "operator alias %r targets unknown operator %r (registered: %s)"
                % (alias, target, known))
        if alias in self._by_name:
            raise ValueError("operator alias %r collides with a registered operator" % alias)
        existing = self._aliases.get(alias)
        if existing is not None:
            if existing == target:
                return alias
            raise ValueError(
                "operator alias %r is already registered for %r, cannot retarget it to %r"
                % (alias, existing, target))
        self._aliases[alias] = target
        return alias

    def aliases(self) -> Any:
        """Detached ``{public_alias: registered_target}`` declaration table."""
        return dict(self._aliases)

    def target_for_handle(self, public_name: Any) -> str:
        """Authenticated registry target for a handle's public local identity."""
        if not isinstance(public_name, str) or not public_name:
            raise TypeError("operator handle name must be a non-empty string")
        if public_name in self._by_name:
            return public_name
        try:
            return self._aliases[public_name]
        except KeyError:
            known = self._order + list(self._aliases)
            raise KeyError(
                "unknown operator handle %r (registered operators/aliases: %s)"
                % (public_name, ", ".join(known) or "<none>")) from None

    def get(self, name: Any) -> Any:
        """Return the operator named ``name`` or raise a clear KeyError."""
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(self._order) or "<none>"
            raise KeyError(
                "unknown operator %r (registered: %s)" % (name, known)) from None

    def names(self) -> Any:
        """Operator names in registration (id) order."""
        return list(self._order)

    def operators_of_kind(self, kind: Any) -> Any:
        """Operators of the given kind, in registration order."""
        return [self._by_name[n] for n in self._order if self._by_name[n].kind == kind]

    def default_of_kind(self, kind: Any) -> Any:
        """The default operator of ``kind`` for model-free resolution.

        Picks the operator flagged ``capabilities["default"]`` if there is exactly
        one; otherwise the sole operator of that kind. Raises a clear error when none
        exists, or when several are compatible and none is privileged -- the caller
        must then disambiguate with an explicit ``P.call(name, ...)``.

        This is a BUILD-TIME resolution (kind -> operator), used only while lowering a Program; it is
        never on a hot kernel path. In a generated kernel operators are addressed by their integer
        ``OperatorId`` (:meth:`id_of` / :meth:`by_id`), so no operator-name string lookup survives into
        the compiled step (ADC-528).
        """
        candidates = self.operators_of_kind(kind)
        privileged = [op for op in candidates if op.capabilities.get("default")]
        if len(privileged) == 1:
            return privileged[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise KeyError("no %s operator registered" % kind)
        names = ", ".join(op.name for op in candidates)
        raise ValueError(
            "multiple %s operators are compatible (%s); call P.call(name, ...) "
            "explicitly" % (kind, names))

    def id_of(self, name: Any) -> int:
        """Integer OperatorId of ``name`` (its registration index)."""
        return self._order.index(name)

    def by_id(self, operator_id: Any) -> Any:
        """Operator at integer id ``operator_id``."""
        return self._by_name[self._order[operator_id]]

    def __contains__(self, name: Any) -> bool:
        return name in self._by_name or name in self._aliases

    def __iter__(self) -> Any:
        return (self._by_name[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return "OperatorRegistry(%s)" % ", ".join(self._order)
