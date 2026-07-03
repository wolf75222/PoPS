"""The ordered, name-keyed :class:`OperatorRegistry` (Spec 2).

Insertion order fixes each operator's integer ``OperatorId`` so the C++ codegen
can dispatch by integer in hot kernels while strings stay for debug / validation.
"""
from __future__ import annotations

from typing import Any

from .operators import Operator


class OperatorRegistry:
    """An ordered, name-keyed registry of :class:`Operator` with stable integer ids.

    Insertion order fixes the ``OperatorId`` (``id_of`` / ``by_id``) so the C++
    codegen (S2-6) can dispatch by integer in hot kernels while strings stay for
    debug / validation only. Re-registering an existing name raises.
    """

    def __init__(self) -> None:
        self._by_name = {}
        self._order = []

    def register(self, operator: Any) -> Any:
        """Register ``operator`` and return it; its id is its insertion index."""
        if not isinstance(operator, Operator):
            raise TypeError("register expects an Operator, got %r" % (operator,))
        if operator.name in self._by_name:
            raise ValueError("operator %r already registered" % (operator.name,))
        self._by_name[operator.name] = operator
        self._order.append(operator.name)
        return operator

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
        return name in self._by_name

    def __iter__(self) -> Any:
        return (self._by_name[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __repr__(self) -> str:
        return "OperatorRegistry(%s)" % ", ".join(self._order)
