"""The typed ``(inputs) -> output`` contract of the operator-first type system.

A :class:`Signature` is structural: it holds a tuple of input types (spaces or
operator-types) and one output type. Equality is by ``(inputs, output)`` so two
signatures built from the same model compare equal. Carries no numerics.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _signature_data(value: Any, *, where: str = "signature type") -> Any:
    """Return the JSON-ready structural form of a signature descriptor.

    Signature extensibility is protocol based: a foreign descriptor is accepted
    when it is immutable/hashable and implements ``to_data()``.  There is no
    central ``isinstance`` branch to update for every new Space family.
    """
    hook = getattr(value, "to_data", None)
    if not callable(hook):
        raise TypeError(
            "%s %r does not implement the signature descriptor protocol "
            "(immutable/hashable + to_data())" % (where, value))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("%s %r must be hashable" % (where, value)) from exc
    data = hook()
    if not isinstance(data, Mapping):
        raise TypeError("%s %r.to_data() must return a mapping" % (where, value))
    return _plain_signature_data(data, where=where)


def _plain_signature_data(value: Any, *, where: str) -> Any:
    """Recursively copy protocol data to plain deterministic JSON values."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("%s to_data() keys must be non-empty strings" % where)
            out[key] = _plain_signature_data(item, where=where)
        return out
    if isinstance(value, (tuple, list)):
        return [_plain_signature_data(item, where=where) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_plain_signature_data(item, where=where) for item in value]
        return sorted(items, key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    hook = getattr(value, "to_data", None)
    if callable(hook):
        return _plain_signature_data(hook(), where=where)
    raise TypeError(
        "%s to_data() returned non-JSON value %r; nested descriptors must also "
        "implement to_data()" % (where, value))


class Signature:
    """A typed contract ``(inputs) -> output``.

    ``inputs`` is a tuple of spaces / operator-types; ``output`` is a space or an
    operator-type. Equality is structural so two signatures built from the same
    model compare equal. The ``>>`` operator-first sugar (``(U, Fields) >> Rate(U)``)
    lands with the public ``pops.model.Module`` API (S2-3); here the canonical
    keyword form is used.
    """

    def __init__(self, inputs: Any, output: Any) -> None:
        normalized = tuple(inputs)
        for index, value in enumerate(normalized):
            _signature_data(value, where="signature input %d" % index)
        _signature_data(output, where="signature output")
        object.__setattr__(self, "inputs", normalized)
        object.__setattr__(self, "output", output)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Signature is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Signature is immutable")

    def _key(self) -> Any:
        return (self.inputs, self.output)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Signature) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        ins = ", ".join(repr(x) for x in self.inputs)
        return "Signature((%s) -> %r)" % (ins, self.output)

    def to_data(self) -> dict[str, Any]:
        """Lossless structural schema; never falls back to address-bearing repr."""
        return {
            "inputs": [_signature_data(value, where="signature input %d" % index)
                       for index, value in enumerate(self.inputs)],
            "output": _signature_data(self.output, where="signature output"),
        }


__all__ = ["Signature"]
