"""Immutable grouping facades for Program values."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class StageStateSet:
    """A coherent, immutable ``block -> State ProgramValue`` stage view."""

    __slots__ = ("name", "_states")
    __pops_ir_immutable__ = True

    def __init__(self, name: Any, mapping: Any) -> None:
        from pops.time.values import ProgramValue
        if not isinstance(name, str) or not name:
            raise TypeError("StageStateSet name must be a non-empty string")
        if not isinstance(mapping, Mapping):
            raise TypeError("StageStateSet mapping must be a block-to-State mapping")
        states = {}
        for block, state in mapping.items():
            if not isinstance(block, str) or not block:
                raise TypeError("StageStateSet block names must be non-empty strings")
            if not (isinstance(state, ProgramValue) and state.vtype == "state"):
                raise ValueError("StageStateSet[%r] must be a State value" % (block,))
            if state.block != block:
                raise ValueError("StageStateSet[%r] must contain that block's State value" % block)
            states[block] = state
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_states", MappingProxyType(states))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StageStateSet is immutable")

    def states(self) -> list[Any]:
        return list(self._states.values())

    def __getitem__(self, block: Any) -> Any:
        return self._states[block]

    def __contains__(self, block: Any) -> bool:
        return isinstance(block, str) and block in self._states

    def keys(self) -> list[str]:
        return list(self._states)

    def items(self) -> list[tuple[str, Any]]:
        return list(self._states.items())

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return "StageStateSet(%r, blocks=%s)" % (self.name, list(self._states))


class _CoupledResult:
    """Immutable multi-output of a coupled-rate operator call."""

    __slots__ = ("_outs",)
    __pops_ir_immutable__ = True

    def __init__(self, outs: Any) -> None:
        if not isinstance(outs, Mapping) or any(not isinstance(k, str) or not k for k in outs):
            raise TypeError("coupled result must be a non-empty-string block mapping")
        object.__setattr__(self, "_outs", MappingProxyType(dict(outs)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("_CoupledResult is immutable")

    def __getitem__(self, block: Any) -> Any:
        return self._outs[block]

    def __contains__(self, block: Any) -> bool:
        return block in self._outs

    def keys(self) -> list[Any]:
        return list(self._outs)

    def items(self) -> list[tuple[Any, Any]]:
        return list(self._outs.items())

    def __len__(self) -> int:
        return len(self._outs)

    def __repr__(self) -> str:
        return "_CoupledResult(blocks=%s)" % list(self._outs)


__all__ = ["StageStateSet", "_CoupledResult"]
