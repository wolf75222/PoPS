"""Typed, authenticated static native functions captured by model libraries.

This describes a source-level interface, not a cross-toolchain binary ABI. The
function is compiled from PreparedNativeComponent's verified tree for the selected
PoPS/Kokkos target; there is no Python evaluation callback or dynamic symbol lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from pops.native_components import PreparedNativeComponent


def _target(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z][A-Za-z0-9_]*)+", value) is None:
        raise ValueError("native target must be an explicit qualified C++ identifier")
    return value


@dataclass(frozen=True, slots=True)
class NativeDerivative:
    """A declared derivative entry point; approximation is never called exact."""
    route: str
    target: str
    __pops_ir_immutable__ = True

    def __post_init__(self):
        if self.route not in ("exact", "approximate"):
            raise ValueError("native derivative declaration requires exact or approximate")
        _target(self.target)

    def to_data(self):
        return {"route": self.route, "target": self.target}


@dataclass(frozen=True, slots=True)
class NativeInputDomain:
    """Finite input with optional explicit open/closed bounds at one flat coordinate."""
    input: int
    component: int
    lower: float | None = None
    upper: float | None = None
    lower_open: bool = False
    upper_open: bool = False
    __pops_ir_immutable__ = True

    def __post_init__(self):
        import math
        if any(type(v) is not int or v < 0 for v in (self.input, self.component)):
            raise TypeError("native domain input/component must be nonnegative integers")
        for value in (self.lower, self.upper):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise TypeError("native domain bounds must be finite numerical constants")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("native domain lower bound must be below upper bound")
        if type(self.lower_open) is not bool or type(self.upper_open) is not bool:
            raise TypeError("native domain open flags must be exact booleans")

    def to_data(self):
        return {name: getattr(self, name) for name in
                ("input", "component", "lower", "upper", "lower_open", "upper_open")}


@dataclass(frozen=True, slots=True)
class NativeFunction:
    """One static target returning ``pops::NativeCallResult<output_width>``.

    Inputs are scalar coordinates, in signature order. Product outputs are flattened
    in their declared order. Omitted reads mean every input component; an explicit
    footprint states which scalar arguments the implementation actually reads.
    A host declaration is refused for device compilation. Device support is a build
    requirement, not qualification evidence.
    """
    component: PreparedNativeComponent
    target: str
    signature: Any
    execution_domains: tuple[str, ...] = ("host",)
    reads: tuple[tuple[int, int], ...] | None = None
    domains: tuple[NativeInputDomain, ...] = ()
    derivatives: tuple[NativeDerivative, ...] = ()
    effects: tuple[str, ...] = ("fallible",)
    interface: str = "pops.native-call@1"
    __pops_ir_immutable__ = True

    def __post_init__(self):
        from pops.model import Signature
        from pops.model.bundles import ProductSpace, RateBundle
        from pops.model.spaces import Space
        if type(self.component) is not PreparedNativeComponent:
            raise TypeError("native function requires an authenticated PreparedNativeComponent")
        _target(self.target)
        if not isinstance(self.signature, Signature) or any(
                not isinstance(space, Space) for space in self.signature.inputs):
            raise TypeError("native function requires a typed scalar-coordinate Signature")
        if not isinstance(self.signature.output, (Space, ProductSpace, RateBundle)):
            raise TypeError("native function requires typed output structure")
        if self.interface != "pops.native-call@1":
            raise ValueError("unsupported native-call source interface")
        if type(self.execution_domains) is not tuple or not self.execution_domains or any(
                domain not in ("host", "device") for domain in self.execution_domains):
            raise ValueError("native execution domains must explicitly contain host and/or device")
        coordinates = {(i, j) for i, space in enumerate(self.signature.inputs)
                       for j in range(len(space.components))}
        if self.reads is not None and (type(self.reads) is not tuple or any(
                type(pair) is not tuple or pair not in coordinates for pair in self.reads)
                or len(set(self.reads)) != len(self.reads)):
            raise ValueError("native read footprint must contain unique declared input components")
        if type(self.domains) is not tuple or any(type(d) is not NativeInputDomain or
                (d.input, d.component) not in coordinates for d in self.domains):
            raise ValueError("native domains must name exact declared input components")
        if type(self.derivatives) is not tuple or any(type(d) is not NativeDerivative for d in self.derivatives):
            raise TypeError("native derivatives require NativeDerivative records")
        if len({d.route for d in self.derivatives}) != len(self.derivatives):
            raise ValueError("native derivative routes must be unique")
        if type(self.effects) is not tuple or any(type(e) is not str or not e for e in self.effects):
            raise TypeError("native effects must be an immutable named contract")

    @property
    def output_entries(self):
        output = self.signature.output
        return output.items() if callable(getattr(output, "items", None)) else (("value", output),)

    @property
    def output_width(self):
        return sum(len(space.components) for _, space in self.output_entries)

    def to_data(self):
        return {"interface": self.interface, "component": self.component.manifest(),
                "target": self.target, "signature": self.signature.to_data(),
                "execution_domains": list(self.execution_domains),
                "reads": None if self.reads is None else [list(pair) for pair in self.reads],
                "domains": [d.to_data() for d in self.domains],
                "derivatives": [d.to_data() for d in self.derivatives], "effects": list(self.effects)}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_data(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def require_execution(self, domain: str):
        if domain not in self.execution_domains:
            raise ValueError("native function %s has no compiled %s route" % (self.target, domain))

    def require_derivative(self, route: str):
        record = next((d for d in self.derivatives if d.route == route), None)
        if record is None and route != "finite_difference":
            raise ValueError("native function %s has no declared %s derivative route" % (self.target, route))
        return {"route": route, "target": None if record is None else record.target,
                "component": self.component.authority(), "function_identity": self.identity}

    def __call__(self, *inputs, context=None):
        from pops._ir.native_call import NativeCall
        return NativeCall(self, inputs, context=context)


__all__ = ["NativeFunction", "NativeDerivative", "NativeInputDomain"]
