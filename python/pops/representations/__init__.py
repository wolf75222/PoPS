"""Representation identities and executable, scientifically typed finite maps.

The legacy ``Representation(name, data)`` descriptor retains its exact v08
identity. The scientific form consumes closed finite graphs; names never select
a conversion, quadrature, duality or native physical implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from collections.abc import Mapping
from types import MappingProxyType
import math
import hashlib


def _freeze_data(value):
    """Canonical inert metadata; executable objects are never retained here."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(type(key) is str and key for key in value):
        return MappingProxyType({key: _freeze_data(value[key]) for key in sorted(value)})
    if type(value) is list or type(value) is tuple:
        return tuple(_freeze_data(item) for item in value)
    raise TypeError("representation metadata must be finite inert JSON data")


def _thaw_data(value):
    if isinstance(value, Mapping):
        return {key: _thaw_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_data(item) for item in value]
    return value


def _data_to_wire(value):
    """Project binary64 values at the owning metadata boundary, not in CBOR."""
    from pops.identity.semantic import semantic_value
    return semantic_value(_thaw_data(value), where="representation metadata")


def _data_identity(value):
    """One exact key for type comparisons, hashes and native serialization."""
    from pops.identity.encoding import canonical_bytes
    # Space keys also occur inside the expression DAG's strict JSON identity.
    # Hex keeps these exact canonical bytes representable at that boundary.
    return canonical_bytes(_data_to_wire(value)).hex()


@dataclass(frozen=True, slots=True, eq=False)
class ScientificType:
    """Frontend meaning of a finite value, checked before representation erasure.

    ``measure`` is an explicit inert mathematical identifier/definition. Its
    analytical interpretation remains a declaration, not a runtime quadrature.
    The finite bodies in ScientificMap are the executed conversion equations.
    """

    finite_type: Any
    space: Any
    kind: str
    variance: tuple[str, ...]
    coordinate_units: tuple[Any, ...]
    measure: Any
    dofs: Any = None
    representation_identity: str | None = None
    affine: str = "linear"
    affine_origin: str | None = None

    __pops_ir_immutable__ = True

    def __post_init__(self):
        from pops.model.spaces import Space
        from pops._ir.quantity import PhysicalDimension
        from pops.time.variant_types import ValueType
        if type(self.finite_type) is not ValueType or not isinstance(self.space, Space):
            raise TypeError("ScientificType requires a finite ValueType and a physical Space")
        if self.affine not in ("linear", "point", "increment"):
            raise ValueError("scientific affine role must be linear, point or increment")
        if (self.affine == "linear") != (self.affine_origin is None):
            raise ValueError("affine points/increments require an origin; linear values have no origin")
        if self.affine_origin is not None and (not isinstance(self.affine_origin, str) or not self.affine_origin):
            raise TypeError("affine origin must be an explicit nonempty identifier")
        if self.space.support is None or any(unit is None for unit in self.space.units):
            raise ValueError("scientific types require explicit support and all component units")
        if type(self.kind) is not str or not self.kind or self.kind.strip() != self.kind:
            raise TypeError("scientific kind must be explicit canonical text")
        if self.representation_identity is not None and (
                type(self.representation_identity) is not str
                or not self.representation_identity.startswith("pops.representation:sha256:")
                or len(self.representation_identity) != 91
                or any(c not in "0123456789abcdef" for c in self.representation_identity[27:])):
            raise TypeError("representation identity requires a canonical graph digest")
        variance = tuple(self.variance)
        if len(variance) != len(self.space.value_shape) or any(
                item not in ("contravariant", "covariant", "component") for item in variance):
            raise ValueError("variance must explicitly classify each value-shape axis")
        coordinate_units = tuple(self.coordinate_units)
        if (len(coordinate_units) != len(self.space.support.coordinates)
                or any(type(unit) is not PhysicalDimension for unit in coordinate_units)):
            raise ValueError("coordinate units must cover every support axis explicitly")
        if not isinstance(self.measure, Mapping) or not self.measure:
            raise TypeError("scientific measure requires an explicit nonempty definition mapping")
        if self.dofs is not None and (type(self.dofs) is not ValueType
                                      or self.dofs.signature["vtype"] != "index_set"):
            raise TypeError("dofs requires the exact finite IndexSet ValueType")
        if self.finite_type.signature["vtype"] == "array":
            if self.dofs is None:
                raise ValueError("an array representation requires its exact dof family")
            domain = {key: value for key, value in self.finite_type.signature.items()
                      if key != "element"} | {"vtype": "index_set"}
            if ValueType(domain) != self.dofs:
                raise ValueError("array finite type and dof family disagree")
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "coordinate_units", coordinate_units)
        object.__setattr__(self, "measure", _freeze_data(self.measure))

    def to_data(self):
        return {"contract": "pops.scientific-type@1", "finite_type": self.finite_type.to_data(),
                "space": self.space.to_data(), "kind": self.kind,
                "variance": list(self.variance),
                "coordinate_units": [unit.to_data() for unit in self.coordinate_units],
                "measure": _data_to_wire(self.measure),
                "dofs": None if self.dofs is None else self.dofs.to_data(),
                "representation_identity": self.representation_identity,
                "affine": self.affine, "affine_origin": self.affine_origin}

    @classmethod
    def from_data(cls, data, *, finite_type=None, dofs=None, space=None):
        """Restore scientific declarations; dependent finite types need authenticated witnesses."""
        from pops.model.spaces import Space
        from pops._ir.quantity import PhysicalDimension
        from pops.time.variant_types import ValueType
        if data.get("contract") != "pops.scientific-type@1":
            raise ValueError("invalid scientific type contract")
        finite_type = ValueType.from_data(data["finite_type"]) if finite_type is None else finite_type
        if _data_identity(finite_type.to_data()) != _data_identity(data["finite_type"]):
            raise TypeError("scientific finite type witness differs from stored declaration")
        if data["dofs"] is not None:
            if dofs is None:
                raise TypeError("scientific dof restoration requires an authenticated domain witness")
            if _data_identity(dofs.to_data()) != _data_identity(data["dofs"]):
                raise TypeError("scientific dof witness differs from stored declaration")
        if space is None:
            definition = dict(data["space"])
            if definition.pop("kind") != "space":
                raise TypeError("scientific state space restoration requires an authenticated witness")
            if "representation_data" in definition:
                raise TypeError("scientific representation data restoration requires an exact Space witness")
            space = Space(**definition)
        if _data_identity(space.to_data()) != _data_identity(data["space"]):
            raise TypeError("scientific Space witness differs from stored declaration")
        return cls(finite_type, space, data["kind"], tuple(data["variance"]),
                   tuple(PhysicalDimension.from_data(dict(x)) for x in data["coordinate_units"]),
                   data["measure"], dofs, data["representation_identity"],
                   data.get("affine", "linear"), data.get("affine_origin"))

    def __eq__(self, other):
        return type(other) is type(self) and _data_identity(self.to_data()) == _data_identity(other.to_data())

    def __hash__(self):
        return hash(_data_identity(self.to_data()))

    def require_compatible(self, other, *, where="scientific compatibility"):
        if type(other) is not ScientificType or self != other:
            raise TypeError(where + ": support, units, variance, measure, representation or finite authority differ")


@dataclass(frozen=True, slots=True)
class ScientificValue:
    """A finite SSA plus declared scientific meaning; no implicit numeric cast.

    Declaring meaning on a bare scalar/array is an authoring assertion. Passing an
    existing ScientificValue cannot silently change that declaration. This object
    proves compatibility of declarations, not analytical validity of the model.
    """

    value: Any
    scientific_type: ScientificType
    __pops_ir_immutable__ = True

    def __post_init__(self):
        from pops.time.values import ProgramValue
        if type(self.scientific_type) is not ScientificType:
            raise TypeError("ScientificValue requires an explicit ScientificType")
        if isinstance(self.value, ScientificValue):
            self.scientific_type.require_compatible(self.value.scientific_type)
            object.__setattr__(self, "value", self.value.value)
        if not isinstance(self.value, ProgramValue) or not self.scientific_type.finite_type.matches(self.value):
            raise TypeError("scientific value has a different finite type or authority")
        if self.value.space is not None and self.value.space != self.scientific_type.space:
            raise TypeError("scientific annotation cannot replace an existing SSA physical Space")
        from pops.scientific_graph import bind_scientific_value
        object.__setattr__(self, "value", bind_scientific_value(self.value, self.scientific_type))


def require_scientific_output(value, requested_type, program):
    """Realize integration boundary: validate meaning before extracting finite SSA."""
    from pops.time._program.value_validation import require_owned
    if type(value) is not ScientificValue or type(requested_type) is not ScientificType:
        raise TypeError("a scientific output requires ScientificValue and requested ScientificType")
    requested_type.require_compatible(value.scientific_type, where="realized output")
    require_owned(program, value.value, "scientific output")
    if not requested_type.finite_type.matches(value.value):
        raise TypeError("scientific output finite authority changed")
    return value.value


@dataclass(frozen=True, slots=True, init=False)
class ScientificMap:
    """A closed GraphFunction with scientific input/output annotations.

    Calling the map actually inlines its equations into the native Program.
    The annotations specify the intended mathematical interpretation; claims
    state what has been verified about that interpretation.
    """

    function: Any
    inputs: Any
    output: ScientificType
    claims: tuple[Any, ...]
    strict: bool
    conversion: Any
    checked_contract: Any

    def __init__(self, function, *, inputs, output, claims=(), strict=False, conversion=None):
        from pops.graph_functions import GraphFunction
        from pops.time.variant_types import Product
        from pops.claims import claims_manifest
        if type(function) is not GraphFunction:
            raise TypeError("ScientificMap requires a closed GraphFunction, not an opaque callable")
        if not isinstance(inputs, Mapping) or any(type(v) is not ScientificType for v in inputs.values()):
            raise TypeError("ScientificMap inputs require named ScientificTypes")
        if type(output) is not ScientificType:
            raise TypeError("ScientificMap output requires ScientificType")
        if function.inputs != Product(**{k: v.finite_type for k, v in inputs.items()}):
            raise TypeError("scientific input annotations differ from the finite graph signature")
        if function.output != output.finite_type:
            raise TypeError("scientific output annotation differs from the finite graph signature")
        claims = tuple(claims)
        claims_manifest(claims)
        if type(strict) is not bool:
            raise TypeError("ScientificMap strict must be an explicit Bool")
        if conversion is not None and not strict:
            raise TypeError("an explicit scientific conversion requires strict graph checking")
        checked_contract = None
        if strict:
            from pops.scientific_graph import check_scientific_graph
            checked_contract = check_scientific_graph(function, inputs=inputs, output=output,
                                                      conversion=conversion)
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "inputs", MappingProxyType(dict(inputs)))
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "strict", strict)
        object.__setattr__(self, "conversion", conversion)
        object.__setattr__(self, "checked_contract", checked_contract)

    def __call__(self, program, **bindings):
        from pops.time._authoring import authoring_transaction
        if set(bindings) != set(self.inputs):
            raise TypeError("scientific map requires exactly its declared named inputs")
        finite = {key: require_scientific_output(bindings[key], typ, program)
                  for key, typ in self.inputs.items()}
        with authoring_transaction(program):
            from pops.scientific_graph import scientific_map_scope
            with scientific_map_scope(strict=self.strict, conversion=self.conversion):
                value = self.function(program, **finite)
            if self.checked_contract is not None:
                from pops.scientific_graph import bind_scientific_value
                value = bind_scientific_value(value, self.output, contract=self.checked_contract, bindings=finite)
            else:
                # Explicit legacy boundary: retained for existing scientific maps, not a proof
                # that the finite body's dimensions or interpretation were checked.
                from pops.scientific_graph import ANNOTATION, CONTRACT, APPLICATION, _body_hash
                attrs = dict(value.attrs, **{ANNOTATION: self.output.to_data(),
                    CONTRACT: {"mode": "legacy", "graph_identity": self.function.identity,
                               "body_hash": _body_hash(value, finite)},
                    APPLICATION: {"input_values": tuple(finite.items())}})
                value = program._replace_value(value, attrs=attrs)
            return ScientificValue(value, self.output)

    def to_data(self):
        from pops.claims import claims_manifest
        return {"contract": "pops.scientific-map@1", "function": self.function.to_data(),
                "inputs": {key: value.to_data() for key, value in self.inputs.items()},
                "output": self.output.to_data(), "claims": claims_manifest(self.claims),
                "checking": "strict" if self.strict else "legacy_annotations",
                "scientific_contract": None if self.checked_contract is None else self.checked_contract.to_data()}


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Representation:
    """Legacy descriptor, or an executable representation of a physical Space.

    Scientific form: ``Representation(space, dofs=D, kind=..., encode=map,
    decode=map, pairing=map)``. Encode/decode must have named input ``value``;
    pairing has ``left`` and ``right``. Signature compatibility never implies
    that encode/decode are inverses or that a pairing is positive definite.
    """

    name: str
    data: Any = None
    space: Any = None
    dofs: Any = None
    kind: Any = None
    encode: Any = None
    decode: Any = None
    pairing: Any = None

    category = "representation"

    def __init__(self, name, data=None, *, dofs=None, kind=None, encode=None, decode=None, pairing=None):
        from pops.model.spaces import Space
        scientific = isinstance(name, Space)
        if scientific:
            space = name
            if data is not None:
                raise TypeError("scientific representations derive data from their finite maps")
            if (type(encode) is not ScientificMap or type(decode) is not ScientificMap
                    or type(pairing) is not ScientificMap):
                raise TypeError("encode, decode and pairing require executable ScientificMaps")
            if set(encode.inputs) != {"value"} or set(decode.inputs) != {"value"}:
                raise TypeError("encode/decode require exactly the scientific input 'value'")
            encode.output.require_compatible(decode.inputs["value"], where="encoded space")
            decode.output.require_compatible(encode.inputs["value"], where="decoded space")
            if decode.output.space != space or encode.output.kind != kind:
                raise TypeError("representation Space/kind disagree with the declared finite maps")
            if dofs != encode.output.dofs:
                raise TypeError("representation dofs differ from the encoded dof authority")
            if set(pairing.inputs) != {"left", "right"}:
                raise TypeError("pairing requires exactly 'left' and 'right' scientific inputs")
            encode.output.require_compatible(pairing.inputs["left"], where="pairing left space")
            if pairing.output.finite_type.signature["vtype"] != "scalar":
                raise TypeError("pairing must return a finite scalar with explicit scientific units")
            if any(typ.representation_identity is not None for typ in (
                    encode.output, decode.inputs["value"], *pairing.inputs.values())):
                raise TypeError("define representation coordinate maps with unqualified scientific types")
            definition = {"contract": "pops.executable-representation-definition@1",
                          "space": space.to_data(),
                          "dofs": None if dofs is None else dofs.to_data(), "kind": kind,
                          "encode": encode.to_data(), "decode": decode.to_data(),
                          "pairing": pairing.to_data()}
            identity = "pops.representation:sha256:" + hashlib.sha256(
                bytes.fromhex(_data_identity(definition))).hexdigest()
            # Coefficients and their declared dual belong to the actual finite
            # maps. Equal widths/kinds do not identify two different bases.
            coefficient = replace(encode.output, representation_identity=identity)
            dual = replace(pairing.inputs["right"], representation_identity=identity)
            encode = ScientificMap(encode.function, inputs=encode.inputs,
                                   output=coefficient, claims=encode.claims, strict=encode.strict,
                                   conversion=encode.conversion)
            decode = ScientificMap(decode.function, inputs={"value": coefficient},
                                   output=decode.output, claims=decode.claims, strict=decode.strict,
                                   conversion=decode.conversion)
            pairing = ScientificMap(pairing.function, inputs={"left": coefficient, "right": dual},
                                    output=pairing.output, claims=pairing.claims, strict=pairing.strict,
                                    conversion=pairing.conversion)
            name = kind
            data = {"contract": "pops.executable-representation@1", "space": space.to_data(),
                    "dofs": None if dofs is None else dofs.to_data(), "kind": kind,
                    "identity": identity,
                    "encode": encode.to_data(), "decode": decode.to_data(), "pairing": pairing.to_data()}
        else:
            space = None
            if any(value is not None for value in (dofs, kind, encode, decode, pairing)):
                raise TypeError("executable representation construction requires a physical Space")
        if not isinstance(name, str) or not name or name.strip() != name:
            raise TypeError("Representation.name must be canonical non-empty text")
        for key, value in dict(name=name, data=_freeze_data(data), space=space, dofs=dofs,
                               kind=kind, encode=encode, decode=decode, pairing=pairing).items():
            object.__setattr__(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        result = {"schema_version": 1, "category": self.category, "name": self.name}
        if self.data is not None:
            result["data"] = _data_to_wire(self.data)
        return result

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self.name == other.name and _data_identity(self.data) == _data_identity(other.data)

    def __hash__(self):
        return hash((self.name, _data_identity(self.data)))

    canonical_identity = to_dict
    inspect = to_dict


def Conservative() -> Representation:
    """Components are conservative densities over their control volumes."""

    return Representation("conservative")


def Primitive() -> Representation:
    """Components are primitive physical variables."""

    return Representation("primitive")


__all__ = ["Representation", "Conservative", "Primitive", "ScientificType",
           "ScientificValue", "ScientificMap", "require_scientific_output"]
