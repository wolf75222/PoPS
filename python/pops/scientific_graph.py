"""Check scientific meaning on the actual finite Program graph before erasure.

This module neither evaluates equations nor introduces a second IR. It elaborates an existing
GraphFunction into a temporary Program, then propagates physical dimensions and representation
contracts over its SSA. Unknown operations fail closed. Analytical meaning of an explicitly
declared conversion remains an author obligation; its numerical body is always inspected.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeAlias
import hashlib
import json

from pops._ir.quantity import PhysicalDimension


DIMENSIONLESS = PhysicalDimension()
ANNOTATION = "scientific_type_v14"
CONTRACT = "scientific_graph_v14"
APPLICATION = "scientific_application_v14"
_MAP_SCOPE: ContextVar[tuple[bool, ScientificConversion | None] | None] = ContextVar(
    "pops_scientific_map_scope", default=None)


@contextmanager
def scientific_map_scope(*, strict, conversion):
    token = _MAP_SCOPE.set((strict, conversion))
    try:
        yield
    finally:
        _MAP_SCOPE.reset(token)


def _key(value):
    from pops.time.canonical_data import _json_ready
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _unit(a, b, sign=1):
    powers = dict(a.powers)
    for name, exponent in b.powers:
        powers[name] = powers.get(name, Fraction()) + sign * exponent
    return PhysicalDimension(tuple(powers.items()))


def _power(a, exponent):
    return PhysicalDimension(tuple((name, power * exponent) for name, power in a.powers))


@dataclass(frozen=True)
class ScientificConversion:
    """Explicit interpretation change of this exact finite map, not a unit-check exemption.

Use for encode/decode, transfer, pairing, contraction or an affine coordinate conversion.
The justification identifies its mathematical interpretation and remaining assumptions. The
checker still verifies every arithmetic operation and the resulting physical dimensions.
``purpose='domain'`` declares an analytical domain obligation for this exact graph. It is
not a proof or an implicit runtime check, and cannot change units or representation.
"""
    graph_identity: str
    purpose: str
    justification: str

    def __post_init__(self):
        if len(self.graph_identity) != 64 or any(c not in "0123456789abcdef" for c in self.graph_identity):
            raise ValueError("scientific conversion requires the exact graph identity")
        if self.purpose not in ("representation", "transfer", "pairing", "contraction", "affine", "domain"):
            raise ValueError("unknown scientific conversion purpose")
        if not isinstance(self.justification, str) or not self.justification.strip():
            raise ValueError("scientific conversion needs an explicit mathematical justification")

    def to_data(self):
        return dict(graph_identity=self.graph_identity, purpose=self.purpose,
                    justification=self.justification)


@dataclass(frozen=True)
class _Scalar:
    unit: PhysicalDimension
    context: tuple | None = None
    affine: str = "linear"
    origin: str | None = None
    literal: int | float | Fraction | Decimal | None = None
    domain: str = "real"


@dataclass(frozen=True)
class _Array:
    domain: str
    element: _Meaning


@dataclass(frozen=True)
class _Sum:
    members: tuple[tuple[str, _Meaning], ...]


@dataclass(frozen=True)
class _Capability:
    kind: str
    payload: _Meaning


@dataclass(frozen=True)
class _Discrete:
    kind: str


class _Bottom:
    """The unreachable member of a finite scientific sum."""


_BOTTOM = _Bottom()
_Meaning: TypeAlias = (_Scalar | _Array | _Sum | _Capability | _Discrete | _Bottom
                       | dict[str, "_Meaning"] | tuple["_Meaning", ...])


class _UnboundScientificInput(TypeError):
    """Finite graph loading has no authority to invent a formal input's units."""



def _literal(value):
    if hasattr(value, "to_python"):
        value = value.to_python()
    return _Scalar(DIMENSIONLESS, literal=value)


def _domain(signature):
    return _key({key: value for key, value in signature.items()
                 if key not in ("element", "vtype", "point")})


def _meaning(scientific_type) -> _Meaning:
    from pops.representations import ScientificType
    if type(scientific_type) is not ScientificType:
        raise TypeError("scientific schema leaves must be ScientificType")
    return _meaning_data(scientific_type.to_data())


def _meaning_data(data) -> _Meaning:
    """Read inert scientific meaning; this never issues a finite domain/field authority."""
    if data.get("contract") != "pops.scientific-type@1":
        raise TypeError("unknown scientific type metadata")
    units = iter(PhysicalDimension.from_data(dict(unit)) for unit in data["space"]["units"])
    context = (_key(data["space"]["support"]), data["kind"],
               _key(data["variance"]), _key(data["measure"]),
               data["representation_identity"], _key(data["coordinate_units"]),
               _key(data["space"]["sampling"]), _key(data["space"]["representation"]))
    domain = data["space"]["domain"]
    def visit(signature) -> _Meaning:
        kind = signature["vtype"]
        if kind == "product":
            return {key: visit(child) for key, child in signature["members"]}
        if kind == "variant":
            return _Sum(tuple((key, visit(child)) for key, child in signature["members"]))
        if kind == "array":
            return _Array(_domain(signature), visit(signature["element"]))
        if kind in ("index", "index_set", "bool", "int64"):
            return _Discrete(kind)
        if kind in ("state", "rhs", "scalar_field"):
            return tuple(_Scalar(unit, context, data.get("affine", "linear"),
                                 data.get("affine_origin"), domain=domain) for unit in units)
        if kind == "scalar":
            try:
                unit = next(units)
            except StopIteration as exc:
                raise TypeError("scientific component units do not cover the finite real leaves") from exc
            return _Scalar(unit, context, data.get("affine", "linear"),
                           data.get("affine_origin"), domain=domain)
        raise TypeError("no scientific schema rule for finite type " + kind)
    result = visit(data["finite_type"])
    if next(units, None) is not None:
        raise TypeError("scientific component units exceed the finite real leaves")
    return result


def _untyped(signature) -> _Meaning:
    kind = signature["vtype"]
    if kind == "product":
        return {key: _untyped(child) for key, child in signature["members"]}
    if kind == "variant":
        return _Sum(tuple((key, _untyped(child)) for key, child in signature["members"]))
    if kind in ("bool", "int64", "index", "index_set"):
        return _Discrete(kind)
    if kind == "scalar": return _Scalar(DIMENSIONLESS)
    if kind == "array": return _Array(_domain(signature), _untyped(signature["element"]))
    if kind in ("state", "rhs", "scalar_field"):
        return tuple(_Scalar(DIMENSIONLESS) for _ in signature["space"].components)
    raise TypeError("untyped scientific real payload needs a declared schema")


def _compatible(a, b, *, interpretation=False, domain_assumption=False) -> _Meaning:
    if a is _BOTTOM or b is _BOTTOM:
        return b if a is _BOTTOM else a
    if isinstance(a, _Scalar) and isinstance(b, _Scalar):
        # A zero with unspecified units is not evidence of membership in an
        # arbitrary refined domain. Branch unification uses this same check.
        if a.domain != b.domain and not domain_assumption:
            raise TypeError("scientific domains differ; provide an explicit domain obligation")
        if a.literal == 0 and a.context is None and b.affine != "point":
            return b
        if b.literal == 0 and b.context is None and a.affine != "point":
            return a
        if a.unit != b.unit:
            raise TypeError("scientific dimensions differ")
        if a.affine != b.affine or a.origin != b.origin:
            raise TypeError("scientific affine value and increment differ")
        if not interpretation and a.context is not None and b.context is not None and a.context != b.context:
            raise TypeError("scientific support, representation, variance or measure differ")
        return replace(a, context=a.context or b.context, literal=None)
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return {key: _compatible(a[key], b[key], interpretation=interpretation,
                                 domain_assumption=domain_assumption) for key in a}
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(_compatible(x, y, interpretation=interpretation,
                                 domain_assumption=domain_assumption) for x, y in zip(a, b))
    if isinstance(a, _Array) and isinstance(b, _Array) and a.domain == b.domain:
        return _Array(a.domain, _compatible(a.element, b.element, interpretation=interpretation,
                                           domain_assumption=domain_assumption))
    if isinstance(a, _Sum) and isinstance(b, _Sum) and dict(a.members).keys() == dict(b.members).keys():
        aa, bb = dict(a.members), dict(b.members)
        return _Sum(tuple((key, _compatible(aa[key], bb[key], interpretation=interpretation,
                                           domain_assumption=domain_assumption)) for key in aa))
    if isinstance(a, _Capability) and isinstance(b, _Capability) and a.kind == b.kind:
        return _Capability(a.kind, _compatible(a.payload, b.payload, interpretation=interpretation,
                                               domain_assumption=domain_assumption))
    if isinstance(a, _Discrete) and a == b:
        return a
    raise TypeError("scientific finite structures or domain authorities differ")


def _real(value):
    if not isinstance(value, _Scalar):
        raise TypeError("scientific arithmetic requires a real scalar component")
    return value


def _product(value: _Meaning) -> dict[str, _Meaning]:
    if not isinstance(value, dict):
        raise TypeError("scientific projection requires a Product")
    return value


def _sum(value: _Meaning) -> _Sum:
    if not isinstance(value, _Sum):
        raise TypeError("scientific projection requires a Variant")
    return value


def _arithmetic(op, a, b=None, *, conversion=None):
    a = _real(a)
    if op == "neg":
        if a.affine == "point":
            raise TypeError("an affine point cannot be negated without an explicit coordinate map")
        return replace(a, literal=None if a.literal is None else -a.literal, domain="real")
    if op in ("exp", "expm1", "log", "sin", "cos"):
        if a.unit != DIMENSIONLESS or a.affine == "point":
            raise TypeError(op + " requires a dimensionless linear scientific argument")
        return replace(a, literal=None, domain="real")
    if op in ("sqrt", "abs"):
        if a.affine == "point":
            raise TypeError(op + " cannot consume an affine point")
        return replace(a, unit=_power(a.unit, Fraction(1, 2)) if op == "sqrt" else a.unit,
                       literal=None, domain="real")
    b = _real(b)
    if op in ("add", "sub", "min", "max", "compare"):
        if a.literal == 0 and a.context is None and b.affine != "point":
            return replace(b, literal=None, domain=b.domain if op == "add" else "real")
        if b.literal == 0 and b.context is None:
            return replace(a, literal=None, domain=a.domain if op in ("add", "sub") else "real")
        if a.unit != b.unit:
            raise TypeError("scientific dimensions differ in " + op)
        if a.context is not None and b.context is not None and a.context != b.context:
            raise TypeError("scientific support, representation, variance or measure differ in " + op)
        if a.origin != b.origin and (a.affine != "linear" or b.affine != "linear"):
            raise TypeError("affine operands belong to different origins")
        if op == "add" and a.affine == b.affine == "point":
            raise TypeError("adding two affine points is not defined")
        if op == "sub" and b.affine == "point" and a.affine != "point":
            raise TypeError("subtracting an affine point from an increment is not defined")
        affine = ("increment" if op == "sub" and a.affine == b.affine == "point"
                  else "point" if "point" in (a.affine, b.affine) else a.affine)
        return replace(a, context=a.context or b.context, affine=affine, literal=None, domain="real")
    if a.affine == "point" or b.affine == "point":
        if conversion is None or conversion.purpose != "affine":
            raise TypeError("affine point arithmetic requires an explicit finite coordinate conversion")
    if a.context is not None and b.context is not None and a.context != b.context:
        if conversion is None or conversion.purpose not in ("pairing", "contraction"):
            raise TypeError("multiplication across scientific supports/representations requires an explicit map")
        if a.context[0] != b.context[0]:
            raise TypeError("pairing/contraction operands must share a support; first provide a transfer")
    if op == "pow":
        if b.unit != DIMENSIONLESS or b.context is not None:
            raise TypeError("scientific exponent must be dimensionless")
        if a.unit == DIMENSIONLESS:
            unit = DIMENSIONLESS
        else:
            if b.literal is None:
                raise TypeError("a dimensional base requires a constant rational exponent")
            unit = _power(a.unit, Fraction(b.literal))
    elif op in ("mul", "div"):
        unit = _unit(a.unit, b.unit, 1 if op == "mul" else -1)
    else:
        raise TypeError("no scientific dimension rule for operation " + op)
    context = a.context or b.context
    return _Scalar(unit, context)


class _Analyzer:
    def __init__(self, bindings=None, *, conversion=None, allow_legacy=False, scientific_ids=None):
        self.known = dict(bindings or {})
        self.conversion = conversion
        self.allow_legacy = allow_legacy
        self.scientific_ids = scientific_ids
        self.visited = set()
        self.contract_receipts = None
        self.detached_bindings = {}

    def scientific_dependency(self, value):
        value = value.prog._canonical_value(value)
        if self.scientific_ids is None:
            declared = _scientific_ancestor(value)
        else:
            declared = (id(value.prog), value.id) in self.scientific_ids
        if declared:
            return True
        # Formal input meanings live in bindings, not in mutable SSA annotations.
        # They remain authoritative when a body tries to relabel a projection.
        pending, seen = [value], set()
        while pending:
            node = pending.pop()
            node = node.prog._canonical_value(node)
            key = (id(node.prog), node.id)
            if key in seen:
                continue
            seen.add(key)
            if node.id in self.known:
                if _has_context(self.known[node.id]):
                    return True
                continue
            pending.extend(_dependencies(node))
        return False

    def block(self, values):
        # Closed GraphFunctions retain checked operations even when their values
        # are unused (they may fail). Inspect those operations under the same
        # bound region arguments; output-only inference cannot type the program.
        for value in values:
            self.get(value)

    def expression(self, node, inputs) -> _Meaning:
        values = []
        for item in node.attrs["expression_nodes"]:
            op = item[0]
            if op == "input":
                source = inputs[item[1]]
                value = source[item[2]] if isinstance(source, tuple) else source
            elif op == "component": value = inputs[item[1]][item[2]]
            elif op == "literal": value = _literal(item[1])
            elif op == "where":
                value = _compatible(values[item[2]], values[item[3]])
            elif op == "compare":
                _arithmetic("compare", values[item[2]], values[item[3]])
                value = _Discrete("bool")
            else:
                value = _arithmetic(op, values[item[1]], values[item[2]] if len(item) > 2 else None,
                                    conversion=self.conversion)
            values.append(value)
        result = tuple(values[index] for index in node.attrs["expressions"])
        return result[0] if node.vtype == "scalar" else result

    def get(self, value) -> _Meaning:
        value = value.prog._canonical_value(value)
        if value.id in self.known:
            if ANNOTATION in value.attrs:
                _compatible(self.known[value.id], _meaning_data(value.attrs[ANNOTATION]))
            return self.known[value.id]
        if value.id in self.visited:
            raise TypeError("cyclic scientific inference without a declared region argument")
        self.visited.add(value.id)
        op, a = value.op, value.attrs
        if ANNOTATION in a and a.get(CONTRACT, {}).get('mode') == 'legacy' and self.allow_legacy:
            _legacy_body(value)
            result = _meaning_data(a[ANNOTATION])
            self.known[value.id] = result
            self.visited.remove(value.id)
            return result
        if ANNOTATION in a and CONTRACT not in a and not any(
                self.scientific_dependency(child) for child in _dependencies(value)):
            # The caller declares an external datum once. No scientific derived edge
            # is permitted to enter this boundary and silently acquire another meaning.
            result = _meaning_data(a[ANNOTATION])
            self.known[value.id] = result
            self.visited.remove(value.id)
            return result
        x = [self.get(child) for child in value.inputs]
        if ANNOTATION in a:
            result = _meaning_data(a[ANNOTATION])
            # An input declaration may assert the meaning of raw constants/data. A derived
            # value with scientific dependencies must still be verified, never just relabeled.
            if a.get(CONTRACT) is None and any(self.scientific_dependency(child) for child in _dependencies(value)):
                inferred = self.operation(value, x)
                _compatible(inferred, result)
            elif a.get(CONTRACT) is not None:
                contract = a[CONTRACT]
                if contract.get("mode") != "strict":
                    raise TypeError("legacy scientific annotation is not a checked finite graph")
                application = a.get(APPLICATION)
                if application is None:
                    raise TypeError("strict scientific contract has no exact applied body")
                references = _application_inputs(value.prog, application)
                if set(references) != set(contract["inputs"]):
                    raise TypeError("scientific application requires its exact named inputs")
                types = {key: _restore_scientific_type(contract["inputs"][key], child)
                         for key, child in references.items()}
                if getattr(value.prog, "_compiled_detached", False):
                    # Registry-free compiler nodes carry the proof issued before
                    # erasure. They never become live authoring capabilities.
                    from pops.time._program.scientific_compile_proof import require_scientific_binding
                    bound = self.detached_bindings.get(id(value.prog))
                    if bound is None:
                        bound = require_scientific_binding(value.prog)
                        self.detached_bindings[id(value.prog)] = bound
                    bound.require_node(value, contract)
                else:
                    output = _restore_scientific_type(contract["output"], value)
                    check_scientific_contract_data(contract, inputs=types, output=output)
                    expected = _application_body_hash(contract, references, inputs=types, output=output)
                    if _body_hash(value, references, canonical_aliases=True) != expected:
                        raise TypeError("scientific conversion contract belongs to a different applied body")
                if self.contract_receipts is not None:
                    self.contract_receipts[value.id] = _key(contract)
                previous = self.conversion
                self.conversion = (None if contract["conversion"] is None
                                   else ScientificConversion(**contract["conversion"]))
                try:
                    inferred = self.operation(value, x)
                finally:
                    self.conversion = previous
                conversion_data = contract.get("conversion")
                domain_assumption = conversion_data is not None and conversion_data["purpose"] == "domain"
                _compatible(inferred, result,
                            interpretation=conversion_data is not None and not domain_assumption,
                            domain_assumption=domain_assumption)
        else:
            result = self.operation(value, x)
        self.known[value.id] = result
        self.visited.remove(value.id)
        return result

    def operation(self, value, x) -> _Meaning:
        from pops.time._program.structured_control import value_signature
        a, op = value.attrs, value.op
        if op == 'provenance_bind':
            from pops.ledger_provenance import validate_contract_operation
            validate_contract_operation(value.inputs, a, value_signature(value))
            return x[0]
        if op == "graph_argument":
            raise _UnboundScientificInput("scientific graph formal needs authoritative input meanings")
        if op == "continuation_open":
            validate_scientific_packet_data(a["schema"])
            declarations = a["schema"].get("scientific", ())
            if declarations:
                expected = _packet_apply(_untyped(a["schema"]["read"]), declarations, initial=True)
                payload = _declare_initial_packet(x[0], expected)
            else:
                payload = x[0]
            return _Capability("base", payload)
        if op == "continuation_read": return x[0].payload
        if op == "continuation_freeze":
            declarations = value_signature(value.inputs[0])["schema"].get("scientific", ())
            expected = _packet_apply(_untyped(value_signature(value.inputs[0])["schema"]["write"]),
                                     declarations, initial=True) if declarations else x[2]["continuation"]
            envelope = dict(x[2], continuation=_compatible(x[2]["continuation"], expected))
            return _Capability("candidate", dict(base=x[0], proposal=x[1], envelope=envelope))
        if op in ("continuation_candidate_value", "continuation_proposal", "continuation_candidate_base"):
            return x[0].payload[{"continuation_candidate_value":"envelope",
                "continuation_proposal":"proposal", "continuation_candidate_base":"base"}[op]]
        if op == "continuation_attest":
            return _Capability("assessment", dict(candidate=x[0], checks=x[1], accepted=x[2]))
        if op == "continuation_assessment_ok": return x[0].payload["accepted"]
        if op == "continuation_assessment_checks": return x[0].payload["checks"]
        if op == "continuation_accept": return _Capability("accepted", x[0])
        if op == "continuation_accepted_candidate": return x[0].payload
        if op == "continuation_publish": return x[0]
        if op == "continuation_child_base":
            return _Capability("base", x[0].payload.payload["envelope"]["continuation"])
        if op == "continuation_consume_increment": return x[0].payload.payload["envelope"]["increment"]
        if op == "continuation_transform": return x[1]
        if op == "state":
            if value.space is None or any(unit is None for unit in value.space.units):
                raise TypeError("scientific native field input needs its declared component units")
            return tuple(_Scalar(unit) for unit in value.space.units)
        if op == "product": return dict(zip(a["fields"], x))
        if op == "product_get": return x[0][a["field"]]
        if op == "bundle_validate": return x[0]
        if op == "bundle_remap_validate":
            # Native validation checks the new dependent Bundle. Its scientific
            # meaning comes from that candidate's actual equations, never from
            # the previous Bundle or the output annotation.
            from pops.bundles import validate_bundle_signature
            from pops.time._program.structured_control import same_signature
            signature = value_signature(value)
            validate_bundle_signature(signature)
            if len(x) != 2 or any(not same_signature(signature, value_signature(child))
                                  for child in value.inputs):
                raise TypeError("scientific remap requires its exact complete Bundle candidate")
            return x[1]
        if op in ("region_argument",): return x[0]
        if op in ("int64_constant", "bool_constant", "region_index", "finite_index_argument",
                  "finite_range_set", "finite_size", "finite_domain", "finite_index",
                  "finite_index_value", "finite_same_domain", "distributed_domain", "distributed_index_argument",
                  "distributed_index", "distributed_gid", "distributed_rank", "distributed_value_domain",
                  "distributed_size", "distributed_offset_argument"):
            return _Discrete(value.vtype)
        if op in ("int64_op", "int64_compare"):
            if any(not isinstance(child, _Discrete) for child in x):
                raise TypeError("integer arithmetic cannot erase scientific dimensions")
            return _Discrete(value.vtype)
        if op == "int64_cast":
            if _real(x[0]).unit != DIMENSIONLESS:
                raise TypeError("integer conversion requires dimensionless input")
            return _Discrete("int64")
        if op == "finite_real":
            if x: return x[0] if isinstance(x[0], _Scalar) else _literal(None)
            return _literal(a.get("value"))
        if op in ("scalar_expression", "pointwise_expression"):
            return self.expression(value, x)
        if op == "scalar_op":
            args = [x[item] if kind == "v" else _literal(item) for kind, item in a["operands"]]
            return _arithmetic(a["fn"], *args, conversion=self.conversion)
        if op == "compare":
            rhs = x[1] if len(x) > 1 else _literal(a["rhs"])
            _arithmetic("compare", x[0], rhs)
            return _Discrete("bool")
        if op == "scalar_isfinite": return _Discrete("bool")
        if op == "bool_op":
            from pops.time._program.structured_control import validate_boolean_operation
            validate_boolean_operation(value.inputs, a, value_signature(value))
            if any(not isinstance(child, _Discrete) for child in x):
                raise TypeError("boolean algebra cannot erase scientific dimensions")
            return _Discrete("bool")
        if op in ("branch", "result_match"):
            self.block(a.get("true_block", ())); self.block(a.get("false_block", ()))
            return _compatible(self.get(a["true_result"]), self.get(a["false_result"]))
        if op in ("finite_at", "distributed_local_at"):
            if not isinstance(x[0], _Array): raise TypeError("scientific access requires Array")
            return x[0].element
        if op == "finite_map":
            self.block(a.get("body_block", ()))
            return _Array(_domain(value_signature(value)), self.get(a["body"]))
        if op == "distributed_map":
            self.block(a.get("body_block", ()))
            body = dict(_sum(self.get(a["body"])).members)
            signature = dict(value_signature(value)["members"])
            runtime = dict(dict(signature["err"]["members"])["reason"]["members"])["runtime"]
            error: dict[str, _Meaning] = dict(rank=_Discrete("int64"), gid=_Discrete("int64"),
                         reason=_Sum((("runtime", _untyped(runtime)), ("user", body["err"]))))
            return _Sum((("ok", _Array(_domain(signature["ok"]), body["ok"])), ("err", error)))
        if op == "distributed_requests":
            self.block(a.get("body_block", ()))
            return _Array(_domain(value_signature(value)), self.get(a["body"]))
        if op in ("distributed_import", "distributed_import_csr"):
            return _Array(_domain(value_signature(value)), x[0].element)
        if op == "distributed_csr":
            self.block(a.get("body_block", ()))
            return _Discrete("csr")
        if op in ("distributed_reduce", "distributed_scatter_reduce", "distributed_scan"):
            initial = x[-1]
            self.known[a["body_argument"].id] = initial
            self.known[a["body_value"].id] = x[0].element
            self.block(a.get("body_block", ()))
            body = self.get(a["body"])
            result = _compatible(initial, _product(body)["state"] if op == "distributed_scan" else body)
            if op == "distributed_reduce": return result
            signature = value_signature(value)
            if op == "distributed_scatter_reduce": return _Array(_domain(signature), result)
            return dict(state=result, values=_Array(_domain(dict(signature["members"])["values"]), _product(body)["value"]))
        if op in ("structured_fold", "finite_scan"):
            self.known[a["body_argument"].id] = x[0]
            self.block(a.get("body_block", ()))
            result = self.get(a["body"])
            if op == "structured_fold":
                return _compatible(x[0], result)
            product = _product(result)
            state = _compatible(x[0], product["state"])
            return dict(state=state, values=_Array(_domain(a["array_signature"]), product["value"]))
        if op == "variant_construct":
            return _Sum(tuple((tag, x[0] if tag == a["tag"] else _BOTTOM)
                              for tag, _ in value_signature(value)["members"]))
        if op == "variant_payload": return dict(x[0].members)[a["tag"]]
        if op == "variant_is": return _Discrete("bool")
        if op == "variant_unreachable": return _BOTTOM
        if op in ("variant_require_success", "result_payload", "result_success", "iteration_payload", "iteration_success"):
            return dict(x[0].members)["ok"]
        if op in ("result_ok",): return _Sum((("ok", x[0]), ("err", _BOTTOM)))
        if op == "result_error": return _Sum((("ok", _BOTTOM), ("err", _Discrete("failure"))))
        if op in ("result_is_ok", "iteration_is_ok"): return _Discrete("bool")
        if op == "capture_result":
            self.block(a.get("body_block", ()))
            body = self.get(a["body"])
            return body if isinstance(body, _Sum) else _Sum((("ok", body), ("err", _BOTTOM)))
        if op == "structured_iterate":
            self.known[a["cond_argument"].id] = self.known[a["body_argument"].id] = x[0]
            self.block(a.get("cond_block", ())); self.block(a.get("body_block", ()))
            if self.get(a["cond"]) != _Discrete("bool"):
                raise TypeError("scientific stop must be Bool")
            body = self.get(a["body"])
            if isinstance(body, _Sum): body = dict(body.members)["ok"]
            state = _compatible(x[0], body)
            return _Sum((("ok", dict(state=state, iterations=_Discrete("int64"))), ("err", _BOTTOM)))
        if op == "field_component": return x[0][a.get("component", a.get("index"))]
        if op == "vector_field": return tuple(x)
        raise TypeError("no checked scientific rule for Program operation " + op)


def _dependencies(value):
    """Numerical/region edges, excluding inert declarations and administrative metadata."""
    from pops.time.values import ProgramValue
    ignored = {ANNOTATION, CONTRACT, APPLICATION, "signature", "state_signature",
               "success_signature", "error_signature", "array_signature"}
    if value.op == 'provenance_bind':
        ignored.add('contract')
    found = {}
    support_only = value.attrs.get("support_only_inputs", ())
    for i, child in enumerate(value.inputs):
        if i not in support_only:
            found[child.id] = child
    pending = list(reversed(tuple(child for key, child in value.attrs.items() if key not in ignored)))
    seen_metadata = set()
    while pending:
        item = pending.pop()
        if isinstance(item, ProgramValue):
            found[item.id] = item
        elif isinstance(item, (Mapping, tuple, list)):
            if id(item) in seen_metadata:
                continue
            seen_metadata.add(id(item))
            children = tuple(item.values()) if isinstance(item, Mapping) else item
            pending.extend(reversed(children))
    return tuple(found.values())


def _scientific_ancestor(value, seen=None):
    seen = set() if seen is None else seen
    pending = [value]
    while pending:
        value = pending.pop()
        value = value.prog._canonical_value(value)
        if value.id in seen:
            continue
        seen.add(value.id)
        if (ANNOTATION in value.attrs
                or bool(value.attrs.get("schema", {}).get("scientific", ()))):
            return True
        pending.extend(reversed(_dependencies(value)))
    return False


def _scientific_reachability(values):
    """Exact dependency reachability, computed once per root validation.

    Propagate declared meaning along reverse edges. Cycles and shared regions
    require no recursive walk per node, and no cache survives a Program edit.
    """
    from collections import defaultdict, deque
    pending = list(values)
    nodes, parents, marked = {}, defaultdict(set), set()
    while pending:
        value = pending.pop()
        value = value.prog._canonical_value(value)
        key = (id(value.prog), value.id)
        if key in nodes:
            continue
        nodes[key] = value
        if ANNOTATION in value.attrs or value.attrs.get('schema', {}).get('scientific'):
            marked.add(key)
        for child in _dependencies(value):
            child = child.prog._canonical_value(child)
            parents[(id(child.prog), child.id)].add(key)
            pending.append(child)
    queue = deque(marked)
    while queue:
        for parent in parents[queue.popleft()]:
            if parent not in marked:
                marked.add(parent)
                queue.append(parent)
    return marked


def _has_context(value):
    if isinstance(value, _Scalar): return value.context is not None
    if isinstance(value, _Array): return _has_context(value.element)
    if isinstance(value, _Capability): return _has_context(value.payload)
    if isinstance(value, _Sum): return any(_has_context(x) for _, x in value.members)
    if isinstance(value, dict): return any(map(_has_context, value.values()))
    if isinstance(value, tuple): return any(map(_has_context, value))
    return False


def _body_hash(value, bindings, *, canonical_aliases=False):
    """Alpha-invariant numerical/effect body identity, with named actual inputs as leaves."""
    from pops.time.values import ProgramValue
    # Repeated named inputs deliberately denote one canonical SSA leaf. The last
    # declared name (sorted for strict applications) preserves the leaf format;
    # legacy applications retain their original ordering. Expected strict bodies replay
    # with the SAME alias partition, never inferred from numerical equality.
    names = sorted(bindings) if canonical_aliases else bindings
    stops = {(id(bindings[key].prog), bindings[key].prog._canonical_value(bindings[key]).id): key
             for key in names}
    memo, nodes = {}, []
    ignored = {ANNOTATION, CONTRACT, APPLICATION, "signature", "state_signature",
               "success_signature", "error_signature", "array_signature"}
    def encode(item):
        if isinstance(item, ProgramValue):
            item = item.prog._canonical_value(item)
            identity = (id(item.prog), item.id)
            if identity in stops: return {"argument": stops[identity]}
            if identity not in memo:
                ordinal = len(nodes)
                memo[identity] = ordinal
                nodes.append(None)
                nodes[ordinal] = dict(op=item.op, vtype=item.vtype,
                    inputs=[encode(child) for child in item.inputs],
                    attrs={key: encode(child) for key, child in item.attrs.items()
                           if key not in ignored and not key.endswith("_region")})
            return {"node": memo[identity]}
        if isinstance(item, Mapping): return {key: encode(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)): return [encode(child) for child in item]
        return item
    root = encode(value)
    return hashlib.sha256(_key(dict(root=root, nodes=nodes)).encode()).hexdigest()


def _application_inputs(program, application):
    from pops.time._program.value_validation import require_owned
    try:
        result = dict(application["input_values"])
        for value in result.values():
            require_owned(program, value, "scientific application input")
        return {key: program._canonical_value(value) for key, value in result.items()}
    except (KeyError, TypeError) as exc:
        raise TypeError("scientific application lost an exact named input") from exc


@dataclass(frozen=True)
class ScientificGraphContract:
    data: str

    def to_data(self): return json.loads(self.data)

    @property
    def identity(self): return hashlib.sha256(self.data.encode()).hexdigest()


def check_scientific_graph(function, *, inputs, output, conversion=None):
    """Check exact finite equations, then return a serializable contract for those bytes."""
    return _check_scientific_graph(function, inputs=inputs, output=output, conversion=conversion)


def _check_scientific_graph(function, *, inputs, output, conversion=None, aliases=None):
    """Replay the closed graph; optional aliases are private application specialization."""
    from pops.graph_functions import GraphFunction, _formals
    if type(function) is not GraphFunction:
        raise TypeError("scientific checker requires a closed GraphFunction")
    if conversion is not None and (type(conversion) is not ScientificConversion
                                  or conversion.graph_identity != function.identity):
        raise TypeError("scientific conversion belongs to a different finite graph")
    p, args = _formals(function.inputs)
    formal = {key: _meaning(typ) for key, typ in inputs.items()}
    analyzer = _Analyzer({args.id: formal}, conversion=conversion)
    formals = {key: args[key] for key in inputs}
    bindings = (formals if aliases is None else
                {key: formals[aliases[key]] for key in inputs})
    result = function(p, **bindings)
    actual = analyzer.get(result)
    analyzer.block(p._values)
    domain_assumption = conversion is not None and conversion.purpose == "domain"
    _compatible(actual, _meaning(output),
                interpretation=conversion is not None and not domain_assumption,
                domain_assumption=domain_assumption)
    data = dict(contract="pops.scientific-graph@1", mode="strict", graph=function.to_data(),
                graph_identity=function.identity,
                inputs={key: typ.to_data() for key, typ in inputs.items()}, output=output.to_data(),
                body_hash=_body_hash(result, bindings, canonical_aliases=True),
                conversion=None if conversion is None else conversion.to_data(),
                domain_obligations=([dict(classification="analytical_assumption",
                    graph_identity=function.identity, justification=conversion.justification,
                    inputs={key: typ.space.domain for key, typ in inputs.items()},
                    output=output.space.domain)] if conversion is not None and domain_assumption else []))
    return ScientificGraphContract(_key(data))


def check_scientific_contract_data(data, *, inputs=None, output=None):
    """Recheck serialized equations; a stored digest is never accepted as a proof by itself."""
    from pops.graph_functions import GraphFunction
    from pops.representations import ScientificType
    data = json.loads(_key(data))
    inputs = ({key: ScientificType.from_data(value) for key, value in data["inputs"].items()}
              if inputs is None else inputs)
    output = ScientificType.from_data(data["output"]) if output is None else output
    inputs = _ordered_scientific_inputs(data, inputs)
    from pops.time.variant_types import Product
    function = GraphFunction.from_data(data["graph"],
        inputs=Product(**{key: typ.finite_type for key, typ in inputs.items()}), output=output.finite_type)
    conversion = None if data["conversion"] is None else ScientificConversion(**data["conversion"])
    checked = check_scientific_graph(function, inputs=inputs, output=output, conversion=conversion)
    if _key(checked.to_data()) != _key(data):
        raise TypeError("scientific contract metadata differs from rechecked graph")
    return checked


def _ordered_scientific_inputs(data, inputs):
    """Keep the declared Product order; keyword/metadata mapping order is irrelevant."""
    names = tuple(key for key, _ in data["graph"]["inputs"]["members"])
    if (len(names) != len(set(names)) or set(names) != set(inputs)
            or set(names) != set(data["inputs"])):
        raise TypeError("scientific application requires its exact named inputs")
    return {key: inputs[key] for key in names}


def _restore_scientific_type(data, actual):
    """Restore dependent finite authorities from this exact application, not names."""
    from pops.representations import ScientificType
    from pops.time.variant_types import type_of, ValueType
    finite = type_of(actual)
    try:
        finite = ValueType.from_data(data["finite_type"])
    except ValueError:
        pass
    dofs = None
    if data["dofs"] is not None:
        dofs = ValueType({k: v for k, v in finite.signature.items() if k != "element"}
                         | {"vtype": "index_set"})
    return ScientificType.from_data(data, finite_type=finite, dofs=dofs, space=actual.space)


def _application_body_hash(data, bindings, *, inputs, output):
    """Specialize only exact named-input SSA aliases; keep the declared graph unchanged.

    A caller's two distinct nodes remain distinct even when their values, types or
    expressions happen to be equal. Replays use real GraphFunction substitution,
    including nested regions. No specialized digest is accepted from metadata.
    """
    data = json.loads(_key(data))
    if set(bindings) != set(data["inputs"]) or set(inputs) != set(bindings):
        raise TypeError("scientific application requires its exact named inputs")
    inputs = _ordered_scientific_inputs(data, inputs)
    representatives, aliases = {}, {}
    for key in sorted(bindings):
        value = bindings[key].prog._canonical_value(bindings[key])
        identity = (id(value.prog), value.id)
        aliases[key] = representatives.setdefault(identity, key)
    if all(key == representative for key, representative in aliases.items()):
        return data["body_hash"]
    for key, representative in aliases.items():
        if _key(inputs[key].to_data()) != _key(inputs[representative].to_data()):
            raise TypeError("aliased scientific inputs require the exact same scientific type")
    from pops.graph_functions import GraphFunction
    from pops.time.variant_types import Product
    function = GraphFunction.from_data(data["graph"],
        inputs=Product(**{key: typ.finite_type for key, typ in inputs.items()}),
        output=output.finite_type)
    conversion = None if data["conversion"] is None else ScientificConversion(**data["conversion"])
    replay = _check_scientific_graph(function, inputs=inputs, output=output,
                                    conversion=conversion, aliases=aliases)
    return replay.to_data()["body_hash"]


def bind_scientific_value(value, scientific_type, *, contract=None, bindings=None):
    """Persist a scientific declaration on canonical SSA; extracting `.value` cannot erase it."""
    from pops.time._program.value_validation import require_owned
    require_owned(value.prog, value, "scientific declaration")
    original = value
    value = value.prog._canonical_value(value)
    data = scientific_type.to_data()
    if ANNOTATION in value.attrs:
        if _key(value.attrs[ANNOTATION]) != _key(data):
            raise TypeError("scientific SSA reannotation changes support, units, variance or representation")
        return original
    if contract is None:
        if any(_scientific_ancestor(child) for child in _dependencies(value)):
            analyzer = _Analyzer()
            _compatible(analyzer.get(value), _meaning(scientific_type))
    attrs = dict(value.attrs, **{ANNOTATION: data})
    if contract is not None:
        if bindings is None:
            raise TypeError("strict scientific annotation requires its exact applied graph body")
        declared = contract.to_data()
        if set(bindings) != set(declared["inputs"]):
            raise TypeError("scientific application requires its exact named inputs")
        types = {key: _restore_scientific_type(declared["inputs"][key], child)
                 for key, child in bindings.items()}
        expected = _application_body_hash(declared, bindings, inputs=types, output=scientific_type)
        if _body_hash(value, bindings, canonical_aliases=True) != expected:
            raise TypeError("strict scientific annotation requires its exact applied graph body")
        attrs[CONTRACT] = declared
        attrs[APPLICATION] = {"input_values": dict(bindings)}
    value.prog._replace_value(value, attrs=attrs)
    # Existing references remain valid aliases of this canonical SSA. Keeping object identity
    # does not erase metadata: inference and serialization consult the Program's canonical node.
    return original


def validate_scientific_graph_call(function, program, bindings, result, *, nodes=()):
    """GraphFunction.__call__ integration hook, after SSA substitution and type checking."""
    scope = _MAP_SCOPE.get()
    if scope is not None and not scope[0]:
        return  # Explicit legacy boundary, never a normative scientific proof.
    if any(_scientific_ancestor(value) for value in bindings.values()):
        analyzer = _Analyzer(conversion=None if scope is None else scope[1])
        analyzer.get(result)
        analyzer.block(nodes)
    return result


def validate_scientific_function(function):
    """GraphFunction.from_data hook: validate annotations using real typed formal arguments."""
    from pops.graph_functions import _formals
    p, args = _formals(function.inputs)
    function(p, **{key: args[key] for key, _ in function.inputs.signature["members"]})
    validate_scientific_program(p, _allow_unbound_formals=True)
    return function


def validate_scientific_program(program, *, strict=False, _allow_unbound_formals=False,
                                _contract_receipts=None):
    """Root validation hook: inspect annotated dependencies after load and before erasure.

    Strict requires at least one declaration and refuses unchecked derived annotations. It does
    not invent scientific declarations for legacy nodes that the case never typed.
    """
    declarations = 0
    def walk(values):
        for value in values:
            yield value
            for key in ("body_block", "cond_block", "true_block", "false_block"):
                yield from walk(value.attrs.get(key, ()))
    values = tuple(walk(program._values))
    scientific_ids = _scientific_reachability(values)
    analyzer = _Analyzer(allow_legacy=not strict, scientific_ids=scientific_ids)
    analyzer.contract_receipts = _contract_receipts
    if getattr(program, "_compiled_detached", False) and getattr(
            program, "_scientific_compile_binding", None) is not None:
        from pops.time._program.scientific_compile_proof import require_scientific_binding
        analyzer.detached_bindings[id(program)] = require_scientific_binding(program)
    def inspect(value):
        try:
            return analyzer.get(value)
        except _UnboundScientificInput:
            if not _allow_unbound_formals or strict:
                raise
            # Only the finite GraphFunction loader can defer a genuinely unbound
            # formal. Strict contracts recheck the full body with authoritative
            # input meanings; executable Program validation never takes this path.
            analyzer.visited.clear()
    legacy_body = set()
    if not strict:
        for value in values:
            if value.attrs.get(CONTRACT, {}).get('mode') == 'legacy':
                legacy_body.update(_legacy_body(value))
    for value in values:
        if value.id in legacy_body:
            continue
        if value.op in ("continuation_open", "continuation_freeze", "continuation_transform"):
            from pops.time._program.structured_control import value_signature
            signature = value_signature(value)
            schema = (signature["schema"] if value.op == "continuation_open"
                      else signature["base"]["schema"])
            if schema.get("scientific"):
                declarations += len(schema["scientific"])
                inspect(value)
        if ANNOTATION in value.attrs:
            declarations += 1
            if not strict and value.attrs.get(CONTRACT, {}).get("mode") == "legacy":
                continue
            inspect(value)
        elif (id(value.prog), value.id) in scientific_ids:
            # Extracting .value and omitting an output annotation must not erase a
            # dimensional error. Administrative sinks are checked through their inputs.
            if value.op in ("commit", "finite_state_commit"):
                for child in value.inputs:
                    if (id(child.prog), child.id) in scientific_ids: inspect(child)
            else:
                inspect(value)
    if strict and not declarations:
        raise TypeError("normative scientific Program has no declared scientific values")
    return True


def _legacy_body(value):
    """Locate only the exact explicitly unchecked map body, stopping at its inputs.

    The legacy annotation is an admitted boundary, never a checked scientific
    conversion. Downstream expressions still inherit its declared units.
    """
    contract = value.attrs.get(CONTRACT, {})
    application = value.attrs.get(APPLICATION)
    if contract.get('mode') != 'legacy' or application is None:
        raise TypeError('legacy scientific boundary lacks its exact applied body')
    references = _application_inputs(value.prog, application)
    if _body_hash(value, references) != contract.get('body_hash'):
        raise TypeError('legacy scientific boundary belongs to a different applied body')
    stops = {node.id for node in references.values()}
    found = set()
    def visit(node):
        node = node.prog._canonical_value(node)
        if node.id in stops or node.id in found:
            return
        found.add(node.id)
        for child in _dependencies(node):
            visit(child)
    visit(value)
    found.discard(value.id)
    return found


__all__ = ["ScientificConversion", "ScientificGraphContract", "check_scientific_graph",
           "check_scientific_contract_data", "bind_scientific_value", "validate_scientific_program",
           "validate_scientific_graph_call", "validate_scientific_function", "scientific_map_scope"]


def scientific_packet_schema(signature, scientific):
    """Compile a complete path schema over the existing finite Product/Array type.

    Each real leaf must be covered exactly once when scientific typing is enabled.
    A path may name a whole nested Array/Product; its ScientificType enumerates
    that payload's real component units and retains its finite domain authorities.
    """
    from pops.representations import ScientificType
    from pops.time.variant_types import instantiate
    from pops.time._program.structured_control import same_signature
    from pops.time._program.versioned_values import _path
    if scientific is None:
        return ()
    if not isinstance(scientific, Mapping) or not scientific:
        raise TypeError('scientific Continuation schema requires a nonempty path mapping')
    entries = []
    for path, typ in scientific.items():
        path = _path(path)
        if type(typ) is not ScientificType:
            raise TypeError('scientific Continuation paths require ScientificType')
        leaf = _scientific_at(signature, path)
        if not same_signature(instantiate(typ.finite_type, leaf.get('point')).signature, leaf):
            raise TypeError('scientific Continuation path has a different finite authority')
        if leaf.get("space") is not None and typ.space != leaf["space"]:
            raise TypeError('scientific Continuation declaration cannot replace a native physical Space')
        _meaning(typ)
        entries.append(dict(path=path, type=typ.to_data()))
    paths = [row['path'] for row in entries]
    if any(a != b and (a[:len(b)] == b or b[:len(a)] == a) for a in paths for b in paths):
        raise TypeError('scientific Continuation paths overlap')
    def covered(sig, path=()):
        if path in paths:
            return
        if sig['vtype'] == 'product':
            for key, child in sig['members']: covered(child, (*path,key))
        elif sig['vtype'] == 'array':
            covered(sig['element'], (*path,'[]'))
        elif sig['vtype'] == 'variant':
            for tag, child in sig['members']: covered(child, (*path,tag))
        elif sig['vtype'] in ('scalar','state','rhs','scalar_field'):
            raise TypeError('scientific Continuation schema does not cover real path '+repr(path))
    covered(signature)
    return tuple(entries)


def _packet_apply(meaning, declarations, *, initial=False) -> _Meaning:
    def change(root, path, expected) -> _Meaning:
        if not path:
            return expected if initial and not _has_context(root) else _compatible(root, expected)
        key, *tail = path
        if isinstance(root, _Array) and key == '[]':
            return _Array(root.domain, change(root.element, tail, expected))
        if isinstance(root, _Sum) and key in dict(root.members):
            return _Sum(tuple((tag, change(child, tail, expected) if tag == key else child)
                              for tag,child in root.members))
        if isinstance(root, dict) and key in root:
            return dict(root, **{key:change(root[key], tail, expected)})
        raise TypeError('scientific Continuation path is absent from actual Product/Array/Variant')
    for entry in declarations:
        meaning = change(meaning, entry['path'], _meaning_data(entry['type']))
    return meaning


def validate_scientific_continuation(value):
    """Recompute candidate meanings from numerical SSA, also after graph reload."""
    _Analyzer().get(value)
    return value


def validate_scientific_packet_data(schema):
    """Authenticate declared path coverage and schema identity after deserialization."""
    if not schema.get('scientific'):
        return
    from pops.representations import ScientificType
    from pops.time.variant_types import ValueType
    from pops.time._program.versioned_values import validate_schema_data
    restored = {}
    for entry in schema['scientific']:
        path = tuple(entry['path']); data = entry['type']; leaf = _scientific_at(schema['read'], path)
        try:
            finite = ValueType.from_data(data['finite_type'])
        except ValueError:
            finite = ValueType(leaf)
        dofs = None if data['dofs'] is None else ValueType({k:v for k,v in leaf.items() if k != 'element'} | {'vtype':'index_set'})
        restored[path] = ScientificType.from_data(data, finite_type=finite, dofs=dofs, space=leaf.get('space'))
    expected = scientific_packet_schema(schema['read'], restored)
    if _key(expected) != _key(schema['scientific']):
        raise TypeError('scientific Continuation schema metadata changed')
    try:
        validate_schema_data(schema)
    except ValueError as exc:
        raise TypeError("scientific Continuation schema identity changed") from exc


def _scientific_at(signature, path):
    for key in path:
        if signature['vtype'] == 'array' and key == '[]':
            signature = signature['element']
        elif signature['vtype'] in ('product','variant') and key in dict(signature['members']):
            signature = dict(signature['members'])[key]
        else:
            raise TypeError('scientific path is not in the finite Product/Array/Variant schema')
    return signature


def _declare_initial_packet(actual, expected) -> _Meaning:
    """An initial datum may acquire meaning once; existing scientific meaning is checked."""
    if actual is _BOTTOM:
        return expected
    if isinstance(actual, _Scalar) and isinstance(expected, _Scalar) and actual.context is None:
        return expected
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return {key:_declare_initial_packet(actual[key], expected[key]) for key in actual}
    if isinstance(actual, _Array) and isinstance(expected, _Array) and actual.domain == expected.domain:
        return _Array(actual.domain, _declare_initial_packet(actual.element, expected.element))
    if isinstance(actual, _Sum) and isinstance(expected, _Sum) and dict(actual.members).keys() == dict(expected.members).keys():
        want = dict(expected.members)
        return _Sum(tuple((tag,_declare_initial_packet(child,want[tag])) for tag,child in actual.members))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return tuple(_declare_initial_packet(a,b) for a,b in zip(actual,expected))
    return _compatible(actual, expected)
