"""Closed, typed Program subgraphs, serialized once and specialized by SSA substitution.

The Python builder is an elaborator, never part of the stored function. Calls copy the closed
operation graph into the caller's actual Program, including nested native control regions. There
is no second evaluator and no runtime function pointer. Unknown operations or unbound scientific
authorities fail closed; a dependent snapshot can be reloaded with explicit type witnesses.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from pops.time import Program
from pops.time._authoring import authoring_transaction, require_program
from pops.time._program.structured_control import new_typed, value_signature
from pops.time._program.value_validation import require_owned, validate_input_regions
from pops.time.variant_types import ValueType, as_type, instantiate
from pops.time.values import ProgramValue, _Coeff


_BLOCKS = ("body_block", "cond_block", "true_block", "false_block")
# This is a language-effect profile, not a list of physical models. In particular, creation of
# state/history resources, publication and implicit reads of an ambient timestep are not calls.
_PURE = frozenset({"provenance_bind",
    "product", "product_get", "scalar_op", "scalar_expression", "compare", "bool_constant", "int64_constant",
    "int64_op", "int64_cast", "int64_compare", "finite_real", "scalar_isfinite",
    "region_argument", "region_index", "linear_combine",
    "pointwise_expression", "field_component", "vector_field", "cell_compare", "where",
    "branch", "structured_fold", "structured_iterate", "iteration_success", "iteration_is_ok",
    "capture_result", "result_ok", "result_error", "result_success", "result_is_ok",
    "result_payload", "iteration_payload", "result_match", "variant_construct", "variant_is",
    "variant_payload", "variant_require_success", "variant_unreachable",
    "finite_range_set", "finite_size", "finite_domain", "finite_same_domain", "finite_index", "finite_index_value",
    "finite_index_argument", "finite_at", "finite_map", "finite_scan",
})
_COLLECTIVE = frozenset({"reduce"})
_RENDEZVOUS = frozenset({"scalar_op", "scalar_expression", "finite_real", "int64_op", "int64_cast", "branch",
                        "structured_fold", "structured_iterate", "iteration_success",
                        "result_success", "variant_require_success", "variant_is", "result_match"})
_FINITE_CHECKED = frozenset({"finite_range_set", "finite_same_domain", "finite_index", "finite_at", "finite_map", "finite_scan"})
_PARTIAL = frozenset({"scalar_op", "scalar_expression", "int64_op", "int64_cast", "structured_fold",
                      "structured_iterate", "iteration_success", "result_success",
                      "variant_require_success", "variant_unreachable", "pointwise_expression",
                      "finite_real", "branch", "variant_is", "result_match"})
from pops.time._program.versioned_values import VERSIONED_OPS
_SCOPED_CAPABILITIES = VERSIONED_OPS - {"continuation_open", "continuation_publish"}
from pops.time._program.distributed_arrays import GLOBAL_OPS as _DISTRIBUTED, LOCAL_OPS as _LOCAL
_PURE = _PURE | _SCOPED_CAPABILITIES
_PURE = _PURE | _LOCAL
_COLLECTIVE = _COLLECTIVE | _DISTRIBUTED
_PARTIAL = _PARTIAL | _SCOPED_CAPABILITIES
_PARTIAL = _PARTIAL | _DISTRIBUTED
_RENDEZVOUS = _RENDEZVOUS | _SCOPED_CAPABILITIES
_SCALARS = frozenset({"scalar", "int64", "bool"})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _compatible(expected, actual):
    """Unbound scalar time is a parameter, never a wildcard for scientific authorities."""
    if expected.get("vtype") != actual.get("vtype"):
        return False
    kind = expected["vtype"]
    if kind in ("product", "variant"):
        left, right = expected["members"], actual["members"]
        return len(left) == len(right) and all(
            a == b and _compatible(x, y) for (a, x), (b, y) in zip(left, right))
    from pops.time.canonical_data import _json_ready
    left, right = dict(expected), dict(actual)
    if kind in _SCALARS and left.get("point") is None:
        left.pop("point", None)
        right.pop("point", None)
    return _json_ready(left) == _json_ready(right)


def _instantiate(signature, point):
    return instantiate(ValueType(signature), point).signature


def _walk_type(value, path=()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_type(item, (*path, key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_type(item, (*path, index))
    elif value is not None and type(value) not in (str, int, bool, float):
        yield path, value


def _at(value, path):
    for key in path:
        value = value[key]
    return value


class _Codec:
    def __init__(self, *, program=None, anchors=(), bindings=None, point=None, local_families=()):
        self.program, self.anchors = program, tuple(anchors)
        self.bindings, self.point = bindings or {}, point
        self.values, self.regions = {}, {}
        self.local_families, self.families = frozenset(local_families), {}

    def encode(self, value):
        from pops.identity.scalar import ScalarLiteral
        from pops.time.points import TimePoint
        from pops.time.stencil import StencilAccess
        from pops.time.value_metadata import CoeffPolynomial
        from decimal import Decimal
        from fractions import Fraction
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("GraphFunction constants must be finite")
            return {"$float": value.hex()}
        if isinstance(value, ProgramValue):
            require_owned(self.program, value, "GraphFunction closed graph")
            return {"$ref": value.id}
        if isinstance(value, ScalarLiteral):
            if value.cpp is not None or value.target not in (None, 'pops::Real', 'float', 'double'):
                raise TypeError("GraphFunction constants cannot contain a custom C++ spelling or target")
            return {"$literal": self.encode(value.__pops_scalar_literal__())}
        if isinstance(value, (Fraction, Decimal)):
            return {"$number": self.encode(ScalarLiteral.from_value(value))}
        if isinstance(value, CoeffPolynomial):
            if any(power != 0 for power in value):
                raise ValueError("GraphFunction cannot capture ambient dt; pass a scalar duration")
            return {"$polynomial": self.encode(dict(value))}
        if type(value) is StencilAccess:
            return {"$stencil": value.required_ghost_depth}
        if isinstance(value, _Coeff):
            if any(power != 0 for power in value.powers):
                raise ValueError("GraphFunction cannot capture ambient dt; pass a scalar duration")
            return {"$coefficient": self.encode(dict(value.powers))}
        if isinstance(value, Mapping):
            if value.get("vtype") in ("index_set", "index", "array") \
                    and value.get("family") in self.local_families:
                value = dict(value)
                family = value.pop("family")
                return {"$finite_signature": [family, self.encode(value)]}
            return {"$map": [[self.encode(key), self.encode(item)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"$tuple": [self.encode(item) for item in value]}
        for path, authority in self.anchors:
            if value is authority:
                return {"$authority": list(path)}
        if isinstance(value, TimePoint) and value.clock == self.program.clock and value.step == 0 \
                and value.offset.to_python() == 0:
            return {"$point": "call"}
        # A numerical rate derives its exact authority from its state formal, never a name lookup.
        if getattr(value, "kind", None) == "rate" and getattr(value, "base_space", None) is not None:
            return {"$rate": self.encode(value.base_space)}
        raise TypeError("GraphFunction contains an unbound authority or unsupported capture: %s"
                        % type(value).__name__)

    def decode(self, data):
        from pops.identity.scalar import ScalarLiteral
        if data is None or type(data) in (str, int, bool):
            return data
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError("malformed closed GraphFunction data")
        key, item = next(iter(data.items()))
        if key == "$float":
            number = float.fromhex(item)
            if not math.isfinite(number) or number.hex() != item:
                raise ValueError("noncanonical GraphFunction float")
            return number
        if key == "$literal":
            value = self.decode(item)
            if value.get("cpp") is not None or value.get('target') not in (None, 'pops::Real', 'float', 'double'):
                raise ValueError("custom C++ literals are not GraphFunction constants")
            return ScalarLiteral(**value)
        if key == "$coefficient":
            return _Coeff(self.decode(item))
        if key == "$polynomial":
            from pops.time.value_metadata import CoeffPolynomial
            value = self.decode(item)
            if any(power != 0 for power in value):
                raise ValueError("GraphFunction cannot capture ambient dt")
            return CoeffPolynomial(value)
        if key == "$number":
            return self.decode(item).to_python()
        if key == "$stencil":
            from pops.time.stencil import StencilAccess
            return StencilAccess(item)
        if key == "$tuple":
            return tuple(self.decode(child) for child in item)
        if key == "$map":
            result = {}
            for a, b in item:
                a = self.decode(a)
                if a in result:
                    raise ValueError("duplicate GraphFunction map key")
                result[a] = self.decode(b)
            return result
        if key == "$ref":
            if type(item) is not int or item not in self.values:
                raise ValueError("unbound or forward GraphFunction value reference")
            return self.values[item]
        if key == "$authority":
            return _at(self.bindings, item)
        if key == "$point" and item == "call":
            return self.point
        if key == "$rate":
            from pops.model.spaces import Rate
            return Rate(self.decode(item))
        if key == "$finite_signature":
            family, signature = item
            if type(family) is not int or family < 0:
                raise ValueError("invalid GraphFunction finite domain family")
            if family not in self.families:
                self.families[family] = self.program._next_finite_domain_identity
                self.program._next_finite_domain_identity += 1
            return dict(self.decode(signature), family=self.families[family])
        raise ValueError("unknown closed GraphFunction data tag %r" % key)

    def node(self, value):
        effects = _effects(value)
        attrs = dict(value.attrs)
        blocks = {}
        for key in _BLOCKS:
            if key in attrs:
                blocks[key] = [self.node(child) for child in attrs.pop(key)]
                attrs.pop(key.replace("_block", "_region"), None)
        return {"id": value.id, "op": value.op, "vtype": value.vtype,
                "inputs": [item.id for item in value.inputs], "attrs": self.encode(attrs),
                "blocks": blocks, "signature": self.encode(_signature(value)),
                "point": self.encode(value.point), "effects": sorted(effects)}

    def inline(self, node):
        if set(node) != {"id", "op", "vtype", "inputs", "attrs", "blocks", "signature", "point", "effects"} \
                or type(node["id"]) is not int or node["id"] < 0:
            raise ValueError("malformed GraphFunction operation record")
        op = node["op"]
        if op not in _PURE | _COLLECTIVE:
            raise ValueError("GraphFunction operation has no declared effect contract: %s" % op)
        if node["id"] in self.values:
            raise ValueError("duplicate GraphFunction SSA definition")
        blocks = {}
        for key, records in node["blocks"].items():
            if key not in _BLOCKS:
                raise ValueError("unsupported GraphFunction region")
            block = []
            region = self.program._region_for_block(block)
            for parent in self.program._recording:
                self.program._allow_region_capture(self.program._region_for_block(parent), region)
            self.program._recording.append(block)
            try:
                for record in records:
                    self.inline(record)
            finally:
                self.program._recording.pop()
            blocks[key] = block
            blocks[key.replace("_block", "_region")] = region
        attrs = self.decode(node["attrs"])
        # A closed graph's default allocator is a context parameter. Bind it at the
        # call site, while an explicitly declared bound remains an exact contract.
        from pops.time._program.distributed_arrays import GLOBAL_OPS, _capacity
        if op in GLOBAL_OPS:
            policy_binding = attrs.get("storage_policy_binding", "explicit")
            if policy_binding not in ("inherited", "explicit"):
                raise ValueError("invalid GraphFunction distributed storage policy binding")
            declared = (_capacity(attrs["storage_bytes"]), _capacity(attrs["storage_versions"]))
            policy = self.program.__dict__.get("_distributed_storage_policy")
            if policy_binding == "explicit":
                if policy is not None and policy != declared:
                    raise ValueError("GraphFunction explicit distributed storage policy differs from its call site")
                self.program.__dict__["_distributed_storage_policy"] = declared
                self.program.__dict__["_distributed_storage_explicit"] = True
                policy = declared
            else:
                policy = policy or (64 * 1024 * 1024, 65536)
                self.program.__dict__["_distributed_storage_policy"] = policy
            attrs.update(storage_bytes=policy[0], storage_versions=policy[1],
                         storage_policy_binding=("explicit" if self.program.__dict__.get(
                             "_distributed_storage_explicit", False) else "inherited"))
        if any(key in attrs for key in (*_BLOCKS, *(k.replace('_block', '_region') for k in _BLOCKS))):
            raise ValueError("GraphFunction regions must use explicit closed block records")
        attrs.update(blocks)
        signature = self.decode(node["signature"])
        if signature["vtype"] != node["vtype"]:
            raise TypeError("GraphFunction operation has inconsistent value types")
        inputs = tuple(self.values[index] for index in node["inputs"])
        _validate_operation(op, inputs, attrs, signature)
        result = new_typed(self.program, signature, op, inputs, attrs,
                           point=self.decode(node["point"]))
        if sorted(_effects(result)) != node["effects"]:
            raise ValueError("GraphFunction effects do not match the serialized operations")
        self.values[node["id"]] = result
        return result


def _signature(value):
    if value.vtype == "iteration_result":
        return {"vtype": value.vtype}
    if value.vtype == "csr":
        # CSR is a closed graph-local adjacency descriptor. This preserves its
        # dependent row/column authorities without making it a stored value type.
        return value.attrs["signature"]
    return value_signature(value)


def _validate_operation(op, inputs, attrs, signature):
    """Check serialized leaf typing before `_new` (which only authenticates ownership/scopes).

    These checks validate language operations. They neither dispatch on a physical model nor
    accept arbitrary Python/C++ callbacks embedded in data.
    """
    from pops.time._program.structured_control import same_signature
    from pops.time._program.versioned_values import validate_signature_schemas
    validate_signature_schemas(signature)
    if op == "provenance_bind":
        from pops.ledger_provenance import validate_contract_operation
        validate_contract_operation(inputs, attrs, signature)
    if op.startswith("finite_"):
        from pops.time._program.finite_arrays import validate_finite_operation
        validate_finite_operation(op, inputs, attrs, signature)
    def expect(condition):
        if not condition:
            raise TypeError('GraphFunction serialized operation %s violates its type contract' % op)
    if op == 'product':
        fields = attrs.get('fields', ())
        expect(signature['vtype'] == 'product' and len(fields) == len(inputs)
               and len(set(fields)) == len(fields)
               and same_signature(signature, {'vtype': 'product', 'members': tuple(
                   (key, value_signature(value)) for key, value in zip(fields, inputs))}))
    elif op == 'product_get':
        expect(len(inputs) == 1 and inputs[0].vtype == 'product')
        members = dict(value_signature(inputs[0])['members'])
        expect(attrs.get('field') in members and same_signature(signature, members[attrs['field']]))
    elif op == 'scalar_op':
        expect(signature['vtype'] == 'scalar' and all(v.vtype == 'scalar' for v in inputs)
               and attrs.get('fn') in ('add', 'sub', 'mul', 'div'))
        operands = attrs.get('operands', ())
        expect(len(operands) == 2)
        from pops.identity.scalar import ScalarLiteral
        for kind, value in operands:
            expect((kind == 'v' and type(value) is int and 0 <= value < len(inputs))
                   or (kind == 'c' and isinstance(value, ScalarLiteral)))
    elif op in ('scalar_expression', 'scalar_isfinite'):
        expect(all(v.vtype == 'scalar' for v in inputs)
               and signature['vtype'] == ('bool' if op == 'scalar_isfinite' else 'scalar'))
    elif op in ('int64_op', 'int64_compare'):
        expect(len(inputs) == 2 and all(v.vtype == 'int64' for v in inputs)
               and signature['vtype'] == ('bool' if op == 'int64_compare' else 'int64'))
    elif op in ('bool_constant', 'int64_constant'):
        kind = bool if op == 'bool_constant' else int
        expect(not inputs and type(attrs.get('value')) is kind
               and signature['vtype'] == ('bool' if kind is bool else 'int64'))
    elif op in ('structured_fold', 'structured_iterate'):
        expect(bool(inputs) and isinstance(attrs.get('body'), ProgramValue)
               and isinstance(attrs.get('body_argument'), ProgramValue))
        expect(same_signature(value_signature(inputs[0]), value_signature(attrs['body_argument'])))
        body = attrs['body']
        if op == 'structured_fold':
            expect(same_signature(signature, value_signature(inputs[0]))
                   and same_signature(signature, value_signature(body)))
    elif op == 'branch':
        expect(len(inputs) == 1 and inputs[0].vtype == 'bool'
               and all(same_signature(signature, value_signature(attrs[key]))
                       for key in ('true_result', 'false_result')))
    elif op == 'variant_construct':
        expect(signature['vtype'] == 'variant' and len(inputs) == 1)
        members = dict(signature['members'])
        expect(attrs.get('tag') in members and same_signature(
            value_signature(inputs[0]), members[attrs['tag']]))


def _effects(value):
    if value.op not in _PURE | _COLLECTIVE:
        raise ValueError("GraphFunction operation has no declared effect contract: %s" % value.op)
    effects = set()
    if value.op in _PARTIAL | _FINITE_CHECKED:
        effects.add("failure")
    if value.op in _COLLECTIVE | _RENDEZVOUS | _FINITE_CHECKED:
        effects.add("collective")
    for key in _BLOCKS:
        for child in value.attrs.get(key, ()):
            effects.update(_effects(child))
    if value.op == "capture_result":
        effects.discard("failure")
    return effects


@dataclass(frozen=True, init=False)
class GraphFunction:
    """Immutable signature and closed operation data, without a retained Python builder."""
    inputs: ValueType
    output: ValueType
    _snapshot: str

    def __init__(self, *, inputs, output, snapshot):
        inputs, output = as_type(inputs), as_type(output)
        if inputs.signature["vtype"] != "product":
            raise TypeError("GraphFunction inputs must be a Product type")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "_snapshot", _json(snapshot))

    @property
    def effects(self):
        return frozenset(self.to_data()["effects"])

    @property
    def identity(self):
        return hashlib.sha256(self._snapshot.encode()).hexdigest()

    def to_data(self):
        return json.loads(self._snapshot)

    @classmethod
    def from_data(cls, data, *, inputs=None, output=None):
        """Reconstruct closed data; dependent authorities require explicit authenticated types."""
        if not isinstance(data, Mapping) or set(data) != {
                "schema_version", "inputs", "output", "formal", "nodes", "result", "effects", "captures"}:
            raise ValueError("malformed GraphFunction snapshot")
        if data["schema_version"] != 1:
            raise ValueError("unsupported GraphFunction schema")
        inputs = ValueType.from_data(data["inputs"]) if inputs is None else as_type(inputs)
        output = ValueType.from_data(data["output"]) if output is None else as_type(output)
        if inputs.to_data() != data["inputs"] or output.to_data() != data["output"]:
            raise TypeError("GraphFunction type witnesses disagree with its serialized contract")
        _Codec().decode(data['captures'])
        result = cls(inputs=inputs, output=output, snapshot=data)
        # Structural/closure/effect verification uses the same real Program authoring path as a
        # call; loading never executes Python user code or fabricates a scientific authority.
        p, args = _formals(inputs)
        result(p, **{key: args[key] for key, _ in inputs.signature["members"]})
        from pops.scientific_graph import validate_scientific_program
        validate_scientific_program(p, _allow_unbound_formals=True)
        return result

    def __call__(self, program, **bindings):
        require_program(program, exact=True, where="GraphFunction call")
        expected = dict(self.inputs.signature["members"])
        if set(bindings) != set(expected):
            raise TypeError("GraphFunction arguments must match its named inputs exactly")
        for key, value in bindings.items():
            require_owned(program, value, "GraphFunction argument %s" % key)
            validate_input_regions(program, (value,), program._current_region(), "GraphFunction call")
            if not _compatible(expected[key], value_signature(value)):
                raise TypeError("GraphFunction argument %r has a different type or authority" % key)
        with authoring_transaction(program):
            args = program.product(**{key: bindings[key] for key in expected})
            snapshot = self.to_data()
            authority_types = {"inputs": self.inputs.signature, "output": self.output.signature}
            codec = _Codec(program=program, bindings=authority_types, point=args.point)
            codec.values[snapshot["formal"]] = args
            effects = set()
            inlined = []
            for node in snapshot["nodes"]:
                value = codec.inline(node)
                inlined.append(value)
                effects.update(_effects(value))
            if sorted(effects) != snapshot["effects"]:
                raise ValueError("GraphFunction effect summary is inconsistent")
            if snapshot["result"] not in codec.values:
                raise ValueError("GraphFunction result is unbound")
            result = codec.values[snapshot["result"]]
            validate_input_regions(program, (result,), program._current_region(), "GraphFunction result")
            if not _compatible(self.output.signature, value_signature(result)):
                raise TypeError("GraphFunction result violates its declared output type")
            from pops.scientific_graph import validate_scientific_graph_call
            validate_scientific_graph_call(self, program, bindings, result, nodes=tuple(inlined))
            return result


def _formals(inputs):
    from pops.time.points import TimePoint
    program = Program("graph_function")
    def explicit_points(signature):
        if signature["vtype"] in ("product", "variant"):
            for _, child in signature["members"]:
                yield from explicit_points(child)
        elif signature.get("point") is not None:
            yield signature["point"]
    # Abstract scalar parameters in a scientifically dependent function share that function's
    # explicit evaluation context. Concrete leaf points remain unchanged and clocks are checked.
    point = next(explicit_points(inputs.signature), TimePoint(program.clock))
    from pops.time.points import point_clock
    program.clock = point_clock(point, "GraphFunction formal context")
    signature = _instantiate(inputs.signature, point)
    args = new_typed(program, signature, "graph_argument", (), {})
    # Scientific leaves have full declarations before any pointwise operation consumes them.
    def declare(sig):
        if sig["vtype"] in ("product", "variant"):
            for _, child in sig["members"]:
                declare(child)
        elif sig.get("state_ref") is not None and sig.get("space") is not None:
            from pops.time._program.value_validation import structural_state_space
            space = structural_state_space(sig["space"])
            if space is not None:
                program._state_spaces[sig["state_ref"]] = space
        if sig["vtype"] in ("index_set", "index", "array"):
            program._next_finite_domain_identity = max(program._next_finite_domain_identity,
                                                       sig["family"]+1)
            if "element" in sig:
                declare(sig["element"])
    declare(signature)
    return program, args


def graph(*, inputs, output, captures=None, effects=None):
    """Elaborate ``builder(program, args[, captures])`` once into a closed native subgraph.

    ``effects``, when supplied, is an exact assertion of the inferred set (``failure`` and/or
    ``collective``). Explicit captures are immutable finite scalar/container data. Their values and
    the resulting operations are serialized. An unlisted ambient resource is rejected.
    """
    inputs, output = as_type(inputs), as_type(output)
    if inputs.signature["vtype"] != "product":
        raise TypeError("graph inputs must be a Product type")
    if effects is not None and (isinstance(effects, str) or set(effects) - {"failure", "collective"}):
        raise ValueError("unknown GraphFunction effect declaration")
    capture_data = _Codec().encode(dict(captures or {}))
    frozen_captures = _freeze(_Codec().decode(capture_data))

    def decorate(builder):
        if not callable(builder):
            raise TypeError("graph requires a Python elaborator")
        program, args = _formals(inputs)
        def elaborate(p):
            return builder(p, args, frozen_captures) if captures is not None else builder(p, args)
        is_result = output.signature["vtype"] == "variant" and set(
            dict(output.signature["members"])) == {"ok", "err"}
        result = program.capture(elaborate) if is_result else elaborate(program)
        require_owned(program, result, "GraphFunction result")
        validate_input_regions(program, (result,), 0, "GraphFunction result")
        if program._commits or program._histories:
            raise ValueError("GraphFunction cannot publish states or create histories")
        if not _compatible(output.signature, value_signature(result)):
            raise TypeError("GraphFunction elaborator result differs from its declared output type")
        anchors = tuple(_walk_type({"inputs": inputs.signature, "output": output.signature}))
        families = {value.attrs['signature']['family'] for value in program._issued_values.values()
                    if (value.op == 'finite_range_set' and len(value.inputs) == 1)
                    or (value.op == 'distributed_domain' and not value.attrs.get('family_witness'))}
        codec = _Codec(program=program, anchors=anchors, local_families=families)
        nodes = [codec.node(value) for value in program._values if value is not args]
        inferred = set()
        for value in program._values:
            if value is not args:
                inferred.update(_effects(value))
        if effects is not None and set(effects) != inferred:
            raise ValueError("GraphFunction declared effects differ from inferred effects")
        return GraphFunction(inputs=inputs, output=output, snapshot={
            "schema_version": 1, "inputs": inputs.to_data(), "output": output.to_data(),
            "formal": args.id, "nodes": nodes, "result": result.id,
            "effects": sorted(inferred), "captures": capture_data})
    return decorate


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def call(function, arguments, *, program=None):
    """Apply a GraphFunction to a named product or mapping of real Program values."""
    if type(function) is not GraphFunction:
        raise TypeError("call requires an exact GraphFunction")
    if isinstance(arguments, ProgramValue):
        if arguments.vtype != "product":
            raise TypeError("call arguments must be a Product")
        program = arguments.prog if program is None else program
        require_owned(program, arguments, "call arguments")
        arguments = {key: arguments[key] for key, _ in value_signature(arguments)["members"]}
    if not isinstance(arguments, Mapping):
        raise TypeError("call arguments must be a named mapping or Product")
    if program is None:
        values = tuple(arguments.values())
        if not values or not isinstance(values[0], ProgramValue):
            raise TypeError("an empty call requires program=Program")
        program = values[0].prog
    return function(program, **arguments)


__all__ = ["GraphFunction", "graph", "call"]
