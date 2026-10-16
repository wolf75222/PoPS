"""Finite immutable arrays in the native Program algebra.

Python records each body once. Cardinalities, checked indices, element operations and
ordered scans execute in generated C++. Distributed ownership dispatches to the
same native algebra with an explicit transport budget at collective boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from collections.abc import Callable

if TYPE_CHECKING:
    from pops.time.value_signatures import ValueSignature

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object

from pops.time._authoring import atomic_authoring
from pops.time.values import ProgramValue
from pops.time._program.structured_control import (
    value_signature, product_signature, scalar_signature, same_signature, new_typed,
)
from pops.time._program.value_validation import require_owned, validate_input_regions


FINITE_VALUE_TYPES = frozenset(("index_set", "index", "array"))


@dataclass(frozen=True, eq=False)
class IndexFamily:
    """Authoring identity shared by distinct native domain instances."""
    capacity: int
    ownership: str = "replicated"

    def __post_init__(self):
        if type(self.capacity) is not int or not 0 <= self.capacity <= (1 << 63)-1:
            raise ValueError("IndexFamily capacity requires a nonnegative exact Int64 constant")
        if self.ownership != "replicated":
            raise ValueError("use Distributed for a distributed domain family")


def finite_signature(value) -> ValueSignature:
    if value.vtype not in FINITE_VALUE_TYPES:
        raise TypeError("expected a finite set, checked index or array")
    return cast("ValueSignature", value.attrs["signature"])


def _element_type(signature):
    kind = signature["vtype"]
    if kind in ("product", "variant"):
        for _, child in signature["members"]:
            _element_type(child)
    elif kind == "array":
        _element_type(signature["element"])
    elif kind not in ("scalar", "int64", "bool", "index", "index_set", "state", "rhs", "scalar_field"):
        raise TypeError("replicated array elements require finite handled values, products or variants")


def _domain_signature(signature) -> ValueSignature:
    return cast("ValueSignature", {key: value for key, value in signature.items() if key != "element"} | {"vtype": "index_set"})


def _array_signature(domain, element) -> ValueSignature:
    _element_type(element)
    return cast("ValueSignature", dict(finite_signature(domain), vtype="array", element=element))


def validate_finite_operation(op, inputs, attrs, signature):
    """Recheck finite language typing after serialization; never trust an opcode's label."""
    def require(ok):
        if not ok:
            raise TypeError("finite operation %s violates its type or domain-family contract" % op)
    def sig(value):
        return value_signature(value)
    if op == "finite_real":
        require(signature["vtype"] == "scalar" and len(inputs) <= 1)
        if inputs:
            require(inputs[0].vtype in ("scalar", "int64"))
        else:
            x = attrs.get("value")
            require(type(x) in (float, int) and math.isfinite(x))
    elif op == "finite_range_set":
        require(signature["vtype"] == "index_set" and signature["ownership"] == "replicated")
        require(type(signature["capacity"]) is int and 0 <= signature["capacity"] < (1 << 63))
        require(len(inputs) in (1, 2) and inputs[0].vtype == "int64")
        if len(inputs) == 2:
            require(inputs[1].vtype == "index_set" and same_signature(signature, sig(inputs[1])))
    elif op in ("finite_size", "finite_same_domain"):
        require(len(inputs) == (2 if op == "finite_same_domain" else 1))
        require(all(v.vtype == "index_set" for v in inputs))
        require(signature["vtype"] == ("bool" if op == "finite_same_domain" else "int64"))
        if len(inputs) == 2:
            require(same_signature(sig(inputs[0]), sig(inputs[1])))
    elif op == "finite_domain":
        require(len(inputs) == 1 and inputs[0].vtype in ("index", "array"))
        require(same_signature(signature, _domain_signature(sig(inputs[0]))))
    elif op in ("finite_index", "finite_index_argument"):
        require(len(inputs) == (2 if op == "finite_index" else 1) and inputs[0].vtype == "index_set")
        require(same_signature(signature, dict(sig(inputs[0]), vtype="index")))
        if op == "finite_index":
            require(inputs[1].vtype == "int64")
    elif op == "finite_index_value":
        require(len(inputs) == 1 and inputs[0].vtype == "index" and signature["vtype"] == "int64")
    elif op == "finite_at":
        require(len(inputs) == 2 and inputs[0].vtype == "array" and inputs[1].vtype == "index")
        require(same_signature(_domain_signature(sig(inputs[0])), _domain_signature(sig(inputs[1]))))
        require(same_signature(signature, sig(inputs[0])["element"]))
    elif op in ("finite_map", "finite_scan"):
        scan = op == "finite_scan"
        require(len(inputs) == (4 if scan else 3))
        domain = inputs[1 if scan else 0]
        require(domain.vtype == "index_set" and all(v.vtype == "int64" for v in inputs[-2:]))
        require(isinstance(attrs.get("body"), ProgramValue) and isinstance(attrs.get("body_index"), ProgramValue))
        require(same_signature(sig(attrs["body_index"]), dict(sig(domain), vtype="index")))
        if scan:
            body = sig(attrs["body"])
            require(body["vtype"] == "product")
            fields = dict(body["members"])
            require(set(fields) == {"state", "value"} and same_signature(fields["state"], sig(inputs[0])))
            require(same_signature(attrs["array_signature"], _array_signature(domain, fields["value"])))
            require(same_signature(signature, product_signature((("state", sig(inputs[0])),
                ("values", attrs["array_signature"])))))
        else:
            require(same_signature(signature, _array_signature(domain, sig(attrs["body"]))))


def validate_finite_arrays(program):
    def walk(values):
        for value in values:
            if value.op.startswith("finite_"):
                validate_finite_operation(value.op, value.inputs, value.attrs, value_signature(value))
            for key in ("body_block", "cond_block", "true_block", "false_block"):
                walk(value.attrs.get(key, ()))
    walk(program._values)


def _record_body(program, domain, initial, callback: Callable[..., ProgramValue]):
    block = []
    region = program._region_for_block(block)
    for parent in program._recording:
        program._allow_region_capture(program._region_for_block(parent), region)
    for capture in (domain, initial):
        if capture is not None and capture.region:
            program._allow_region_capture(capture.region, region)
    program._recording.append(block)
    try:
        signature = dict(finite_signature(domain), vtype="index")
        index = new_typed(program, signature, "finite_index_argument", (domain,), {}, point=domain.point)
        if initial is None:
            argument = None
            result = callback(program, index)
        else:
            argument = new_typed(program, value_signature(initial), "region_argument", (initial,), {},
                                 point=initial.point)
            result = callback(program, argument, index)
        require_owned(program, result, "finite array body")
        validate_input_regions(program, (result,), region, "finite array body result")
    finally:
        program._recording.pop()
    return block, region, argument, index, result


class _ProgramFiniteArrays(_ProgramBase):
    @atomic_authoring
    def float64(self, value):
        """Explicit finite Float64 constant/cast; Int64 conversion may round by IEEE rules."""
        if isinstance(value, ProgramValue):
            require_owned(self, value, "float64")
            if value.vtype not in ("int64", "scalar"):
                raise TypeError("float64 cast requires Int64 or Float64")
            return new_typed(self, scalar_signature("scalar"), "finite_real", (value,), {})
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("float64 requires a numeric constant or native Int64")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("float64 constant must be finite")
        return new_typed(self, scalar_signature("scalar"), "finite_real", (), {"value": value})

    @atomic_authoring
    def range_set(self, cardinality, *, capacity=None, ownership=None, family=None):
        """Issue a fresh finite set instance with native size in [0, capacity]."""
        witness = None
        if isinstance(family, ProgramValue):
            require_owned(self, family, "range_set family witness", vtype="index_set")
            witness = family
            fs = finite_signature(family)
            if capacity is not None and capacity != fs["capacity"]:
                raise ValueError("range_set cannot change family capacity")
            if ownership is not None and ownership != fs["ownership"]:
                raise ValueError("range_set cannot change family ownership")
            capacity, ownership, family_id = fs["capacity"], fs["ownership"], fs["family"]
        elif isinstance(family, IndexFamily):
            if capacity is not None and capacity != family.capacity:
                raise ValueError("range_set cannot change family capacity")
            if ownership is not None and ownership != family.ownership:
                raise ValueError("range_set cannot change family ownership")
            capacity, ownership = family.capacity, family.ownership
            registry = self.__dict__.setdefault("_finite_families", {})
            if family not in registry:
                registry[family] = self._next_finite_domain_identity
                self._next_finite_domain_identity += 1
            family_id = registry[family]
        elif family is not None:
            raise TypeError("family requires IndexFamily or an owned IndexSet witness")
        else:
            family_id = self._next_finite_domain_identity
            self._next_finite_domain_identity += 1
        if ownership != "replicated":
            raise NotImplementedError("finite array ownership currently requires explicit 'replicated'")
        if type(capacity) is not int or not 0 <= capacity <= (1 << 63)-1:
            raise ValueError("range_set capacity requires a nonnegative exact Int64-range constant")
        count = self.int64(cardinality)
        signature = {"vtype": "index_set", "family": family_id, "capacity": capacity,
                     "ownership": "replicated", "point": count.point}
        inputs = (count,) if witness is None else (count, witness)
        return new_typed(self, signature, "finite_range_set", inputs,
                         {"fresh_family": family is None or isinstance(family, IndexFamily)}, point=count.point)

    @atomic_authoring
    def same_domain(self, left, right):
        """Compare live instance identity, including empty domains; cardinality is insufficient."""
        for value in (left, right):
            require_owned(self, value, "same_domain", vtype="index_set")
        if not same_signature(finite_signature(left), finite_signature(right)):
            raise TypeError("same_domain requires the same static domain family")
        return new_typed(self, scalar_signature("bool"), "finite_same_domain", (left, right), {})

    @atomic_authoring
    def size(self, domain):
        require_owned(self, domain, "size")
        if domain.vtype != "index_set":
            raise TypeError("size requires an IndexSet")
        if finite_signature(domain)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).size(domain)
        return new_typed(self, scalar_signature("int64"), "finite_size", (domain,), {}, point=domain.point)

    @atomic_authoring
    def domain(self, value):
        require_owned(self, value, "domain")
        if value.vtype not in ("index", "array"):
            raise TypeError("domain requires an Index or Array")
        if finite_signature(value)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).domain_of(value)
        return new_typed(self, _domain_signature(finite_signature(value)), "finite_domain", (value,), {},
                         point=value.point)

    @atomic_authoring
    def index(self, domain, offset):
        require_owned(self, domain, "index domain")
        if domain.vtype != "index_set":
            raise TypeError("index requires an IndexSet")
        if finite_signature(domain)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).index(domain, offset)
        offset = self.int64(offset)
        return new_typed(self, dict(finite_signature(domain), vtype="index"),
                         "finite_index", (domain, offset), {}, point=domain.point)

    @atomic_authoring
    def index_value(self, index):
        """Explicit forgetful projection to Int64; indexing still requires P.index(D,k)."""
        require_owned(self, index, "index_value")
        if index.vtype != "index":
            raise TypeError("index_value requires a checked Index")
        if finite_signature(index)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).gid(index)
        return new_typed(self, scalar_signature("int64"), "finite_index_value", (index,), {}, point=index.point)

    @atomic_authoring
    def at(self, array, index):
        require_owned(self, array, "at array")
        require_owned(self, index, "at index")
        if array.vtype != "array" or index.vtype != "index":
            raise TypeError("at requires an Array and a checked Index, never a naked Int64")
        if finite_signature(array)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).at_local(array, index)
        if not same_signature(_domain_signature(finite_signature(array)),
                              _domain_signature(finite_signature(index))):
            raise ValueError("at index belongs to a different finite domain family")
        # A container may retain values from different exact points. Restore the stored
        # element signature rather than relabeling it with the domain's construction point.
        return new_typed(self, finite_signature(array)["element"], "finite_at", (array,index), {})

    @atomic_authoring
    def map(self, domain=None, body: Callable[..., ProgramValue] | None = None, *, physical_map=None, source=None, target=None, byte_capacity=None):
        # Preserve the existing physical-support map route. The type selects a language
        # operation, never a physical model, and its original validator owns that contract.
        if physical_map is not None:
            if domain is not None:
                raise TypeError("map accepts one physical map or finite domain")
            domain = physical_map
        if not isinstance(domain, ProgramValue) or domain.vtype != "index_set":
            if body is not None:
                raise TypeError("finite map requires an IndexSet and body(P, Index)")
            from .physical_maps import _ProgramPhysicalMaps
            return _ProgramPhysicalMaps.map(self, domain, source=source, target=target)
        if source is not None or target is not None:
            raise TypeError("finite map does not accept physical source/target stages")
        require_owned(self, domain, "map domain")
        if domain.vtype != "index_set" or not callable(body):
            raise TypeError("map requires an IndexSet and body(P, Index)")
        if finite_signature(domain)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return self.require_success(Distributed(self).local_map(domain,
                lambda p, i: p.capture(lambda q: body(q, i)), byte_capacity=byte_capacity))
        if byte_capacity is not None:
            raise TypeError("replicated map does not use a transport budget")
        validate_input_regions(self, (domain,), self._current_region(), "map")
        count, budget = self.size(domain), self.int64(finite_signature(domain)["capacity"])
        block, region, _, index, result = _record_body(self, domain, None, body)
        signature = _array_signature(domain, value_signature(result))
        return new_typed(self, signature, "finite_map", (domain,count,budget),
                         {"body_block": block, "body_region": region, "body_index": index,
                          "body": result}, point=domain.point)

    @atomic_authoring
    def scan(self, domain, initial, body: Callable[..., ProgramValue], *, order, byte_capacity=None):
        """Ordered scan returning product(state=final, values=Array(D,T)).

        body(P, state, Index) returns product(state=next_state, value=emitted_element).
        No partially constructed array is returned on failure.
        """
        require_owned(self, domain, "scan domain or values")
        if domain.vtype == "array":
            values = domain
            if finite_signature(values)["ownership"] != "replicated":
                from .distributed_arrays import Distributed
                return Distributed(self).scan(values, initial, body, order=order, byte_capacity=byte_capacity)
            return self.scan(self.domain(values), initial, lambda p, state, i: body(p, state, p.at(values, i)),
                             order=order, byte_capacity=byte_capacity)
        if domain.vtype != "index_set" or finite_signature(domain)["ownership"] != "replicated":
            raise TypeError("distributed scan requires materialized Array values, not local index captures")
        if order != "ascending" or byte_capacity is not None:
            raise NotImplementedError("replicated scan supports explicit ascending order")
        require_owned(self, domain, "scan domain")
        require_owned(self, initial, "scan initial")
        if domain.vtype != "index_set" or not callable(body):
            raise TypeError("scan requires an IndexSet and body(P, state, Index)")
        validate_input_regions(self, (domain,initial), self._current_region(), "scan")
        state_signature = value_signature(initial)
        if initial.vtype in ("result", "iteration_result"):
            raise TypeError("scan requires a handled initial Result")
        count, budget = self.size(domain), self.int64(finite_signature(domain)["capacity"])
        block, region, argument, index, result = _record_body(self, domain, initial, body)
        if result.vtype != "product":
            raise TypeError("scan body must return product(state=..., value=...)")
        fields = dict(value_signature(result)["members"])
        if set(fields) != {"state", "value"} or not same_signature(fields["state"], state_signature):
            raise TypeError("scan body must preserve carried state type and return exactly state/value")
        array_signature = _array_signature(domain, fields["value"])
        signature = product_signature((("state",state_signature),("values",array_signature)))
        return new_typed(self, signature, "finite_scan", (initial,domain,count,budget),
                         {"body_block": block, "body_region": region, "body_argument": argument,
                          "body_index": index, "body": result, "array_signature": array_signature},
                         point=initial.point)

    def gather(self, values, indices, *, byte_capacity=None):
        """Replicated gather, composed solely from map and checked at."""
        require_owned(self, values, "gather values")
        require_owned(self, indices, "gather indices")
        if values.vtype != "array" or indices.vtype != "array" \
                or finite_signature(indices)["element"]["vtype"] != "index":
            raise TypeError("gather requires values:Array(E,T), indices:Array(D,Index(E))")
        if finite_signature(values)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).import_values(values, indices, byte_capacity=byte_capacity)
        if byte_capacity is not None:
            raise TypeError("replicated gather does not use a transport budget")
        return self.map(self.domain(indices), lambda p,i: p.at(values,p.at(indices,i)))

    def reduce(self, values, initial, body: Callable[..., ProgramValue], *, order, byte_capacity=None):
        """Local ordered fold on replicated entries; never sums a replica per MPI rank."""
        require_owned(self, values, "reduce values")
        if values.vtype != "array" or not callable(body):
            raise TypeError("reduce requires Array and body(P,carry,element)")
        if finite_signature(values)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).reduce(values, initial, body, order=order, byte_capacity=byte_capacity)
        if order != "ascending" or byte_capacity is not None:
            raise ValueError("replicated reduction requires ascending order and no transport budget")
        domain = self.domain(values)
        return self.fold(initial, self.size(domain),
                         lambda p,s,k: body(p,s,p.at(values,p.index(domain,k))),
                         max_steps=finite_signature(domain)["capacity"])

    @atomic_authoring
    def scatter_reduce(self, values, indices, domain, initial, body, *, order, byte_capacity=None):
        """Functional ordered accumulation; duplicate destinations retain source order."""
        for value in (values, indices, domain, initial):
            require_owned(self, value, "scatter_reduce")
        if values.vtype != "array" or indices.vtype != "array" or domain.vtype != "index_set":
            raise TypeError("scatter_reduce requires arrays of values/indices and an IndexSet")
        if finite_signature(values)["ownership"] != "replicated":
            from .distributed_arrays import Distributed
            return Distributed(self).scatter_reduce(values, indices, domain, initial, body,
                                                     order=order, byte_capacity=byte_capacity)
        if order != "ascending" or byte_capacity is not None or not callable(body):
            raise ValueError("replicated scatter_reduce requires ascending order and a combine body")
        target = finite_signature(indices)["element"]
        if target["vtype"] != "index" or not same_signature(_domain_signature(target), finite_signature(domain)):
            raise TypeError("scatter indices must belong to the destination family")
        source = self.domain(values)
        compatible = self.same_domain(source, self.domain(indices))
        self.require_success(self.branch(compatible, lambda p: p.success(initial),
                                         lambda p: p.failure(initial, 1)))
        # Check every destination even if the output domain is empty. Otherwise an invalid
        # nonempty scatter into an empty destination could silently become an empty result.
        def check(p, state, k):
            index = p.at(indices, p.index(source, k))
            valid = p.same_domain(p.domain(index), domain)
            return p.require_success(p.branch(valid, lambda q: q.success(state),
                                               lambda q: q.failure(state, 1)))
        self.fold(initial, self.size(source), check, max_steps=finite_signature(source)["capacity"])
        def destination(p, j):
            def accumulate(q, carry, k):
                i = q.index(source, k)
                selected = q.index_value(q.at(indices, i)) == q.index_value(j)
                return q.branch(selected, lambda z: body(z, carry, z.at(values, i)), lambda z: carry)
            return p.fold(initial, p.size(source), accumulate, max_steps=finite_signature(source)["capacity"])
        return self.map(domain, destination)
