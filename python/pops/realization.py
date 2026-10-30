"""Occurrence-qualified realization builders on the real temporal Program IR.

Bodies execute once while authoring. Their results are materialized by
``Program.value``; no realization callback survives into a native step. This
module does not imply a discretization for an unresolved continuous operator.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops._ir.balance import BalanceOccurrence
from pops._ir.symbolic import freeze_symbolic_metadata
from pops.time._authoring import authoring_transaction, require_program
from pops.time._program.value_validation import require_owned
from pops.time.values import ProgramValue


@dataclass(frozen=True, slots=True, eq=False)
class Use:
    """One structural use of one retained physical balance occurrence."""

    occurrence: BalanceOccurrence
    locus: str
    temporal_role: str
    representation: str
    partition: str = "whole"
    scientific_type: Any = None
    __pops_ir_immutable__ = True

    def __post_init__(self):
        if not isinstance(self.occurrence, BalanceOccurrence):
            raise TypeError("Use requires a retained BalanceOccurrence, not a model name")
        for field in ("locus", "temporal_role", "representation", "partition"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise TypeError("Use.%s must be an explicit nonempty identifier" % field)
        if self.scientific_type is not None:
            from pops.representations import ScientificType
            if type(self.scientific_type) is not ScientificType:
                raise TypeError("Use.scientific_type requires an exact ScientificType")

    @property
    def key(self):
        return (*self.occurrence.identity, self.locus, self.temporal_role,
                self.representation, self.partition, self.scientific_type)

    def to_data(self):
        result = {"occurrence": self.occurrence.to_data(), "locus": self.locus,
                "temporal_role": self.temporal_role, "representation": self.representation,
                "partition": self.partition}
        if self.scientific_type is not None:
            result["scientific_type"] = self.scientific_type.to_data()
        return result


def _uses(values, where):
    values = tuple(values)
    if any(type(value) is not Use for value in values):
        raise TypeError(where + " requires Use objects")
    if len({value.key for value in values}) != len(values):
        raise ValueError(where + " repeats a structural use")
    return values


def _leaves(value):
    from pops.representations import ScientificValue
    if isinstance(value, ProgramValue):
        yield value
    elif type(value) is ScientificValue:
        yield value.value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)



# A realization may append a closed SSA/region graph, not change the caller's
# execution policy, publication, persistent history, or resource declarations.
_BODY_ALLOCATION_STATE = frozenset({
    "_values", "_issued_values", "_next_id", "_next_finite_domain_identity",
    "_recording", "_next_region", "_recording_regions", "_region_imports",
    "_finite_families",
})


def _body_controls(program):
    def key(value):
        if isinstance(value, dict):
            return ("dict", tuple((key(k), key(v)) for k, v in value.items()))
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(key(v) for v in value))
        if isinstance(value, (set, frozenset)):
            return (type(value).__name__, frozenset(key(v) for v in value))
        # No overloaded symbolic equality, serialization callback, or digest proof.
        return (type(value), id(value))
    return {name: key(value) for name, value in program.__dict__.items()
            if name not in _BODY_ALLOCATION_STATE}


def _validate_body_closure(program, nodes, *, allowed, before):
    nodes = tuple(nodes)
    seen = set()
    from pops.time.value_metadata import CoeffPolynomial
    from pops.time.values import _Coeff
    def metadata(value):
        if isinstance(value, (CoeffPolynomial, _Coeff)):
            powers = value.powers if isinstance(value, _Coeff) else value
            if any(power != 0 for power in powers):
                raise ValueError("Realize body cannot capture ambient dt; pass a scalar duration binding")
        elif isinstance(value, Mapping):
            for item in value.values():
                metadata(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                metadata(item)
    distributed_arguments = tuple(node for node in nodes
        if node.op in {"finite_index_argument", "distributed_index_argument", "distributed_offset_argument"}
        and node.attrs.get("signature", {}).get("ownership") == "contiguous_distributed")
    def visit(node):
        require_owned(program, node, "Realize body dependency")
        node = program._canonical_value(node)
        if node.id in allowed or node.id in seen:
            return
        seen.add(node.id)
        if node.id < before:
            raise ValueError("Realize body reads an undeclared temporal binding")
        if node.op in {"commit", "continuation_open", "continuation_publish", "finite_state_commit"}:
            raise ValueError("Realize body cannot publish or initialize accepted state")
        metadata(node.attrs)
        dependencies = (*node.inputs, *tuple(_leaves(node.attrs)))
        if not dependencies and node.op not in {
                "scalar_op", "finite_real", "int64_constant", "bool_constant",
                "region_index", "product", "variant_unreachable"}:
            # Rank is a property of an authenticated local-domain region, not a
            # free ambient input. Nested regions may explicitly import that binder.
            local = tuple(argument for argument in distributed_arguments
                          if argument.region == node.region or argument.region in
                          program._region_imports.get(node.region, ()))
            if node.op != "distributed_rank" or not local:
                raise ValueError("Realize body creates an undeclared native input")
            dependencies = local
        for dependency in dependencies:
            visit(dependency)
    for node in nodes:
        visit(node)


def _at_path(value, path):
    for key in path:
        if isinstance(value, Mapping) or isinstance(value, (tuple, list)):
            value = value[key]
        else:
            raise TypeError("Realize output paths traverse only explicit products/tuples")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Realized:
    """Native SSA results and the exact use-to-result binding."""

    name: str
    outputs: tuple[tuple[Use, ProgramValue], ...]
    reads: tuple[Use, ...]

    def __getitem__(self, use):
        if type(use) is not Use:
            raise TypeError("a realized output is selected by its Use")
        for declared, value in self.outputs:
            if declared.key == use.key:
                return value
        raise KeyError("this realization does not own the requested Use")


class Realize:
    """Elaborate an explicitly bound body to existing native Program operations.

    ``body(program, **bindings)`` builds scalar/field expressions or returns
    existing Program values. ``outputs`` maps every owned Use to a tuple path
    in the returned product. The body is an authoring callable, not a native
    terminal. External temporal captures must occur in bindings. Committing or
    modifying the caller's accepted-output declarations inside the body is refused.
    Current support is the support admitted by the existing Program operations;
    an arbitrary topology, quadrature or trace is never inferred from a Use label.
    """

    __slots__ = ("name", "owns", "reads", "body", "bindings", "outputs", "claims", "strict")

    def __init__(self, name, *, owns, reads=(), body, bindings, outputs, claims=(), strict=False):
        if type(name) is not str or not name:
            raise TypeError("Realize requires an explicit name")
        if not callable(body):
            raise TypeError("Realize body must build native Program expressions")
        if not isinstance(bindings, Mapping) or any(type(k) is not str for k in bindings):
            raise TypeError("Realize bindings must be a named product")
        self.name = name
        self.owns, self.reads = _uses(owns, "owns"), _uses(reads, "reads")
        if type(strict) is not bool:
            raise TypeError("Realize.strict requires an explicit Bool")
        self.strict = strict
        if strict and any(use.scientific_type is None for use in (*self.owns, *self.reads)):
            raise TypeError("strict Realize requires a scientific type for every owned/read Use")
        if not self.owns:
            raise ValueError("Realize must own at least one requested use")
        if not isinstance(outputs, Mapping) or any(type(k) is not Use for k in outputs):
            raise TypeError("Realize outputs must map Uses to explicit tuple paths")
        by_key = {use.key: tuple(path) for use, path in outputs.items()}
        if len(by_key) != len(outputs) or set(by_key) != {use.key for use in self.owns}:
            raise ValueError("Realize output bindings must cover each owned Use exactly once")
        if any(any(type(k) not in (str, int) or isinstance(k, bool) for k in p)
               for p in by_key.values()):
            raise TypeError("Realize output paths require string fields or integer indices")
        self.body = body
        from pops.claims import claims_manifest
        self.claims = tuple(claims)
        claims_manifest(self.claims)
        self.bindings = MappingProxyType({key: freeze_symbolic_metadata(value)
                                         for key, value in bindings.items()})
        self.outputs = tuple((use, by_key[use.key]) for use in self.owns)

    def evaluate(self, program):
        require_program(program, exact=False, where="Realize.evaluate")
        bound = tuple(_leaves(self.bindings))
        for value in bound:
            require_owned(program, value, "Realize binding")
        # Every pre-existing temporal dependency must be a declared input. Walking
        # stops at bound values, so their own valid provenance is not redefined.
        allowed = {value.id for value in bound}
        before = program._next_id
        with authoring_transaction(program):
            controls = _body_controls(program)
            product = self.body(program, **dict(self.bindings))
            if _body_controls(program) != controls:
                raise ValueError("Realize body changed Program policy, publication, or declarations")
            # Effects survive even when their numerical result is unused. Check
            # every created canonical node, not only the output dependency cone.
            body_nodes = tuple(program._canonical_value(value)
                               for value in program._issued_values.values() if value.id >= before)
            _validate_body_closure(program, body_nodes, allowed=allowed, before=before)
            results = []
            for index, (use, path) in enumerate(self.outputs):
                expression = _at_path(product, path)
                if use.scientific_type is not None:
                    from pops.representations import require_scientific_output
                    expression = require_scientific_output(expression, use.scientific_type, program)
                if self.strict:
                    from pops.scientific_graph import CONTRACT, validate_scientific_program, bind_scientific_value
                    canonical = program._canonical_value(expression)
                    if CONTRACT not in canonical.attrs or canonical.attrs[CONTRACT].get("mode") != "strict":
                        raise TypeError("strict Realize output requires checked equations from a strict ScientificMap")
                    validate_scientific_program(program, strict=True)
                    expression = program.product(value=expression)["value"]
                    bind_scientific_value(expression, use.scientific_type)
                elif isinstance(expression, ProgramValue) and expression.vtype == "scalar":
                    expression = expression + 0  # fresh scalar SSA, no alteration of the input
                value = program.value("%s_output_%d" % (self.name, index), expression)
                _validate_body_closure(program, (value,), allowed=allowed, before=before)
                attrs = dict(value.attrs)
                attrs["realization_use_v02"] = {
                    "name": self.name, "use": use.to_data(),
                    "reads": tuple(read.to_data() for read in self.reads),
                    "input_values": tuple(v.id for v in bound),
                    "output_path": path, "invocation": before,
                    "checking": "strict" if self.strict else "legacy_annotations",
                }
                if self.claims:
                    from pops.claims import claims_manifest
                    attrs["realization_use_v02"]["claims"] = claims_manifest(self.claims)
                value = program._replace_value(value, attrs=attrs)
                results.append((use, value))
            return Realized(self.name, tuple(results), self.reads)


def require_coverage(required, realizations):
    """Check delivery ownership; reading a collision does not consume its volume use."""
    required = _uses(required, "required uses")
    counts = {}
    for realization in realizations:
        if not isinstance(realization, Realize):
            raise TypeError("coverage requires concrete Realize builders")
        for use in realization.owns:
            if use.key in counts:
                raise ValueError("a required contribution has multiple owning realizations")
            counts[use.key] = realization
    if set(counts) != {use.key for use in required}:
        raise ValueError("realization coverage has missing or extra owned uses")
    return True


def validate_realized_outputs(program):
    """Require each declared output to reach an actual commit/effect consumer.

    Metadata alone is not consumption. The compiler calls this before lowering;
    unused ownership declarations cannot stand in for numerical contributions.
    """
    require_program(program, exact=False, where="realized output validation")
    values = {value.id: value for value in program._values}
    for value in program._values:
        for nested in program._subblock_value_refs(value):
            values.setdefault(nested.id, nested)
    _validate_continuation_coverage(program, values)
    declared = {value.id for value in values.values()
                if "realization_use_v02" in value.attrs}
    if not declared:
        return True
    live, seen = set(), set()

    def visit(value):
        if not isinstance(value, ProgramValue) or value.id in seen:
            return
        seen.add(value.id)
        live.add(value.id)
        support_only = value.attrs.get("support_only_inputs", ())
        for index, dependency in enumerate(value.inputs):
            if index not in support_only:
                visit(dependency)
        # Region outputs/conditions are real data dependencies even though their
        # SSA references live in attributes. Merely belonging to a body_block is
        # not consumption of every intermediate inside that block.
        for key, attribute in value.attrs.items():
            if not key.endswith("_block"):
                for dependency in _leaves(attribute):
                    visit(dependency)

    for value in program._commits.values():
        visit(value)
    # Explicit reductions/diagnostics and retained history are also legitimate
    # consumers. Merely materializing a dead intermediate is not.
    for value in values.values():
        if value.op in {"record_scalar", "record_balance_term", "store_history", "continuation_publish"}:
            visit(value)
    # Provisional evaluations may be superseded or never selected. They are not
    # missing deliveries when another evaluation of the exact same qualified Use
    # reaches a real consumer. ProgramRealization additionally checks the exact
    # target commit and the uniqueness of the final numerical frontier.
    from pops.identity.encoding import canonical_bytes
    from pops.identity.semantic import semantic_value
    def group(value):
        return (value.block, canonical_bytes(semantic_value(
            value.attrs["realization_use_v02"]["use"], where="realization Use")))
    consumed_groups = {group(values[value_id]) for value_id in declared & live}
    missing = {value_id for value_id in declared - live
               if group(values[value_id]) not in consumed_groups}
    if missing:
        raise ValueError("owned realization outputs are not consumed by the native program: %s"
                         % sorted(missing))
    return True


def require_continuation_coverage(publication, deliveries):
    """Bind physical Uses to exact paths of the published finite continuation.

    This library contract emits no numerical terminal. The compiler checks the
    actual candidate data flow on each successful path. It proves structural
    coverage, not the consistency/order or linearity of authored equations.
    """
    if not isinstance(publication, ProgramValue) or publication.op != "continuation_publish":
        raise TypeError("finite realization coverage requires the root continuation publication")
    require_owned(publication.prog, publication, "continuation realization coverage")
    if not isinstance(deliveries, Mapping) or not deliveries:
        raise TypeError("coverage requires nonempty path-to-Uses bindings")
    rows = []
    keys = set()
    for path, uses in deliveries.items():
        if not isinstance(path, tuple) or any(type(key) is not str or not key for key in path):
            raise TypeError("continuation delivery paths are tuples of Product field names")
        uses = _uses(uses, "continuation delivery")
        if not uses:
            raise ValueError("continuation delivery requires at least one Use")
        for use in uses:
            if use.key in keys:
                raise ValueError("one Use cannot have two distinct owned continuation deliveries")
            keys.add(use.key)
        rows.append({"path": path, "uses": tuple(use.to_data() for use in uses)})
    attrs = dict(publication.attrs)
    attrs["realization_continuation_v14"] = {"version": 1, "deliveries": tuple(rows)}
    replacement = publication.prog._replace_value(publication, attrs=attrs)
    return replacement


def _validate_continuation_coverage(program, values):
    from pops.codegen.program_realization import _key
    from pops.codegen.typed_delivery import require_single_delivery, term_label
    from pops.time._program.structured_control import value_signature
    publications = [value for value in values.values() if value.op == "continuation_publish"]
    for publication in publications:
        contract = publication.attrs.get("realization_continuation_v14")
        finite_outputs = [value for value in values.values()
                          if "realization_use_v02" in value.attrs and value.block is None]
        if contract is None:
            if finite_outputs:
                raise ValueError("finite realized Uses require exact continuation delivery paths")
            continue
        if set(contract) != {"version", "deliveries"} or contract["version"] != 1:
            raise ValueError("invalid continuation realization contract")
        expected = {}
        signature = value_signature(publication)["candidate"]["base"]["schema"]["write"]
        for row in contract["deliveries"]:
            if set(row) != {"path", "uses"} or not row["uses"]:
                raise ValueError("invalid continuation delivery binding")
            path = tuple(row["path"])
            target_type = signature
            for field in path:
                if target_type["vtype"] != "product" or field not in dict(target_type["members"]):
                    raise ValueError("continuation delivery path is absent from its exact schema")
                target_type = dict(target_type["members"])[field]
            for use in row["uses"]:
                key = _key(use)
                if key in expected:
                    raise ValueError("continuation coverage duplicates an owned Use")
                expected[key] = (path, term_label(use))
        outputs = {}
        for value in finite_outputs:
            key = _key(value.attrs["realization_use_v02"]["use"])
            outputs.setdefault(key, []).append(value)
        if set(outputs) != set(expected):
            raise ValueError("continuation coverage has missing or foreign finite Uses")
        for key, (path, term) in expected.items():
            output_ids = {value.id for value in outputs[key]}
            message = "Use needs exactly one delivered SSA evaluation on every continuation path"
            require_single_delivery(publication, lambda value, output_ids=output_ids: value.id in output_ids,
                                    term=term, path=("continuation",) + path,
                                    missing=message, duplicate=message)
    return True


__all__ = ["Use", "Realize", "Realized", "require_coverage", "require_continuation_coverage",
           "validate_realized_outputs"]
