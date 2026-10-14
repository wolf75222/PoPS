"""Native continuation capabilities; staging implementation, never an integer token DSL.

Every operation below is an ordinary Program IR operation lowered to the native owned store.
Structural metadata describes the payload; it cannot manufacture a live capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import new_typed, value_signature, same_signature
from pops.time._program.value_validation import require_owned, require_top_level
from pops.time.canonical_data import _json_ready
from pops.time.variant_types import ValueType
from pops.bundles import BUNDLE_OPS, validate_bundle_signature


CAPABILITY_TYPES = frozenset(("continuation_base", "continuation_candidate",
                             "continuation_assessment", "continuation_accepted"))
VERSIONED_OPS = frozenset(("continuation_open", "continuation_read", "continuation_freeze",
                          "continuation_candidate_value", "continuation_proposal",
                          "continuation_candidate_base", "continuation_attest",
                          "continuation_assessment_ok", "continuation_assessment_checks", "continuation_accept",
                          "continuation_accepted_candidate", "continuation_child_base",
                          "continuation_transform", "continuation_consume_increment", "continuation_publish")) | BUNDLE_OPS


def _digest(value):
    data = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()



def validate_schema_data(schema):
    """Reject a changed structural contract retaining an old compiled identity.

    This is integrity of a declaration, not evidence that its equations are true.
    Re-authoring another schema changes its identity and remains an explicit model change.
    """
    from collections.abc import Mapping
    keys = {"identity", "read", "write", "fields", "time_path", "step_path", "roles",
            "byte_capacity", "scientific", "current_dependencies"}
    if not isinstance(schema, Mapping) or set(schema) != keys:
        raise ValueError("continuation schema has incomplete mandatory declarations")
    data = {key: schema[key] for key in keys - {"identity", "time_path", "step_path"}}
    data.update(time=schema["time_path"], step=schema["step_path"])
    if schema["identity"] != _digest(data):
        raise ValueError("continuation schema declarations differ from their compiled identity")


def validate_signature_schemas(signature, seen=None):
    """Traverse structural capability types; each exact schema object is checked once."""
    from collections.abc import Mapping
    seen = set() if seen is None else seen
    def visit(item):
        if not isinstance(item, (Mapping, tuple, list)) or id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, Mapping):
            if item.get("vtype") == "continuation_base":
                validate_schema_data(item["schema"])
                return
            for child in item.values():
                visit(child)
        else:
            for child in item:
                visit(child)
    visit(signature)


def _path(value):
    if not isinstance(value, (tuple, list)) or not value or any(type(x) is not str or not x for x in value):
        raise TypeError("continuation paths must be nonempty tuples of named Product members")
    return tuple(value)


def _at(signature, path):
    for key in _path(path):
        if signature["vtype"] != "product":
            raise TypeError("continuation binding traverses a non-Product")
        try:
            signature = dict(signature["members"])[key]
        except KeyError as exc:
            raise ValueError("continuation binding names a missing member") from exc
    return signature


def _replace(signature, path, replacement):
    key, *tail = path
    if signature["vtype"] != "product" or key not in dict(signature["members"]):
        raise ValueError("continuation binding names a missing Product member")
    return dict(signature, members=tuple((name, (_replace(child, tail, replacement) if tail else replacement)
                                         if name == key else child)
                                        for name, child in signature["members"]))


def _fields(signature, path=()):
    kind = signature["vtype"]
    if kind in ("state", "rhs", "scalar_field"):
        yield path, signature
    elif kind in ("product", "variant"):
        for key, child in signature["members"]:
            yield from _fields(child, (*path, key))
    elif kind == "array":
        yield from _fields(signature["element"], (*path, "[]"))


def _persistable(signature):
    kind = signature["vtype"]
    if kind in CAPABILITY_TYPES:
        raise TypeError("live continuation capabilities cannot be persisted as scientific leaves")
    if kind in ("product", "variant"):
        if signature.get("bundle") is not None:
            validate_bundle_signature(signature)
        for _, child in signature["members"]:
            _persistable(child)
    elif kind == "array":
        if signature.get("ownership") not in ("replicated", "contiguous_distributed"):
            raise NotImplementedError("continuation Array ownership has no native structural codec")
        _persistable(signature["element"])
    elif kind not in ("scalar", "int64", "bool", "index_set", "index", "state", "scalar_field"):
        raise TypeError("unsupported persistent continuation leaf %r" % kind)


@dataclass(frozen=True)
class ContinuationSchema:
    """Compiled structure with distinct authenticated current and outgoing field views.

    Bindings are pairs `(U.next, path)`. They cover every declared state block exactly once.
    Nested fields in Array/Variant require further publication support and fail closed here.
    """
    read_type: ValueType
    write_type: ValueType
    fields: tuple
    time_path: tuple
    step_path: tuple
    roles: tuple
    byte_capacity: int
    scientific: tuple
    current_dependencies: tuple
    identity: str
    __pops_ir_immutable__ = True

    def __init__(self, initial, *, fields=(), time, step, roles=(), byte_capacity=64 * 1024 * 1024, scientific=None, current_dependencies=()):
        from pops.time.handles import StateEndpointHandle
        signature = value_signature(initial)
        if signature["vtype"] != "product":
            raise TypeError("ContinuationSchema requires the complete native Product")
        _persistable(signature)
        time, step = _path(time), _path(step)
        if _at(signature, time)["vtype"] != "scalar" or _at(signature, step)["vtype"] != "int64":
            raise TypeError("continuation time must be Float64 and step must be Int64")
        if type(byte_capacity) is not int or byte_capacity <= 0 or byte_capacity >= 2**63:
            raise ValueError("continuation byte capacity must be a positive int64 bound")
        bindings, owners, paths = [], set(), set()
        outgoing = signature
        for endpoint, path in fields:
            path = _path(path)
            if not isinstance(endpoint, StateEndpointHandle):
                raise TypeError("continuation field binding requires U.next")
            leaf = _at(signature, path)
            if leaf["vtype"] != "state" or leaf.get("block") != endpoint.block:
                raise ValueError("continuation field binding must own the exact state block")
            if endpoint.state in owners or path in paths:
                raise ValueError("duplicate continuation state projection")
            from pops.time._program.value_validation import require_compatible_spaces
            require_compatible_spaces(endpoint.space, leaf.get("space"), "continuation field binding", typed_pair=True)
            owners.add(endpoint.state)
            paths.add(path)
            bindings.append((endpoint, path))
            outgoing = _replace(outgoing, path, dict(leaf, point=endpoint.point))
        if {path for path, _ in _fields(signature)} != paths:
            raise NotImplementedError("every persistent field needs one exact fixed Product publication binding")
        roles = tuple(( _path(path), role) for path, role in roles)
        if len({path for path, _ in roles}) != len(roles):
            raise ValueError("duplicate continuation role path")
        for path, role in roles:
            _at(signature, path)
            if role not in ("scientific", "retry_control", "reconstructible_cache"):
                raise ValueError("unknown continuation role")
        from pops.scientific_graph import scientific_packet_schema
        scientific = scientific_packet_schema(signature, scientific)
        from pops.current_dependencies import current_dependencies_data
        current_dependencies = current_dependencies_data(outgoing, current_dependencies)
        data = dict(read=signature, write=outgoing, fields=tuple((endpoint.state, path) for endpoint, path in bindings),
                    time=time, step=step, roles=roles, byte_capacity=byte_capacity, scientific=scientific,
                    current_dependencies=current_dependencies)
        for name, value in dict(read_type=ValueType(signature), write_type=ValueType(outgoing),
                                fields=tuple(bindings), time_path=time, step_path=step, roles=roles,
                                byte_capacity=byte_capacity, scientific=scientific, current_dependencies=current_dependencies, identity=_digest(data)).items():
            object.__setattr__(self, name, value)

    def to_data(self):
        return dict(identity=self.identity, read=self.read_type.signature, write=self.write_type.signature,
                    fields=tuple((endpoint.state, path) for endpoint, path in self.fields),
                    time_path=self.time_path, step_path=self.step_path, roles=self.roles,
                    byte_capacity=self.byte_capacity, scientific=self.scientific, current_dependencies=self.current_dependencies)


def _cap(kind, **metadata):
    return dict(vtype=kind, **metadata)


def _require(program, value, kind, where):
    require_owned(program, value, where, vtype=kind)
    signature = value_signature(value)
    validate_signature_schemas(signature)
    return signature


def _same(expected, actual, where):
    if not same_signature(expected, actual):
        raise ValueError(where + ": exact structural payload/authority mismatch")


def validate_continuations(program, *, target=None):
    """Check the closed executable profile and unique root publication after deserialization too."""
    def walk(values):
        for value in values:
            yield value
            for key in ("body_block", "cond_block", "true_block", "false_block", "apply_block", "residual_block"):
                yield from walk(value.attrs.get(key, ()))
    values = tuple(walk(program._values))
    seen_schemas = set()
    for value in values:
        # Validate type metadata where each IR family actually stores it. Legacy
        # geometry and opaque iteration results need not be structured-control
        # carriers in order to participate in an ordinary Program.
        validate_signature_schemas(value.attrs, seen_schemas)
        if value.vtype in CAPABILITY_TYPES:
            # A live capability must still carry its mandatory structural type.
            validate_signature_schemas(value_signature(value), seen_schemas)
        if value.op in ("continuation_open", "continuation_publish"):
            validate_schema_data(value.attrs["schema"])
    from pops.ledger_provenance import validate_program_contracts
    validate_program_contracts(program)
    for value in values:
        if value.op in BUNDLE_OPS:
            signature = value_signature(value)
            validate_bundle_signature(signature)
            expected_inputs = 1 if value.op == "bundle_validate" else 2
            if len(value.inputs) != expected_inputs:
                raise ValueError("Bundle validation has an incomplete native input contract")
            if value.op == "bundle_remap_validate":
                for item in value.inputs:
                    _same(signature, value_signature(item), "remapped Bundle")
                path = tuple(value.attrs.get("source_instance", ()))
                if path == ("instance",) or path not in tuple(map(tuple, signature["bundle"]["domains"])):
                    raise ValueError("remap must retain a separately bound source instance")
                histories = dict(signature["members"])["histories"]["members"]
                policies = tuple(value.attrs.get("history_policies", ()))
                if len(policies) != len(histories) or {name for name, _ in policies} != {name for name, _ in histories} \
                        or any(policy not in ("transfer", "reconstruct") for _, policy in policies):
                    raise ValueError("remap must execute one transfer/reconstruction for every retained history")
            else:
                actual = value_signature(value.inputs[0])
                _same({k: v for k, v in signature.items() if k != "bundle"},
                      {k: v for k, v in actual.items() if k != "bundle"}, "Bundle input")
    opened = [value for value in values if value.op == "continuation_open"]
    active = any(value.op in VERSIONED_OPS - BUNDLE_OPS or value.attrs.get("continuation_scope") for value in values)
    if not active:
        return
    if target is not None and target != "system":
        raise NotImplementedError("Continuation publication currently requires fixed-topology Uniform")
    publications = [value for value in values if value.op == "continuation_publish"]
    if len(opened) != 1 or opened[0].region or len(publications) != 1 or publications[0].region:
        raise ValueError("Continuation requires one initialization and one root publication")
    from pops.graph_functions import _PURE, _COLLECTIVE
    allowed = _PURE | _COLLECTIVE | VERSIONED_OPS | {"state", "requested_dt", "reached_duration"}
    for value in values:
        if value.op not in allowed:
            raise NotImplementedError("Continuation operation %r is not a qualified pure trial/projection" % value.op)
        if value.op == "reached_duration" and value.id > publications[0].id:
            raise ValueError("reached duration must precede the complete continuation publication")
        if value.attrs.get("continuation_scope"):
            if value.op != "capture_result" or value.attrs["success_signature"]["vtype"] != "continuation_accepted":
                raise ValueError("continuation scope must return its complete accepted provisional")
    schema = opened[0].attrs["schema"]
    if schema != publications[0].attrs["schema"]:
        raise ValueError("continuation publication changed the installed schema")
    fields = {state_ref for state_ref, _ in schema["fields"]}
    if fields != set(program._commits):
        raise ValueError("ordinary state commits cannot coexist with the complete continuation publication")
    if getattr(program, "_histories", None) or getattr(program, "_post_sync_commits", None):
        raise NotImplementedError("legacy retained history and post-synchronization effects are not Continuation leaves")
    # Authenticate commit roots as exact projections of the one accepted candidate, not merely
    # values with matching shapes. This check is repeated after graph load/rebuild.
    publication = publications[0]
    for state_ref, path in schema["fields"]:
        node = program._commits[state_ref]
        for key in reversed(("continuation", *path)):
            if node.op != "product_get" or node.attrs.get("field") != key or len(node.inputs) != 1:
                raise ValueError("continuation carrier commit lost its authenticated projection")
            node = node.inputs[0]
        if node.op != "continuation_candidate_value" or len(node.inputs) != 1:
            raise ValueError("continuation carrier does not read the complete candidate")
        node = node.inputs[0]
        if node.op != "continuation_accepted_candidate" or node.inputs != publication.inputs:
            raise ValueError("continuation carrier projects a different accepted candidate")


class _ProgramVersionedValues:
    @atomic_authoring
    def continuation_capture(self, body):
        """A real native child scope, revoked on Failure and promoted on accepted success."""
        from pops.time._program.structured_control import success_signature
        from pops.time.variant_types import Failure, Result, type_of
        def wrapped(p):
            output = body(p)
            if output.vtype == "continuation_accepted":
                return p.variant(Result(type_of(output), Failure()), "ok", output)
            if success_signature(output)["vtype"] != "continuation_accepted":
                raise TypeError("a scoped continuation trial must return its complete AcceptedProvisional")
            return output
        return self.capture(wrapped, _continuation_scope=True)

    @atomic_authoring
    def continuation_open(self, initial, schema):
        require_top_level(self, initial, "continuation initialization")
        if type(schema) is not ContinuationSchema:
            raise TypeError("continuation_open requires an exact ContinuationSchema")
        _same(schema.read_type.signature, value_signature(initial), "continuation initialization")
        if any(v.op == "continuation_open" for v in self._values):
            raise ValueError("one root continuation schema is allowed per Program")
        declared = {v.state_ref for v in self._values if v.op == "state"}
        if {endpoint.state for endpoint, _ in schema.fields} != declared:
            raise ValueError("continuation projections must cover every declared native state")
        validate_schema_data(schema.to_data())
        signature = _cap("continuation_base", schema=schema.to_data(), view="read")
        value = new_typed(self, signature, "continuation_open", (initial,),
                          {"schema": schema.to_data()}, point=initial.point)
        if schema.scientific:
            from pops.scientific_graph import validate_scientific_continuation
            validate_scientific_continuation(value)
        return value

    @atomic_authoring
    def continuation_read(self, base):
        signature = _require(self, base, "continuation_base", "continuation read")
        return new_typed(self, signature["schema"][signature["view"]], "continuation_read", (base,), {}, point=base.point)

    @atomic_authoring
    def continuation_freeze(self, base, proposal, envelope):
        signature = _require(self, base, "continuation_base", "candidate base")
        require_owned(self, proposal, "candidate proposal", vtype="product")
        require_owned(self, envelope, "complete candidate", vtype="product")
        members = dict(value_signature(envelope)["members"])
        if tuple(members) != ("continuation", "increment", "observations", "checks"):
            raise TypeError("Candidate requires ordered continuation, increment, observations, checks")
        _same(signature["schema"]["write"], members["continuation"], "candidate continuation")
        for item in members.values():
            _persistable(item)
        _persistable(value_signature(proposal))
        output = _cap("continuation_candidate", base=signature, proposal=value_signature(proposal),
                      envelope=value_signature(envelope))
        value = new_typed(self, output, "continuation_freeze", (base, proposal, envelope), {}, point=envelope.point)
        from pops.ledger_provenance import validate_candidate
        proof = validate_candidate(value)
        value = self._replace_value(value, attrs=dict(value.attrs, increment_proof_v15=proof))
        if signature["schema"].get("scientific"):
            from pops.scientific_graph import validate_scientific_continuation
            validate_scientific_continuation(value)
        return value

    @atomic_authoring
    def continuation_candidate_value(self, candidate):
        signature = _require(self, candidate, "continuation_candidate", "candidate value")
        return new_typed(self, signature["envelope"], "continuation_candidate_value", (candidate,), {}, point=candidate.point)

    @atomic_authoring
    def continuation_proposal(self, candidate):
        signature = _require(self, candidate, "continuation_candidate", "candidate proposal")
        return new_typed(self, signature["proposal"], "continuation_proposal", (candidate,), {}, point=candidate.point)

    @atomic_authoring
    def continuation_candidate_base(self, candidate):
        signature = _require(self, candidate, "continuation_candidate", "candidate base")
        return new_typed(self, signature["base"], "continuation_candidate_base", (candidate,), {}, point=candidate.point)

    @atomic_authoring
    def continuation_attest(self, candidate, assessment):
        signature = _require(self, candidate, "continuation_candidate", "assessment candidate")
        from pops.graph_functions import GraphFunction
        if type(assessment) is not GraphFunction:
            raise TypeError("continuation assessment requires the exact closed executed GraphFunction")
        base = self.continuation_candidate_base(candidate)
        decision = assessment(self, base=self.continuation_read(base),
                              proposal=self.continuation_proposal(candidate),
                              candidate=self.continuation_candidate_value(candidate))
        decision_type = value_signature(decision)
        if decision_type["vtype"] != "product" or set(dict(decision_type["members"])) != {"accepted", "checks"}:
            raise TypeError("Assess must return exactly Product(accepted,checks)")
        accepted, checks = decision["accepted"], decision["checks"]
        require_owned(self, checks, "assessment payload", vtype="product")
        require_owned(self, accepted, "assessment decision", vtype="bool")
        _persistable(value_signature(checks))
        output = _cap("continuation_assessment", candidate=signature, checks=value_signature(checks))
        return new_typed(self, output, "continuation_attest", (candidate, checks, accepted),
                         {"assessment_identity": assessment.identity}, point=candidate.point)

    @atomic_authoring
    def continuation_assessment_ok(self, assessment):
        _require(self, assessment, "continuation_assessment", "assessment decision")
        from pops.time._program.structured_control import scalar_signature
        return new_typed(self, scalar_signature("bool"), "continuation_assessment_ok", (assessment,), {}, point=assessment.point)

    @atomic_authoring
    def continuation_assessment_checks(self, assessment):
        signature = _require(self, assessment, "continuation_assessment", "assessment checks")
        return new_typed(self, signature["checks"], "continuation_assessment_checks", (assessment,), {}, point=assessment.point)

    @atomic_authoring
    def continuation_accept(self, candidate, assessment):
        candidate_type = _require(self, candidate, "continuation_candidate", "accepted candidate")
        assessment_type = _require(self, assessment, "continuation_assessment", "accepted assessment")
        _same(candidate_type, assessment_type["candidate"], "candidate/assessment type")
        # Runtime also checks exact dynamic identity: equal structural types never suffice.
        return new_typed(self, _cap("continuation_accepted", candidate=candidate_type,
                                   checks=assessment_type["checks"]), "continuation_accept",
                         (candidate, assessment), {}, point=candidate.point)

    @atomic_authoring
    def continuation_accepted_candidate(self, accepted):
        signature = _require(self, accepted, "continuation_accepted", "accepted provisional candidate")
        return new_typed(self, signature["candidate"], "continuation_accepted_candidate", (accepted,), {}, point=accepted.point)

    @atomic_authoring
    def continuation_transform(self, previous, envelope, proposal=None):
        signature = _require(self, previous, "continuation_candidate", "candidate transformation")
        base = self.continuation_candidate_base(previous)
        proposal = self.continuation_proposal(previous) if proposal is None else proposal
        replacement = self.continuation_freeze(base, proposal, envelope)
        _same(signature, value_signature(replacement), "transformed candidate")
        return new_typed(self, signature, "continuation_transform", (previous, replacement), {}, point=previous.point)

    @atomic_authoring
    def continuation_child_base(self, accepted):
        signature = _require(self, accepted, "continuation_accepted", "accepted child base")
        schema = signature["candidate"]["base"]["schema"]
        if schema["fields"]:
            raise NotImplementedError("nested evolving field-frontier rebinding is not yet qualified")
        return new_typed(self, _cap("continuation_base", schema=schema, view="write"),
                         "continuation_child_base", (accepted,), {}, point=accepted.point)

    @atomic_authoring
    def continuation_consume_increment(self, accepted):
        """Consume the exact accepted child delta once in this native parent scope."""
        signature = _require(self, accepted, "continuation_accepted", "consume child increment")
        increment = dict(signature["candidate"]["envelope"]["members"])["increment"]
        return new_typed(self, increment, "continuation_consume_increment", (accepted,), {}, point=accepted.point)

    @atomic_authoring
    def continuation_publish(self, accepted):
        require_top_level(self, accepted, "root continuation publication")
        signature = _require(self, accepted, "continuation_accepted", "root continuation publication")
        if self._commits or any(v.op == "continuation_publish" for v in self._values):
            raise ValueError("continuation publication must be the unique root scientific commit")
        schema = signature["candidate"]["base"]["schema"]
        candidate = self.continuation_accepted_candidate(accepted)
        root = self.continuation_candidate_value(candidate)["continuation"]
        # These roots preserve existing model/partition routing. The emitter must suppress the
        # legacy commit_many and publish only through the native authenticated image adapter.
        for state_ref, path in schema["fields"]:
            value = root
            for key in path:
                value = value[key]
            self._commit_state(state_ref, value)
        return new_typed(self, value_signature(accepted), "continuation_publish", (accepted,),
                         {"schema": schema}, point=accepted.point)
