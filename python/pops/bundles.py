"""Dependent finite mesh packages, checked by the native structural value algebra.

A Bundle is an ordinary native Product with a closed dependency schema.  Its
arrays are not accepted because their sizes happen to agree: each one must own
the exact runtime domain named by its binding.  Remapping is an authored graph
which returns the complete package, including geometry, maps and histories.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import new_typed, same_signature, value_signature
from pops.time._program.value_validation import require_owned
from pops.time.variant_types import ValueType, type_of


BUNDLE_OPS = frozenset(("bundle_validate", "bundle_remap_validate"))
_ROLES = ("instance", "fields", "geometry", "maps", "histories")
_DOMAIN_KEYS = ("family", "capacity", "ownership")


def _path(value):
    if not isinstance(value, (tuple, list)) or not value or any(
            type(item) is not str or not item for item in value):
        raise TypeError("Bundle paths must be nonempty sequences of member names")
    return tuple(value)


def _project_type(signature, path, *, domain=False):
    for key in path:
        kind = signature["vtype"]
        if kind == "array" and key == "[]" and not domain:
            signature = signature["element"]
        elif kind in ("product", "variant") and (kind == "product" or not domain):
            try:
                signature = dict(signature["members"])[key]
            except KeyError as error:
                raise ValueError("Bundle path names an absent member") from error
        else:
            raise TypeError("Bundle path traverses an incompatible structural type")
    return signature


def _dependent_paths(signature, prefix=(), *, root=True):
    # An inner Bundle owns its own relative binders and is validated recursively.
    if not root and signature.get("bundle") is not None:
        return
    kind = signature["vtype"]
    if kind in ("array", "index", "index_set"):
        yield prefix, signature
    if kind in ("product", "variant"):
        for name, child in signature["members"]:
            yield from _dependent_paths(child, (*prefix, name), root=False)
    elif kind == "array":
        yield from _dependent_paths(signature["element"], (*prefix, "[]"), root=False)
    elif kind not in ("index", "index_set", "scalar", "int64", "bool"):
        raise TypeError("Bundle data requires finite arrays and structural leaves; "
                        "a legacy Field needs an explicit representation map")


def _unmarked(signature):
    return {key: value for key, value in signature.items() if key != "bundle"}


def validate_bundle_signature(signature):
    """Validate the complete dependency contract, including after graph loading."""
    contract = signature.get("bundle")
    if signature.get("vtype") != "product" or not isinstance(contract, Mapping):
        raise TypeError("expected a dependent Bundle signature")
    members = dict(signature["members"])
    if tuple(members) != _ROLES or members["instance"]["vtype"] != "index_set":
        raise TypeError("Bundle requires ordered instance, fields, geometry, maps, histories")
    if any(members[role]["vtype"] != "product" for role in _ROLES[1:]):
        raise TypeError("Bundle fields, geometry, maps and histories must be Products")
    if set(contract) != {"domains", "bindings"}:
        raise ValueError("Bundle dependency contract has unknown or missing members")
    domains = tuple(_path(path) for path in contract["domains"])
    if len(set(domains)) != len(domains) or ("instance",) not in domains:
        raise ValueError("Bundle domains must uniquely include the current instance")
    domain_types = {}
    for path in domains:
        leaf = _project_type(signature, path, domain=True)
        if leaf["vtype"] != "index_set":
            raise TypeError("Bundle domain binder must name an IndexSet")
        domain_types[path] = leaf
    bindings = tuple((_path(path), _path(domain)) for path, domain in contract["bindings"])
    if len({path for path, _ in bindings}) != len(bindings):
        raise ValueError("Bundle dependent paths may be bound only once")
    dependents = dict(_dependent_paths(signature))
    if set(dependents) != {path for path, _ in bindings}:
        raise ValueError("Bundle bindings must cover every Array, Index and domain exactly")
    for path, domain in bindings:
        if domain not in domain_types:
            raise ValueError("Bundle binding names an undeclared domain")
        leaf, target = dependents[path], domain_types[domain]
        if any(leaf.get(key) != target.get(key) for key in _DOMAIN_KEYS):
            raise ValueError("Bundle binding changes a domain family, capacity or ownership")
        if leaf["vtype"] == "index_set" and path != domain:
            raise ValueError("Bundle domain declarations bind themselves, not another instance")
    return contract


@dataclass(frozen=True)
class BundleSchema:
    """Closed dependency paths for a complete finite package.

    ``domains`` maps role names to Product paths, including ``('instance',)``.
    ``bindings`` maps every Array/Index path to one of those names. ``'[]'``
    traverses each array element; a Variant member applies only to its active
    arm. Domain leaves are bound to themselves automatically. An inner Bundle
    carries and checks its own relative domain bindings.
    """
    input_type: ValueType
    value_type: ValueType
    __pops_ir_immutable__ = True

    def __init__(self, prototype, *, domains, bindings):
        if type(prototype) is ValueType:
            signature = prototype.signature
        else:
            signature = value_signature(prototype)
        if signature.get("bundle") is not None:
            raise TypeError("define a BundleSchema from an unmarked complete Product")
        if not isinstance(domains, dict) or not isinstance(bindings, dict):
            raise TypeError("Bundle domains and bindings require explicit mappings")
        names = {name: _path(path) for name, path in domains.items()}
        if any(type(name) is not str or not name for name in names):
            raise TypeError("Bundle domain roles require nonempty names")
        bound = []
        for path, name in bindings.items():
            if name not in names:
                raise ValueError("Bundle binding names an undeclared domain role")
            bound.append((_path(path), names[name]))
        bound.extend((path, path) for path in names.values())
        marked = dict(signature, bundle={"domains": tuple(names.values()),
                                         "bindings": tuple(bound)})
        validate_bundle_signature(marked)
        object.__setattr__(self, "input_type", ValueType(signature))
        object.__setattr__(self, "value_type", ValueType(marked))

    def bind(self, program, value):
        return _bind(program, self, value)


@atomic_authoring
def _bind(program, schema, value):
    require_owned(program, value, "Bundle input", vtype="product")
    if not same_signature(schema.input_type.signature, _unmarked(value_signature(value))):
        raise TypeError("Bundle construction changed its complete structural schema")
    return new_typed(program, schema.value_type.signature, "bundle_validate", (value,), {},
                     point=value.point)


@atomic_authoring
def Bundle(program, schema, *, instance, fields, geometry, maps, histories):
    """Construct all five native roles and execute exact-instance validation."""
    if type(schema) is not BundleSchema:
        raise TypeError("Bundle requires an exact BundleSchema")
    value = program.product(instance=instance, fields=fields, geometry=geometry,
                            maps=maps, histories=histories)
    return schema.bind(program, value)


@dataclass(frozen=True)
class Remap:
    """Closed graph plus explicit source-domain and history obligations.

    ``body(bundle, data)`` returns Result of the complete Bundle. Every history
    member has a declared ``transfer``, ``reconstruct`` or ``reject`` policy.
    These labels do not prove conservation: the graph supplies the actual
    equations and the enclosing Assess must test the numerical requirements.
    A ``reject`` policy refuses this remap, rather than erasing that history.
    """
    body: object
    source_instance: tuple
    histories: tuple

    def __init__(self, body, *, source_instance, histories):
        from pops.graph_functions import GraphFunction
        if type(body) is not GraphFunction:
            raise TypeError("Remap body must be a closed GraphFunction")
        if not isinstance(histories, dict) or any(
                type(name) is not str or not name or policy not in ("transfer", "reconstruct", "reject")
                for name, policy in histories.items()):
            raise TypeError("Remap requires one explicit policy per history member")
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "source_instance", _path(source_instance))
        object.__setattr__(self, "histories", tuple(histories.items()))


@atomic_authoring
def remap(program, method, bundle, *, data):
    """Return a native Result; failure cannot expose a partial new Bundle."""
    if type(method) is not Remap:
        raise TypeError("remap requires a closed Remap descriptor")
    require_owned(program, bundle, "remap base", vtype="product")
    signature = value_signature(bundle)
    validate_bundle_signature(signature)
    history_type = dict(signature["members"])["histories"]
    if {name for name, _ in history_type["members"]} != {name for name, _ in method.histories}:
        raise ValueError("Remap policy must account for every retained history member")
    if method.source_instance not in tuple(map(tuple, signature["bundle"]["domains"])):
        raise ValueError("Remap source_instance must be a declared domain binder")
    if method.source_instance == ("instance",):
        raise ValueError("Remap source instance must be retained separately from its new instance")

    def attempt(p):
        from pops.time.variant_types import Failure, Int64, Product, Result
        if any(policy == "reject" for _, policy in method.histories):
            errors = Failure(remap_history=Product(code=Int64))
            result = Result(type_of(bundle), errors)
            return p.require_success(p.variant(result, "err", p.variant(errors, "remap_history",
                                                                        p.product(code=p.int64(1)))))
        result = method.body(p, bundle=bundle, data=data)
        candidate = p.require_success(result)
        if not same_signature(signature, value_signature(candidate)):
            raise TypeError("Remap must return the complete same-family Bundle schema")
        return new_typed(p, signature, "bundle_remap_validate", (bundle, candidate),
                         {"source_instance": method.source_instance,
                          "history_policies": method.histories}, point=candidate.point)

    return program.capture(attempt)


__all__ = ("BundleSchema", "Bundle", "Remap", "remap")
