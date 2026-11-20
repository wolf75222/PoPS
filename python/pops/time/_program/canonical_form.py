"""One canonical form per frozen authoring Program (N11, F5).

A frozen authoring Program projects its canonical form (``_serialize``) once per provenance
variant and owns the result in a private slot, together with a witness of everything
that projection read. Every later read checks the witness first, and projects again when it fails,
so a read returns what a fresh projection would return.

A form is kept only when nothing it depends on can change in contract after it was computed:

- the Program is frozen and is an authoring Program, not a detached compile clone;
- every Case registry that ``canonical_handle`` consulted is frozen;
- both projections are closed builtin JSON with finite floats, so the decode is exact;
- the reachable projections are issued by stock implementations with a closed observer, and
  all the values they read are authenticated by the closure (``_immutable_closure``).

The witness holds the model-definition fingerprints the projection observed, the registries it
consulted, and every reachable object with the objects bound to its attributes, compared by
identity. A definition changed after freeze, or an out-of-contract ``object.__setattr__`` on the
Program, a node or a leaf, makes the witness fail.
"""
from __future__ import annotations

import gc
import json
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType, MethodType
from typing import Any, cast

# Private switch for the bit-for-bit test: False projects every read, as before this lot.
_ENABLED = True

# A form is built and checked outside the lock. Only its attachment to its exact Program is
# changed under the lock.
_LOCK = threading.Lock()
_FORM_SLOT = "_sealed_canonical_form"

_SCALARS = {id(kind): kind for kind in (
    type(None), bool, int, float, complex, str, bytes, Fraction, Decimal)}
_IMMUTABLE_CLASSES: dict[type, bool] = {}
_SLOT_NAMES: dict[type, tuple[str, ...]] = {}
_PROBE = "_pops_canonical_form_probe"
# Program attributes kept outside the IR and never read by the projection: the authoring body of
# a matrix-free operator (``Program.set_apply``) is a Python function that a reference solver
# calls again; the projection reads the traced ``apply_block`` nodes instead. The witness still
# holds the attribute itself, so rebinding it drops the form, but its functions are not walked.
_AUTHORING_ONLY = frozenset({"_operator_apply_bodies"})
# Instance and field registries are observed by canonical_handle; the Handle witness only
# traverses the values that the stock handle projections themselves can read.
_HANDLE_INPUTS = frozenset({
    "owner_path", "local_id", "kind", "schema_version", "_declaration_ref", "_block_ref",
    "param_kind", "signature", "category", "_registered_operator_name", "model_owner_path",
})
_UNSET = object()


_ObjectWitness = tuple[Any, type, tuple[str, ...], tuple[Any, ...]]
_MappingWitness = tuple[dict[Any, Any], tuple[tuple[Any, Any], ...]]


@dataclass(frozen=True, slots=True, eq=False)
class _Form:
    texts: tuple[str, str]
    ir_hash: str
    authorities: tuple[tuple[Any, Any], ...]
    registries: tuple[Any, ...]
    objects: tuple[_ObjectWitness, ...]
    registry_guards: tuple[tuple[Any, MethodType], ...]
    mappings: tuple[_MappingWitness, ...]


class _RecordingObserved(dict):
    """The fingerprint memo of one form build: records each authority the projection reads."""

    def __init__(self, inherited: Any) -> None:
        super().__init__(inherited)
        self.touched: dict[Any, None] = {}

    def __contains__(self, key: Any) -> bool:
        self.touched[key] = None
        return super().__contains__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.touched[key] = None
        super().__setitem__(key, value)


def eligible(program: Any) -> bool:
    return (_ENABLED and getattr(program, "_frozen", False) is True
            and getattr(program, "_compiled_detached", False) is not True)


def serialize(program: Any, include_provenance: bool) -> dict[str, Any]:
    """The canonical form of ``program``, from its kept form when the witness holds."""
    if not eligible(program):
        return _fresh_projection(program, include_provenance)
    form = _current_form(program)
    if form is not None:
        return json.loads(form.texts[include_provenance])
    return _build(program, include_provenance)[1]


def ir_hash(program: Any) -> str:
    """``canonical_json_sha256(_serialize(False))``, from the kept form when the witness holds."""
    from pops._structural_data import canonical_json_sha256

    if not eligible(program):
        return projected_ir_hash(program)
    form = _current_form(program)
    if form is None:
        form, data = _build(program, False)
        if form is None:
            return canonical_json_sha256(data)
    return form.ir_hash


def has_form(program: Any) -> bool:
    with _LOCK:
        return _attached_form(program) is not None


def _attached_form(program: Any) -> _Form | None:
    form = getattr(program, _FORM_SLOT, None)
    return form if type(form) is _Form and form.objects[0][0] is program else None


def projected_ir_hash(program: Any) -> str:
    """A real projection of the IR hash, bypassing any kept form."""
    from pops._structural_data import canonical_json_sha256

    return canonical_json_sha256(_fresh_projection(program, False))


def _fresh_projection(program: Any, include_provenance: bool) -> dict[str, Any]:
    from pops.time import references

    if references._REGISTRY_READS.get() is None:
        return program._canonical_projection(include_provenance=include_provenance)
    # A nested projection that is not kept cannot supply its registries to its caller's form.
    token = references._REGISTRY_READS.set(None)
    try:
        return program._canonical_projection(include_provenance=include_provenance)
    finally:
        references._REGISTRY_READS.reset(token)


def _current_form(program: Any) -> _Form | None:
    from .serialization import _IR_IDENTITY_REUSE

    with _LOCK:
        form = _attached_form(program)
    if form is None:
        return None
    scope = _IR_IDENTITY_REUSE.get()
    if scope is not None and not scope.applies():
        scope = None
    checked = scope is not None and form in scope.checked_forms
    if not (_mappings_hold(form.mappings) if checked else _witness_holds(program, form)):
        with _LOCK:
            if _attached_form(program) is form:
                object.__delattr__(program, _FORM_SLOT)
        return None
    if scope is not None:
        scope.checked_forms.add(form)
    return form


def _witness_holds(program: Any, form: _Form) -> bool:
    return _mappings_hold(form.mappings) and _owners_hold(program, form)


def _owners_hold(program: Any, form: _Form) -> bool:
    if form.objects[0][0] is not program or getattr(program, "_frozen", False) is not True:
        return False
    return (
        _keepable_registries(form.registries)
        and _objects_hold(program, form)
        and all(authority.fingerprint() == fingerprint
                for authority, fingerprint in form.authorities)
    )


def _objects_hold(program: Any, form: _Form) -> bool:
    for obj, kind, names, values in form.objects:
        if type(obj) is not kind:
            return False
        current_names, current_values = _state(obj, omit_form=obj is program)
        if current_names != names or len(current_values) != len(values):
            return False
        for current, recorded in zip(current_values, values):
            if current is not recorded:
                return False
    return all(_sealed_registry_guard(registry, guard)
               for registry, guard in form.registry_guards)


def _mapping_witness(proxy: MappingProxyType) -> _MappingWitness | None:
    """Observe the exact backing dict without dispatching into a provider Mapping."""
    if sys.implementation.name != "cpython":
        return None
    try:
        # CPython's mappingproxy tp_traverse visits exactly its backing mapping. The audit
        # event can be refused by the application, in which case we project normally.
        references = gc.get_referents(proxy)
        if len(references) == 1 and type(references[0]) is dict:
            backing = references[0]
            return backing, tuple(backing.items())
    except Exception:
        return None
    return None


def _mappings_hold(records: tuple[_MappingWitness, ...]) -> bool:
    try:
        for backing, entries in records:
            if len(backing) != len(entries):
                return False
            for actual, recorded in zip(backing.items(), entries, strict=True):
                if actual[0] is not recorded[0] or actual[1] is not recorded[1]:
                    return False
    except (RuntimeError, ValueError):
        # A size or key change during iteration cannot authenticate an earlier projection.
        return False
    return True


def _build(program: Any, first: bool) -> tuple[_Form | None, dict[str, Any]]:
    """Project ``program`` for variant ``first``; keep a form when everything allows it."""
    from pops._structural_data import canonical_json_sha256
    from pops.model import ownership

    closure = _immutable_closure(program)
    if closure is None:
        return None, _fresh_projection(program, first)
    authorities: dict[Any, Any] = {}
    registries: dict[int, Any] = {}
    data = _observed_projection(program, first, authorities, registries)
    if not _keepable_registries(registries.values()) or not _closed_json(data):
        return None, data
    other_authorities: dict[Any, Any] = {}
    other = _observed_projection(program, not first, other_authorities, registries)
    if any(other_authorities[authority] != authorities[authority]
           for authority in other_authorities.keys() & authorities.keys()):
        raise ownership.UnresolvedOwnershipError(
            "model definition changed during canonical projection")
    authorities.update(other_authorities)
    if not _keepable_registries(registries.values()) or not _closed_json(other):
        return None, data
    by_variant = {first: data, not first: other}
    try:
        without_provenance = _closed_text(by_variant[False])
        with_provenance = _closed_text(by_variant[True])
    except (ValueError, RecursionError):
        return None, data
    objects, registry_guards, mappings = closure
    form = _Form(
        (without_provenance, with_provenance),
        canonical_json_sha256(by_variant[False]),
        tuple(authorities.items()),
        tuple(registries.values()),
        objects,
        registry_guards,
        mappings,
    )
    # An inherited definition memo cannot authenticate publication. Re-read each definition at
    # the boundary, including authorities that only one of the two variants reached.
    token = ownership._FINGERPRINT_SCOPE.set(None)
    try:
        for authority, fingerprint in form.authorities:
            if authority.fingerprint() != fingerprint:
                raise ownership.UnresolvedOwnershipError(
                    "model definition changed during canonical projection")
    finally:
        ownership._FINGERPRINT_SCOPE.reset(token)
    if not _objects_hold(program, form) or not _keepable_registries(form.registries):
        raise RuntimeError("Program IR identity changed during canonical projection")
    if not _mappings_hold(form.mappings):
        return None, _fresh_projection(program, first)

    with _LOCK:
        if _attached_form(program) is None:
            object.__setattr__(program, _FORM_SLOT, form)
    return form, data


def _closed_text(value: dict[str, Any]) -> str:
    text = json.dumps(value, sort_keys=False, separators=(",", ":"), allow_nan=False)
    json.loads(text)  # an integer beyond the decoder's digit limit keeps no form
    return text


def _observed_projection(
    program: Any, include_provenance: bool, authorities: dict[Any, Any],
    registries: dict[int, Any],
) -> dict[str, Any]:
    """Project once, recording the definitions and registries the projection reads.

    The projection runs in its own fingerprint scope, seeded with the enclosing scope's memo so it
    reads exactly the fingerprints a projection inside that scope would read. Its observations go
    back to the enclosing scope, which checks them when it ends, or are checked here, as the
    outermost ``_definition_fingerprint_scope`` does, when there is none.
    """
    from pops.model import ownership
    from pops.time import references

    outer = ownership._FINGERPRINT_SCOPE.get()
    outer_observed = outer.observed if outer is not None and outer.applies() else None
    projection = ownership._FingerprintProjection()
    observed = _RecordingObserved(outer_observed or {})
    projection.observed = observed
    reads = references._RegistryReads()
    token = ownership._FINGERPRINT_SCOPE.set(projection)
    reads_token = references._REGISTRY_READS.set(reads)
    try:
        return program._canonical_projection(include_provenance=include_provenance)
    finally:
        recorded_registries = reads.close()
        references._REGISTRY_READS.reset(reads_token)
        ownership._FINGERPRINT_SCOPE.reset(token)
        projection.active = False
        try:
            touched = {authority: dict.__getitem__(observed, authority)
                       for authority in observed.touched if dict.__contains__(observed, authority)}
        finally:
            observed.clear()
            observed.touched.clear()
        authorities.update(touched)
        for registry in recorded_registries:
            registries[id(registry)] = registry
        if outer_observed is not None:
            for authority, fingerprint in touched.items():
                outer_observed.setdefault(authority, fingerprint)
        else:
            for authority, fingerprint in touched.items():
                if authority.fingerprint() != fingerprint:
                    raise ownership.UnresolvedOwnershipError(
                        "model definition changed during canonical projection")


def _keepable_registries(registries: Any) -> bool:
    # One frozen registry means its whole Case graph is sealed: freeze_problem_graph freezes the
    # block, field, time, param and initial registries in one transaction (freeze_atomically).
    return all(getattr(registry, "_frozen", False) is True for registry in registries)


def _closed_json(value: Any) -> bool:
    """Exact dict/list/str/int/finite float/bool/None, so ``json`` round-trips it exactly."""
    import math

    stack = [value]
    while stack:
        node = stack.pop()
        kind = type(node)
        if node is None or kind is str or kind is int or kind is bool:
            continue
        if kind is float:
            if not math.isfinite(node):
                return False
            continue
        if kind is dict:
            if any(type(key) is not str for key in node):
                return False
            stack.extend(node.values())
        elif kind is list:
            stack.extend(node)
        else:
            return False
    return True


def _slot_names(kind: type) -> tuple[str, ...]:
    names = _SLOT_NAMES.get(kind)
    if names is None:
        found: list[str] = []
        for klass in kind.__mro__:
            slots = klass.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot not in ("__weakref__", "__dict__") and slot not in found:
                    found.append(slot)
        names = _SLOT_NAMES[kind] = tuple(found)
    return names


def _state(obj: Any, *, omit_form: bool = False) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """The attribute names of ``obj`` and the objects bound to them."""
    slots = _slot_names(type(obj))
    if omit_form:
        slots = tuple(name for name in slots if name != _FORM_SLOT)
    values = [getattr(obj, name, _UNSET) for name in slots]
    names: tuple[str, ...] = slots
    attributes = getattr(obj, "__dict__", None)
    if attributes:
        names = slots + tuple(attributes)
        values.extend(attributes.values())
    return names, tuple(values)


def _immutable_class(kind: type, sample: Any) -> bool:
    known = _IMMUTABLE_CLASSES.get(kind)
    if known is not None:
        return known
    from pops._ir.symbolic import ImmutableSymbolic

    params = getattr(kind, "__dataclass_params__", None)
    if params is not None and params.frozen:
        result = True
    elif all(klass.__dict__.get("__slots__", None) == () for klass in kind.__mro__[:-1]):
        # Stateless: every class but object declares empty __slots__, so an instance has no
        # attribute to assign (CopyCurrent). A builtin container or an ndarray declares none.
        result = True
    elif issubclass(kind, ImmutableSymbolic):
        result = True
    elif kind.__setattr__ is not object.__setattr__:
        # A class with its own __setattr__, marked __pops_ir_immutable__ or not (AdvanceControl
        # is not). The marker is only a claim: check once per class that assignment is refused.
        try:
            setattr(sample, _PROBE, None)
        except Exception:
            result = True
        else:
            try:
                object.__delattr__(sample, _PROBE)
            except AttributeError:
                pass
            result = False
    else:
        result = False
    _IMMUTABLE_CLASSES[kind] = result
    return result


def _sealed_registry_guard(registry: Any, guard: MethodType) -> bool:
    from pops.model._module_freeze import ModuleFreezable
    from pops.model.module import Module
    from pops.model.registry import OperatorRegistry
    from pops.physics._freeze import _FROZEN_CLASSES

    module = cast(Module, guard.__self__)
    return (
        type(registry) is _FROZEN_CLASSES.get(OperatorRegistry)
        and type(module) is _FROZEN_CLASSES.get(Module)
        and registry.frozen and module.frozen
        and guard.__func__ is ModuleFreezable._guard_mutable
        and module._registry is registry
        and type(registry._by_name) is MappingProxyType
        and type(registry._order) is tuple
        and type(registry._aliases) is MappingProxyType
    )


def _immutable_closure(
    program: Any,
) -> tuple[tuple[_ObjectWitness, ...], tuple[tuple[Any, MethodType], ...],
           tuple[_MappingWitness, ...]] | None:
    """Every object reachable from ``program``, with its attribute bindings, or None.

    None when a source cannot authenticate its projection. Case registries and
    ``_AuthoringAuthority`` are observed by canonical_handle and the fingerprint scope.
    The node's ``prog`` reference and other authoring-only inputs are outside the projection.
    """
    from pops._ir.symbolic import ImmutableSymbolic
    from pops.model.ownership import _AuthoringAuthority
    from pops.time._step.strategy import StepStrategy
    from .api import Program
    from .canonical_form_authorities import projection_authorities

    if type(program) is not Program:
        return None
    admitted, leaf_types = projection_authorities()

    root_names, root_values = _state(program, omit_form=True)
    records: list[_ObjectWitness] = [(program, type(program), root_names, root_values)]
    registry_guards: list[tuple[Any, MethodType]] = []
    mappings: list[_MappingWitness] = []
    stack = [value for name, value in zip(root_names, root_values)
             if name not in _AUTHORING_ONLY]
    seen: set[int] = {id(program)}
    while stack:
        node = stack.pop()
        kind = type(node)
        if _SCALARS.get(id(kind)) is kind:
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if kind is tuple or kind is frozenset:
            stack.extend(node)
            continue
        if kind is MappingProxyType:
            observed = _mapping_witness(node)
            if observed is None:
                return None
            mappings.append(observed)
            for key, value in observed[1]:
                stack.append(key)
                stack.append(value)
            continue
        reference = leaf_types.get(id(kind)) is kind
        if (reference and isinstance(node, Enum)) or kind is _AuthoringAuthority:
            continue
        if not reference and (admitted.get(id(kind)) is not kind or not _immutable_class(kind, node)):
            return None
        names, values = _state(node)
        if isinstance(node, StepStrategy) and any(
            _SCALARS.get(id(type(value))) is not type(value) for value in values
        ):
            return None
        records.append((node, kind, names, values))
        symbolic = isinstance(node, ImmutableSymbolic)
        for name, value in zip(names, values):
            if (value is _UNSET or (symbolic and name == "prog")
                    or (reference and name not in _HANDLE_INPUTS)):
                continue
            if name == "_mutation_guard" and type(value) is MethodType \
                    and _sealed_registry_guard(node, value):
                registry_guards.append((node, value))
                continue
            stack.append(value)
    return tuple(records), tuple(registry_guards), tuple(mappings)


__all__ = ["eligible", "has_form", "ir_hash", "projected_ir_hash", "serialize"]
