"""One canonical form per frozen authoring Program (N11, F5).

A frozen authoring Program projects its canonical form (``_serialize``) once per provenance
variant and keeps the result here, outside the Program, together with a witness of everything
that projection read. Every later read checks the witness first, and projects again when it fails,
so a read returns what a fresh projection would return.

A form is kept only when nothing it depends on can change in contract after it was computed:

- the Program is frozen and is an authoring Program, not a detached compile clone;
- every Case registry that ``canonical_handle`` consulted is frozen;
- both projections are closed builtin JSON with finite floats, so the decode is exact;
- every object reachable from the Program's attributes and nodes is of an immutable kind
  (``_immutable_closure``).

The witness holds the model-definition fingerprints the projection observed, the registries it
consulted, and every reachable object with the objects bound to its attributes, compared by
identity. A definition changed after freeze, or an out-of-contract ``object.__setattr__`` on the
Program, a node or a leaf, makes the witness fail.
"""
from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

# Private switch for the bit-for-bit test: False projects every read, as before this lot.
_ENABLED = True

_FORMS: WeakKeyDictionary[Any, _Form] = WeakKeyDictionary()
# Forms are built outside the lock; the dictionary is read and written only under it. Two threads
# that build the same form at once keep one of two equal forms.
_LOCK = threading.Lock()

_SCALARS = frozenset({type(None), bool, int, float, complex, str, bytes, Fraction, Decimal})
_IMMUTABLE_CLASSES: dict[type, bool] = {}
_SLOT_NAMES: dict[type, tuple[str, ...]] = {}
_PROBE = "_pops_canonical_form_probe"
# Program attributes kept outside the IR and never read by the projection: the authoring body of
# a matrix-free operator (``Program.set_apply``) is a Python function that a reference solver
# calls again; the projection reads the traced ``apply_block`` nodes instead. The witness still
# holds the attribute itself, so rebinding it drops the form, but its functions are not walked.
_AUTHORING_ONLY = frozenset({"_operator_apply_bodies"})
_UNSET = object()


class _Form:
    __slots__ = ("texts", "ir_hash", "authorities", "registries", "objects", "trusted_scope")

    def __init__(self) -> None:
        self.texts: dict[bool, str] = {}
        self.ir_hash = ""
        self.authorities: tuple[tuple[Any, Any], ...] = ()
        self.registries: tuple[Any, ...] = ()
        self.objects: tuple[tuple[Any, tuple[str, ...], tuple[Any, ...]], ...] = ()
        # The ir_identity_reuse scope that already checked this witness. Within that scope a
        # read trusts it; the scope projects or checks every identity again when it ends.
        self.trusted_scope: Any = None


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
        return program._canonical_projection(include_provenance=include_provenance)
    form = _current_form(program)
    if form is not None:
        return json.loads(form.texts[include_provenance])
    return _build(program, include_provenance)[1]


def ir_hash(program: Any) -> str:
    """``canonical_json_sha256(_serialize(False))``, from the kept form when the witness holds."""
    from pops._structural_data import canonical_json_sha256

    if not eligible(program):
        return canonical_json_sha256(program._canonical_projection(include_provenance=False))
    form = _current_form(program)
    if form is None:
        form, data = _build(program, False)
        if form is None:
            return canonical_json_sha256(data)
    return form.ir_hash


def has_form(program: Any) -> bool:
    with _LOCK:
        return program in _FORMS


def projected_ir_hash(program: Any) -> str:
    """A real projection of the IR hash, bypassing any kept form."""
    from pops._structural_data import canonical_json_sha256

    return canonical_json_sha256(program._canonical_projection(include_provenance=False))


def _current_form(program: Any) -> _Form | None:
    from .serialization import _IR_IDENTITY_REUSE

    with _LOCK:
        form = _FORMS.get(program)
    if form is None:
        return None
    scope = _IR_IDENTITY_REUSE.get()
    if scope is not None and scope.applies() and form.trusted_scope is scope:
        return form
    if not _witness_holds(program, form):
        with _LOCK:
            if _FORMS.get(program) is form:
                del _FORMS[program]
        return None
    form.trusted_scope = scope if scope is not None and scope.applies() else None
    return form


def _witness_holds(program: Any, form: _Form) -> bool:
    if getattr(program, "_frozen", False) is not True:
        return False
    for registry in form.registries:
        if getattr(registry, "_frozen", False) is not True:
            return False
    for obj, names, values in form.objects:
        current_names, current_values = _state(obj)
        if current_names != names or len(current_values) != len(values):
            return False
        for current, recorded in zip(current_values, values):
            if current is not recorded:
                return False
    for authority, fingerprint in form.authorities:
        if authority.fingerprint() != fingerprint:
            return False
    return True


def _build(program: Any, first: bool) -> tuple[_Form | None, dict[str, Any]]:
    """Project ``program`` for variant ``first``; keep a form when everything allows it."""
    authorities: dict[Any, Any] = {}
    registries: dict[int, Any] = {}
    data = _observed_projection(program, first, authorities, registries)
    if not _keepable_registries(registries) or not _closed_json(data):
        return None, data
    objects = _immutable_closure(program)
    if objects is None:
        return None, data
    other = _observed_projection(program, not first, authorities, registries)
    if not _keepable_registries(registries) or not _closed_json(other):
        return None, data
    by_variant = {first: data, not first: other}
    form = _Form()
    try:
        for variant, value in by_variant.items():
            text = json.dumps(value, sort_keys=False, separators=(",", ":"), allow_nan=False)
            json.loads(text)  # an integer beyond the decoder's digit limit keeps no form
            form.texts[variant] = text
    except (ValueError, RecursionError):
        return None, data
    from pops._structural_data import canonical_json_sha256

    form.ir_hash = canonical_json_sha256(by_variant[False])
    form.authorities = tuple(authorities.items())
    form.registries = tuple(registries.values())
    form.objects = objects
    with _LOCK:
        _FORMS[program] = form
    from .serialization import _IR_IDENTITY_REUSE

    scope = _IR_IDENTITY_REUSE.get()
    form.trusted_scope = scope if scope is not None and scope.applies() else None
    return form, data


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
    reads: list[Any] = []
    token = ownership._FINGERPRINT_SCOPE.set(projection)
    reads_token = references._REGISTRY_READS.set(reads)
    try:
        return program._canonical_projection(include_provenance=include_provenance)
    finally:
        references._REGISTRY_READS.reset(reads_token)
        ownership._FINGERPRINT_SCOPE.reset(token)
        projection.active = False
        touched = {authority: dict.__getitem__(observed, authority)
                   for authority in observed.touched if dict.__contains__(observed, authority)}
        authorities.update(touched)
        for registry in reads:
            registries[id(registry)] = registry
        if outer_observed is not None:
            for authority, fingerprint in touched.items():
                outer_observed.setdefault(authority, fingerprint)
        else:
            for authority, fingerprint in touched.items():
                if authority.fingerprint() != fingerprint:
                    raise ownership.UnresolvedOwnershipError(
                        "model definition changed during canonical projection")


def _keepable_registries(registries: dict[int, Any]) -> bool:
    # One frozen registry means its whole Case graph is sealed: freeze_problem_graph freezes the
    # block, field, time, param and initial registries in one transaction (freeze_atomically).
    return all(getattr(registry, "_frozen", False) is True for registry in registries.values())


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


def _state(obj: Any) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """The attribute names of ``obj`` and the objects bound to them."""
    slots = _slot_names(type(obj))
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


def _immutable_closure(program: Any) -> tuple[tuple[Any, tuple[str, ...], tuple[Any, ...]], ...] | None:
    """Every object reachable from ``program``, with its attribute bindings, or None.

    None when anything reachable is of a kind outside the allowlist below: a mutable container,
    a NumPy array, a nested Program, a callable, or any unknown type. The walk stops at Handles
    (their fields are set once and canonicalized through the frozen Case registries that the
    witness holds), at ``_AuthoringAuthority`` (read only through its serial and its fingerprint,
    which the witness holds) and at the ``prog`` back reference of a node.
    """
    from pops._ir.symbolic import ImmutableSymbolic
    from pops.model.handles import Handle
    from pops.model.ownership import _AuthoringAuthority

    records = [(program, *_state(program))]
    stack = [value for name, value in zip(records[0][1], records[0][2])
             if name not in _AUTHORING_ONLY]
    seen: set[int] = {id(program)}
    while stack:
        node = stack.pop()
        kind = type(node)
        if kind in _SCALARS:
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if kind is tuple or kind is frozenset:
            stack.extend(node)
            continue
        if kind is MappingProxyType:
            # freeze_symbolic_metadata and freeze_containers always build the proxy over a fresh
            # dict that nothing else holds, so the proxy is as immutable as its contents.
            for key, value in node.items():
                stack.append(key)
                stack.append(value)
            continue
        if isinstance(node, (Enum, Handle)) or kind is _AuthoringAuthority:
            continue
        if not _immutable_class(kind, node):
            return None
        names, values = _state(node)
        records.append((node, names, values))
        symbolic = isinstance(node, ImmutableSymbolic)
        for name, value in zip(names, values):
            if value is _UNSET or (symbolic and name == "prog"):
                continue
            stack.append(value)
    return tuple(records)


__all__ = ["eligible", "has_form", "ir_hash", "projected_ir_hash", "serialize"]
