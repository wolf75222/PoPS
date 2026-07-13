"""Typed brick descriptors, factories, and the strict external-brick catalog."""
from __future__ import annotations

from typing import Any

from pops._manifest_protocol import strict_json_loads

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")

BRICK_MANIFEST_SCHEMA_VERSION = 3
_BRICK_MANIFEST_TOP_KEYS = frozenset({"schema_version", "abi_key", "annotations", "bricks"})
_BRICK_MANIFEST_TOP_REQUIRED = ("schema_version", "abi_key", "annotations", "bricks")
_BRICK_MANIFEST_ENTRY_REQUIRED = (
    "id", "category", "requirements", "capabilities", "native_id", "supported_layouts",
    "supported_platforms", "params", "options", "exported_symbols",
)
_BRICK_MANIFEST_ENTRY_KEYS = frozenset(_BRICK_MANIFEST_ENTRY_REQUIRED)


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name: str, brick_type: str, *, category: str = "brick",
                 native_id: str = "", scheme: Any = None, requirements: Any = None,
                 capabilities: Any = None, options: Any = None, available: bool = True,
                 expression: Any = None, builder: Any = None) -> None:
        if brick_type not in BRICK_TYPES:
            raise ValueError("brick_type %r must be one of %s"
                             % (brick_type, ", ".join(BRICK_TYPES)))
        self.name = str(name)
        self.brick_type = str(brick_type)
        self.category = str(category)
        self.native_id = str(native_id)
        self.scheme = scheme
        self.requirements = dict(requirements or {})
        self.capabilities = dict(capabilities or {})
        self.options = dict(options or {})
        # ADC-625: availability is the EXPLAINED route (available(context) -> Availability), not a
        # public bool. The constructor flag is stored privately as the single source the explaining
        # route derives from; consumers ask available(), never a bool attribute.
        self._available = bool(available)
        # Optional board value carried by a generated/macro brick; kept OFF the
        # identity key (it may be an unhashable board node).
        self.expression = expression
        # Optional Python builder of a GENERATED-brick solver (``@pops.lib.solver``):
        # the function that AUTHORS the solver IR. Like ``expression`` it is kept OFF
        # the identity key (a callable is not part of the brick's value identity).
        self.builder = builder

    def freeze(self) -> BrickDescriptor:
        """Freeze this brick descriptor: a later attribute mutation RAISES (ADC-563). Returns ``self``.

        The :class:`BrickDescriptor` counterpart of :meth:`pops.descriptors.Descriptor.freeze`; the
        assembly that holds it (a frozen ``Problem`` / ``Program``) seals it so a route the compiled
        artifact committed cannot be silently re-pointed. Idempotent."""
        from pops._descriptor_protocol import _freeze_descriptor_value

        for name, value in tuple(self.__dict__.items()):
            if name != "_frozen":
                object.__setattr__(self, name, _freeze_descriptor_value(value))
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, key: str, value: Any) -> None:
        """Refuse an attribute mutation after :meth:`freeze` (ADC-563), naming the frozen brick."""
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "brick descriptor %r [%s] is frozen (ADC-563): cannot set %r after the assembly was "
                "frozen by pops.compile; author a fresh descriptor and recompile."
                % (getattr(self, "name", "?"), getattr(self, "category", "brick"), key))
        object.__setattr__(self, key, value)

    def _key(self) -> tuple:
        return (self.category, self.name, self.brick_type, self.native_id,
                self.scheme, tuple(sorted(self.options.items())))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BrickDescriptor) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "BrickDescriptor(%r, %r, scheme=%r)" % (
            self.name, self.brick_type, self.scheme)

    # --- DescriptorProtocol surface (Spec 5 sec.6). The metadata stays carried by the
    # ``requirements`` / ``capabilities`` / ``options`` ATTRIBUTES above (this descriptor's
    # documented identity); availability is the EXPLAINED ``available(context) -> Availability``
    # route (ADC-625), matching the protocol member the ``Descriptor`` base exposes. No computation.
    def lower(self, context: Any = None) -> Any:
        """The inert :class:`~pops.descriptors_report.LoweredDescriptor` for this brick (ADC-527).

        Metadata only, no computation. The typed record carries ``name`` / ``category`` /
        ``native_id`` / ``scheme`` / ``options`` as attributes (and via ``to_dict``; ADC-625). A route
        with an empty ``native_id`` (a catalogued-but-not-native brick) is left to the loud
        :meth:`validate` refusal upstream -- never a silent fallback.
        """
        return LoweredDescriptor(name=self.name, category=self.category,
                                 native_id=self.native_id or None, options=dict(self.options),
                                 scheme=self.scheme)

    def available(self, context: Any = None) -> Availability:
        """The EXPLAINABLE availability status of this brick (ADC-625: the ONE availability route).

        Matches the ``DescriptorProtocol`` ``available(context) -> Availability`` member (the same
        method the base :class:`Descriptor` and the mesh descriptors expose): a route is chosen by an
        EXPLAINED status, never a bare bool. There is no public ``.available`` bool attribute anymore;
        this is the single source of truth. A native brick is ``yes``; a catalogued brick with no
        native symbol yet is ``no`` with the reason and the typed alternative (mirrors the
        :meth:`validate` message), so a rejection is explainable before the runtime is touched. A
        consumer reads ``brick.available(context).ok`` (or ``bool(brick.available(context))``).
        """
        if self._available:
            return Availability.yes()
        return Availability.no(
            "%s [%s] has no native C++ symbol yet" % (self.name, self.category),
            missing=["native_id"],
            alternatives=["choose an available descriptor from pops.inspect_capabilities()"])

    def inspect(self) -> dict:
        """A plain-dict view of the brick descriptor (Spec 5 sec.12.1).

        ``available`` is derived from the ONE availability route (``available().ok``), so the
        inspect view never diverges from the explained status (ADC-625).
        """
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "scheme": self.scheme, "options": dict(self.options),
                "requirements": dict(self.requirements),
                "capabilities": dict(self.capabilities), "available": self.available().ok}

    def to_data(self) -> dict[str, Any]:
        """Exact inert identity data for any consumer implementing the descriptor protocol."""
        if self.expression is not None or self.builder is not None:
            raise TypeError(
                "BrickDescriptor with expression/builder payload cannot claim an exact data identity"
            )
        return {
            "name": self.name,
            "brick_type": self.brick_type,
            "category": self.category,
            "native_id": self.native_id,
            "scheme": self.scheme,
            "requirements": dict(self.requirements),
            "capabilities": dict(self.capabilities),
            "options": dict(self.options),
            "available": self.available().ok,
        }

    def validate(self, context: Any = None) -> bool:
        """Raise a clear error when this brick has no native symbol yet (unavailable route)."""
        if not self.available(context).ok:
            raise ValueError(
                "%s [%s] is not available: it has no native C++ symbol yet; unsupported route: "
                "requested %s:%s; available route: native %s descriptors with a non-empty "
                "native_id; alternative: choose an available descriptor from "
                "pops.inspect_capabilities()."
                % (self.name, self.category, self.category, self.name, self.category))
        return True

    def capability_matrix(self, context: Any = None) -> Any:
        """One-row ADC-549 capability matrix for this brick descriptor (metadata only)."""
        from pops._capabilities import CapabilityRouteMatrix, CapabilityRouteRow
        ok = self.available(context).ok
        status = "available" if ok else "unavailable"
        limitation = "" if ok else "catalogued descriptor has no native C++ symbol"
        error = ""
        if not ok:
            error = ("unsupported route: requested %s:%s; available route: native %s "
                     "descriptors with a non-empty native_id; alternative: choose an available "
                     "descriptor from pops.inspect_capabilities()."
                     % (self.category, self.name, self.category))
        row = CapabilityRouteRow(
            "%s:%s" % (self.category, self.name),
            layout="context", backend="native" if self.native_id else "none",
            platform="context", mpi=None, gpu=None, status=status,
            limitation=limitation, error_message=error, source="descriptor")
        return CapabilityRouteMatrix(self.name, "context", [row])


# --- shared descriptor factories (imported by every catalog namespace) ------
# Native ids below (in the catalog modules) are the REAL C++ symbols in include/pops
# (verified): the FV bricks live at top level in ``namespace pops`` (e.g. pops::HLLCFlux),
# not under a numerics/fv namespace. Some catalogued bricks have no native type yet --
# they are emitted with ``available=False`` and an EMPTY native_id rather than a
# fabricated symbol.
def _native(name: str, native_id: str, scheme: Any, *, category: str, caps: Any = None,
            capabilities: Any = None, **options: Any) -> BrickDescriptor:
    """A native-brick descriptor.

    ``caps`` lists the model capabilities the brick REQUIRES (folded into ``requirements``);
    ``capabilities`` is the brick's own PROVIDED-capability dict -- the ``supports_<route>``
    view a typed solver advertises (built with
    :func:`pops.solvers.requirements.capability_map`) so an introspection / route check can see
    whether the brick supports uniform / amr / mpi / gpu (Spec 6 sec.4 / sec.9). Both default to
    none, so an unannotated brick is unchanged.
    """
    req = {"capabilities": list(caps)} if caps is not None else {}
    return BrickDescriptor(name, "native", category=category, native_id=native_id,
                           scheme=scheme, requirements=req, capabilities=capabilities,
                           options=options or None)


def _planned(name: str, scheme: Any, *, category: str, **options: Any) -> BrickDescriptor:
    """A catalogued brick with NO native C++ symbol yet (available=False, no id).

    It names the slot in the catalog without overclaiming a symbol; wiring a native
    type for it is tracked as a follow-up.
    """
    return BrickDescriptor(name, "native", category=category, native_id="",
                           scheme=scheme, options=options or None, available=False)


# --- external C++ bricks (Spec 3 section 21-22 / criterion 20) -------------
# A user ships a brick in a standalone ``.so`` that registers a manifest entry at
# static-init time (the C++ ``POPS_REGISTER_BRICK`` macro -> ``BrickRegistry``) and exports
# a C ``pops_brick_manifest()`` returning JSON. ``load_cpp_library`` dlopens it, parses that
# JSON and registers the ids in this in-process catalog; ``riemann.User(id)`` /
# ``external(id)`` then surface an ``external_cpp`` descriptor carrying the manifest's
# requirements/capabilities. An id that was never loaded raises a clear error -- a
# descriptor is NEVER fabricated for an unregistered brick.
_EXTERNAL_BRICKS: dict = {}
_EXTERNAL_BRICK_ORIGINS: dict = {}


def _clear_external_catalog() -> None:
    """Drop every loaded external brick (test isolation; not part of the public API)."""
    _EXTERNAL_BRICKS.clear()
    _EXTERNAL_BRICK_ORIGINS.clear()


def _split_csv(value: Any) -> list:
    """Split one canonical CSV string; the empty string denotes an explicit empty list."""
    if not isinstance(value, str):
        raise ValueError("manifest CSV field (requirements / capabilities / supported_layouts / "
                         "supported_platforms / params / options / exported_symbols) must be a CSV "
                         "string; got %r" % (value,))
    if not value:
        return []
    tokens = value.split(",")
    if any(not token or token != token.strip() for token in tokens):
        raise ValueError("manifest CSV field must be canonical (no whitespace or empty tokens): %r"
                         % value)
    if len(tokens) != len(set(tokens)):
        raise ValueError("manifest CSV field contains duplicate token(s): %r" % value)
    return tokens


def _validate_annotations(value: Any) -> dict:
    """Validate documentary extension keys without interpreting or normalising their values."""
    from urllib.parse import urlparse

    if not isinstance(value, dict):
        raise ValueError("external brick manifest 'annotations' must be an object")
    for key in value:
        if not isinstance(key, str):
            raise ValueError("external brick manifest annotation keys must be strings")
        parsed = urlparse(key)
        if not (key.startswith("x-") and len(key) > 2) and not parsed.scheme:
            raise ValueError(
                "external brick manifest annotation key %r must be a namespace URI or start with 'x-'"
                % key
            )
    return value


def _parse_brick_manifest_document(manifest_json: Any) -> tuple:
    """Return canonical records, ABI key, and exact documentary annotations for schema v3."""
    try:
        doc = strict_json_loads(manifest_json, where="external brick manifest")
    except ValueError as err:
        if "duplicate" in str(err) or "non-finite" in str(err):
            raise
        raise ValueError("external brick manifest is not valid JSON: %s" % err) from err
    if not isinstance(doc, dict):
        raise ValueError("external brick manifest must be a JSON object with 'schema_version' and "
                         "'bricks'; got %r" % (manifest_json,))
    unknown_top = sorted(set(doc) - _BRICK_MANIFEST_TOP_KEYS)
    if unknown_top:
        raise ValueError("external brick manifest has unknown top-level field(s) %s; the strict schema "
                         "allows only %s" % (unknown_top, sorted(_BRICK_MANIFEST_TOP_KEYS)))
    for field in _BRICK_MANIFEST_TOP_REQUIRED:
        if field not in doc:
            remedy = ("; it predates the versioned schema -- migrate the manifest offline"
                      if field == "schema_version" else "")
            raise ValueError("external brick manifest is missing the required %r field%s"
                             % (field, remedy))
    version = doc["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("external brick manifest 'schema_version' must be an integer")
    if version != BRICK_MANIFEST_SCHEMA_VERSION:
        raise ValueError("external brick manifest 'schema_version' is %r, incompatible with the "
                         "supported version %d; migrate or regenerate the brick library"
                         % (version, BRICK_MANIFEST_SCHEMA_VERSION))
    abi_key = doc["abi_key"]
    if not isinstance(abi_key, str) or not abi_key:
        raise ValueError("external brick manifest 'abi_key' must be a non-empty string")
    annotations = _validate_annotations(doc["annotations"])
    bricks = doc["bricks"]
    if not isinstance(bricks, list):
        raise ValueError("external brick manifest 'bricks' must be a list; got %r" % bricks)

    records = []
    seen_ids = set()
    seen_native_ids = set()
    csv_fields = {
        "requirements", "capabilities", "supported_layouts", "supported_platforms", "params",
        "options", "exported_symbols",
    }
    for entry in bricks:
        if not isinstance(entry, dict):
            raise ValueError("external brick manifest entry must be an object; got %r" % (entry,))
        unknown_entry = sorted(set(entry) - _BRICK_MANIFEST_ENTRY_KEYS)
        if unknown_entry:
            raise ValueError("external brick manifest entry %r has unknown field(s) %s; the strict "
                             "schema allows only %s"
                             % (entry.get("id"), unknown_entry,
                                list(_BRICK_MANIFEST_ENTRY_REQUIRED)))
        for field in _BRICK_MANIFEST_ENTRY_REQUIRED:
            if field not in entry:
                raise ValueError("external brick manifest entry %r is missing the required '%s' field"
                                 % (entry.get("id"), field))
        for field in ("id", "category", "native_id"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise ValueError("external brick manifest entry field %r must be a non-empty string"
                                 % field)
        for field in csv_fields:
            if not isinstance(entry[field], str):
                raise ValueError("external brick manifest entry field %r must be a CSV string" % field)
        brick_id = entry["id"]
        native_id = entry["native_id"]
        if brick_id in seen_ids:
            raise ValueError("external brick manifest contains duplicate brick id %r" % brick_id)
        if native_id in seen_native_ids:
            raise ValueError("external brick manifest contains duplicate native_id %r" % native_id)
        seen_ids.add(brick_id)
        seen_native_ids.add(native_id)
        records.append({
            "id": brick_id,
            "category": entry["category"],
            "requirements": _split_csv(entry["requirements"]),
            "capabilities": _split_csv(entry["capabilities"]),
            "native_id": native_id,
            "supported_layouts": _split_csv(entry["supported_layouts"]),
            "supported_platforms": _split_csv(entry["supported_platforms"]),
            "params": _split_csv(entry["params"]),
            "options": _split_csv(entry["options"]),
            "exported_symbols": _split_csv(entry["exported_symbols"]),
        })
    return records, abi_key, annotations


def parse_brick_manifest(manifest_json: Any) -> tuple:
    """Strictly parse schema v3 into canonical records and the required ABI key."""
    records, abi_key, _annotations = _parse_brick_manifest_document(manifest_json)
    return records, abi_key


def _register_manifest(manifest_json: Any) -> int:
    """Parse a manifest and register its bricks transactionally in the in-process catalog.

    Identical repetitions are idempotent; conflicting ids or manifest authorities are refused. Delegates the
    strict parse (schema_version / required fields / unknown-field refusal) to :func:`parse_brick_manifest`;
    this is the seam ``load_cpp_library`` calls after dlopen (also usable directly, no compiled ``.so``)."""
    from copy import deepcopy

    records, abi_key, annotations = _parse_brick_manifest_document(manifest_json)
    conflicts = [record["id"] for record in records
                 if record["id"] in _EXTERNAL_BRICKS
                 and (_EXTERNAL_BRICKS[record["id"]] != record
                      or _EXTERNAL_BRICK_ORIGINS[record["id"]]
                      != (abi_key, annotations))]
    if conflicts:
        raise ValueError("external brick id collision has different metadata: %s"
                         % ", ".join(sorted(conflicts)))
    for record in records:
        _EXTERNAL_BRICKS.setdefault(record["id"], dict(record))
        _EXTERNAL_BRICK_ORIGINS.setdefault(
            record["id"], (abi_key, deepcopy(annotations)))
    return len(records)


def load_cpp_library(path: Any) -> int:
    """Load an external C++ brick ``.so`` and register the bricks it manifests (criterion 20).

    Opens @p path with :func:`ctypes.CDLL` (its static initializers run the
    ``POPS_REGISTER_BRICK`` registrations), calls the exported C function
    ``const char* pops_brick_manifest()`` to read the registered bricks as JSON, and registers
    the ids in the in-process catalog so ``riemann.User(id)`` / :func:`external` resolve. The
    ``.so`` must export ``pops_brick_manifest`` (a missing symbol is a clear ``ValueError``).
    Returns the number of bricks registered.
    """
    import ctypes
    import os
    os.stat(path)  # normalize every missing-path route to the exact FileNotFoundError contract
    handle = ctypes.CDLL(str(path))  # raises OSError if the existing path is not loadable
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("external brick library %r does not export pops_brick_manifest(); it "
                         "is not an pops brick .so" % (path,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("external brick library %r: pops_brick_manifest() returned NULL"
                         % (path,))
    return _register_manifest(raw.decode("utf-8"))


def load_compiled_manifest(path: Any) -> Any:
    """Read the per-artifact NativeManifest JSON of a compiled block ``.so`` (Spec 5 sec.13.12, #36).

    The sibling of :func:`load_cpp_library` for an AOT model artifact (one built by the DSL through
    ``POPS_DEFINE_COMPILED_BLOCK``): it dlopens @p path with :func:`ctypes.CDLL` and calls the
    exported C function ``const char* pops_compiled_manifest()`` -- the JSON the macro emits at the
    ``.so``'s OWN compile time from the model traits ({abi_version, n_vars, n_aux, n_params,
    ghost_depth, supports_stride, supports_partial_imex_mask, supports_named_fields, roles,
    native_entrypoints}). Returns the parsed dict.

    Missing symbols, ``NULL``, legacy schemas and malformed data are hard errors. Runtime loading is
    not a migration seam; rebuild or migrate the artifact offline before installation.
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_compiled_manifest
    except AttributeError as error:
        raise ValueError(
            "compiled artifact %r has no current pops_compiled_manifest(); rebuild or migrate "
            "the artifact offline" % (path,)
        ) from error
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("compiled artifact %r returned a NULL native manifest" % (path,))
    try:
        from pops._manifest_protocol import strict_json_loads
        doc = strict_json_loads(raw, where="compiled native manifest JSON")
    except (ValueError, TypeError) as err:
        raise ValueError("compiled-artifact manifest of %r is not valid JSON: %s"
                         % (path, err)) from err
    from pops.external._artifact_manifest_ops import validate_native_manifest
    return validate_native_manifest(doc)


def _external_descriptor(brick_id: Any, *, expect_category: Any = None) -> BrickDescriptor:
    """The ``external_cpp`` descriptor for a loaded brick @p brick_id (raise if not loaded).

    An unloaded id raises :class:`LookupError` naming the id and :func:`load_cpp_library`; a
    category mismatch (selecting via ``riemann.User`` a brick registered as a preconditioner)
    raises :class:`ValueError`. The manifest requirements/capabilities become list metadata on
    the descriptor (mirroring the native bricks' ``requirements={"capabilities": [...]}``).
    """
    entry = _EXTERNAL_BRICKS.get(str(brick_id))
    if entry is None:
        raise LookupError(
            "external brick %r not loaded; call pops.lib.load_cpp_library(...) on the brick "
            ".so first (loaded: %s)" % (brick_id, sorted(_EXTERNAL_BRICKS) or "none"))
    if expect_category is not None and entry["category"] != expect_category:
        raise ValueError("external brick %r is registered as category %r, not %r"
                         % (brick_id, entry["category"], expect_category))
    req = {"capabilities": list(entry["requirements"])} if entry["requirements"] else {}
    caps = {"provides": list(entry["capabilities"])} if entry["capabilities"] else {}
    return BrickDescriptor(entry["id"], "external_cpp", category=entry["category"],
                           native_id=entry["native_id"], scheme="user",
                           requirements=req or None, capabilities=caps or None)


def external(brick_id: Any) -> BrickDescriptor:
    """An ``external_cpp`` descriptor for a loaded brick of ANY category (criterion 20).

    The category-agnostic counterpart of ``riemann.User`` / ``preconditioner.User``: it surfaces
    whatever category the manifest registered. An unloaded id raises a clear :class:`LookupError`.
    """
    return _external_descriptor(brick_id)


# --- generic typed-descriptor protocol (Spec 5 sec.6) -----------------------------------
# Spec 5 stabilizes "every object that chooses a route is a typed descriptor that declares
# its requirements/capabilities/options and answers available(context) with an EXPLAINABLE
# status". The native-brick :class:`BrickDescriptor` above is one family; the params / output /
# external (and the mesh) descriptors are another. That protocol family (``Availability`` /
# ``Descriptor`` / ``DescriptorProtocol`` / ``reject_string_selector``) lives in
# ``_descriptor_protocol`` (split out for the 500-line cap) and is re-exported here so the
# ``from pops.descriptors import Availability, Descriptor, ...`` paths stay unchanged.
from pops._descriptor_protocol import (  # noqa: E402,F401  (re-exported at the historical path)
    Availability,
    Descriptor,
    DescriptorProtocol,
    reject_string_selector,
)

# ADC-527: the typed DescriptorProtocol result objects. Re-exported here so the one public home of
# the descriptor surface (pops.descriptors) exposes them alongside Availability / Descriptor.
from pops.descriptors_report import (  # noqa: E402,F401
    CapabilitySet,
    LoweredDescriptor,
    Requirement,
    RequirementSet,
    ReportTree,
)
