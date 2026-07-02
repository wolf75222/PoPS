"""pops.descriptors -- the typed brick descriptor and the external-brick catalog.

This is the canonical home of :class:`BrickDescriptor` (the inert, numerics-free
metadata record of a numerical brick) and of the EXTERNAL C++ brick catalog
(:func:`load_cpp_library` / :func:`external` / :func:`_external_descriptor`).

It also owns the shared descriptor factories ``_native`` / ``_planned`` that the
catalog namespace modules (riemann, reconstruction, ...) import, so those
factories are defined ONCE here instead of in six files.

The hybrid/native brick CLASSES (``NativeBrick`` / ``HybridModel`` / the partial
DSL bricks) are NOT here: they live permanently in :mod:`pops.physics.bricks` and
:mod:`pops.physics.hybrid`. ``lib.descriptors`` is the Spec-3 catalog descriptor
only.
"""
import json

BRICK_TYPES = ("native", "generated", "macro", "external_cpp")

# STRICT versioned schema of the external-brick manifest (ADC-611). The JSON pops_brick_manifest()
# exports carries schema_version at the top level; the parser refuses a manifest without it (legacy /
# pre-ADC-611 -> "regenerate the brick library"), a wrong version, a missing required field, or an
# UNKNOWN field (top-level or per entry). Emitter (POPS_DEFINE_BRICK_MANIFEST / BrickRegistry::to_json)
# and parser stay in LOCKSTEP: they share this version and this field set.
BRICK_MANIFEST_SCHEMA_VERSION = 1
# Allowed keys at the top level and per brick entry (strict allow-lists -- anything else is refused).
_BRICK_MANIFEST_TOP_KEYS = frozenset({"schema_version", "abi_key", "bricks"})
_BRICK_MANIFEST_ENTRY_REQUIRED = ("id", "category", "requirements", "capabilities")
_BRICK_MANIFEST_ENTRY_KEYS = frozenset(_BRICK_MANIFEST_ENTRY_REQUIRED)


class BrickDescriptor:
    """A typed, numerics-free descriptor of a numerical brick.

    Identity is by all metadata fields so two descriptors of the same brick
    compare equal (used to detect a re-selected brick and to key the artifact
    hash). It is intentionally inert: it has no ``eval`` / ``compile`` / call.
    """

    def __init__(self, name, brick_type, *, category="brick", native_id="",
                 scheme=None, requirements=None, capabilities=None, options=None,
                 available=True, expression=None, builder=None):
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
        self.available = bool(available)
        # Optional board value carried by a generated/macro brick; kept OFF the
        # identity key (it may be an unhashable board node).
        self.expression = expression
        # Optional Python builder of a GENERATED-brick solver (``@pops.lib.solver``):
        # the function that AUTHORS the solver IR. Like ``expression`` it is kept OFF
        # the identity key (a callable is not part of the brick's value identity).
        self.builder = builder

    def _key(self):
        return (self.category, self.name, self.brick_type, self.native_id,
                self.scheme, tuple(sorted(self.options.items())))

    def __eq__(self, other):
        return isinstance(other, BrickDescriptor) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return "BrickDescriptor(%r, %r, scheme=%r)" % (
            self.name, self.brick_type, self.scheme)

    # --- DescriptorProtocol surface (Spec 5 sec.6). The metadata stays carried by the
    # ``requirements`` / ``capabilities`` / ``options`` / ``available`` ATTRIBUTES above
    # (this descriptor's documented identity); these inert methods only expose the same
    # protocol member NAMES the ``Descriptor`` base does. They add no computation.
    def lower(self, context=None):
        """The inert lowering record for this brick (metadata only; no computation)."""
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "scheme": self.scheme, "options": dict(self.options)}

    def inspect(self):
        """A plain-dict view of the brick descriptor (Spec 5 sec.12.1)."""
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "scheme": self.scheme, "options": dict(self.options),
                "requirements": dict(self.requirements),
                "capabilities": dict(self.capabilities), "available": self.available}

    def validate(self, context=None):
        """Raise a clear error when this brick has no native symbol yet (``available`` False)."""
        if not self.available:
            raise ValueError(
                "%s [%s] is not available: it has no native C++ symbol yet; unsupported route: "
                "requested %s:%s; available route: native %s descriptors with a non-empty "
                "native_id; alternative: choose an available descriptor from "
                "pops.inspect_capabilities()."
                % (self.name, self.category, self.category, self.name, self.category))
        return True

    def capability_matrix(self, context=None):
        """One-row ADC-549 capability matrix for this brick descriptor (metadata only)."""
        from pops._capabilities import CapabilityRouteMatrix, CapabilityRouteRow
        status = "available" if self.available else "unavailable"
        limitation = "" if self.available else "catalogued descriptor has no native C++ symbol"
        error = ""
        if not self.available:
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
def _native(name, native_id, scheme, *, category, caps=None, capabilities=None, **options):
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


def _planned(name, scheme, *, category, **options):
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
_EXTERNAL_BRICKS = {}


def _clear_external_catalog():
    """Drop every loaded external brick (test isolation; not part of the public API)."""
    _EXTERNAL_BRICKS.clear()


def _split_csv(value):
    """Split a manifest CSV field into a stripped, non-empty token list ([] when absent)."""
    if value is None:
        return []
    if not isinstance(value, str):
        raise ValueError("manifest requirements/capabilities must be a CSV string; got %r"
                         % (value,))
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def parse_brick_manifest(manifest_json):
    """Parse a brick manifest (the JSON ``pops_brick_manifest()`` returns) under the STRICT versioned
    schema (ADC-611) into ``(records, abi_key)`` WITHOUT registering anything.

    The manifest is ``{"schema_version": 1, "abi_key": <opt str>, "bricks": [{"id", "category",
    "requirements", "capabilities"}, ...]}``. STRICT policy -- the error always NAMES the offending field:
      - not valid JSON / not an object -> ValueError;
      - missing ``schema_version`` -> ValueError ("regenerate the brick library"): a manifest without it
        is legacy (pre-ADC-611); the in-tree emitter always writes it now;
      - ``schema_version`` != BRICK_MANIFEST_SCHEMA_VERSION -> ValueError (naming got vs expected);
      - an UNKNOWN top-level key or an UNKNOWN entry key -> ValueError (no permissive silent-ignore);
      - a brick entry missing any of id / category / requirements / capabilities -> ValueError (naming
        the field and the brick id). requirements/capabilities are CSV strings (possibly empty "").
    ``abi_key`` is carried as inert metadata (the .so's dlopen-time ABI guard is enforced separately for
    the library-.so path; a brick .so rebuilds against the headers, so it is documented-optional here).
    """
    try:
        doc = json.loads(manifest_json)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError("external brick manifest is not valid JSON: %s" % (err,)) from err
    if not isinstance(doc, dict):
        raise ValueError("external brick manifest must be a JSON object with 'schema_version' and "
                         "'bricks'; got %r" % (manifest_json,))
    unknown_top = sorted(set(doc) - _BRICK_MANIFEST_TOP_KEYS)
    if unknown_top:
        raise ValueError("external brick manifest has unknown top-level field(s) %s; the strict schema "
                         "allows only %s" % (unknown_top, sorted(_BRICK_MANIFEST_TOP_KEYS)))
    if "schema_version" not in doc:
        raise ValueError("external brick manifest is missing the required 'schema_version' field "
                         "(expected %d); it predates the versioned schema -- regenerate the brick "
                         "library against the current headers" % (BRICK_MANIFEST_SCHEMA_VERSION,))
    version = doc["schema_version"]
    if version != BRICK_MANIFEST_SCHEMA_VERSION:
        raise ValueError("external brick manifest 'schema_version' is %r, incompatible with the "
                         "supported version %d; regenerate the brick library"
                         % (version, BRICK_MANIFEST_SCHEMA_VERSION))
    bricks = doc.get("bricks")
    if not isinstance(bricks, list):
        raise ValueError("external brick manifest 'bricks' must be a list; got %r"
                         % (manifest_json,))
    records = []
    for entry in bricks:
        if not isinstance(entry, dict):
            raise ValueError("external brick manifest entry must be an object; got %r" % (entry,))
        unknown_entry = sorted(set(entry) - _BRICK_MANIFEST_ENTRY_KEYS)
        if unknown_entry:
            raise ValueError("external brick manifest entry %r has unknown field(s) %s; the strict "
                             "schema allows only %s"
                             % (entry.get("id"), unknown_entry, list(_BRICK_MANIFEST_ENTRY_REQUIRED)))
        for field in _BRICK_MANIFEST_ENTRY_REQUIRED:
            if field not in entry:
                raise ValueError("external brick manifest entry %r is missing the required '%s' field"
                                 % (entry.get("id"), field))
        if not entry.get("id"):
            raise ValueError("external brick manifest entry must carry a non-empty 'id'; got %r"
                             % (entry,))
        records.append({
            "id": str(entry["id"]),
            "category": str(entry.get("category") or "brick"),
            "requirements": _split_csv(entry.get("requirements")),
            "capabilities": _split_csv(entry.get("capabilities")),
        })
    return records, doc.get("abi_key")


def _register_manifest(manifest_json):
    """Parse a brick manifest under the strict versioned schema and register its bricks in the in-process
    catalog (last load wins on a repeated id). Returns the number of bricks registered. Delegates the
    strict parse (schema_version / required fields / unknown-field refusal) to :func:`parse_brick_manifest`;
    this is the seam ``load_cpp_library`` calls after dlopen (also usable directly, no compiled ``.so``)."""
    records, _abi_key = parse_brick_manifest(manifest_json)
    for record in records:
        _EXTERNAL_BRICKS[record["id"]] = dict(record)
    return len(records)


def load_cpp_library(path):
    """Load an external C++ brick ``.so`` and register the bricks it manifests (criterion 20).

    Opens @p path with :func:`ctypes.CDLL` (its static initializers run the
    ``POPS_REGISTER_BRICK`` registrations), calls the exported C function
    ``const char* pops_brick_manifest()`` to read the registered bricks as JSON, and registers
    the ids in the in-process catalog so ``riemann.User(id)`` / :func:`external` resolve. The
    ``.so`` must export ``pops_brick_manifest`` (a missing symbol is a clear ``ValueError``).
    Returns the number of bricks registered.
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
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


def load_compiled_manifest(path):
    """Read the per-artifact NativeManifest JSON of a compiled block ``.so`` (Spec 5 sec.13.12, #36).

    The sibling of :func:`load_cpp_library` for an AOT model artifact (one built by the DSL through
    ``POPS_DEFINE_COMPILED_BLOCK``): it dlopens @p path with :func:`ctypes.CDLL` and calls the
    exported C function ``const char* pops_compiled_manifest()`` -- the JSON the macro emits at the
    ``.so``'s OWN compile time from the model traits ({abi_version, n_vars, n_aux, n_params,
    ghost_depth, supports_stride, supports_partial_imex_mask, supports_named_fields, roles,
    native_entrypoints}). Returns the parsed dict.

    GRACEFUL FALLBACK for an OLD ``.so`` (built before this work) that does NOT export the symbol:
    returns ``None`` rather than raising, so a manifest builder can fall back to its honest-None set.
    A path that is not a loadable library still raises ``OSError`` (it is not an pops artifact at
    all); a present-but-malformed JSON raises ``ValueError`` (a corrupt manifest must not pass).
    """
    import ctypes
    handle = ctypes.CDLL(str(path))  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_compiled_manifest
    except AttributeError:
        return None  # old .so without the symbol: honest fallback, not an error
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError("compiled-artifact manifest of %r is not valid JSON: %s"
                         % (path, err)) from err
    if not isinstance(doc, dict):
        raise ValueError("compiled-artifact manifest of %r must be a JSON object; got %r"
                         % (path, raw))
    return doc


def _external_descriptor(brick_id, *, expect_category=None):
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
                           native_id=entry["id"], scheme="user",
                           requirements=req or None, capabilities=caps or None)


def external(brick_id):
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
