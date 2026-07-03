"""pops.external.manifests -- read + register a compiled-brick manifest (Spec 5 sec.5.17).

A manifest is the JSON ``pops_brick_manifest()`` exports under the STRICT versioned schema (ADC-611 /
ADC-544): ``{"schema_version": 2, "abi_key": <opt>, "bricks": [{"id", "category", "requirements",
"capabilities", <optional native_id / supported_layouts / supported_platforms / params / options /
exported_symbols>}, ...]}``. It can be read from a ``.json`` file or from a ``.so`` (dlopened).
:func:`register` / :func:`register_manifest_file` register the ids in the in-process catalog owned by
:mod:`pops.descriptors`; :func:`read_manifest` is the read-only counterpart that returns the metadata
WITHOUT registering or executing anything. The strict parse (schema_version / required fields /
unknown-field refusal) lives ONCE in :func:`pops.descriptors.parse_brick_manifest`. Nothing here computes.
"""
import ctypes

from pops.descriptors import (BRICK_MANIFEST_SCHEMA_VERSION, load_cpp_library, _register_manifest,
                              parse_brick_manifest)


def register_manifest_file(path):
    """Register the bricks in a manifest ``.json`` file. Returns the count registered."""
    with open(str(path), "r", encoding="utf-8") as handle:
        return _register_manifest(handle.read())


def register(path):
    """Register a manifest from a ``.json`` file or a brick ``.so`` (dlopen). Returns the count.

    A ``.json`` path is parsed directly; anything else is treated as a loadable ``.so`` and
    dlopened via :func:`pops.descriptors.load_cpp_library` (its static initializers register
    the bricks and the exported ``pops_brick_manifest()`` is read).
    """
    p = str(path)
    if p.endswith(".json"):
        return register_manifest_file(p)
    return load_cpp_library(p)


class CompiledManifest:
    """The read-only metadata of a compiled-brick manifest (Spec 5 sec.5.17).

    A plain value holding the parsed manifest: the ABI key (when the manifest carries one) and
    the per-brick records (id / category / requirements / capabilities). It is inert -- it
    NEITHER registers the bricks in the in-process catalog NOR dlopens / executes anything, so a
    caller can inspect a third-party brick before deciding to load it. Use
    :func:`pops.external.register` to actually register the bricks.
    """

    def __init__(self, bricks, *, abi_key=None):
        self.bricks = list(bricks)
        self.abi_key = abi_key

    @property
    def ids(self):
        """The brick ids in declaration order."""
        return [b["id"] for b in self.bricks]

    @property
    def categories(self):
        """The set of brick categories the manifest declares."""
        return sorted({b["category"] for b in self.bricks})

    def to_dict(self):
        """A plain-dict view of the manifest (abi_key + brick records)."""
        return {"abi_key": self.abi_key, "bricks": [dict(b) for b in self.bricks]}

    def __repr__(self):
        return "CompiledManifest(ids=%r, abi_key=%r)" % (self.ids, self.abi_key)


def _parse_manifest_metadata(manifest_json):
    """Parse manifest JSON into a :class:`CompiledManifest` WITHOUT registering it, under the STRICT
    versioned schema (ADC-611).

    Delegates to :func:`pops.descriptors.parse_brick_manifest` (the single strict parser: schema_version
    check, required fields, unknown-field refusal, each error naming the offending field), building an
    inert value instead of mutating the in-process catalog. Any schema violation raises ``ValueError``.
    """
    records, abi_key = parse_brick_manifest(manifest_json)
    return CompiledManifest(records, abi_key=abi_key)


def read_manifest(path):
    """Read a brick manifest (``.json`` or ``.so``) for INSPECTION only (Spec 5 sec.5.17).

    Returns a :class:`CompiledManifest` with the manifest's ids / categories / requirements /
    capabilities (and the ``abi_key`` when present). It does NOT register the bricks in the
    in-process catalog and does NOT execute any brick code: a ``.json`` path is parsed directly;
    a ``.so`` path is dlopened ONLY to read the exported ``pops_brick_manifest()`` string (its
    static initializers run as a side effect of any dlopen, but no brick is registered or
    invoked). Use :func:`register` when you actually want to register the bricks.
    """
    p = str(path)
    if p.endswith(".json"):
        with open(p, "r", encoding="utf-8") as handle:
            return _parse_manifest_metadata(handle.read())
    handle = ctypes.CDLL(p)  # raises OSError if the path is not a loadable library
    try:
        manifest_fn = handle.pops_brick_manifest
    except AttributeError as err:
        raise ValueError("brick library %r does not export pops_brick_manifest(); it is not an "
                         "pops brick .so" % (p,)) from err
    manifest_fn.restype = ctypes.c_char_p
    raw = manifest_fn()
    if raw is None:
        raise ValueError("brick library %r: pops_brick_manifest() returned NULL" % (p,))
    return _parse_manifest_metadata(raw.decode("utf-8"))


__all__ = ["register", "register_manifest_file", "read_manifest", "CompiledManifest",
           "BRICK_MANIFEST_SCHEMA_VERSION"]
