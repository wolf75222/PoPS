"""pops.codegen.compile_library -- the Spec 3 brick-library manifest / ABI layer.

``pops.codegen.compile_library("my_numerics.so", objects=[...], backend=pops.codegen.Production())``
collects generated / macro / native brick descriptors (from :mod:`pops.numerics` /
:mod:`pops.solvers`, the ``@pops.codegen.solvers.solver`` registry, IR macros) into a
reusable-library MANIFEST: the
library name, the loaded-module ABI key, the brick list (id, type, category,
scheme, native id, requirements, capabilities), the generated symbols a future
``.so`` would export, and a stable content hash. The manifest is consumed by the
library-descriptor reader (:func:`read_library_manifest`) and by
``pops.codegen.compile_problem(..., libraries=[...])``.

The manifest, the ABI key and the content hash are numerics-free (no Python solve).
``compile_library(..., emit=True)`` ALSO emits the C++ of the library's bricks
(:mod:`pops.codegen.library_codegen`) and compiles a real ``.so`` with the same Kokkos toolchain a
problem ``.so`` uses (:func:`pops.codegen.toolchain.pops_loader_build_flags`, ``POPS_KOKKOS_ROOT``), exporting
the metadata, the ABI key, the brick list / signatures / requirements / capabilities and the
generated symbols. :func:`read_library_manifest` reads that descriptor back from the ``.so``
(dlopen) and rejects an ABI / Kokkos mismatch as a HARD error. ``pops.codegen.compile_problem(...,
libraries=[...])`` reads + validates the compiled ``.so`` (the consume path).
"""
from __future__ import annotations

from typing import Any

import hashlib
import json

from pops._manifest_immutability import (
    canonical_manifest_json,
    thaw_manifest_json,
)
from pops.descriptors import BrickDescriptor
from pops.codegen._library_manifest_data import freeze_bricks as _freeze_bricks

__all__ = ["LibraryManifest", "compile_library", "read_library_manifest"]

# Manifest schema version: bumped if the serialized shape changes, so a stale
# round-trip is rejected loud rather than silently mis-read.
_MANIFEST_VERSION = 2

_REQUIRED_KEYS = ("manifest_version", "name", "backend", "abi_key", "bricks",
                  "generated_symbols", "content_hash")
_ALLOWED_KEYS = frozenset(_REQUIRED_KEYS + ("so_path",))


def _lower_library_backend(backend: Any) -> Any:
    """Lower a typed backend descriptor to its token, rejecting a bare string (Spec 5 sec.7).

    Mirrors ``pops.compile``'s typed-backend handling: ``backend`` is a typed
    :class:`pops.codegen.backends._Backend` (``Production()`` / ``AOT()`` / ``JIT()``) and lowers
    to its canonical token (``"production"`` ...), exactly the string the manifest always
    recorded -- so the manifest / content hash stay byte-identical. ``None`` defaults to
    ``Production()``. A bare backend string is REJECTED (the public string form is removed; the
    error names the typed alternative).
    """
    from pops.codegen.backends import Production, _Backend
    if backend is None:
        backend = Production()
    if isinstance(backend, str):
        raise TypeError(
            "compile_library: backend must be a typed pops.codegen backend descriptor "
            "(pops.codegen.Production() -- the only one supported yet), not the string %r"
            % (backend,))
    if not isinstance(backend, _Backend):
        raise TypeError(
            "compile_library: backend must be a typed pops.codegen backend descriptor "
            "(e.g. Production()); got %r" % (backend,))
    return backend.lower()


def _brick_entry(obj: Any) -> dict:
    """The serializable manifest entry for one brick descriptor.

    Folds the descriptor's identity metadata (id, type, category, scheme, native
    id) and its requirements / capabilities. It carries NO numerics and no Python
    callable (the ``@pops.codegen.solvers.solver`` builder is kept off the manifest,
    mirroring how it is kept off the descriptor identity key).
    """
    if not isinstance(obj, BrickDescriptor):
        raise TypeError(
            "compile_library objects must be brick descriptors "
            "(e.g. pops.solvers.GMRES(), pops.numerics.riemann.HLLC(), an "
            "@pops.codegen.solvers.solver generated brick); got %r" % (obj,))
    return {
        "id": obj.name,
        "brick_type": obj.brick_type,
        "category": obj.category,
        "scheme": obj.scheme,
        "native_id": obj.native_id,
        "available": obj.available().ok,
        "requirements": dict(obj.requirements),
        "capabilities": dict(obj.capabilities),
        "options": dict(obj.options),
    }


def _generated_symbols(bricks: Any) -> Any:
    """The sorted ids of the GENERATED bricks -- the symbols the compiled ``.so``
    would export. Native bricks reference EXISTING ``pops::`` symbols (already in the
    loaded module) and external bricks reference a user ``.so``, so neither adds a
    generated symbol; only a generated brick (e.g. an ``@pops.codegen.solvers.solver`` solver)
    contributes one."""
    return sorted({b["id"] for b in bricks if b["brick_type"] == "generated"})


def _content_hash(name: Any, backend: Any, abi_key: Any, bricks: Any) -> str:
    """Stable content hash of the manifest: sha256 over the name, backend, ABI key
    and the SORTED brick entries.

    Mirrors the ``_model_hash`` / ``module_hash`` idiom (sha256 of a structured,
    sort-stable text blob). Sorting the brick entries by id makes the hash
    order-insensitive (a library is a SET of bricks, not a sequence); it is
    sensitive to the brick set, the name, the backend and the ABI key. The
    ``available`` flag and native id fold in so a planned brick gaining a real
    symbol re-keys the library.
    """
    payload = {
        "protocol": "pops.library-manifest.v%d" % _MANIFEST_VERSION,
        "name": str(name),
        "backend": str(backend),
        "abi_key": str(abi_key),
        "bricks": [
            thaw_manifest_json(brick)
            for brick in sorted(bricks, key=lambda entry: entry["id"])
        ],
    }
    return hashlib.sha256(canonical_manifest_json(payload).encode("utf-8")).hexdigest()


class LibraryManifest:
    """The descriptor of a compiled brick library (Spec 3 section 21).

    An inert metadata record: the library ``name``, the ``backend``, the loaded-module
    ``abi_key`` (header signature + compiler + std), the ``bricks`` (serialized brick
    entries), the ``generated_symbols`` the ``.so`` exports, a stable ``content_hash``, and
    -- once ``compile_library(..., emit=True)`` has compiled it -- the ``so_path`` of the
    real artifact (``None`` for a manifest-only build). It computes nothing; the codegen /
    runtime and the library reader consume it. :func:`compile_library` builds it;
    :func:`read_library_manifest` reconstructs it from :meth:`to_dict` OR from a compiled
    ``.so`` path.
    """

    __slots__ = (
        "name", "backend", "abi_key", "bricks", "generated_symbols", "content_hash", "so_path",
    )

    def __init__(self, name: Any, backend: Any, abi_key: Any, bricks: Any, generated_symbols: Any,
                 content_hash: Any, so_path: Any = None) -> None:
        frozen_bricks = _freeze_bricks(bricks)
        expected_symbols = tuple(_generated_symbols(frozen_bricks))
        try:
            supplied_symbols = tuple(str(symbol) for symbol in generated_symbols)
        except TypeError:
            raise TypeError("LibraryManifest generated_symbols must be an iterable") from None
        if supplied_symbols != expected_symbols:
            raise ValueError(
                "LibraryManifest generated_symbols do not match the generated brick records "
                "(got %r, expected %r)" % (supplied_symbols, expected_symbols)
            )
        normalized_hash = str(content_hash)
        expected_hash = _content_hash(name, backend, abi_key, frozen_bricks)
        if normalized_hash != expected_hash:
            raise ValueError(
                "LibraryManifest content_hash does not match its canonical payload "
                "(got %r, expected %r); the manifest is corrupt or stale"
                % (normalized_hash, expected_hash)
            )
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "backend", str(backend))
        object.__setattr__(self, "abi_key", str(abi_key))
        object.__setattr__(self, "bricks", frozen_bricks)
        object.__setattr__(self, "generated_symbols", expected_symbols)
        object.__setattr__(self, "content_hash", normalized_hash)
        # Path of the compiled .so, or None for a manifest-only (emit=False) build. It is
        # provenance, NOT identity: it stays OUT of __eq__ / the content hash (the same
        # library compiled to two paths is the same library) but IS carried on to_dict.
        object.__setattr__(self, "so_path", None if so_path is None else str(so_path))

    def _validate_integrity(self) -> None:
        expected_symbols = tuple(_generated_symbols(self.bricks))
        if self.generated_symbols != expected_symbols:
            raise ValueError(
                "LibraryManifest generated_symbols no longer match its brick records "
                "(got %r, expected %r)" % (self.generated_symbols, expected_symbols)
            )
        expected = _content_hash(self.name, self.backend, self.abi_key, self.bricks)
        if self.content_hash != expected:
            raise ValueError(
                "LibraryManifest content_hash no longer matches its canonical payload "
                "(got %r, expected %r)" % (self.content_hash, expected)
            )

    def to_dict(self) -> dict:
        """The serialized manifest (round-trips through :func:`read_library_manifest`)."""
        return {
            "manifest_version": _MANIFEST_VERSION,
            "name": self.name,
            "backend": self.backend,
            "abi_key": self.abi_key,
            "bricks": thaw_manifest_json(self.bricks),
            "generated_symbols": list(self.generated_symbols),
            "content_hash": self.content_hash,
            "so_path": self.so_path,
        }

    def _identity(self) -> dict:
        """The manifest dict WITHOUT the so_path (the artifact path is provenance, not
        identity: the same library compiled to two paths compares equal)."""
        d = self.to_dict()
        d.pop("so_path", None)
        return d

    def artifact_data(self) -> dict:
        """Compile identity without location-only ``so_path`` provenance."""
        return self._identity()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, LibraryManifest) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self.content_hash)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LibraryManifest is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LibraryManifest is immutable")

    def __repr__(self) -> str:
        return "LibraryManifest(%r, bricks=%d, hash=%s)" % (
            self.name, len(self.bricks), self.content_hash[:12])


def compile_library(name: Any, objects: Any, *, backend: Any = None, emit: bool = False,
                    so_path: Any = None, cxx: Any = None, force: bool = False) -> Any:
    """Build a reusable brick library from a set of brick descriptors.

    @p name is the library ``.so`` name; @p objects is a non-empty list of
    :class:`pops.descriptors.BrickDescriptor` (native / generated / macro / external bricks,
    e.g. ``pops.solvers.GMRES()``, ``pops.numerics.riemann.HLLC()``, an
    ``@pops.codegen.solvers.solver`` generated brick). @p backend is a TYPED backend descriptor
    (``pops.codegen.Production()`` -- the only one supported yet; ``None`` defaults to it); a bare
    backend string is REJECTED (Spec 5 sec.7, mirroring ``pops.compile``). Returns a
    :class:`LibraryManifest` carrying the brick metadata, the loaded-module ABI key and a stable
    content hash.

    With ``emit=False`` (default) it returns the MANIFEST only (numerics-free, no
    compiler needed). With ``emit=True`` it ALSO emits the library C++
    (:func:`pops.codegen.library_codegen.emit_library_cpp`) and compiles a REAL ``.so`` with the
    same Kokkos toolchain a problem ``.so`` uses (:func:`pops.codegen.toolchain.pops_loader_build_flags`,
    ``POPS_KOKKOS_ROOT``); the returned manifest carries the artifact ``so_path``. Without
    an explicit ``so_path`` the ``.so`` is cached out-of-source keyed by the content hash +
    ABI key (``force=True`` recompiles). The ``.so`` exports the metadata, the ABI key, the
    brick list / requirements / capabilities and the generated symbols; an ABI / Kokkos
    mismatch when it is later read back is a HARD error -- never a silent fallback.
    """
    backend = _lower_library_backend(backend)
    if backend != "production":
        raise ValueError(
            "compile_library currently supports the production backend only; got %r"
            % (backend,))
    if not objects:
        raise ValueError("compile_library requires a non-empty objects= list of "
                         "pops.lib brick descriptors")
    bricks = [_brick_entry(obj) for obj in objects]
    abi_key = _abi_key()
    manifest = LibraryManifest(
        name=name, backend=backend, abi_key=abi_key, bricks=bricks,
        generated_symbols=_generated_symbols(bricks),
        content_hash=_content_hash(name, backend, abi_key, bricks))
    if emit:
        compiled_path = _emit_and_compile(manifest, so_path=so_path, cxx=cxx, force=force)
        manifest = LibraryManifest(
            name=manifest.name, backend=manifest.backend, abi_key=manifest.abi_key,
            bricks=manifest.bricks, generated_symbols=manifest.generated_symbols,
            content_hash=manifest.content_hash, so_path=compiled_path,
        )
    return manifest


def _emit_and_compile(manifest: Any, *, so_path: Any = None, cxx: Any = None,
                      force: bool = False) -> Any:
    """Emit @p manifest's C++ and compile the library ``.so``; return its path.

    Reuses the production toolchain helpers (:mod:`pops.codegen.toolchain`,
    :mod:`pops.codegen.cache`): the same Kokkos compiler, flags and ABI-keyed cache path a
    problem ``.so`` uses, so the library ``.so`` is ABI-compatible with the loaded ``_pops``
    module. Kokkos is mandatory (PoPS is Kokkos-only); a missing ``POPS_KOKKOS_ROOT`` is a
    clear error from :func:`pops.codegen.toolchain.pops_loader_build_flags`.
    """
    import os
    import tempfile

    from . import toolchain, cache
    from .library_codegen import emit_library_cpp

    src = emit_library_cpp(manifest)
    include = toolchain.pops_include()
    sig = toolchain.pops_header_signature(include)
    cc, cflags, lflags = toolchain.pops_loader_build_flags(cxx)
    eff_std = toolchain._probe_cxx_std(cc, toolchain.loader_cxx_std())
    source_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(
        ("%s|%s|%s|%s|%s" % (
            manifest.content_hash, source_hash, sig, cc, eff_std)).encode("utf-8")
    ).hexdigest()

    if so_path is None:
        key = "%s|%s|%s" % (sig, cc, eff_std)
        so_path = cache._cache_so_path(manifest.content_hash, key, "library-production",
                                       "library", manifest.name)
        if not force and os.path.isfile(so_path):
            from .compile_provenance import verify_cached_program_so

            verify_cached_program_so(
                so_path, cache_key=cache_key, abi_key=manifest.abi_key)
            loaded = _read_so_manifest(so_path)
            if loaded != manifest:
                raise RuntimeError(
                    "pops.compile_library: cached .so manifest does not match the requested "
                    "library content hash %s" % manifest.content_hash
                )
            return so_path

    optflags = cache._dsl_optflags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "library.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        flags = ["-shared", "-fPIC", "-std=" + eff_std, *optflags,
                 "-DPOPS_HEADER_SIG=\"%s\"" % sig, *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *lflags]
        toolchain._run_compile(cmd, "compile_library (backend production)")
    from .compile_provenance import write_cachekey_sidecar

    write_cachekey_sidecar(
        so_path, cache_key=cache_key, abi_key=manifest.abi_key,
        toolchain="%s|%s" % (cc, eff_std))
    loaded = _read_so_manifest(so_path)
    if loaded != manifest:
        raise RuntimeError(
            "pops.compile_library: freshly compiled .so manifest does not match requested "
            "library content hash %s" % manifest.content_hash
        )
    return so_path


def read_library_manifest(manifest: Any) -> Any:
    """Reconstruct a :class:`LibraryManifest` from a serialized dict, a compiled ``.so`` path,
    or a :class:`LibraryManifest` (idempotent).

    * a :class:`LibraryManifest` is revalidated and returned unchanged;
    * a dict produced by :meth:`to_dict` round-trips; a dict missing a required key, or
      carrying an unknown manifest version, is rejected loud (a corrupt / stale manifest is
      never silently half-read);
    * a ``str`` / ``os.PathLike`` is treated as a compiled library ``.so`` path: it is
      dlopen'd (:func:`_read_so_manifest`), its exported descriptor is read back, and its ABI
      key is compared against the loaded ``_pops`` module -- an ABI / Kokkos mismatch is a HARD
      error (the bricks would otherwise crash the loader with a cryptic symbol failure).

    ``pops.codegen.compile_problem(..., libraries=[...])`` uses this to accept a manifest, a
    serialized descriptor, OR a compiled ``.so`` path.
    """
    import os

    if isinstance(manifest, LibraryManifest):
        manifest._validate_integrity()
        return manifest
    if isinstance(manifest, (str, os.PathLike)):
        return _read_so_manifest(os.fspath(manifest))
    if not isinstance(manifest, dict):
        raise TypeError("read_library_manifest expects a manifest dict, a compiled .so path, "
                        "or a LibraryManifest; got %r" % (manifest,))
    missing = [k for k in _REQUIRED_KEYS if k not in manifest]
    if missing:
        raise KeyError("library manifest is missing required keys: %s"
                       % ", ".join(missing))
    version = manifest["manifest_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("library manifest_version must be an integer")
    if version != _MANIFEST_VERSION:
        raise ValueError("unsupported library manifest version %r (expected %d)"
                         % (version, _MANIFEST_VERSION))
    if any(not isinstance(key, str) for key in manifest):
        raise TypeError("library manifest keys must be strings")
    unknown = sorted(set(manifest) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError("library manifest has unknown field(s): %s" % ", ".join(unknown))
    return LibraryManifest(
        name=manifest["name"], backend=manifest["backend"],
        abi_key=manifest["abi_key"], bricks=manifest["bricks"],
        generated_symbols=manifest["generated_symbols"],
        content_hash=manifest["content_hash"], so_path=manifest.get("so_path"))


def _read_so_manifest(so_path: Any) -> Any:
    """Read a compiled library ``.so`` descriptor back into a :class:`LibraryManifest` (dlopen).

    Opens @p so_path with :func:`ctypes.CDLL` and reads the ``pops_library_*`` exports the
    codegen emitted (name / backend / content hash / ABI key + the per-brick string tables).
    Enforces the ABI / Kokkos guard FIRST: the ``.so``'s ``pops_library_abi_key()`` is compared
    against the loaded ``_pops`` module's ABI key, and a mismatch raises a HARD :class:`RuntimeError`
    (the bricks were compiled against a different toolchain -- dlopen-ing them into a problem would
    fail with a cryptic symbol error or, worse, silent UB). A ``.so`` lacking ``pops_library_*``
    exports is not an pops library (clear error). The static-init ``POPS_REGISTER_BRICK`` calls also
    populate the in-process external-brick catalog as a side effect of the load.
    """
    import ctypes

    handle = ctypes.CDLL(str(so_path))  # raises OSError if the path is not a loadable library

    def cstr(symbol: Any) -> str:
        try:
            fn = getattr(handle, symbol)
        except AttributeError as err:
            raise ValueError(
                "library %r does not export %s(); it is not an pops compiled brick library "
                "(pops.codegen.compile_library(..., emit=True))" % (so_path, symbol)) from err
        fn.restype = ctypes.c_char_p
        raw = fn()
        return "" if raw is None else raw.decode("utf-8")

    def cint(symbol: Any) -> int:
        fn = getattr(handle, symbol)
        fn.restype = ctypes.c_int
        return int(fn())

    def cstr_i(symbol: Any, i: Any) -> str:
        fn = getattr(handle, symbol)
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int]
        raw = fn(i)
        return "" if raw is None else raw.decode("utf-8")

    try:
        version = cint("pops_library_manifest_version")
    except AttributeError as err:
        raise ValueError(
            "library %r does not export pops_library_manifest_version(); regenerate it with "
            "the current pops.codegen.compile_library" % (so_path,)
        ) from err
    if version != _MANIFEST_VERSION:
        raise ValueError(
            "library %r has manifest version %r (expected %d); regenerate it with the current "
            "pops.codegen.compile_library" % (so_path, version, _MANIFEST_VERSION)
        )

    so_abi = cstr("pops_library_abi_key")
    module_abi = _abi_key()
    # HARD ABI / Kokkos guard: never silently load bricks compiled against a different toolchain.
    if module_abi not in ("", "abi_key=unavailable") and so_abi != module_abi:
        raise RuntimeError(
            "pops.codegen.read_library_manifest: library %r was compiled with an ABI key DIFFERENT "
            "from the loaded _pops module (library %r vs module %r). The bricks were built "
            "against another compiler / C++ standard / header tree / Kokkos build; dlopen-ing "
            "them into a problem would fail with a cryptic symbol error or undefined behavior. "
            "Recompile the library with pops.codegen.compile_library(..., emit=True) using the SAME "
            "toolchain (POPS_KOKKOS_ROOT) that built _pops." % (so_path, so_abi, module_abi))

    n = cint("pops_library_brick_count")
    bricks = []
    for i in range(n):
        scheme = cstr_i("pops_library_brick_scheme", i)
        bricks.append({
            "id": cstr_i("pops_library_brick_id", i),
            "brick_type": cstr_i("pops_library_brick_type", i),
            "category": cstr_i("pops_library_brick_category", i),
            "scheme": scheme or None,
            "native_id": cstr_i("pops_library_brick_native_id", i),
            "available": cstr_i("pops_library_brick_available", i) == "1",
            "requirements": json.loads(cstr_i("pops_library_brick_requirements", i) or "{}"),
            "capabilities": json.loads(cstr_i("pops_library_brick_capabilities", i) or "{}"),
            "options": json.loads(cstr_i("pops_library_brick_options", i) or "{}"),
        })
    gen = [cstr_i("pops_library_generated_symbol", i)
           for i in range(cint("pops_library_generated_symbol_count"))]
    return LibraryManifest(
        name=cstr("pops_library_name"), backend=cstr("pops_library_backend"),
        abi_key=so_abi, bricks=bricks, generated_symbols=gen,
        content_hash=cstr("pops_library_content_hash"), so_path=so_path)


def _abi_key() -> str:
    """The loaded ``_pops`` module ABI key (header signature + compiler + std), or a
    stable placeholder when ``_pops`` is unavailable (a numpy-free / module-free
    interpreter exercising the pure-Python manifest layer). The key namespaces a
    library to the exact toolchain that will dlopen its bricks."""
    try:
        from .. import abi_key as _key  # pops.abi_key delegates to _pops.abi_key()
        return _key()
    except Exception:
        return "abi_key=unavailable"
