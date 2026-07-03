"""pops.codegen.compile_provenance : the debug provenance sidecar for a compiled Program.

``compile_problem(debug=True)`` (or ``POPS_KEEP_GENERATED``) persists the generated ``.cpp`` next
to the ``.so`` for inspection. ADC-536 makes that persisted source SELF-DESCRIBING: a leading
C++ block-comment banner carries the serialized Program IR, the program / ABI / cache hashes, the
compile flags, the toolchain and the redacted compile command, so the ``.cpp`` on disk documents
exactly WHAT was built and HOW.

STRICT invariant (R5): the banner is written ONLY into the persisted sidecar ``.cpp``, never into
the source fed to the compiler. The ``.so`` bytes and the cache key are therefore unchanged whether
``debug`` is on or off -- the banner is inert provenance, not an input to the build. ``compile_problem``
proves this by compiling the banner-free ``src`` and only decorating the sidecar copy.

The sidecar and its ``.cachekey`` companion are written atomically (temp file + ``os.replace``) so a
crashed / concurrent compile never leaves a half-written provenance file that the cache-HIT guard
would then read.
"""
from __future__ import annotations

from typing import Any

import json
import os


# The cache-key sidecar suffix: ``<so-name>.cachekey`` sits next to the ``.so`` and records the
# keys + toolchain line the cache HIT guard re-verifies (CONTRACTS6 decision 1). Plain text, one
# field per line, so a human (or a shell one-liner) can read it without importing pops.
CACHEKEY_SUFFIX = ".cachekey"


def cachekey_path(so_path: Any) -> Any:
    """The ``<so-name>.cachekey`` sidecar path next to @p so_path (ADC-536 stale/ABI guard)."""
    return so_path + CACHEKEY_SUFFIX


def _atomic_write(path: Any, text: Any) -> None:
    """Write @p text to @p path atomically (temp file in the same dir + ``os.replace``).

    Same-directory temp keeps the replace atomic (a cross-filesystem rename is not). A failed write
    leaves the pre-existing file untouched rather than a truncated one -- the cache HIT guard then
    reads a whole sidecar or none, never a half-written one."""
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, ".%s.tmp-%d" % (os.path.basename(path), os.getpid()))
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def write_cachekey_sidecar(so_path: Any, *, cache_key: Any, abi_key: Any, toolchain: Any) -> None:
    """Atomically write the ``<so>.cachekey`` sidecar the cache-HIT guard re-verifies (ADC-536).

    Records the full ``cache_key``, the ``abi_key`` and a single ``toolchain`` line (compiler + std),
    one ``key=value`` per line. Written on every fresh compile so a subsequent cache HIT can prove
    the on-disk ``.so`` was built for the SAME key; a missing sidecar (a legacy ``.so``) or a
    mismatch is a loud refusal, never a silent reuse."""
    text = "cache_key=%s\nabi_key=%s\ntoolchain=%s\n" % (cache_key, abi_key, toolchain)
    _atomic_write(cachekey_path(so_path), text)


def read_cachekey_sidecar(so_path: Any) -> Any:
    """Read the ``<so>.cachekey`` sidecar into a dict, or ``None`` when it is absent (ADC-536).

    ``None`` means the ``.so`` predates the sidecar (a legacy artifact); the caller treats that as a
    stale/ABI-unverifiable artifact and refuses it. A present-but-malformed sidecar yields whatever
    ``key=value`` lines parsed (the guard compares the fields it needs and refuses on a mismatch)."""
    path = cachekey_path(so_path)
    if not os.path.isfile(path):
        return None
    fields = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


class StaleArtifactError(RuntimeError):
    """A cached ``.so`` whose sidecar is missing or disagrees with the freshly computed keys."""


def verify_cached_program_so(so_path: Any, *, cache_key: Any, abi_key: Any) -> None:
    """Fail LOUD on a cache HIT whose sidecar is missing or disagrees with the fresh keys (ADC-536).

    On a cache HIT the ``.so`` is reused WITHOUT recompiling, so nothing else re-checks that the
    on-disk artifact matches the current headers / compiler / route set. This guard reads the
    ``<so>.cachekey`` sidecar and compares the recorded ``cache_key`` / ``abi_key`` with the ones
    just computed from the live inputs. A missing sidecar (a ``.so`` built before this guard) or any
    mismatch RAISES :class:`StaleArtifactError` naming the ``.so``, the expected-vs-found key, and
    how to clear the cache -- never a warn-and-reuse (owner directive: fail loud).
    """
    found = read_cachekey_sidecar(so_path)
    if found is None:
        raise StaleArtifactError(
            "pops.compile: the cached .so %r has no %s sidecar, so its ABI / cache identity cannot "
            "be verified. It was built before the stale-artifact guard (ADC-536) and is refused as "
            "unverifiable rather than reused blindly.\n"
            "Remedy: delete the stale artifact to force a clean rebuild:\n"
            "  rm %s%s*\n"
            "or set POPS_CACHE_DIR to a fresh directory."
            % (so_path, CACHEKEY_SUFFIX, os.path.splitext(so_path)[0], ".*"))
    found_cache = found.get("cache_key")
    found_abi = found.get("abi_key")
    if found_cache != cache_key or found_abi != abi_key:
        raise StaleArtifactError(
            "pops.compile: the cached .so %r is STALE -- its %s sidecar does not match the freshly "
            "computed keys (a foreign/corrupt .so at the keyed path, or a toolchain / header change "
            "that the cache file name did not capture).\n"
            "  cache_key expected=%s found=%s\n"
            "  abi_key   expected=%s found=%s\n"
            "It is refused (never reused) to avoid a cryptic dlopen 'symbol not found'.\n"
            "Remedy: delete the stale artifact to force a clean rebuild:\n"
            "  rm %s %s"
            % (so_path, CACHEKEY_SUFFIX, cache_key, found_cache, abi_key, found_abi,
               so_path, cachekey_path(so_path)))


def build_debug_banner(program: Any, model: Any, *, program_hash: Any, abi_key: Any,
                       cache_key: Any, cflags: Any, lflags: Any, cxx: Any, std: Any,
                       command: Any, registry: Any) -> str:
    """Return the C++ block-comment provenance banner for the persisted debug ``.cpp`` (ADC-536).

    The banner documents WHAT the ``.so`` was built from and HOW: the serialized Program IR (the
    exact ``_serialize()`` blob ``_ir_hash`` digests), the program / ABI / cache hashes, the compile
    + link flags, the compiler and C++ standard, the redacted compile command and the route registry
    components. It is a C++ block comment (``/* ... */``), inert to the compiler.

    STRICT (R5): this string is prepended ONLY to the persisted sidecar ``.cpp``, never to the source
    fed to the compiler -- so the ``.so`` bytes and the cache key are byte-identical whether ``debug``
    is on or off. A ``*/`` in a serialized field is defanged to ``* /`` so the block comment cannot be
    closed early by the content.
    """
    ir = "(no Program IR: this handle carries no serializable time Program)"
    if program is not None and hasattr(program, "_serialize"):
        ir = json.dumps(program._serialize(), indent=2, sort_keys=True)
    model_name = getattr(model, "name", None) or getattr(program, "name", None) or "problem"
    lines = [
        "pops.compile provenance banner (ADC-536) -- INERT, sidecar-only, not compiled",
        "",
        "model            : %s" % model_name,
        "program          : %s" % (getattr(program, "name", None) or "problem"),
        "program_hash     : %s" % program_hash,
        "abi_key          : %s" % abi_key,
        "cache_key        : %s" % cache_key,
        "cxx              : %s" % cxx,
        "std              : %s" % std,
        "cflags           : %s" % " ".join(cflags or []),
        "lflags           : %s" % " ".join(lflags or []),
        "compile_command  : %s" % command,
        "route_registry   : %s" % registry,
        "",
        "serialized Program IR (the WHAT _ir_hash digests):",
        ir,
    ]
    body = "\n".join(lines).replace("*/", "* /")
    return "/*\n%s\n*/\n" % body
