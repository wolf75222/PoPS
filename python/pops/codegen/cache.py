"""pops.codegen.cache : out-of-source .so cache helpers.

Extracted verbatim from pops.dsl (bodies byte-for-byte); only import lines adjusted.
Public API re-exported from pops.codegen.__init__.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
import re
import shlex
from threading import Lock
import tempfile
import time
from typing import Any



# Optimization flags shared by generated libraries on the sole production path.
# Default -O3 -DNDEBUG: hot-loop asserts disarmed + full vectorization -> parity with a native block (at
# -O2 without -DNDEBUG the generated kernel is ~1.48x). $POPS_DSL_OPTFLAGS may override this only
# through the closed path-free vocabulary below. The flags do not alter the ABI contract; explicit
# ISA choices and every other accepted codegen option remain part of the artifact identity.
_DSL_OPTFLAGS_DEFAULT = "-O3 -DNDEBUG"

# This is deliberately a closed vocabulary. ``POPS_DSL_OPTFLAGS`` participates in a native
# compiler command whose other inputs are content-authenticated. Accepting a generic compiler token
# here would re-open hidden build inputs through response files, forced includes, search paths,
# compiler plugins, object files or linker options. Additions therefore belong here one semantic
# flag at a time and must remain path-free.
_DSL_CODEGEN_EXACT_FLAGS = frozenset({
    "-DNDEBUG",
    "-UNDEBUG",
    "-fassociative-math",
    "-ffast-math",
    "-ffinite-math-only",
    "-finline-functions",
    "-finline-functions-called-once",
    "-fmath-errno",
    "-fno-associative-math",
    "-fno-fast-math",
    "-fno-finite-math-only",
    "-fno-inline-functions",
    "-fno-inline-functions-called-once",
    "-fno-math-errno",
    "-fno-omit-frame-pointer",
    "-fno-reciprocal-math",
    "-fno-semantic-interposition",
    "-fno-signed-zeros",
    "-fno-slp-vectorize",
    "-fno-strict-aliasing",
    "-fno-trapping-math",
    "-fno-tree-vectorize",
    "-fno-unroll-loops",
    "-fno-vectorize",
    "-fomit-frame-pointer",
    "-freciprocal-math",
    "-fsigned-zeros",
    "-fslp-vectorize",
    "-fstrict-aliasing",
    "-ftrapping-math",
    "-ftree-vectorize",
    "-funroll-loops",
    "-fvectorize",
})
_DSL_CODEGEN_FLAG_PATTERNS = (
    re.compile(r"-O(?:0|1|2|3|s|z|g|fast)\Z"),
    re.compile(r"-(?:march|mcpu|mtune)=[A-Za-z0-9][A-Za-z0-9_.+-]*\Z"),
    re.compile(r"-ffp-contract=(?:fast|off|on)\Z"),
)


def _is_safe_dsl_codegen_flag(flag: str) -> bool:
    # ``native`` is path-free but host-dependent. The artifact cache is architecture-scoped, not
    # CPU-feature scoped, so accepting it could reuse an instruction-set-specific binary on a
    # different machine. Require an explicit CPU/ISA token that enters the artifact identity.
    if flag.endswith("=native"):
        return False
    return flag in _DSL_CODEGEN_EXACT_FLAGS or any(
        pattern.fullmatch(flag) is not None for pattern in _DSL_CODEGEN_FLAG_PATTERNS
    )


def _dsl_optflags() -> list[str]:
    """Return the closed, path-free optimization flags for a production DSL artifact."""
    raw = os.environ.get("POPS_DSL_OPTFLAGS", _DSL_OPTFLAGS_DEFAULT)
    try:
        flags = shlex.split(raw, posix=True)
    except ValueError as exc:
        raise ValueError("POPS_DSL_OPTFLAGS is not a valid shell-style token list") from exc
    for flag in flags:
        if not _is_safe_dsl_codegen_flag(flag):
            raise ValueError(
                "POPS_DSL_OPTFLAGS rejects unsupported token %r; only the closed path-free "
                "optimization/codegen allowlist is accepted (no include/search paths, forced "
                "includes, object/response files, linker options, plugins or toolchain overrides)"
                % flag
            )
    return flags


def _platform_cache_key() -> str:
    """MACHINE traits that change the binary code of the .so without changing the C++ ABI key (__VERSION__
    is identical cross-arch): CPU architecture + optimization flags. Without them in the cache
    key, a .so x86_64 (Rosetta) or -march=native would be reused on another machine/arch via
    a shared cache (NFS, synchronized home) -> SIGILL (illegal instruction) or cryptic dlopen."""
    import platform
    return "arch=%s;optflags=%s" % (platform.machine(), " ".join(_dsl_optflags()))


# --- Out-of-source build cache -----------------------------------------------
# When the caller does not provide so_path, m.compile(...) writes the .so into a SHARED out-of-source
# cache (never next to the temporary .cpp), indexed by a stable model key: model_hash (formulas
# + roles + n_aux + params) AND abi_key (header signature + compiler + std). Two compilations
# of the SAME model (same key) reuse the cached .so (cache HIT, no recompilation); changing the
# model OR a parameter OR the toolchain changes the key -> new .so (cache MISS, recompilation). The
# file name carries the key, so several variants coexist without collision. cf. the same idea
# in adc_cases/common/native.py (ABI key = compiler + flags + header signature).
def _configured_cache_root() -> tuple[str, bool]:
    """Return the single configured PoPS cache authority and whether it is explicit."""
    override = os.environ.get("POPS_CACHE_DIR")
    if override:
        return override, True
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(xdg, "pops"), False


def pops_cache_dir() -> str:
    """Cache directory for the .so files generated by m.compile() without an explicit so_path.

    $POPS_CACHE_DIR (override), else $XDG_CACHE_HOME/pops/dsl, else ~/.cache/pops/dsl. Created as needed.
    Out-of-source by construction (never inside the repo tree), so nothing to ignore on the git side."""
    root, explicit = _configured_cache_root()
    base = root if explicit else os.path.join(root, "dsl")
    os.makedirs(base, exist_ok=True)
    return base


def component_store_dir() -> str:
    """Immutable installed-component store under the configured PoPS cache authority.

    ``POPS_CACHE_DIR`` is the explicit authority when set. Otherwise this resolves to
    ``$XDG_CACHE_HOME/pops/component-store-v1`` (or ``~/.cache/pops/...``).
    Artifact installation owns content addressing, atomic publication and digest revalidation.
    """
    root, _ = _configured_cache_root()
    directory = os.path.join(root, "component-store-v1")
    os.makedirs(directory, exist_ok=True)
    return directory


def _identity_cache_so_path(spec_identity: Any) -> str:
    """Return the collision-safe path addressed by one artifact specification.

    The complete digest is retained in the filename.  Human-readable names and paths are not
    identity inputs; any emitted symbol name that changes bytes belongs in the artifact-spec
    component payload before this function is called.
    """
    from pops.identity import Identity

    if not isinstance(spec_identity, Identity) or spec_identity.domain != "artifact-spec":
        raise TypeError("cache path requires a pops.artifact-spec Identity")
    return os.path.join(pops_cache_dir(), spec_identity.hexdigest + ".so")


@contextmanager
def _artifact_cache_lock(so_path: Any):
    """Serialize publication of one content-addressed native artifact across processes.

    MPI ranks and independent Python workers share the out-of-source cache.  The binary and its
    identity sidecar form one authenticated cache entry, so checking, compiling and publishing them
    must be one critical section.  The lock file is deliberately persistent: unlinking it would let
    a late opener lock a different inode while the current publisher still owns the old one.

    POSIX uses ``flock``.  Windows uses a one-byte non-blocking ``msvcrt`` lock with an explicit
    retry loop because ``LK_LOCK`` has a bounded implementation-defined retry count.  Both locks are
    released by the operating system if a compiler process exits unexpectedly.
    """
    path = os.path.abspath(os.fspath(so_path)) + ".pops-cache.lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = open(path, "a+b")
    windows_locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    windows_locked = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                if windows_locked:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _artifact_cache_staging_path(so_path: Any) -> str:
    """Reserve a unique same-directory compiler output file for one cache publication."""
    destination = os.path.abspath(os.fspath(so_path))
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    stem, suffix = os.path.splitext(os.path.basename(destination))
    descriptor, staging = tempfile.mkstemp(
        prefix=".%s.pops-stage-" % stem,
        suffix=suffix or ".so",
        dir=directory,
    )
    os.close(descriptor)
    return staging


def _precision_cache_key() -> str:
    """The floating-point precision component of a compiled artifact's cache key (ADC-536).

    The native runtime is fixed to double precision today (``NATIVE_REAL_BYTES`` = 8), but the token
    enters the key so a FUTURE precision switch (single / mixed) is a cache MISS, never a silent
    reuse of a double-precision ``.so`` under a single-precision request. Readable
    ("precision=double;real_bytes=8") so the mismatching field is nameable in a diagnostic. When a
    typed ``Production(precision=...)`` backend descriptor lands (538/537) it becomes the token's
    source; until then it mirrors the single native fact."""
    from pops.runtime_environment import NATIVE_PRECISION, NATIVE_REAL_BYTES
    return "precision=%s;real_bytes=%d" % (NATIVE_PRECISION, NATIVE_REAL_BYTES)


def _registry_cache_key() -> str:
    """Route registry + report vocabulary components of EVERY cache key (ADC-599).

    The typed native route registry (pops.runtime.routes / route_ids.hpp) and the
    capabilities/reports vocabulary participate in the artifact identity: an artifact built
    against a different route set (a route added/removed/re-tokenized, a native entry renamed)
    or an older report vocabulary must be a cache MISS, never a silent reuse. The component is
    readable ("routes=v2:<hash16>;capvocab=1") so the mismatching field is nameable in
    diagnostics and in compiled.inspect()."""
    from pops.runtime.routes import (CAPABILITY_VOCAB_VERSION, ROUTE_REGISTRY_VERSION,
                                     route_registry_hash)
    return "routes=v%d:%s;capvocab=%d" % (ROUTE_REGISTRY_VERSION, route_registry_hash()[:16],
                                          CAPABILITY_VOCAB_VERSION)


_process_so_identity: dict[str, str] = {}
_process_so_identity_lock = Lock()


def _artifact_distinct_so_path(so_path: Any, spec_identity: Any) -> Any:
    """Keep explicit native paths safe from the dynamic loader's path-based handle cache.

    Recompiling a different production artifact over an already used path can make ``dlopen``
    return the old handle. The first identity retains the requested path; later identities use a
    deterministic sibling derived from their authenticated artifact identity. Rebuilding the same
    identity keeps the same path because its native contract is unchanged.
    """
    import hashlib
    import os

    requested = os.path.abspath(os.fspath(so_path))
    identity = str(spec_identity)
    with _process_so_identity_lock:
        previous = _process_so_identity.get(requested)
        if previous is None:
            _process_so_identity[requested] = identity
            return so_path
        if previous == identity:
            return so_path

        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        root, ext = os.path.splitext(os.fspath(so_path))
        alternate = "%s.%s%s" % (root, digest, ext or ".so")
        _process_so_identity.setdefault(os.path.abspath(alternate), identity)
        return alternate


def _record_artifact_identity(so_path: Any, spec_identity: Any) -> None:
    """Record the authenticated artifact occupying a native loader path in this process."""
    import os

    with _process_so_identity_lock:
        _process_so_identity[os.path.abspath(os.fspath(so_path))] = str(spec_identity)
