"""pops.codegen.toolchain : BUILD-INFRA / toolchain helpers for DSL compilation.

Extracted verbatim from pops.dsl (bodies byte-for-byte); only import lines adjusted.
Public API re-exported from pops.codegen.__init__.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from collections.abc import Mapping
from typing import Any


# --- _pops access (mirrors dsl.py: try top-level then relative) ---
def _pops_module() -> Any:
    """Return ``_pops`` only when it is absent, never when its load failed.

    A binary-loader or dependency error from an installed extension must remain visible: treating it
    as an unavailable optional module would select a different toolchain and hide the real cause.
    """
    try:
        return importlib.import_module("_pops")
    except ModuleNotFoundError as exc:
        if exc.name != "_pops":
            raise
    try:
        return importlib.import_module("pops._pops")
    except ModuleNotFoundError as exc:
        if exc.name == "pops._pops":
            return None
        raise


_NATIVE_LOADER_CONTRACT_FIELDS = frozenset({"schema_version", "compile_definitions"})
_NATIVE_LOADER_SHARED_DEFINITIONS = ("POPS_RUNTIME_SHARED_EXCEPTION_ABI",)


def _native_loader_manifest_compile_flags(module: Any) -> list[str]:
    """Replay the closed host manifest shared by every generated native plugin route."""
    raw = getattr(module, "__native_loader_contract__", None)
    if not isinstance(raw, Mapping) or set(raw) != _NATIVE_LOADER_CONTRACT_FIELDS:
        raise RuntimeError(
            "loaded pops._pops exposes no exact __native_loader_contract__ schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise RuntimeError("unsupported pops._pops.__native_loader_contract__ schema_version")
    definitions = raw["compile_definitions"]
    if type(definitions) is not tuple or definitions != _NATIVE_LOADER_SHARED_DEFINITIONS:
        raise RuntimeError(
            "pops._pops native-loader compile definitions differ from the supported shared "
            "exception ABI contract")
    return ["-D" + definition for definition in definitions]


# --- Signature of the core header tree (ABI key of the "production" path) -------------
# The "production" backend (compile_native) emits a .so loader that inlines the header template
# pops::add_compiled_model and calls off-line methods of the already-loaded _pops module. Loader and
# module MUST share the same C++ ABI (same headers, compiler, standard). We materialize the
# "header signature" in the ABI key (pops/runtime/abi_key.hpp, token POPS_HEADER_SIG) ; the
# module build bakes it (CMake) and compile_native re-bakes it (-D flag) by computing it IDENTICALLY.
# The computation MUST be bit-for-bit identical on the CMake side (python/CMakeLists.txt) and here:
# sha256 of the sorted concatenation
# "<category> <relpath>\n<sha256(content)>\n" for every installed row in
# include/pops_headers.manifest. api, abi, sdk-root and sdk-support all enter the published ABI;
# test-only or untracked files never do.
def _installed_pops_header_rows(include: Any) -> tuple[tuple[str, str], ...]:
    """Return the exact installed PoPS SDK rows authenticated by the header signature."""
    manifest = os.path.join(include, "pops_headers.manifest")
    try:
        with open(manifest, encoding="utf-8") as source:
            rows = source.read().splitlines()
    except OSError as exc:
        raise RuntimeError("PoPS installed-header manifest is missing from %s" % include) from exc

    categories = ("api", "abi", "sdk-root", "sdk-support", "test-only")
    installed_categories = categories[:-1]
    installed = []
    seen = set()
    for line_number, raw in enumerate(rows, 1):
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        parts = row.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in categories:
            raise RuntimeError("invalid PoPS installed-header manifest row %d" % line_number)
        category, rel = parts
        if rel.startswith("/") or ".." in rel.split("/") or not rel.startswith("pops/") \
                or not rel.endswith((".hpp", ".h", ".inc")):
            raise RuntimeError("invalid header path in PoPS manifest: %s" % rel)
        if rel in seen:
            raise RuntimeError("duplicate header path in PoPS manifest: %s" % rel)
        seen.add(rel)
        if category != "test-only":
            installed.append((category, rel))

    present = {category for category, _ in installed}
    missing = [category for category in installed_categories if category not in present]
    if missing:
        raise RuntimeError(
            "PoPS installed-header manifest has empty categories: %s" % ", ".join(missing))

    return tuple(sorted(installed))


def pops_authenticated_header_paths(include: Any) -> frozenset[str]:
    """Canonical paths of every compiler header covered by ``pops_header_signature``.

    The root directory itself is deliberately not an authority: an unmanifested file placed next to
    the SDK must not become an implicit build input merely because the compiler can find it.
    """
    paths = []
    for _category, rel in _installed_pops_header_rows(include):
        path = os.path.join(include, *rel.split("/"))
        if not os.path.isfile(path):
            raise RuntimeError("PoPS installed header is missing: %s" % rel)
        paths.append(os.path.realpath(path))
    return frozenset(paths)


def pops_header_signature(include: Any) -> str:
    """Signature of the exact normalized installed-header contract under ``include``."""
    import hashlib

    entries = []
    for category, rel in _installed_pops_header_rows(include):
        path = os.path.join(include, *rel.split("/"))
        try:
            with open(path, "rb") as source:
                digest = hashlib.sha256(source.read()).hexdigest()
        except OSError as exc:
            raise RuntimeError("PoPS installed header is missing: %s" % rel) from exc
        entries.append("%s %s\n%s\n" % (category, rel, digest))
    blob = "".join(entries).encode()
    return hashlib.sha256(blob).hexdigest()


# --- Auto-detection of the pops include directory -----------------------------------
# To make m.compile(...) ergonomic, the pops headers directory is deduced automatically
# when the caller does not pass it. MIRROR of adc_cases/common/native.py::pops_include : we try
# $POPS_INCLUDE (explicit override), then we climb from the installed `pops` package (build-py/python/
# pops/ -> ../../../include), then the neighboring repo ../PoPS/include. Validity criterion : the
# canonical file pops/mesh/multifab.hpp exists. No hard import of pops here (the dsl module may be loaded
# outside the package) : we resolve `pops.__file__` lazily.
def pops_include() -> str:
    """include/ directory of PoPS (header-only headers of the core), auto-detected.

    Priority : $POPS_INCLUDE (override), otherwise from the installed `pops` package
    (.../pops -> ../../../include), otherwise the neighboring repo (.../PoPS/include from this module).
    Requires that pops/mesh/storage/multifab.hpp exists. Raises RuntimeError if not found (diagnostic listing the
    candidates), so as to NEVER compile against a silently wrong include."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))           # .../python/pops/codegen
    candidates = []
    env = os.environ.get("POPS_INCLUDE")
    if env:
        candidates.append(env)
    try:
        import pops as _pops_pkg
        pkg = os.path.dirname(os.path.abspath(_pops_pkg.__file__))   # .../pops
        candidates.append(os.path.join(pkg, "include"))  # wheel-owned, exact signed header tree
        candidates.append(os.path.normpath(os.path.join(pkg, "..", "..", "..", "include")))
    except Exception:
        pass
    # from this file (python/pops/codegen/toolchain.py) : python/pops/codegen -> python/pops -> python -> repo root -> include
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "..", "include")))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "pops", "mesh", "storage", "multifab.hpp")):
            return c
    raise RuntimeError(
        "pops headers not found (looking for pops/mesh/storage/multifab.hpp). "
        "Pass include=<PoPS>/include or set POPS_INCLUDE. Candidates tried : "
        + ", ".join(repr(c) for c in candidates))


# --- C++ standard of the native loader (ABI boundary of the "production" path) ----------
# The "production" backend generates a .so loader that inlines add_compiled_model<> and calls off-line
# methods of the ALREADY-loaded _pops module. The ABI key (pops/runtime/abi_key.hpp) encodes __cplusplus :
# the loader and the module must therefore share the SAME C++ standard, otherwise add_native_block rejects
# ("incompatible ABI"). The module bakes its real standard (POPS_CXX_STD : 20 under Kokkos because CUDA 12.x
# has no -std=c++23, 23 otherwise) and exposes it as _pops.__cxx_std__. We derive the expected -std flag of the
# native model from it INSTEAD OF freezing c++23 (which broke the native path under Kokkos/GH200, where the module is
# in c++20). Direct MIRROR of the build, so never a silent gap between loader and model.
def loader_cxx_std() -> str:
    """Flag '-std=c++NN' that the native model (backend="production") MUST use to share the ABI
    of the loaded _pops module. Source of truth : _pops.__cxx_std__ (integer 20/23 baked by the build, =
    POPS_CXX_STD : 20 under Kokkos, 23 otherwise). Graceful fallbacks if the attribute is missing (old module) :
    we parse __cplusplus from _pops.abi_key() (>202002L -> c++23, otherwise c++20) ; failing all that,
    we fall back to the historical default c++23 (non-Kokkos host case, unchanged)."""
    _pops = _pops_module()
    std = _pops_cxx_std_from_module(_pops) if _pops is not None else None
    return std or "c++23"


def _pops_cxx_std_from_module(mod: Any) -> Any:
    """C++ standard of the module @p mod as 'c++NN', or None if undeterminable. Priority to the integer
    __cxx_std__ (baked by the build) ; otherwise we extract std=<__cplusplus> from the ABI key."""
    n = getattr(mod, "__cxx_std__", None)
    if isinstance(n, int) and n in (20, 23):
        return "c++%d" % n
    # Fallback : parse "...;std=<__cplusplus>;..." from the ABI key (old module without __cxx_std__).
    abi_key = getattr(mod, "abi_key", None)
    if callable(abi_key):
        try:
            key = abi_key()
        except Exception:
            return None
        for tok in str(key).split(";"):
            if tok.startswith("std="):
                val = tok[len("std="):].rstrip("Ll")
                if val.isdigit():
                    return "c++23" if int(val) > 202002 else "c++20"
    return None


# --- Compiler of the DSL .so files (ABI boundary, counterpart of loader_cxx_std) -------------------------
# REAL BUG fixed here : in an active conda env, `which c++` often points to ANOTHER compiler
# than the one that built _pops (old gcc/clang from the conda PATH). Symptom : the runtime compilation
# of the production DSL loader fails with the raw compiler error ("error: invalid value 'c++23'
# in '-std=c++23'") ; and even if it passed, the ABI key (which encodes __VERSION__ of the compiler,
# cf. abi_key.hpp) would reject the .so ("incompatible ABI"). The ONLY guaranteed-compatible compiler
# is the one from the _pops build : CMake bakes it (POPS_CXX_COMPILER -> _pops.__cxx_compiler__) and we
# prefer it here over the PATH. $POPS_CXX remains the conscious override (chosen conda toolchain, wrapper...).
def loader_cxx_compiler() -> Any:
    """Path of the compiler that BUILT the _pops module (baked by CMake as __cxx_compiler__),
    or None if it is unknown (old module, manual build) or absent from this machine.

    macOS : CMake often bakes the INTERNAL c++ of the Xcode / CommandLineTools toolchain
    (.../XcodeDefault.xctoolchain/usr/bin/c++), which invokes clang WITHOUT an SDK sysroot -> every DSL
    .so fails on \"'string' file not found\". The /usr/bin/c++ shim (xcrun) runs THE SAME
    clang while resolving the SDK : same __VERSION__, hence same ABI key -- so we prefer the shim
    (pitfall and remedy identical to compile_loader of the native C++ tests)."""
    import sys
    mod = _pops_module()
    cc = getattr(mod, "__cxx_compiler__", "") if mod is not None else ""
    if not (cc and os.path.isfile(cc) and os.access(cc, os.X_OK)):
        return None
    if sys.platform == "darwin" and (".xctoolchain/" in cc or "/CommandLineTools/" in cc) \
            and os.path.isfile("/usr/bin/c++"):
        return "/usr/bin/c++"
    return cc


def _check_headers_match_module(include: Any) -> str:
    """PRE-DLOPEN GUARD of the native path (real bug) : if the headers under @p include have changed since
    the build of _pops (recent pull, another clone...), the loader compiled against them references
    C++ signatures that the OLD module does not export -> the dlopen of add_native_block fails BEFORE the
    ABI guard, with a cryptic error ("symbol not found in flat namespace '__ZN3adc6System13
    install_block...'"). So we compare HERE, before any compilation, the header signature baked
    into the module with that of the @p include tree, and we fail with a clear remedy. No-op if the
    module is not loadable or has no signature (manual build : historical degradation)."""
    from .abi import module_header_signature  # intra-package; avoids circular at module level
    baked = module_header_signature()
    current = pops_header_signature(include)
    if baked is not None and current != baked:
        mod = _pops_module()
        so = getattr(mod, "__file__", "(unknown)")
        raise RuntimeError(
            "pops.dsl : the pops headers of %r DO NOT MATCH those with which the _pops module "
            "was built (%s).\n"
            "  current header signature : %s\n"
            "  signature baked in _pops  : %s\n"
            "Typical cause : `git pull` / headers edited AFTER the module build -> the DSL loader "
            "would reference C++ signatures absent from the module (dlopen : 'symbol not found').\n"
            "Remedy : REBUILD the module with these headers :\n"
            "  cmake --preset python && cmake --build --preset python   (or the usual build-py)\n"
            "or point POPS_INCLUDE at the headers of the build that produced this module."
            % (include, so, current[:16], baked[:16]))
    return current  # signature of the @p include tree, reusable (avoids a 2nd walk+sha256)


def _default_cxx(cxx: Any = None) -> Any:
    """CENTRALIZED resolution of the DSL .so compiler (all backends). Priority :
      1. explicit cxx (caller argument) ;
      2. $POPS_CXX (conscious environment override) ;
      3. the compiler that built _pops (the only one guaranteed ABI-compatible, cf. above) ;
      4. c++ / g++ / clang++ from the PATH (historical behavior, last resort)."""
    return (cxx or os.environ.get("POPS_CXX") or loader_cxx_compiler()
            or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))


# Historical spellings of the same language levels : clang < 17 / gcc < 11 know
# only 'c++2b'/'c++2a'. Same level requested ; on an OLD compiler __cplusplus may differ from the
# module -> if applicable, explicit ABI rejection downstream (never silent UB).
_STD_ALIAS = {"c++23": "c++2b", "c++20": "c++2a"}


_NATIVE_COMPILE_ENVIRONMENT_DENYLIST = frozenset({
    # Compiler include search injection.
    "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "INCLUDE",
    # Linker/library injection. Native plugin libraries are supplied as authenticated absolute
    # paths or closed toolchain flags, never by ambient process search paths.
    "LIBRARY_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES", "LD_PRELOAD", "LIB", "LIBPATH",
    # Ambient flags and compiler-driver redirection.
    "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS", "GCC_EXEC_PREFIX", "COMPILER_PATH",
    "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES",
})


def native_compile_environment(environment: Any = None) -> dict[str, str]:
    """Return the explicit native-build environment with ambient injection channels removed.

    PATH, SDKROOT, the active conda prefix and platform deployment variables remain available: they
    select the compiler/toolchain already recorded by the artifact contract. Include and link search
    paths, however, must come from the authenticated command line.
    """
    source = os.environ if environment is None else environment
    result = {str(key): str(value) for key, value in source.items()}
    for name in _NATIVE_COMPILE_ENVIRONMENT_DENYLIST:
        result.pop(name, None)
    return result


def _run_compile(cmd: Any, what: Any) -> None:
    """Run the compilation command @p cmd CAPTURING stderr : on failure, raises a
    SELF-CONTAINED RuntimeError (command + compiler output + remedies) instead of the raw
    CalledProcessError whose message contains only the command line (real bug : the user sees
    only a 'returned non-zero exit status 1' drowned in the traceback)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=True, env=native_compile_environment())
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip()
        out = (r.stdout or b"").decode(errors="replace").strip()  # MSVC cl writes errors on STDOUT
        err = (err + "\n" + out).strip() if out else err
        raise RuntimeError(
            "pops.dsl: compiling the .so (%s) failed (exit %d).\n"
            "Command: %s\n"
            "Compiler output:\n%s\n"
            "Hints: `python -c \"import pops; pops.doctor()\"` diagnoses the environment "
            "(compiler/standard/headers); POPS_CXX forces a specific compiler."
            % (what, r.returncode, " ".join(cmd), err[:4000] or "(empty)"))


_probe_cache = {}  # (cc, std) -> effective std: avoids re-probing repeatedly (N compiled models)


def _probe_cxx_std(cc: Any, std: Any) -> str:
    """Checks BEFORE compilation that @p cc accepts -std=@p std (probe -fsyntax-only on empty source).

    Returns the EFFECTIVE std: @p std if it passes, otherwise its historical alias (c++23 -> c++2b) if
    it passes, otherwise raises an ACTIONABLE RuntimeError (compiler used, build compiler,
    solutions) instead of the raw compiler error. Skipped for nvcc_wrapper (different -x
    semantics; explicit GPU path, already gated by POPS_KOKKOS_CXX/POPS_KOKKOS_USE_NVCC_WRAPPER).
    Result memoized per (cc, std): a single probe even if N models are compiled in a row."""
    import subprocess
    if "nvcc" in os.path.basename(cc or ""):
        return std
    if sys.platform == "win32":
        return std  # cl/clang-cl: -fsyntax-only probe inapplicable; std translated to /std: at compile
    cached = _probe_cache.get((cc, std))
    if cached is not None:
        return cached

    def accepts(s: Any) -> tuple:
        try:
            r = subprocess.run([cc, "-x", "c++", "-std=" + s, "-fsyntax-only", "-"],
                               input=b"", capture_output=True, timeout=60)
            return r.returncode == 0, (r.stderr or b"").decode(errors="replace")
        except Exception as exc:  # not found, not executable, timeout: same actionable diagnostic
            return False, str(exc)

    good, err = accepts(std)
    if good:
        _probe_cache[(cc, std)] = std
        return std
    alias = _STD_ALIAS.get(std)
    if alias:
        good_alias, _ = accepts(alias)
        if good_alias:
            _probe_cache[(cc, std)] = alias
            return alias
    baked = loader_cxx_compiler()
    raise RuntimeError(
        "pops.dsl: the compiler %r does not support -std=%s (standard required to share the ABI of "
        "the _pops module).\nCompiler output:\n%s\n"
        "Compiler of the _pops build: %s\n"
        "Solutions:\n"
        "  - use the build compiler: export POPS_CXX=%r (or cxx=... in m.compile);\n"
        "  - macOS: update Xcode / the Command Line Tools (recent AppleClang);\n"
        "  - conda: `conda install -c conda-forge cxx-compiler` (gcc>=13 / clang>=17) then "
        "export POPS_CXX=$CONDA_PREFIX/bin/clang++ (macOS) or $CONDA_PREFIX/bin/g++ (Linux).\n"
        "NB: a compiler DIFFERENT from the build one may compile but then be rejected "
        "('incompatible ABI': the ABI key encodes the compiler version); prefer the build one."
        % (cc, std, (err or "").strip()[:800],
           baked or "(unknown: module without __cxx_compiler__, rebuild _pops to bake it)",
           baked or "<path/to/build/compiler>"))


def _native_kokkos_root() -> Any:
    """Concrete Kokkos prefix selected by the authenticated module contract.

    An explicit environment override remains supported for relocatable installations, but its two
    defining headers must match the hashes baked into ``_pops``.  With no override, the exact include
    tree selected by CMake is replayed.  No ``sys.prefix``/conda heuristic is accepted.
    """
    selected = _native_kokkos_selection()
    return None if selected is None else selected[0]


_KOKKOS_CONTRACT_FIELDS = frozenset({
    "schema_version", "abi_sha256", "include_dirs", "header_paths", "header_sha256",
})


def _file_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_kokkos_contract(module: Any = None) -> Mapping[str, Any] | None:
    """Validate and reauthenticate the Kokkos development manifest baked into ``_pops``."""
    import hashlib
    import re

    if module is None:
        module = _pops_module()
    if module is None:
        return None
    enabled = getattr(module, "__has_kokkos__", None)
    raw = getattr(module, "__kokkos_contract__", None)
    if enabled is False and raw is None:
        return None
    if enabled is not True:
        raise RuntimeError("loaded pops._pops exposes no exact __has_kokkos__ boolean contract")
    if not isinstance(raw, Mapping) or set(raw) != _KOKKOS_CONTRACT_FIELDS:
        raise RuntimeError(
            "Kokkos-enabled pops._pops must expose the exact __kokkos_contract__ schema; "
            "rebuild PoPS")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise RuntimeError("unsupported pops._pops.__kokkos_contract__ schema_version")
    abi = raw["abi_sha256"]
    if not isinstance(abi, str) or re.fullmatch(r"[0-9a-f]{64}", abi) is None:
        raise RuntimeError("Kokkos contract abi_sha256 must be a lowercase SHA-256")

    tuples = {}
    for name in ("include_dirs", "header_paths", "header_sha256"):
        value = raw[name]
        if type(value) is not tuple or not value or any(
                not isinstance(item, str) or not item for item in value):
            raise RuntimeError(
                "pops._pops.__kokkos_contract__[%r] must be a non-empty tuple of text" % name)
        tuples[name] = value
    includes = tuples["include_dirs"]
    paths = tuples["header_paths"]
    hashes = tuples["header_sha256"]
    if len(set(includes)) != len(includes):
        raise RuntimeError("Kokkos contract include_dirs must be unique")
    if len(paths) != 2 or len(hashes) != 2:
        raise RuntimeError("Kokkos contract must authenticate exactly two defining headers")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        raise RuntimeError("Kokkos contract header hashes must be lowercase SHA-256 values")
    if any(any(token in value for token in ("|", ";", "\r", "\n"))
           for value in (*includes, *paths)):
        raise RuntimeError("Kokkos contract contains an ambiguous serialized delimiter")
    if any(not os.path.isabs(include) for include in includes):
        raise RuntimeError("Kokkos contract include directories must be absolute")
    if any(not os.path.isabs(path) or os.path.dirname(path) not in includes for path in paths):
        raise RuntimeError("Kokkos contract headers must belong to its include directories")
    if {os.path.basename(path) for path in paths} != {"Kokkos_Core.hpp", "KokkosCore_config.h"}:
        raise RuntimeError("Kokkos contract does not identify its two defining headers")
    ordered_paths = tuple(
        os.path.join(include, name)
        for include in includes
        for name in ("Kokkos_Core.hpp", "KokkosCore_config.h")
        if os.path.join(include, name) in paths
    )
    if ordered_paths != paths:
        raise RuntimeError("Kokkos contract header order is not canonical")
    header_hashes = dict(zip(paths, hashes, strict=True))
    material_lines = []
    for include in includes:
        material_lines.append("include=%s\n" % include)
        for name in ("Kokkos_Core.hpp", "KokkosCore_config.h"):
            path = os.path.join(include, name)
            if path in header_hashes:
                material_lines.append(
                    "header=%s;sha256=%s\n" % (path, header_hashes[path]))
    material = "".join(material_lines).encode()
    if hashlib.sha256(material).hexdigest() != abi:
        raise RuntimeError("Kokkos contract payload does not authenticate its abi_sha256")
    return raw


def _native_kokkos_selection() -> tuple[str, tuple[str, ...], str] | None:
    """Return ``(root, include_dirs, abi)`` after closed contract validation."""
    contract = _native_kokkos_contract()
    override = next((os.environ[key] for key in (
        "POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT") if os.environ.get(key)), None)
    if contract is None:
        if override:
            raise RuntimeError(
                "a Kokkos root override cannot be authenticated because loaded pops._pops has no "
                "Kokkos contract; rebuild PoPS")
        return None

    includes = tuple(contract["include_dirs"])
    paths = tuple(contract["header_paths"])
    hashes = tuple(contract["header_sha256"])
    if override:
        root = os.path.realpath(override)
        include = os.path.join(root, "include")
        relocated = tuple(os.path.join(include, os.path.basename(path)) for path in paths)
        if any(not os.path.isfile(path) for path in relocated):
            raise RuntimeError(
                "explicit Kokkos root does not contain the defining headers: %s" % root)
        if tuple(_file_sha256(path) for path in relocated) != hashes:
            raise RuntimeError(
                "explicit Kokkos root differs from the installation used to build pops._pops: %s"
                % root)
        return root, (include,), str(contract["abi_sha256"])

    if any(not os.path.isdir(include) for include in includes):
        raise RuntimeError("baked Kokkos include directory is unavailable; rebuild PoPS")
    if any(not os.path.isfile(path) for path in paths):
        raise RuntimeError("baked Kokkos defining header is unavailable; rebuild PoPS")
    for path, expected in zip(paths, hashes, strict=True):
        if _file_sha256(path) != expected:
            raise RuntimeError(
                "Kokkos header changed in place after pops._pops was built: %s; rebuild PoPS" % path)
    core = next(path for path in paths if os.path.basename(path) == "Kokkos_Core.hpp")
    core_include = os.path.dirname(core)
    root = os.path.dirname(core_include) if os.path.basename(core_include) == "include" \
        else core_include
    return root, includes, str(contract["abi_sha256"])


def _native_kokkos_include_dirs() -> tuple[str, ...]:
    selected = _native_kokkos_selection()
    return () if selected is None else selected[1]


def _libomp_prefix() -> Any:
    """Homebrew libomp prefix on macOS (for -Xpreprocessor -fopenmp), or None. AppleClang does not
    handle `-fopenmp` alone: it needs -Xpreprocessor -fopenmp + the libomp include/lib (cf. CMakeLists)."""
    if sys.platform != "darwin":
        return None
    import subprocess
    try:
        p = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        prefix = p.stdout.strip()
        if prefix and os.path.isdir(os.path.join(prefix, "lib")):
            return prefix
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _native_feature_key() -> str:
    """Traits that change the inline code of the native loader and must therefore enter the cache (else
    a cached SERIAL .so would be reused on a Kokkos module -> silent serial fallback).

    Beyond on/off, the key carries the authenticated Kokkos contract ABI.  That identity covers both
    defining headers, including the generated backend/version configuration, so an in-place update or
    Serial/OpenMP switch cannot reuse an old loader."""
    root = _native_kokkos_root()
    if root is None:
        kk = "kokkos=off"
    else:
        selected = _native_kokkos_selection()
        if selected is None:  # Defensive: root and contract selection must remain one authority.
            raise RuntimeError("Kokkos root selected without an authenticated Kokkos contract")
        kk = "kokkos=on;kabi=%s" % selected[2]
    # The native-loader manifest changes cross-DSO declarations and must partition every cached
    # plugin even in serial builds. Replaying it here also fails closed before a stale host contract
    # can select an artifact built under the header-only exception mode.
    mod = _pops_module()
    loader = "loader_defs=" + ",".join(
        flag.removeprefix("-D") for flag in _native_loader_manifest_compile_flags(mod))
    # The MPI seam changes both inline code and ABI.  Partition by the concrete CMake-authenticated
    # mpi.h/library fingerprint, not only an on/off bit: Open MPI and MPICH artifacts must never share
    # one cache slot merely because both define POPS_HAS_MPI.
    from pops.codegen._native_mpi import native_mpi_abi_key
    mpi = native_mpi_abi_key(mod)
    return "%s;%s;%s" % (kk, loader, mpi)


def _warn_kokkos_parity() -> None:
    """Compatibility preflight kept at the compile-driver seam; validation now fails closed."""
    _native_kokkos_selection()


def _env_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in ("1", "on", "true", "yes", "y")


def _native_kokkos_compiler(cxx: Any) -> Any:
    """Effective compiler of the native loader. CUDA must be EXPLICIT (POPS_KOKKOS_CXX=<nvcc_wrapper>
    or POPS_KOKKOS_USE_NVCC_WRAPPER=1): many Kokkos OpenMP installs also provide a
    nvcc_wrapper, choosing it by default would break CPU jobs without nvcc. For Kokkos OpenMP, the host suffices."""
    if cxx:
        return cxx
    env = os.environ.get("POPS_KOKKOS_CXX")
    if env:
        return env
    root = _native_kokkos_root()
    wrapper = os.path.join(root, "bin", "nvcc_wrapper") if root else ""
    if wrapper and os.path.exists(wrapper) and _env_truthy(os.environ.get("POPS_KOKKOS_USE_NVCC_WRAPPER")):
        return wrapper
    # Centralized fallback: $POPS_CXX, then the _pops build compiler (the only ABI-compatible one
    # guaranteed for a native loader), then the PATH (historical). cf. _default_cxx.
    return _default_cxx(None)


def _pops_import_lib() -> Any:
    """(Windows, ADC-100) Path of the import library _pops.lib (System POPS_EXPORT symbols) against
    which to link the DSL .dll. Searched next to the _pops module. None if absent."""
    mod = _pops_module()
    if mod is None:
        return None
    d = os.path.dirname(getattr(mod, "__file__", "") or "")
    cand = os.path.join(d, "_pops.lib")
    return cand if os.path.exists(cand) else None


def _native_kokkos_flags() -> tuple:
    """Compile/link flags so the production DSL loader instantiates add_compiled_model WITH Kokkos.

    The loader contains the header-only templates (make_block / assemble_rhs / for_each_cell). Compiled
    without POPS_HAS_KOKKOS while _pops is built WITH Kokkos would diverge from its allocator and
    execution ABI, so the authenticated contract is mandatory."""
    include_dirs = _native_kokkos_include_dirs()
    if not include_dirs:
        return [], []
    if sys.platform == "win32":
        # MSVC/clang-cl: Kokkos as a SHARED DLL -> link the import lib kokkoscore.lib (ONE single runtime;
        # _pops loads the same kokkoscore.dll). cl accepts -D/-I. No -fopenmp/-ldl/-pthread (POSIX).
        root = _native_kokkos_root()
        compile_flags = ["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE"]
        for include in include_dirs:
            compile_flags.extend(("-I", include))
        return compile_flags, [os.path.join(root, "lib", "kokkoscore.lib")]
    compile_flags = ["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE"]
    for include in include_dirs:
        compile_flags.extend(("-I", include))
    # Do NOT link libkokkos* INTO the .so: the _pops module has already loaded the Kokkos runtime, a
    # SINGLETON (global registry of execution spaces), and add_native_block promotes it to global
    # scope (RTLD_GLOBAL). Linking a 2nd copy of Kokkos into the loader gives two runtimes: the
    # computation runs, but on exit Kokkos::finalize() aborts "Execution space instance to be removed
    # couldn't be found!" (SIGABRT, atexit). So we leave the Kokkos symbols UNDEFINED in the .so
    # (resolved at load time against the module, like install_block/grid_context). Only -fopenmp (the
    # OpenMP backend of the generated kernel) + -ldl/-pthread remain. ROMEO validation: DSL warm scales and
    # clean exit (exit 0), ratio ~0.96x the bricks.
    link_flags = ["-ldl", "-pthread"]
    if "nvcc_wrapper" not in os.path.basename(_native_kokkos_compiler(None) or ""):
        # OpenMP required for the Kokkos OpenMP exec space (harmless/ignored under Kokkos Serial).
        if sys.platform == "darwin":
            # macOS / AppleClang: `-fopenmp` alone is rejected -> -Xpreprocessor -fopenmp (+ include
            # Homebrew libomp if present). We do NOT link libomp (-lomp) into the .so: a 2nd copy of
            # libomp gives TWO OpenMP runtimes ("mutex lock failed: Invalid argument" at runtime).
            # The omp_*/__kmpc_* symbols resolve at load time (flat namespace, -undefined
            # dynamic_lookup set by compile_aot/compile_native) against the libomp already loaded by _pops.
            libomp = _libomp_prefix()
            compile_flags += ["-Xpreprocessor", "-fopenmp"]
            if libomp is not None:
                compile_flags += ["-I", os.path.join(libomp, "include")]
        else:
            # ELF/Linux: libgomp is shared by soname (no double-runtime), -fopenmp on both sides.
            compile_flags.append("-fopenmp")
            link_flags.append("-fopenmp")
    return compile_flags, link_flags


def pops_loader_build_flags(cxx: Any = None) -> tuple:
    """Flags to compile OUTSIDE CMake a .so that INCLUDES the pops headers and will be loaded into the
    _pops module (DSL loaders, ABI tests). PoPS being Kokkos-only, the .so MUST be compiled with
    Kokkos (for_each.hpp #error otherwise). Returns (compiler, compile_flags, link_flags): Kokkos +
    (macOS) -undefined dynamic_lookup. The Kokkos symbols stay UNDEFINED, resolved at load time
    against the Kokkos runtime already loaded by _pops (no 2nd copy). Raises if no installed Kokkos is
    visible through the authenticated contract baked into _pops (an explicit relocated root may
    override it only when the defining headers match). The host's central native
    loader manifest is replayed for every route before the optional MPI manifest, so serial and MPI
    plugins consume the same exported exception RTTI without acquiring the producer definition."""
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "pops_loader_build_flags: PoPS is Kokkos-only and loaded pops._pops exposes no "
            "authenticated Kokkos development contract; rebuild PoPS.")
    cc = _native_kokkos_compiler(cxx)
    cflags, lflags = _native_kokkos_flags()
    module = _pops_module()
    from pops.codegen._native_host import ensure_native_host_global
    ensure_native_host_global(module)
    loader_cflags = _native_loader_manifest_compile_flags(module)
    from pops.codegen._native_mpi import native_mpi_build_flags
    mpi_cflags, mpi_lflags = native_mpi_build_flags(module)
    cflags = [*loader_cflags, *cflags, *mpi_cflags]
    lflags = [*lflags, *mpi_lflags]
    if sys.platform == "darwin":
        cflags = list(cflags) + ["-undefined", "dynamic_lookup"]
    return cc, cflags, lflags
