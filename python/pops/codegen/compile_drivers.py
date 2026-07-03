"""pops.codegen.compile_drivers : the compiler-invocation + facade layer of the pipeline.

Extracted verbatim from ``pops.codegen.compile`` so the model compile pipeline fits the
Spec-4 file-size budget.  These drivers receive a ``HyperbolicModel`` (or a Program /
Module) and invoke the C++ compiler on the source the ``compile_emit`` emitters produce:
``compile_so`` / ``compile_aot`` / ``compile_native`` (one per backend), the
``compile_or_jit`` mode dispatcher, the ``compile_model`` facade, ``_module_to_model``
(lower a ``pops.model.Module`` to a dsl ``Model``) and ``compile_problem`` (compile a
``pops.time.Program`` into a ``problem.so``).  ``pops.codegen.compile`` re-imports every
name so its public surface is unchanged.

Does NOT import pops.physics at module level to avoid import cycles; the physics facade and
aux helpers are imported lazily inside the functions that need them.
"""

import os
import sys

from pops.codegen.toolchain import (
    pops_include,
    loader_cxx_std,
    _default_cxx,
    _probe_cxx_std,
    _check_headers_match_module,
    _warn_kokkos_parity,
    _native_kokkos_root,
    _native_kokkos_compiler,
    _native_kokkos_flags,
    _native_feature_key,
    _run_compile,
    _pops_import_lib,
    pops_header_signature,
    pops_loader_build_flags,
)
from pops.codegen.cache import (
    _cache_so_path,
    _backend_distinct_so_path,
    _record_so_backend,
    _registry_cache_key,
    _precision_cache_key,
    _native_mpi_flags,
    _dsl_optflags,
)
from pops.codegen.compile_provenance import (
    build_debug_banner,
    verify_cached_program_so,
    write_cachekey_sidecar,
)
from pops.codegen.abi import _abi_key_python
from pops.codegen.compile_emit import (
    _BACKENDS,
    model_hash,
    emit_cpp_so_source,
    emit_cpp_aot_source,
    emit_cpp_native_loader,
)
from pops.codegen.backends import lower_backend
from pops.codegen._compile_command_redact import _redact_compile_command  # noqa: F401
# _module_to_model moved to module_lowering.py (500-line budget); re-exported here so
# ``from pops.codegen.compile_drivers import _module_to_model`` (and pops.codegen.compile) is unchanged.
from pops.codegen.module_lowering import _module_to_model  # noqa: F401


# ---------------------------------------------------------------------------
# Compiler runners
# ---------------------------------------------------------------------------

def compile_so(model, so_path, include=None, name=None, cxx=None, std="c++20",
               hoist_reciprocals=False):
    """JIT: generate the FULL MODEL (emit_cpp_so_source) and compile a shared
    library loadable by System.add_dynamic_block (dlopen). Returns so_path.
    """
    import tempfile
    if include is None:
        include = pops_include()
    src = emit_cpp_so_source(model, name=name, hoist_reciprocals=hoist_reciprocals)
    cc = _default_cxx(cxx)
    if not cc:
        raise RuntimeError("compile_so: no C++ compiler found")
    std = _probe_cxx_std(cc, std)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        _run_compile([cc, "-shared", "-fPIC", "-std=" + std, "-O2", "-I", include, cpp,
                      "-o", so_path], "backend jit, compile_so")
    return so_path


def compile_aot(model, so_path, include=None, name=None, cxx=None, std="c++20",
                hoist_reciprocals=False):
    """Backend "compile" (AOT): generate the FULL MODEL (emit_cpp_aot_source)
    and compile a .so loadable by System.add_compiled_block. Returns so_path.

    KOKKOS-ONLY: the AOT model includes the pops headers (multifab/for_each),
    which do NOT compile without POPS_HAS_KOKKOS. So we compile the .so WITH
    Kokkos (same flags as the native loader), which also aligns its ABI with
    the _pops module.
    """
    import tempfile
    if include is None:
        include = pops_include()
    src = emit_cpp_aot_source(model, name=name, hoist_reciprocals=hoist_reciprocals)
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "compile_aot: PoPS is Kokkos-only -- the AOT model includes the pops headers which "
            "require Kokkos. Point at an installed Kokkos via POPS_KOKKOS_ROOT (or Kokkos_ROOT), e.g. "
            "`export POPS_KOKKOS_ROOT=/path/to/kokkos` (Serial is enough on CPU). "
            "Run `python -c \"import pops; pops.doctor()\"` for a full diagnosis and copy-paste fixes.")
    cc = _native_kokkos_compiler(cxx)
    if not cc:
        raise RuntimeError("compile_aot: no C++ compiler found")
    std = _probe_cxx_std(cc, std)
    kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
    mpi_compile_flags = _native_mpi_flags()
    link_extra = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model_aot.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        _run_compile([cc, "-shared", "-fPIC", "-std=" + std, *_dsl_optflags(), "-I", include]
                     + kokkos_compile_flags + mpi_compile_flags + link_extra
                     + [cpp, "-o", so_path] + kokkos_link_flags,
                     "backend aot, compile_aot")
    return so_path


def compile_native(model, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                   hoist_reciprocals=False):
    """Backend "production": generate the NATIVE LOADER (emit_cpp_native_loader)
    and compile it into a .so loadable by System.add_native_block
    (target="system") or AmrSystem.add_native_block (target="amr_system").
    Returns so_path.
    """
    import tempfile
    if include is None:
        include = pops_include()
    sig = _check_headers_match_module(include)
    _warn_kokkos_parity()
    src = emit_cpp_native_loader(model, name=name, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
    cc = _native_kokkos_compiler(cxx)
    if not cc:
        raise RuntimeError(
            "compile_native: no C++ compiler found. The PRODUCTION native route is REQUIRED for "
            "the compile/bind target surface; the prototype/host routes are NOT a fallback (ADC-600).")
    std = _probe_cxx_std(cc, std)
    kokkos_compile_flags, kokkos_link_flags = _native_kokkos_flags()
    mpi_compile_flags = _native_mpi_flags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "model_native.cpp")
        src_eff = ('#define POPS_HEADER_SIG "%s"\n' % sig + src) if sys.platform == "win32" else src
        with open(cpp, "w") as f:
            f.write(src_eff)
        if sys.platform == "win32":
            pops_lib = _pops_import_lib()
            if not pops_lib:
                raise RuntimeError(
                    "compile_native: _pops.lib not found next to the _pops module (required to "
                    "link the DSL .dll; rebuild _pops with POPS_EXPORT_BUILDING_MODULE). The "
                    "PRODUCTION native route is REQUIRED for the compile/bind target surface; the "
                    "prototype/host routes are NOT a fallback (ADC-600).")
            cl_flags = (["/nologo", "/LD", "/std:" + std, "/O2", "/DNDEBUG", "/EHsc",
                         "/permissive-", "/Zc:preprocessor", "/DNOMINMAX", "/bigobj"]
                        + kokkos_compile_flags + mpi_compile_flags)
            cmd = ([cc] + cl_flags + ["-I", include, cpp,
                    "/Fe:" + so_path, "/Fo" + tmp + os.sep,
                    "/link"] + kokkos_link_flags + [pops_lib])
        else:
            optflags = _dsl_optflags()
            flags = ["-shared", "-fPIC", "-std=" + std, *optflags,
                     "-DPOPS_HEADER_SIG=\"%s\"" % sig, *kokkos_compile_flags, *mpi_compile_flags]
            if sys.platform == "darwin":
                flags += ["-undefined", "dynamic_lookup"]
            cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *kokkos_link_flags]
        _run_compile(cmd, "backend production, compile_native")
    return so_path


def compile_or_jit(model, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
                   target="system", hoist_reciprocals=False):
    """Unified API selecting the backend by mode:

    - mode="jit"     -> compile_so (IModel, virtual dispatch: host prototyping);
    - mode="compile" -> compile_aot (AOT production path, numerically identical to native);
    - mode="native"  -> compile_native (native zero-copy loader; target consumed here).

    @p target: "system" (default) | "amr_system". ONLY consumed by mode="native".
    """
    if mode == "jit":
        if target != "system":
            raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'jit' "
                             "(the AMR path exists only for mode='native')")
        return compile_so(model, so_path, include, name=name, cxx=cxx, std=std,
                          hoist_reciprocals=hoist_reciprocals)
    if mode == "compile":
        if target != "system":
            raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'compile' "
                             "(the AMR path exists only for mode='native')")
        return compile_aot(model, so_path, include, name=name, cxx=cxx, std=std,
                           hoist_reciprocals=hoist_reciprocals)
    if mode == "native":
        return compile_native(model, so_path, include, name=name, cxx=cxx, std=std, target=target,
                              hoist_reciprocals=hoist_reciprocals)
    raise ValueError("compile_or_jit: mode 'jit' | 'compile' | 'native' (received %r)" % mode)


# ---------------------------------------------------------------------------
# compile_model -- full facade (mirrors HyperbolicModel.compile logic)
# ---------------------------------------------------------------------------

def compile_model(model, so_path=None, include=None, backend="auto", name=None, cxx=None,
                  std=None, require_metadata=False, target="system", hoist_reciprocals=False):
    """Compilation facade by INTENTION: compiles *model* (a ``HyperbolicModel``)
    into a .so via the engine designated by *backend* and returns its path.

    This is the free-function equivalent of ``HyperbolicModel.compile``.
    ``dsl.HyperbolicModel.compile`` is a thin wrapper that calls this.

    @p backend: "prototype" | "aot" | "production" | "auto".
    @p target:  "system" (default) | "amr_system".
    @p require_metadata: if True, requires physical roles AND explicit gamma.
    Returns so_path.
    """
    from pops.codegen.toolchain import resolve_auto_backend

    m = model
    # ADDITIVE (Spec 5 sec.8.15): accept a typed backend descriptor (Production()/AOT()/JIT())
    # as well as the legacy string; lower it to the token the _BACKENDS table keys on. A plain
    # string passes through unchanged so the existing consumers keep working.
    backend = lower_backend(backend)
    if backend == "auto":
        backend, _auto_reason = resolve_auto_backend(include)
    if backend not in _BACKENDS:
        raise ValueError("compile: backend %r unknown (expected %s + 'auto')"
                         % (backend, sorted(_BACKENDS)))
    if target not in ("system", "amr_system"):
        raise ValueError("compile: target 'system' | 'amr_system' (received %r)" % (target,))
    mode, adder = _BACKENDS[backend]
    if target == "amr_system" and mode != "native":
        raise ValueError("compile: target='amr_system' exists only for backend='production' "
                         "(native AMR path); received backend=%r" % (backend,))
    if std is None:
        std = loader_cxx_std() if mode == "native" else "c++20"
    if include is None:
        include = pops_include()

    # Metadata guard rails (before any cache).
    # _check_require_metadata lives on the HyperbolicModel: call it via the model.
    m._check_require_metadata(require_metadata, backend)

    # Out-of-source CACHE when so_path is omitted.
    if so_path is None:
        kokkos_like = backend in ("production", "aot")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        abi_key = _abi_key_python(include, eff_cxx, std)
        cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
        if hoist_reciprocals:
            cache_backend += ";hoist"
        so_path = _cache_so_path(model_hash(m), abi_key, cache_backend, target, name)
        if os.path.exists(so_path):
            _record_so_backend(so_path, backend)
            return so_path
    else:
        so_path = _backend_distinct_so_path(so_path, backend)

    out_path = compile_or_jit(m, so_path, include, mode=mode, name=name, cxx=cxx, std=std,
                              target=target, hoist_reciprocals=hoist_reciprocals)
    _record_so_backend(out_path, backend)
    return out_path


# ---------------------------------------------------------------------------
# compile_problem -- compile a pops.time.Program into a problem.so
# ---------------------------------------------------------------------------

def compile_problem(so_path=None, *, model=None, time=None, backend="production", target="system",
                    force=False, cxx=None, include=None, std=None, debug=False, libraries=None):
    """Compile a ``pops.time.Program`` into a ``problem.so`` the runtime loads
    via ``sim.install_program``.

    Lowers the Program IR to C++ (``Program.emit_cpp_program``) and compiles
    it against the pops headers with the SAME Kokkos toolchain as the loaded
    _pops module (``pops_loader_build_flags``), so the ``.so`` is ABI-compatible
    and runs in-process. Returns a ``CompiledProblem`` (``.so_path`` + metadata).

    The physical ``model`` is validated here (fail-loud) and carried on the
    handle, but in this MVP it is added as a normal block
    (``sim.add_equation``) while the Program drives the step via
    ``ProgramContext`` (``ctx.rhs_into`` uses the block RHS); a single combined
    model+program ``.so`` is a later phase. Constraints (spec): ``backend``
    must be "production"; ``target`` is "system" (the .so exports
    ``pops_install_program``) or "amr_system" (it ALSO exports
    ``pops_install_program_amr``, the AMR install entry, epic ADC-511 / ADC-508).
    Without an explicit ``so_path``
    the ``.so`` is cached out-of-source keyed by [program source + header
    signature + compiler + std]; ``force=True`` recompiles. ``debug=True`` also
    writes the generated ``.cpp`` next to the ``.so`` for inspection.
    """
    import hashlib
    import tempfile
    from pops.codegen.loader import CompiledProblem
    from pops.codegen.env import CodegenEnv

    # ADDITIVE (Spec 5 sec.12.4, #47-48): resolve the codegen POPS_* environment ONCE. An explicit
    # argument wins over the env -- debug=True forces keep-generated regardless of POPS_KEEP_GENERATED,
    # and the resolver leaves the JIT-backdoor gate OFF unless POPS_JIT_BACKDOOR is itself set (loud
    # warning emitted in from_env when it is). The snapshot is recorded on the returned handle so the
    # active env state is inspectable (criterion #47), never hidden.
    cenv = CodegenEnv.from_env(keep_generated=debug)
    cenv.log("compile_problem: backend=%s target=%s force=%s" % (backend, target, force))

    # ADDITIVE (Spec 5 sec.8.15): accept a typed backend descriptor (Production()) as well as the
    # legacy string; lower it before the production-only guard so both selectors behave the same.
    backend = lower_backend(backend)
    if backend != "production":
        raise ValueError("compiled time programs require backend='production'")
    if target not in ("system", "amr_system"):
        raise ValueError("compiled time programs support target='system' | 'amr_system' "
                         "(received %r)" % (target,))

    library_manifests = []
    external_brick_records = []
    if libraries:
        # Lazy import to avoid a top-level library chain at import time.
        from pops.codegen.library import read_library_manifest  # type: ignore[attr-defined]
        from pops.external.bricks import CompiledBrickRef
        for lib_obj in libraries:
            # ADC-544: a CompiledBrickRef among libraries= is VALIDATED here (the four compile-time
            # gates fire BEFORE any use -- ABI mismatch / missing capability / unsupported layout /
            # missing symbol, all RAISE never warn) and its manifest record is captured so the
            # artifact's manifest() can list its external bricks. Anything else is a brick LIBRARY
            # manifest (LibraryManifest / dict / compiled .so path).
            if isinstance(lib_obj, CompiledBrickRef):
                lib_obj.validate()  # gates fire; raises on ABI / capability / layout / symbol failure
                record = lib_obj.manifest_record()
                if record is not None:
                    external_brick_records.append(record)
                continue
            library_manifests.append(read_library_manifest(lib_obj))

    if time is None or not hasattr(time, "emit_cpp_program"):
        raise ValueError("compile_problem: time must be an pops.time.Program (got %r)" % (time,))

    # ONE VALIDATION (ADC-557): lower_and_validate is the SINGLE validate + lower step. It validates
    # the model ONCE (a raw Module's embedded checks, or a dsl / physics Model's check() dependency
    # validation) and returns the emit model + the operator-first Module (the canonical compile-IR
    # authority carried as the lowered-module trace). The divergent standalone model.check() compile
    # step is gone -- there is no second validation path. A lowering error is remapped onto the user's
    # facade handles. The emit model is byte-identical to before (a Module still lowers via
    # _module_to_model; a dsl / physics Model is consumed as-is).
    from pops.codegen.module_lowering import lower_and_validate
    model, source_module = lower_and_validate(model, facade=model)

    # VALIDATED-OR-ABSENT INVARIANT (ADC-558): every structural check runs HERE, before a handle
    # exists. lower_and_validate validated the model above; emit_cpp_program calls program.validate()
    # + _check_lowerable, so a malformed Program raises now; a compiler error raises at _run_compile
    # below; a stale cache HIT raises at verify_cached_program_so. A CompiledProblem is therefore
    # returned ONLY after validation AND a successful (or already-cached-and-verified) compile -- both
    # return points below hand back a fully-valid, directly bindable handle, never a "to-check" one.
    # There is no public check() to run after compile: the handle's validity is guaranteed by its
    # existence, and the single status signal is the "compiled, waiting for pops.bind(...)" line in
    # inspect(). A failure NEVER lets a partially-validated handle escape.
    src = time.emit_cpp_program(model=model, target=target)

    include = include or pops_include()
    sig = pops_header_signature(include)
    cc, cflags, lflags = pops_loader_build_flags(cxx)
    eff_std = _probe_cxx_std(cc, std or loader_cxx_std())
    abi_key = "%s|%s|%s" % (sig, cc, eff_std)
    # Stable program (problem) hash + cache key (Spec 5 sec.12.4, #48): the program-source hash is
    # the WHAT, the cache key combines it with the abi_key (the HOW) -- the same identity the
    # out-of-source .so cache file name carries. Computed unconditionally so the metadata is present
    # on BOTH the cache-hit and the fresh-compile path (and even when an explicit so_path is given).
    # The route registry / report vocabulary component (ADC-599) enters the key too: a native
    # route change invalidates cached Programs exactly like model .so files. Numerics DESCRIPTOR
    # changes are already covered by program_hash (they change the emitted source).
    # ADC-536: the native Kokkos/MPI feature-key and the precision token join the PROGRAM cache key
    # (the model .so path already folds the feature-key via _native_feature_key; the program key
    # omitted both). A SERIAL-stub .so must not be reused on an MPI module, a .so built against a
    # different Kokkos must be a MISS, and a future precision switch must not reuse a double .so.
    # These tokens were NOT in the key before, so adding them is a ONE-TIME cache invalidation (the
    # .so BYTES are unchanged -- the emitted source is byte-identical; only the keyed file name and
    # the manifest cache_key move once). The same tokens enter _cache_so_path below.
    feature_key = _native_feature_key()
    precision_key = _precision_cache_key()
    program_hash = hashlib.sha256(src.encode()).hexdigest()
    cache_key = hashlib.sha256(("%s|%s|program-production|%s|%s|%s|%s"
                                % (program_hash, abi_key, target, _registry_cache_key(),
                                   feature_key, precision_key))
                               .encode()).hexdigest()

    # The Module manifest (ADC-585): attached on BOTH the cache-hit and fresh-compile path; its
    # abi_key slot is bound in CompiledProblem. None for a bare dsl.Model with no backing Module.
    # source_module is now the operator-first Module for a facade / dsl Model too (ADC-557), so the
    # manifest is ALWAYS the operator-first trace on the standard flow.
    from pops.model.manifest import module_manifest_of
    module_manifest = module_manifest_of(source_module if source_module is not None else model)
    # module_hash (ADC-557 I5): the stable hash of the operator-first Module, carried on the handle so
    # a post-compile in-place model mutation can be DETECTED loudly by bind (parity with the block
    # drift check). None for a model with no backing Module.
    module_hash = source_module.module_hash() if (source_module is not None
                                                   and hasattr(source_module, "module_hash")) else None

    if so_path is None:
        # Fold the feature-key + precision token into the backend slot of the cache file name (the
        # model .so path folds the feature-key the same way), so the keyed .so is distinct once the
        # tokens are non-default -- a one-time re-key, the .so bytes unchanged (ADC-536).
        cache_backend = "program-production;%s;%s" % (feature_key, precision_key)
        so_path = _cache_so_path(program_hash, abi_key, cache_backend, target,
                                 getattr(time, "name", "problem"))
        # POPS_CODEGEN_DIR (sec.12.4, #47): redirect the out-of-source .so (and any kept source /
        # dump) into the requested directory, keeping the collision-free cache file name. An explicit
        # so_path bypasses this -- the caller pinned the path. Created on demand, never inside the repo.
        if cenv.codegen_dir:
            os.makedirs(cenv.codegen_dir, exist_ok=True)
            so_path = os.path.join(cenv.codegen_dir, os.path.basename(so_path))
        if not force and os.path.isfile(so_path):
            # STALE / ABI GUARD (ADC-536, CONTRACTS6 decision 1): a cache HIT reuses the .so WITHOUT
            # recompiling, so nothing else re-checks it against the current keys. verify the sidecar
            # <so>.cachekey matches the freshly computed cache_key / abi_key; a missing sidecar (a
            # legacy .so) or any mismatch RAISES (never a silent warn-and-reuse).
            verify_cached_program_so(so_path, cache_key=cache_key, abi_key=abi_key)
            cenv.log("compile_problem: cache HIT -> %s" % so_path)
            compiled = CompiledProblem(so_path, time, model, abi_key, cc, eff_std,
                                       libraries=library_manifests, problem_hash=program_hash,
                                       cache_key=cache_key, codegen_env=cenv,
                                       module_manifest=module_manifest, module_hash=module_hash,
                                       external_bricks=external_brick_records)
            cenv.run_dumps(compiled)
            return compiled

    optflags = _dsl_optflags()
    # POPS_KEEP_GENERATED (sec.12.4, #47): keep the emitted .cpp next to the .so -- the same effect
    # debug=True has (debug=True already set keep_generated in cenv, explicit-arg-wins). When neither
    # is set the source lives only in the TemporaryDirectory below and is discarded.
    gen_src_path = os.path.splitext(so_path)[0] + ".cpp" if cenv.keep_generated else None
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "problem.cpp")
        # The compiler ALWAYS reads the banner-free src (the temp .cpp). The debug banner rides ONLY
        # the persisted sidecar below, so the .so bytes and the cache key are byte-identical whether
        # debug is on or off (ADC-536 R5).
        with open(cpp, "w") as f:
            f.write(src)
        flags = ["-shared", "-fPIC", "-std=" + eff_std, *optflags,
                 "-DPOPS_HEADER_SIG=\"%s\"" % sig, *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *lflags]
        # Record the compile command for introspection (Spec 5 sec.12.4, #49). The temporary .cpp is
        # in a TemporaryDirectory that is gone after this block, so report the persistent debug .cpp
        # (or "<generated>") rather than the vanished temp path; redact secrets/env in the tokens.
        compile_command = _redact_compile_command(cmd, tmp_cpp=cpp,
                                                  gen_src=gen_src_path or "<generated>")
        # Persist the sidecar .cpp with a leading provenance banner (ADC-536): serialized IR, hashes,
        # flags, toolchain and the redacted command. Written to gen_src_path ONLY, never to cpp -- so
        # the compiled bytes stay banner-free. Now that compile_command is known the banner is complete.
        if gen_src_path:
            banner = build_debug_banner(
                time, model, program_hash=program_hash, abi_key=abi_key, cache_key=cache_key,
                cflags=cflags, lflags=lflags, cxx=cc, std=eff_std, command=compile_command,
                registry=_registry_cache_key())
            try:
                with open(gen_src_path, "w") as f:
                    f.write(banner + src)
            except OSError:
                gen_src_path = None
        cenv.log("compile_problem: invoking %s" % compile_command, level="debug")
        # C++ ERROR CONTEXT (ADC-536): _run_compile raises a self-contained RuntimeError with the
        # compiler output, but the ephemeral temp .cpp it names is gone after this block. On failure
        # persist the GENERATED source next to the .so (unless debug already did) and re-raise citing
        # it, so a compiler error in the emitted code is always inspectable and clearly flagged as
        # generated (re-run with debug=True for the full provenance banner).
        try:
            _run_compile(cmd, "compile_problem (backend production)")
        except RuntimeError as exc:
            failed_src = os.path.splitext(so_path)[0] + ".failed.cpp"
            try:
                with open(failed_src, "w") as f:
                    f.write(src)
            except OSError:
                failed_src = "<generated (not persisted: write failed)>"
            raise RuntimeError(
                "%s\nThis is GENERATED code emitted by pops.time.Program %r; the failing source was "
                "written to %s. Re-run pops.compile(..., debug=True) to keep the .cpp with a full "
                "provenance banner (IR + hashes + flags + command)."
                % (exc, getattr(time, "name", "problem"), failed_src)) from exc
    # Sidecar cache-key file (ADC-536): record the keys + toolchain next to the fresh .so so a later
    # cache HIT can prove the on-disk artifact matches (verify_cached_program_so). Atomic write.
    write_cachekey_sidecar(so_path, cache_key=cache_key, abi_key=abi_key,
                           toolchain="%s|%s" % (cc, eff_std))
    cenv.log("compile_problem: compiled -> %s" % so_path)
    compiled = CompiledProblem(so_path, time, model, abi_key, cc, eff_std,
                               libraries=library_manifests, problem_hash=program_hash,
                               cache_key=cache_key, compile_command=compile_command,
                               generated_sources=[gen_src_path] if gen_src_path else [],
                               codegen_env=cenv, module_manifest=module_manifest,
                               module_hash=module_hash, external_bricks=external_brick_records)
    cenv.run_dumps(compiled)
    return compiled
