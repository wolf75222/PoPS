"""Internal compiler-invocation layer; public users enter through ``compile_problem`` only."""

import hashlib
import json
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
    _native_mpi_flags,
    _dsl_optflags,
)
from pops.codegen.abi import _abi_key_python
from pops.codegen.compile_emit import (
    _BACKENDS,
    model_hash,
    emit_cpp_so_source,
    emit_cpp_aot_source,
    emit_cpp_native_loader,
)
from pops.codegen.backends import lower_backend, lower_problem_backend
from pops.codegen._compile_command_redact import _redact_compile_command  # noqa: F401

__all__ = ["compile_problem"]


# Compiler runners

def _compile_so(model, so_path, include=None, name=None, cxx=None, std="c++20",
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


def _compile_aot(model, so_path, include=None, name=None, cxx=None, std="c++20",
                 hoist_reciprocals=False):
    """AOT model loader compiled with the same Kokkos ABI as ``_pops``."""
    import tempfile
    if include is None:
        include = pops_include()
    src = emit_cpp_aot_source(model, name=name, hoist_reciprocals=hoist_reciprocals)
    if _native_kokkos_root() is None:
        raise RuntimeError(
            "compile_aot: adc_cpp is Kokkos-only -- the AOT model includes the pops headers which "
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


def _compile_native(model, so_path, include=None, name=None, cxx=None, std="c++23", target="system",
                    hoist_reciprocals=False):
    """Production native loader for System or AmrSystem block install."""
    import tempfile
    if include is None:
        include = pops_include()
    sig = _check_headers_match_module(include)
    _warn_kokkos_parity()
    src = emit_cpp_native_loader(model, name=name, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
    cc = _native_kokkos_compiler(cxx)
    if not cc:
        raise RuntimeError("compile_native: no C++ compiler found")
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
                    "link the DSL .dll; rebuild _pops with POPS_EXPORT_BUILDING_MODULE).")
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


def _compile_or_jit(model, so_path, include=None, mode="jit", name=None, cxx=None, std="c++20",
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
        return _compile_so(model, so_path, include, name=name, cxx=cxx, std=std,
                           hoist_reciprocals=hoist_reciprocals)
    if mode == "compile":
        if target != "system":
            raise ValueError("compile_or_jit: target='amr_system' not supported in mode 'compile' "
                             "(the AMR path exists only for mode='native')")
        return _compile_aot(model, so_path, include, name=name, cxx=cxx, std=std,
                            hoist_reciprocals=hoist_reciprocals)
    if mode == "native":
        return _compile_native(model, so_path, include, name=name, cxx=cxx, std=std, target=target,
                               hoist_reciprocals=hoist_reciprocals)
    raise ValueError("compile_or_jit: mode 'jit' | 'compile' | 'native' (received %r)" % mode)


# compile_model -- full facade (mirrors HyperbolicModel.compile logic)

def _compile_model(model, so_path=None, include=None, backend=None, name=None, cxx=None,
                   std=None, require_metadata=False, target="system", hoist_reciprocals=False):
    """Compilation facade by INTENTION: compiles *model* (a ``HyperbolicModel``)
    into a .so via the engine designated by *backend* and returns its path.

    This is the free-function equivalent of ``HyperbolicModel.compile``.
    ``dsl.HyperbolicModel.compile`` is a thin wrapper that calls this.

    @p backend: typed ``Production()`` | ``AOT()`` | ``JIT()`` descriptor; ``None`` means
       ``Production()``.
    @p target: internal native-loader token: "system" (default) | "amr_system".
    @p require_metadata: if True, requires physical roles AND explicit gamma.
    Returns so_path.
    """
    m = model
    backend = lower_backend(backend)
    if backend not in _BACKENDS:
        raise ValueError("compile: backend %r unknown (expected %s)"
                         % (backend, sorted(_BACKENDS)))
    if target not in ("system", "amr_system"):
        raise ValueError("compile: target 'system' | 'amr_system' (received %r)" % (target,))
    mode, adder = _BACKENDS[backend]
    if target == "amr_system" and mode != "native":
        raise ValueError("compile: target='amr_system' exists only for backend=pops.codegen.Production() "
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

    out_path = _compile_or_jit(m, so_path, include, mode=mode, name=name, cxx=cxx, std=std,
                               target=target, hoist_reciprocals=hoist_reciprocals)
    _record_so_backend(out_path, backend)
    return out_path


# compile_problem -- compile a model + pops.time.Program into a problem.so

def _problem_target_from_layout(layout):
    """Return the native problem ABI target selected by a typed mesh layout."""
    if layout is None:
        return "system"
    from pops.mesh.layouts import AMR, Uniform
    if isinstance(layout, AMR):
        return "amr_system"
    if isinstance(layout, Uniform):
        return "system"
    raise TypeError(
        "compile_problem: layout must be a typed pops.mesh.layouts.Uniform(...) or AMR(...) "
        "descriptor; got %r" % type(layout).__name__)


def _stable_identity_value(value):
    """JSON-stable, side-effect-free representation for problem identity records."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_identity_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable_identity_value(value[k]) for k in sorted(value, key=str)}
    if hasattr(value, "inspect") and callable(value.inspect):
        return _stable_identity_value(value.inspect())
    if hasattr(value, "options") and callable(value.options):
        return {
            "type": type(value).__name__,
            "category": getattr(value, "category", None),
            "options": _stable_identity_value(value.options()),
        }
    raise TypeError(
        "compiled problem identity cannot serialize %s: route descriptors must expose "
        "inspect() or options(), and identity values must be JSON primitives, lists or dicts; "
        "repr() is intentionally rejected because it is not a stable identity"
        % type(value).__name__)


def _library_identity(manifests):
    out = []
    for manifest in manifests or []:
        if hasattr(manifest, "to_dict") and callable(manifest.to_dict):
            out.append(_stable_identity_value(manifest.to_dict()))
        elif hasattr(manifest, "as_dict") and callable(manifest.as_dict):
            out.append(_stable_identity_value(manifest.as_dict()))
        else:
            out.append(_stable_identity_value(manifest))
    return out


_GENERATED_SOURCE_IDENTITY_VERSION = "pops-generated-source-v1"


def _semantic_problem_hash(record):
    """Digest only the semantic part of a compiled problem identity."""
    blob = json.dumps(record["semantic"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _compiled_problem_cache_key(record):
    """Digest the full binary cache identity: semantic + provenance + generated-source guard."""
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _compiled_problem_identity(*, source, model, program, layout, backend, target,
                               include, compiler, std, abi_key, optflags,
                               library_manifests,
                               codegen_version=_GENERATED_SOURCE_IDENTITY_VERSION):
    """Structured identity of the combined problem artifact.

    ``problem_hash`` is the semantic identity: Module IR, Program IR, route descriptors, layout,
    backend/platform and libraries. Toolchain provenance and generated-source guards live beside it
    in ``problem_identity`` and participate in ``cache_key`` only. This keeps equivalent headers
    under a different include path from changing the semantic hash while still preventing a stale
    binary cache hit.
    """
    source_identity = {
        "version": str(codegen_version),
        "source": source,
    }
    source_hash = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    module_hash = model.module_hash() if hasattr(model, "module_hash") else None
    program_hash = program._ir_hash() if hasattr(program, "_ir_hash") else None
    record = {
        "schema": "pops-compiled-problem-v1",
        "semantic": {
            "name": getattr(program, "name", "problem"),
            "module": {
                "name": getattr(model, "name", None),
                "hash": module_hash,
            },
            "program": {
                "name": getattr(program, "name", None),
                "hash": program_hash,
            },
            "descriptors": {
                "layout": _stable_identity_value(layout),
                "backend": _stable_identity_value(backend),
                "libraries": _library_identity(library_manifests),
            },
            "toolchain": {
                "compiler": compiler,
                "std": std,
                "abi_key": abi_key,
                "native_features": _native_feature_key(),
                "optflags": list(optflags),
            },
            "runtime_route": {
                "target": target,
            },
        },
        "provenance": {
            "include": include,
            "compiler": compiler,
            "std": std,
            "abi_key": abi_key,
            "native_features": _native_feature_key(),
            "optflags": list(optflags),
        },
        "generated_source": {
            "version": source_identity["version"],
            "hash": source_hash,
            "language": "c++",
        },
    }
    problem_hash = _semantic_problem_hash(record)
    return record, problem_hash, module_hash, program_hash, source_hash


def compile_problem(so_path=None, *, model=None, program=None, time=None, backend=None, layout=None,
                    force=False, cxx=None, include=None, std=None, debug=False, libraries=None):
    """Compile a physical model + ``pops.time.Program`` into one ``problem.so``.

    This is the public Spec corrective route. It does not compile a Program in isolation and it
    does not accept string route selectors: pass ``backend=Production()`` (or omit it for the
    default) and select the runtime ABI with ``layout=Uniform(...)`` / ``layout=AMR(...)``. A
    ``layout=None`` compile emits the uniform ``System`` ABI for compact scripts.

    The produced ``.so`` carries the GeneratedProgram plus GeneratedModule metadata and is later
    installed by ``sim.install(compiled, ...)``. ``program=`` is the public Program argument;
    ``time=`` is accepted only as an internal transition alias while older tests are migrated.
    """
    import tempfile
    from pops.codegen.loader import CompiledProblem
    from pops.codegen.env import CodegenEnv
    from pops.model import Module

    # ADDITIVE (Spec 5 sec.12.4, #47-48): resolve the codegen POPS_* environment ONCE. An explicit
    # argument wins over the env -- debug=True forces keep-generated regardless of POPS_KEEP_GENERATED,
    # and the resolver leaves the JIT-backdoor gate OFF unless POPS_JIT_BACKDOOR is itself set (loud
    # warning emitted in from_env when it is). The snapshot is recorded on the returned handle so the
    # active env state is inspectable (criterion #47), never hidden.
    cenv = CodegenEnv.from_env(keep_generated=debug)

    backend_descriptor = backend if backend is not None else {
        "type": "Production",
        "default": True,
    }
    backend = lower_problem_backend(backend)
    target = _problem_target_from_layout(layout)
    cenv.log("compile_problem: backend=%s target=%s force=%s" % (backend, target, force))
    if backend != "production":
        raise ValueError("compile_problem: compiled problems require backend=Production()")

    library_manifests = []
    if libraries:
        # Lazy import to avoid a top-level library chain at import time.
        from pops.codegen.library import read_library_manifest  # type: ignore[attr-defined]
        for lib_obj in libraries:
            library_manifests.append(read_library_manifest(lib_obj))

    if program is not None and time is not None:
        raise TypeError("compile_problem: pass program=, not both program= and legacy time=")
    if program is None:
        program = time
    if program is None or not hasattr(program, "emit_cpp_program"):
        raise ValueError(
            "compile_problem: program must be an pops.time.Program (got %r)" % (program,))
    if model is None:
        raise ValueError(
            "compile_problem: model is required; compiling a time Program without a physical "
            "model is no longer supported")

    if not isinstance(model, Module) and hasattr(model, "_m"):
        raise TypeError(
            "compile_problem: legacy physics/codegen facades carrying private _m are not "
            "accepted. Author with pops.physics.Model and pass model=m.to_module(), or build a "
            "pops.model.Module directly.")
    if not isinstance(model, Module) and hasattr(model, "to_module"):
        model = model.to_module()
    if not isinstance(model, Module):
        raise TypeError(
            "compile_problem: model must be a pops.model.Module (or a modern "
            "pops.physics.Model with to_module()). Legacy physics/codegen facades are not "
            "accepted; build a Module and compile that.")

    if model is not None and hasattr(model, "check"):
        model.check()

    include = include or pops_include()
    sig = pops_header_signature(include)
    cc, cflags, lflags = pops_loader_build_flags(cxx)
    eff_std = _probe_cxx_std(cc, std or loader_cxx_std())
    abi_key = "%s|%s|%s" % (sig, cc, eff_std)
    optflags = _dsl_optflags()

    # Compute the semantic problem hash first, then embed that hash into the generated problem ABI.
    # A second identity pass records the final generated-source guard/cache key. The semantic digest
    # is independent of the generated source bytes, so the second pass must return the same hash.
    src_probe = program._emit_cpp_program_for_target(model=model, target=target)
    problem_identity, problem_hash, module_hash, program_hash, source_hash = (
        _compiled_problem_identity(
            source=src_probe,
            model=model,
            program=program,
            layout=layout,
            backend=backend_descriptor,
            target=target,
            include=include,
            compiler=cc,
            std=eff_std,
            abi_key=abi_key,
            optflags=optflags,
            library_manifests=library_manifests,
        )
    )
    src = program._emit_cpp_program_for_target(model=model, target=target, problem_hash=problem_hash)
    problem_identity, final_problem_hash, module_hash, program_hash, source_hash = (
        _compiled_problem_identity(
            source=src,
            model=model,
            program=program,
            layout=layout,
            backend=backend_descriptor,
            target=target,
            include=include,
            compiler=cc,
            std=eff_std,
            abi_key=abi_key,
            optflags=optflags,
            library_manifests=library_manifests,
        )
    )
    if final_problem_hash != problem_hash:
        raise RuntimeError(
            "compile_problem: internal problem hash instability while embedding pops_problem_hash")
    problem_hash = final_problem_hash
    cache_key = _compiled_problem_cache_key(problem_identity)

    if so_path is None:
        so_path = _cache_so_path(cache_key, abi_key, "problem-production", target,
                                 getattr(program, "name", "problem"))
        # POPS_CODEGEN_DIR (sec.12.4, #47): redirect the out-of-source .so (and any kept source /
        # dump) into the requested directory, keeping the collision-free cache file name. An explicit
        # so_path bypasses this -- the caller pinned the path. Created on demand, never inside the repo.
        if cenv.codegen_dir:
            os.makedirs(cenv.codegen_dir, exist_ok=True)
            so_path = os.path.join(cenv.codegen_dir, os.path.basename(so_path))
        if not force and os.path.isfile(so_path):
            cenv.log("compile_problem: cache HIT -> %s" % so_path)
            compiled = CompiledProblem(so_path, program, model, abi_key, cc, eff_std,
                                       libraries=library_manifests, problem_hash=problem_hash,
                                       module_hash=module_hash, source_hash=source_hash,
                                       problem_identity=problem_identity, cache_key=cache_key,
                                       codegen_env=cenv)
            cenv.run_dumps(compiled)
            return compiled

    # POPS_KEEP_GENERATED (sec.12.4, #47): keep the emitted .cpp next to the .so -- the same effect
    # debug=True has (debug=True already set keep_generated in cenv, explicit-arg-wins). When neither
    # is set the source lives only in the TemporaryDirectory below and is discarded.
    gen_src_path = os.path.splitext(so_path)[0] + ".cpp" if cenv.keep_generated else None
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "problem.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        if gen_src_path:
            try:
                with open(gen_src_path, "w") as f:
                    f.write(src)
            except OSError:
                gen_src_path = None
        flags = ["-shared", "-fPIC", "-std=" + eff_std, *optflags,
                 "-DPOPS_HEADER_SIG=\"%s\"" % sig, *cflags]
        cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *lflags]
        # Record the compile command for introspection (Spec 5 sec.12.4, #49). The temporary .cpp is
        # in a TemporaryDirectory that is gone after this block, so report the persistent debug .cpp
        # (or "<generated>") rather than the vanished temp path; redact secrets/env in the tokens.
        compile_command = _redact_compile_command(cmd, tmp_cpp=cpp,
                                                  gen_src=gen_src_path or "<generated>")
        cenv.log("compile_problem: invoking %s" % compile_command, level="debug")
        _run_compile(cmd, "compile_problem (backend production)")
    cenv.log("compile_problem: compiled -> %s" % so_path)
    compiled = CompiledProblem(so_path, program, model, abi_key, cc, eff_std,
                               libraries=library_manifests, problem_hash=problem_hash,
                               module_hash=module_hash, source_hash=source_hash,
                               problem_identity=problem_identity, cache_key=cache_key,
                               compile_command=compile_command,
                               generated_sources=[gen_src_path] if gen_src_path else [],
                               codegen_env=cenv)
    cenv.run_dumps(compiled)
    return compiled
