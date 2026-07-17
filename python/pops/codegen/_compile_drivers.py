"""Compiler invocation and facade layer extracted from :mod:`pops.codegen._compile`.
Physics and facade helpers stay lazy to preserve the import graph."""

from __future__ import annotations

import os
import sys
from typing import Any

from pops.codegen.toolchain import (
    pops_include,
    loader_cxx_std,
    _probe_cxx_std,
    _check_headers_match_module,
    _warn_kokkos_parity,
    _native_kokkos_compiler,
    _run_compile,
    _pops_import_lib,
    pops_header_signature,
    pops_loader_build_flags,
)
from pops.codegen.cache import (
    _artifact_cache_lock,
    _artifact_cache_staging_path,
    _identity_cache_so_path,
    _artifact_distinct_so_path,
    _record_artifact_identity,
    _registry_cache_key,
    _dsl_optflags,
)
from pops.codegen.compile_provenance import (
    artifact_sidecar_path,
    build_debug_banner,
    publish_staged_artifact,
    verify_cached_artifact,
    write_artifact_sidecar,
)
from pops.codegen.abi import _abi_key_python
from pops.codegen._compile_emit import emit_cpp_native_loader
from pops.codegen._backends import lower_backend
from pops.codegen._compile_command_redact import _redact_compile_command  # noqa: F401
from pops.codegen.compile_link_flags import deterministic_program_link_flags
# _module_to_model moved to module_lowering.py (500-line budget); re-exported here so
# ``from pops.codegen._compile_drivers import _module_to_model`` (and pops.codegen._compile) is unchanged.
from pops.codegen.module_lowering import _module_to_model  # noqa: F401


def compile_native(model: Any, so_path: Any, include: Any = None, name: Any = None, cxx: Any = None,
                   std: Any = "c++23", target: Any = "system",
                   hoist_reciprocals: Any = False) -> Any:
    """Backend "production": generate the NATIVE LOADER (emit_cpp_native_loader)
    and compile it into a .so loadable by System.add_native_block
    (target="system") or AmrSystem.add_native_block (target="amr_system").
    Returns so_path.
    """
    import tempfile
    from pops.codegen._compile_platform import require_shared_library_compile_platform
    require_shared_library_compile_platform("compile_native", windows_supported=True)
    if include is None:
        include = pops_include()
    sig = _check_headers_match_module(include)
    _warn_kokkos_parity()
    src = emit_cpp_native_loader(model, name=name, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
    cc, native_compile_flags, native_link_flags = pops_loader_build_flags(cxx)
    if not cc:
        raise RuntimeError(
            "compile_native: no C++ compiler found. The PRODUCTION native route is REQUIRED for "
            "the compile/bind target surface; the prototype/host routes are NOT a fallback (ADC-600).")
    std = _probe_cxx_std(cc, std)
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
                        + native_compile_flags)
            cmd = ([cc] + cl_flags + ["-I", include, cpp,
                    "/Fe:" + so_path, "/Fo" + tmp + os.sep,
                    "/link"] + native_link_flags + [pops_lib])
        else:
            optflags = _dsl_optflags()
            flags = ["-shared", "-fPIC", "-std=" + std, *optflags,
                     "-DPOPS_HEADER_SIG=\"%s\"" % sig, *native_compile_flags]
            cmd = [cc, *flags, "-I", include, cpp, "-o", so_path, *native_link_flags]
        _run_compile(cmd, "backend production, compile_native")
    return so_path


# ---------------------------------------------------------------------------
# compile_model -- full facade (mirrors HyperbolicModel.compile logic)
# ---------------------------------------------------------------------------

def compile_model(model: Any, so_path: Any = None, include: Any = None, backend: Any = "production",
                  name: Any = None, cxx: Any = None, std: Any = None,
                  require_metadata: Any = False, target: Any = "system",
                  hoist_reciprocals: Any = False) -> Any:
    """Compilation facade by INTENTION: compiles *model* (a ``HyperbolicModel``)
    into a native fixed-ABI package and returns its path.

    This is the free-function equivalent of ``HyperbolicModel.compile``.
    ``dsl.HyperbolicModel.compile`` is a thin wrapper that calls this.

    @p backend: the private ``"production"`` token or ``Production()``.
    @p target:  "system" (default) | "amr_system".
    @p require_metadata: if True, requires physical roles AND explicit gamma.
    Returns so_path.
    """
    m = model
    backend = lower_backend(backend)
    if target not in ("system", "amr_system"):
        raise ValueError("compile: target 'system' | 'amr_system' (received %r)" % (target,))
    if std is None:
        std = loader_cxx_std()
    if include is None:
        include = pops_include()

    # Metadata guard rails (before any cache).
    # _check_require_metadata lives on the HyperbolicModel: call it via the model.
    m._check_require_metadata(require_metadata, backend)

    eff_cxx = _native_kokkos_compiler(cxx)
    abi_key = _abi_key_python(include, eff_cxx, std)
    from pops.codegen._artifact_identity import model_artifact_spec

    semantic_identity, spec_identity = model_artifact_spec(
        m, backend=str(backend), target=str(target), name=name, compiler=eff_cxx,
        standard=std, abi_key=str(abi_key),
        hoist_reciprocals=hoist_reciprocals)

    def _compile_and_authenticate(path: Any, destination: Any = None) -> Any:
        out_path = compile_native(m, path, include, name=name, cxx=cxx, std=std,
                                  target=target, hoist_reciprocals=hoist_reciprocals)
        if destination is None:
            write_artifact_sidecar(
                out_path, semantic_identity=semantic_identity, spec_identity=spec_identity)
            published = out_path
        else:
            publish_staged_artifact(
                out_path,
                destination,
                semantic_identity=semantic_identity,
                spec_identity=spec_identity,
            )
            published = destination
        _record_artifact_identity(published, spec_identity)
        return published

    # Out-of-source CACHE when so_path is omitted.  The check and publication share one
    # cross-process critical section: MPI ranks must never compile over the same content-addressed
    # binary after another rank has sealed it into a CompiledSimulationArtifact.
    if so_path is None:
        so_path = _identity_cache_so_path(spec_identity)
        with _artifact_cache_lock(so_path):
            if os.path.exists(so_path):
                verify_cached_artifact(
                    so_path, semantic_identity=semantic_identity, spec_identity=spec_identity)
                return so_path
            staging = _artifact_cache_staging_path(so_path)
            try:
                return _compile_and_authenticate(staging, so_path)
            finally:
                for leftover in (staging, artifact_sidecar_path(staging)):
                    try:
                        os.remove(leftover)
                    except FileNotFoundError:
                        pass
    else:
        so_path = _artifact_distinct_so_path(so_path, spec_identity)
    return _compile_and_authenticate(so_path)


# compile_problem -- compile a pops.time.Program into a problem.so
def compile_problem(so_path: Any = None, *, model: Any = None, model_graph: Any = None,
                    time: Any = None,
                    backend: Any = "production", target: Any = "system", force: Any = False,
                    cxx: Any = None, include: Any = None, std: Any = None, debug: Any = False,
                    libraries: Any = None, problem_snapshot: Any = None,
                    field_plans: Any = None) -> Any:
    """Compile a time Program into an ABI-compatible native ``problem.so``.

    Only the production backend is supported; ``target`` selects system or AMR entrypoints. An
    omitted path uses the content-addressed cache, ``force`` recompiles, and ``debug`` retains C++.
    The returned ``CompiledProblem`` carries the validated physical model and compile metadata.
    """
    import tempfile
    from pops.codegen._compile_platform import require_shared_library_compile_platform
    require_shared_library_compile_platform("compile_problem", windows_supported=False)
    from pops.codegen.loader import CompiledProblem
    from pops.codegen.env import CodegenEnv
    if problem_snapshot is not None:
        from pops.problem._snapshot import validate_problem_snapshot
        validate_problem_snapshot(problem_snapshot)
    # Resolve the codegen POPS_* environment once. An explicit argument wins over the environment;
    # debug=True forces keep-generated regardless of POPS_KEEP_GENERATED. The immutable snapshot is
    # recorded on the returned handle so the active compile settings remain inspectable.
    cenv = CodegenEnv.from_env(keep_generated=debug)
    cenv.log("compile_problem: backend=%s target=%s force=%s" % (backend, target, force))

    # Authenticate the sole final compiler route before inspecting the program graph.
    backend = lower_backend(backend)
    if target not in ("system", "amr_system"):
        raise ValueError("compiled time programs support target='system' | 'amr_system' "
                         "(received %r)" % (target,))

    if libraries:
        raise TypeError(
            "compile_problem(libraries=) was removed; compile authenticated source components "
            "with pops.external.compile_component and reference their canonical descriptors")
    library_manifests = []
    external_brick_records = []

    from pops.time._program.contract import require_program
    try:
        require_program(time, exact=True, where="compile_problem: time")
    except (TypeError, RuntimeError):
        raise ValueError(
            "compile_problem: time must be an pops.time.Program (got %r)" % (time,)
        ) from None
    from pops.codegen.program_models import prepare_program_authority
    model, source_module, lowering_coverage, compile_authority = (
        prepare_program_authority(model, model_graph)
    )
    from pops.time._program.detach import detach_compiled_program
    time = detach_compiled_program(time)
    program_graph = time.to_graph()
    from pops.codegen.program_graph_lowering import emit_program_graph
    src = emit_program_graph(
        program_graph, lowering_program=time, model=model,
        model_graph=model_graph, target=target, field_plans=field_plans)

    include = include or pops_include()
    sig = pops_header_signature(include)
    cc, cflags, lflags = pops_loader_build_flags(cxx)
    lflags = deterministic_program_link_flags(lflags)
    eff_std = _probe_cxx_std(cc, std or loader_cxx_std())
    abi_key = "%s|%s|%s" % (sig, cc, eff_std)
    # Semantic, artifact-spec and final binary identities remain independently versioned.
    optflags = _dsl_optflags()
    from pops.codegen._artifact_identity import program_artifact_spec

    semantic, spec_identity = program_artifact_spec(
        snapshot=problem_snapshot,
        model_authority=(
            model_graph
            if model_graph is not None
            else source_module if source_module is not None else model
        ),
        program=time,
        program_graph=program_graph,
        target=target,
        abi_key=abi_key,
        compiler=cc,
        standard=eff_std,
        source=src,
        cflags=cflags,
        lflags=lflags,
        optflags=optflags,
        libraries=library_manifests,
    )
    program_hash = semantic.hexdigest
    cache_key = spec_identity.hexdigest

    # The Module manifest (ADC-585): attached on BOTH the cache-hit and fresh-compile path; its
    # abi_key slot is bound in CompiledProblem. None for a bare dsl.Model with no backing Module.
    # source_module is now the operator-first Module for a facade / dsl Model too (ADC-557), so the
    # manifest is ALWAYS the operator-first trace on the standard flow.
    from pops.model.manifest import module_manifest_of
    module_manifest = (
        None
        if model_graph is not None
        else module_manifest_of(source_module if source_module is not None else model)
    )
    # module_hash (ADC-557 I5): the stable hash of the operator-first Module, carried on the handle so
    # a post-compile in-place model mutation can be DETECTED loudly by bind (parity with the block
    # drift check). None for a model with no backing Module.
    module_hash = source_module.module_hash() if (source_module is not None
                                                   and hasattr(source_module, "module_hash")) else None
    # Capture the program-parameter ABI table while the compiler still owns the full model IR.
    # Public orchestration replaces the live model builder by a CompiledModel metadata/loader value;
    # bind consumes this immutable table and never re-enters authoring analysis.
    from pops.codegen.program_emit_params import program_param_entries
    program_param_routes = tuple(program_param_entries(time, compile_authority))

    def _compiled_handle(binary, artifact, *, compile_command=None, generated_source=None):
        compiled = CompiledProblem(
            so_path,
            time,
            compile_authority,
            abi_key,
            cc,
            eff_std,
            libraries=library_manifests,
            problem_hash=program_hash,
            cache_key=cache_key,
            compile_command=compile_command,
            generated_sources=[generated_source] if generated_source else [],
            codegen_env=cenv,
            module_manifest=module_manifest,
            module_hash=module_hash,
            external_bricks=external_brick_records,
            problem_snapshot=problem_snapshot,
            program_param_routes=program_param_routes,
            generated_cpp=src,
            lowering_coverage=lowering_coverage,
            program_graph=program_graph,
        )
        compiled.semantic_identity = semantic
        compiled.artifact_spec_identity = spec_identity
        compiled.binary_identity = binary
        compiled.artifact_identity = artifact
        cenv.run_dumps(compiled)
        return compiled

    def _compile_fresh(*, staged: bool):
        # Compiler writes never target a shared content-addressed path.  Only a fully compiled and
        # authenticated staging pair is atomically promoted while the identity lock is held.
        compile_path = _artifact_cache_staging_path(so_path) if staged else so_path
        gen_src_path = os.path.splitext(so_path)[0] + ".cpp" if cenv.keep_generated else None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cpp = os.path.join(tmp, "problem.cpp")
                # The compiler ALWAYS reads the banner-free src. The retained source and failure
                # diagnostics use the logical final path, never the private staging name.
                with open(cpp, "w") as f:
                    f.write(src)
                flags = [
                    "-shared", "-fPIC", "-std=" + eff_std, *optflags,
                    "-DPOPS_HEADER_SIG=\"%s\"" % sig, *cflags,
                ]
                cmd = [cc, *flags, "-I", include, cpp, "-o", compile_path, *lflags]
                reported_cmd = [so_path if item == compile_path else item for item in cmd]
                compile_command = _redact_compile_command(
                    reported_cmd,
                    tmp_cpp=cpp,
                    gen_src=gen_src_path or "<generated>",
                )
                if gen_src_path:
                    banner = build_debug_banner(
                        time,
                        compile_authority,
                        program_hash=program_hash,
                        abi_key=abi_key,
                        cache_key=cache_key,
                        cflags=cflags,
                        lflags=lflags,
                        cxx=cc,
                        std=eff_std,
                        command=compile_command,
                        registry=_registry_cache_key(),
                    )
                    try:
                        with open(gen_src_path, "w") as f:
                            f.write(banner + src)
                    except OSError:
                        gen_src_path = None
                cenv.log("compile_problem: invoking %s" % compile_command, level="debug")
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
                        "%s\nThis is GENERATED code emitted by pops.time.Program %r; the failing "
                        "source was written to %s. Re-run pops.compile(..., debug=True) to keep "
                        "the .cpp with a full provenance banner (IR + hashes + flags + command)."
                        % (exc, getattr(time, "name", "problem"), failed_src)
                    ) from exc
            if staged:
                binary, artifact = publish_staged_artifact(
                    compile_path,
                    so_path,
                    semantic_identity=semantic,
                    spec_identity=spec_identity,
                )
            else:
                binary, artifact = write_artifact_sidecar(
                    so_path,
                    semantic_identity=semantic,
                    spec_identity=spec_identity,
                )
            _record_artifact_identity(so_path, spec_identity)
            cenv.log("compile_problem: compiled -> %s" % so_path)
            return _compiled_handle(
                binary,
                artifact,
                compile_command=compile_command,
                generated_source=gen_src_path,
            )
        finally:
            if staged:
                for leftover in (compile_path, artifact_sidecar_path(compile_path)):
                    try:
                        os.remove(leftover)
                    except FileNotFoundError:
                        pass

    if so_path is None:
        so_path = _identity_cache_so_path(spec_identity)
        # POPS_CODEGEN_DIR redirects the logical final artifact, not the private staging output.
        if cenv.codegen_dir:
            os.makedirs(cenv.codegen_dir, exist_ok=True)
            so_path = os.path.join(cenv.codegen_dir, os.path.basename(so_path))
        with _artifact_cache_lock(so_path):
            if not force and os.path.isfile(so_path):
                binary, artifact = verify_cached_artifact(
                    so_path,
                    semantic_identity=semantic,
                    spec_identity=spec_identity,
                )
                cenv.log("compile_problem: cache HIT -> %s" % so_path)
                return _compiled_handle(binary, artifact)
            return _compile_fresh(staged=True)

    so_path = _artifact_distinct_so_path(so_path, spec_identity)
    return _compile_fresh(staged=False)
