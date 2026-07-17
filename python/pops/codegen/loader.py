"""pops.codegen.loader : result/wrapper classes for compiled .so artefacts.

``CompiledModel`` packages a model ``.so`` with the metadata needed to wire
it (adder, names, roles, gamma, n_aux, params, caps, abi_key, model_hash).

``CompiledProblem`` packages a program ``.so`` (a compiled time
``pops.time.Program``) plus the metadata to install + reproduce it.

Neither class imports ``pops.dsl`` or ``pops.physics`` at module level.
"""
from __future__ import annotations

from typing import Any

from pops.codegen._loader_dump import CompiledProblemDumpMixin


class CompiledProblem(CompiledProblemDumpMixin):
    """An advanced, INTERNAL compiled handle: a generated ``problem.so`` (a
    compiled time Program) plus the metadata to install + reproduce it. It is
    produced by the low-level ``pops.codegen.compile_problem(...)`` driver and,
    for the public path, wrapped by ``pops.compile(...)``; wire a runnable
    simulation from it with ``pops.bind(compiled, ...)`` (ADC-523). The concrete
    class stays off the top-level surface; callers consume the authenticated object returned by
    ``pops.compile`` through its inspection protocol. The bound Program drives
    ``sim.step(dt)`` entirely in C++ via ``ProgramContext``.

    The ``.so`` is compiled against the pops headers with the SAME Kokkos
    toolchain as the loaded _pops module (cf. ``pops_loader_build_flags``),
    so its ABI key matches and the internal install seam accepts it.
    ``os.fspath(compiled)`` returns ``so_path`` (it can be passed where a
    path is expected).

    VALIDATED-OR-ABSENT (ADC-558): a ``CompiledProblem`` is returned ONLY after
    ``compile_problem`` has validated the Program / model and successfully
    compiled (or verified a cached) ``.so`` -- so a handle in hand is always
    fully validated and directly ``pops.bind``-able. There is NO public
    ``check()`` to run afterwards; the inspectable surface is
    :meth:`inspect` / :meth:`requirements` / :meth:`manifest`, and the single
    validity signal is the ``"compiled, waiting for pops.bind(...)"`` status
    line. A failed compile raises before any handle exists.
    """

    def __init__(self, so_path: Any, program: Any, model: Any, abi_key: Any, cxx: Any, std: Any,
                 libraries: Any = None, problem_hash: Any = None, cache_key: Any = None,
                 compile_command: Any = None, generated_sources: Any = None,
                 codegen_env: Any = None, module_manifest: Any = None,
                 module_hash: Any = None, external_bricks: Any = None,
                 problem_snapshot: Any = None, bind_schema: Any = None,
                 program_param_routes: Any = None, generated_cpp: Any = None,
                 lowering_coverage: Any = None, program_graph: Any = None) -> None:
        self.so_path = so_path
        # Code emission has completed before the loader is constructed.  Retain a clone-owned,
        # registry-free Program for inspection/runtime metadata, never the authoring Program graph.
        from pops.time._program.api import Program
        resolved_program = program
        if isinstance(program, Program):
            from pops.time._program.detach import detach_compiled_program
            resolved_program = detach_compiled_program(program)
        self.program = resolved_program
        if program_graph is None and self.program is not None:
            program_graph = self.program.to_graph()
        if program_graph is not None:
            from pops.time import ProgramGraph

            if type(program_graph) is not ProgramGraph:
                raise TypeError("CompiledProblem program_graph must be an exact ProgramGraph")
            if self.program is None:
                raise ValueError("CompiledProblem cannot carry a ProgramGraph without a Program")
            resolved_graph = self.program.to_graph()
            if resolved_graph.graph_hash != program_graph.graph_hash:
                raise ValueError(
                    "CompiledProblem ProgramGraph does not match the resolved lowering Program"
                )
        self.program_graph = program_graph
        if self.program is None:
            self.program_block_routes = ()
        else:
            from pops.time.references import block_name

            self.program_block_routes = tuple(
                sorted(
                    (index, block_name(reference))
                    for reference, index in self.program._block_indices().items()
                )
            )
        self.model = model              # the physical model (optional; added as a block in the MVP)
        # The self-describing Module manifest (ADC-585): the operator-first central representation of
        # the resolved model (spaces / params / aux / typed operators / native routes), superseding
        # the legacy flat ModelSpec. None when the model is a bare dsl.Model with no backing Module,
        # or when the handle is built outside compile_problem. Its abi_requirements abi_key slot is
        # bound below from this handle's abi_key (a compile-time fact the manifest builder leaves
        # open).
        self.module_manifest = module_manifest
        if module_manifest is not None and abi_key is not None:
            binder = getattr(module_manifest, "with_abi_key", None)
            if not callable(binder):
                raise TypeError(
                    "CompiledProblem: module_manifest must provide with_abi_key() so ABI binding "
                    "does not mutate shared manifest state")
            self.module_manifest = binder(abi_key)
        from pops.codegen._compiled_module_view import CompiledModuleView
        self._module_view = CompiledModuleView(model, self.module_manifest)
        # The operator-first Module hash (ADC-557 I5): the compile-time identity of the canonical
        # compile IR, frozen on the handle so a post-compile in-place model mutation can be DETECTED
        # loudly (parity with the block-name drift check). None for a model with no backing Module.
        self._module_hash = module_hash
        self.program_name = getattr(self.program, "name", None)
        program_ir_hash = getattr(self.program, "_ir_hash", None)
        self.program_hash = program_ir_hash() if callable(program_ir_hash) else None
        self.program_graph_hash = (
            self.program_graph.graph_hash if self.program_graph is not None else None
        )
        self.abi_key = abi_key          # cache key: header signature | compiler | C++ standard
        self.cxx = cxx
        self.std = std
        # Validated brick libraries (Spec 3 section 21, ADC-464): the LibraryManifests read +
        # ABI-checked from libraries=[...]. Empty when none were passed. Their bricks (and their
        # generated symbols) are exposed to the problem; a compiled library .so was already
        # dlopen'd (and ABI-guarded) by read_library_manifest.
        self.libraries = list(libraries) if libraries else []
        # ADC-544: the external compiled-brick manifest records (native_id / category / requirements
        # / capabilities / supported_layouts / supported_platforms / exported_symbols) bound into this
        # artifact via CompiledBrickRef entries in libraries=. They were VALIDATED (the four
        # compile-time gates) by compile_problem before the handle existed. Empty when none were
        # passed. manifest() lists them so the artifact self-describes its external dependencies.
        self.external_bricks = [dict(r) for r in external_bricks] if external_bricks else []
        # Compiled-artifact metadata (Spec 5 sec.12.4, #48-49): set by compile_problem. The
        # problem hash is the artifact-input hash (program source plus an optional frozen Problem
        # snapshot -- distinct from program_hash, the in-memory Program IR hash); cache_key is the
        # identity the out-of-source cache file name/sidecar carries; compile_command is the redacted
        # compiler invocation; generated_sources are the .cpp files written for inspection (debug=).
        # None on a route that does not record a value (e.g. an externally constructed handle) -- a
        # documented absence, not a fabricated value (cf. the property accessors below).
        self._problem_hash = problem_hash
        self._cache_key = cache_key
        self._problem_snapshot = problem_snapshot
        self._compile_command = compile_command
        self._generated_sources = list(generated_sources) if generated_sources else []
        # Exact banner-free translation unit lowered by compile_problem.  Public orchestration
        # replaces the authoring model by an immutable native loader after compilation; retaining
        # source text lets dump_cpp() remain exact without re-entering that discarded builder.
        self._generated_cpp = generated_cpp
        self.lowering_coverage = lowering_coverage
        # Active codegen POPS_* environment snapshot (Spec 5 sec.12.4, #47-48): the resolved
        # CodegenEnv that governed this compile (log level, codegen dir, keep-generated, dump flags,
        # cache dir, profile, and the UNSAFE jit-backdoor gate). Recorded so the env state
        # is inspectable in inspect(); None for a handle built outside compile_problem (no env was
        # resolved -- a documented absence, not a fabricated default).
        self._codegen_env = codegen_env
        # The immutable parameter contract. The low-level compile_problem driver may not have a
        # whole Problem and therefore leaves it absent; pops.compile attaches the schema captured
        # from its frozen Problem before sealing this handle.
        self.bind_schema = bind_schema
        # Immutable ABI routing facts captured while the compiler still owns the full authoring
        # model. Public orchestration may replace ``model`` by a CompiledModel loader afterwards;
        # bind therefore never has to re-run model analysis from a builder.
        # ``None`` means the low-level handle never captured this ABI table; an empty tuple means
        # compilation captured it and proved that the Program reads no runtime parameter.  Keeping
        # that distinction lets bind fail closed instead of re-entering codegen from ``model``.
        self.program_param_routes = (
            None if program_param_routes is None else tuple(program_param_routes)
        )
        self.install_plan = None
        self.semantic_identity = None
        self.artifact_spec_identity = None
        self.binary_identity = None
        self.artifact_identity = None

    @property
    def target(self) -> str:
        plan = self.install_plan
        return plan.target if plan is not None else "system"

    @property
    def layout(self) -> Any:
        plan = self.install_plan
        return plan.layout if plan is not None else None

    @property
    def authoring_snapshot(self) -> Any:
        """Complete immutable authoring identity used to compile this artifact."""
        return self._problem_snapshot

    def _seal(self) -> None:
        """Make this handle deeply immutable after orchestration's last attach.

        The advanced ``pops.codegen.compile_problem`` route returns an unsealed handle (its callers
        legitimately attach orchestration metadata); the PUBLIC artifact is sealed."""
        if self.program is not None and not getattr(
                self.program, "_compiled_detached", False):
            raise TypeError(
                "CompiledProblem cannot be sealed with a live/foreign time builder; public "
                "pops.compile requires a detached pops.Program"
            )
        from pops.codegen._artifact_freeze import seal_attributes

        seal_attributes(self)

    def _discard_authoring(self) -> None:
        """Drop the last formula-model builder after all compile metadata was captured."""
        if getattr(self, "_sealed", False):
            raise RuntimeError("cannot discard authoring from a sealed compiled program")
        self.model = None

    def __setattr__(self, name: Any, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "CompiledProblem is immutable after pops.compile (ADC-563): cannot set %r. The "
                "artifact is frozen to what was compiled; assemble a new Problem and recompile to "
                "change it." % (name,))
        object.__setattr__(self, name, value)

    def __fspath__(self) -> Any:
        return self.so_path

    # --- compiled-artifact metadata (Spec 5 sec.12.4, #48-49) ----------------
    @property
    def codegen_dir(self) -> Any:
        """Directory the compiled ``.so`` (and any generated source) lives in (sec.12.4, #48).

        The out-of-source cache directory the .so was written to (``os.path.dirname(so_path)``);
        the generated ``.cpp`` -- when ``compile_problem(debug=True)`` wrote one -- sits beside it.
        ``None`` only if the handle carries no ``so_path``."""
        import os
        return os.path.dirname(self.so_path) if self.so_path else None

    @property
    def problem_hash(self) -> Any:
        """Stable hash of the artifact inputs the ``.so`` was compiled for (sec.12.4, #48).

        The sha256 of the emitted C++ source plus the optional frozen AuthoringSnapshot identity --
        the cache identity (the WHAT). ``None`` for a handle built outside ``compile_problem``; use
        :attr:`program_hash` for the in-memory Program's IR hash, which is always available."""
        return self._problem_hash

    @property
    def cache_key(self) -> Any:
        """The (problem source | abi_key | backend/target) cache key of the ``.so`` (sec.12.4, #48).

        The sha256 the out-of-source build cache keys the artifact on; reproducing it requires the
        same program, headers, compiler and C++ standard. ``None`` for an externally built handle."""
        return self._cache_key

    @property
    def compile_command(self) -> Any:
        """The REDACTED compiler invocation that built the ``.so`` (sec.12.4, #49).

        A single command string with the ephemeral temp source replaced by the generated-source
        path and any secret-looking token masked (cf. ``_redact_compile_command``). ``None`` on a
        cache HIT (the .so was not rebuilt this call -- a documented absence, never a fabricated
        command) or for an externally constructed handle. Recompile with ``force=True`` to populate
        it."""
        return self._compile_command

    @property
    def generated_sources(self) -> Any:
        """The generated source files written for inspection (sec.12.4, #49).

        The ``.cpp`` files ``compile_problem(debug=True)`` (or ``POPS_KEEP_GENERATED``) persisted next
        to the ``.so`` (the default keeps the source only in a TemporaryDirectory, so this is empty
        unless one of those was set). A list (possibly empty), never ``None``."""
        return list(self._generated_sources)

    @property
    def codegen_env(self) -> Any:
        """The resolved codegen ``POPS_*`` environment snapshot that governed this compile (sec.12.4).

        A :class:`pops.codegen.env.CodegenEnv` recording the EFFECTIVE settings (env defaults already
        overridden by any explicit argument): log level, codegen dir, keep-generated, dump-IR /
        dump-CPP, cache dir and profile. Surfaced in :meth:`inspect` so the active
        environment state is never hidden (criterion #47).
        ``None`` for a handle built outside ``compile_problem``."""
        return self._codegen_env

    def module_hash(self) -> Any:
        """The compile-time hash of the operator-first Module (ADC-557 I5), or ``None``.

        The stable ``pops.model.Module.module_hash`` captured at compile time: the identity of the
        canonical compile IR the artifact was built from. bind compares the LIVE model's Module hash
        against it to DETECT a post-compile in-place mutation loudly (a compiled artifact is frozen at
        compile; the model object is held by reference, so a later mutation would otherwise silently
        change what bind lowers). ``None`` for a model with no backing Module."""
        return self._module_hash

    # --- operator introspection (Spec 2, S2-5): metadata read from the carried model,
    # no need to load or run the .so.
    def _intro_model(self) -> Any:
        if self.model is None:
            raise ValueError("this CompiledProblem carries no model; operator introspection "
                             "is unavailable")
        return self.model

    def list_operators(self) -> Any:
        """Names of the typed operators of the compiled module (registration order)."""
        if self._module_view.available:
            return list(self._module_view.operator_names())
        return self._intro_model().operator_registry().names()

    def list_state_spaces(self) -> Any:
        """Names of the compiled module's state spaces."""
        if self._module_view.available:
            return list(self._module_view.state_spaces)
        return self._intro_model().list_state_spaces()

    def list_field_spaces(self) -> Any:
        """Names of the compiled module's field spaces."""
        if self._module_view.available:
            return list(self._module_view.field_spaces)
        return self._intro_model().list_field_spaces()

    def operator_signature(self, name: Any) -> Any:
        """The pops.model.Signature of operator ``name`` in the compiled module."""
        if self._module_view.available:
            return self._module_view.signature(name)
        return self._intro_model().operator_registry().get(name).signature

    def operator_requirements(self, name: Any) -> dict:
        """The requirements dict of operator ``name``."""
        if self._module_view.available:
            return self._module_view.requirements(name)
        return dict(self._intro_model().operator_registry().get(name).requirements)

    def operator_capabilities(self, name: Any) -> dict:
        """The capabilities dict of operator ``name``."""
        if self._module_view.available:
            return self._module_view.capabilities(name)
        return dict(self._intro_model().operator_registry().get(name).capabilities)

    # --- bind-input + memory introspection (Spec 5 sec.12.2 / 12.3, #44-46) ---
    # These read the carried metadata (the lowered Program + the physical model); they do NOT
    # compile, bind, dlopen or read any runtime array.
    def arguments(self) -> Any:
        """The runtime inputs this artifact expects at ``pops.bind`` (Spec 5 sec.12.2, #44-45).

        Returns an :class:`pops.codegen.inspect_compiled.Arguments` listing -- WITHOUT any bind or
        runtime data -- the instances (state space / components / required), params (type / kind /
        required), aux (layout / required), outputs and the resolved runtime layout. Field
        discretizations and solver providers are compile-time plan evidence, never bind inputs.
        It is DISTINCT from :meth:`requirements`-style compile constraints: ``arguments`` lists
        only concrete values you must SUPPLY to bind. It allocates and reads nothing."""
        from pops.codegen.inspect_compiled import build_component_arguments
        return build_component_arguments(self)

    def manifest(self) -> Any:
        """The RICH self-describing manifest of this artifact (Spec 5 sec.13.12, #36).

        Returns a :class:`pops.external.CompiledArtifactManifest`: the ABI identity (``abi_key`` /
        ``required_headers_sig``), the model name, the blocks / variables / roles, the required
        aux, the const / runtime params, the ghost depth, the field outputs and the ``supports_*``
        capability flags (uniform / AMR / MPI / GPU known from the backend caps; stride /
        partial-IMEX mask / named fields honestly ``None`` until the C++ codegen emits them). It
        AGGREGATES the metadata this handle already carries (via :meth:`arguments` + the carried
        model + ``abi_key``); it binds, dlopens and runs nothing. The widening of the thin
        :class:`pops.external.CompiledManifest` (a brick-id / category list) into the full
        artifact self-description Spec 5 sec.13.12 requires."""
        from pops.external.artifact_manifest import build_compiled_manifest
        return build_compiled_manifest(self)

    def estimate_memory(self, mesh: Any, *, platform: Any = None, layout: Any = None) -> Any:
        """A FORMULA-based memory estimate on ``mesh`` (Spec 5 sec.12.3, #46).

        Returns an :class:`pops.codegen.inspect_compiled.MemoryEstimate`: the state /
        field-output / aux / RHS-scratch / state-scratch / scalar-field / Krylov / multigrid /
        AMR-patch / halo / MPI-buffer byte budgets, computed as a FORMULA over the mesh shape and
        the artifact's static cost (``Program.estimate``) + component counts. It NEVER allocates a
        ``MultiFab``; every assumption is in :attr:`MemoryEstimate.assumptions` and the estimate is
        CONSERVATIVE. @p mesh an exact ``pops.mesh.CartesianGrid`` over a framed Rectangle; @p
        platform an optional hint (``"mpi"`` adds the halo-exchange buffer); @p layout an optional
        ``pops.layouts.AMR`` / ``Uniform`` for an AMR hierarchy estimate (conservative;
        full-refinement worst case)."""
        from pops.codegen.inspect_compiled import build_memory_estimate
        return build_memory_estimate(
            self, mesh, platform=platform, layout=layout or self.layout)

    def scratch_plan(self) -> Any:
        """The scratch-buffer liveness plan of this artifact's time Program (Spec 5 sec.13.11.3, #38).

        Returns a :class:`pops.codegen.scratch_plan.ScratchPlan`: the per-category scratch counts
        (state / rhs / scalar-field), the PROVABLY-reusable buffers (scratch nodes whose SSA live
        ranges are disjoint, so the codegen may share one buffer), the REJECTED reuse (with the
        reason -- a still-live occupant or an aux/field barrier) and the PERSISTENT Krylov / multigrid
        solver buffers. Computed by a liveness analysis over the carried Program IR
        (``Program.scratch_liveness`` / ``buffer_reuse_report``); the step-body reuse is EXACT, the
        persistent solver counts are conservative and labelled so. It NEVER binds, dlopens or
        allocates -- it is inspectable BEFORE ``pops.bind``. Raises a clear error if this handle
        carries no Program."""
        from pops.codegen.scratch_plan import build_scratch_plan
        return build_scratch_plan(self._require_program("scratch_plan"), model=self.model)

    # --- inspection completeness (Spec 5 sec.12.1, criterion #15) -------------
    # The print(compiled) reports + the codegen/IR dumps. All INERT metadata-reading (they aggregate
    # the carried Program + model + compile artifacts), EXCEPT dump_cpp which REUSES the existing
    # emit_cpp_program codegen. None binds, dlopens, allocates or runs.
    def inspect(self) -> Any:
        """A printable :class:`pops.codegen.inspect_report.CompiledReport` of this artifact (sec.12.1).

        The ``print(compiled.inspect())`` summary: name, backend, platform, layout, blocks (+ state /
        components), fields (+ solver), program (+ commits), the REQUIRED runtime inputs (states /
        params / aux from :meth:`arguments`), the on-disk artifacts (so_path / abi_key / cache_key)
        and the bind-pending status line. It AGGREGATES the metadata this handle already carries (no
        compile / bind / runtime read); :meth:`~CompiledReport.to_dict` serialises it."""
        from pops.codegen.inspect_report import build_compiled_report
        return build_compiled_report(self)

    def requirements(self) -> Any:
        """The COMPILE-TIME constraints of this artifact (sec.12.1), DISTINCT from :meth:`arguments`.

        Returns a :class:`pops.codegen.inspect_report.RequirementsReport`: the model capabilities the
        lowered route relies on (``wave_speeds`` / ``hllc_star_state`` / ``roe_dissipation``, read
        from the carried model's emitted flags), the required descriptors (the spatial scheme is a
        BIND input, reported as such), and the layout / backend / ABI constraints. A piece genuinely
        unknowable from today's metadata is stated honestly in
        :attr:`~RequirementsReport.unknown`, never fabricated. :meth:`arguments` lists what you SUPPLY
        at bind; ``requirements`` lists what the compiled route NEEDS from the model + toolchain."""
        from pops.codegen.inspect_report import build_requirements
        return build_requirements(self)

    def __str__(self) -> str:
        """A short, deterministic, array-free summary (Spec 5 sec.12.1, #40-41).

        Prints the program name, a short program-source/IR hash, a short ABI key and the validated
        library count -- never the ``.so`` contents and never a ``<...object at 0x...>`` repr. The
        hash makes two artifacts of the same program look identical run to run (deterministic)."""
        hash_value = self._problem_hash or self.program_hash
        short_hash = hash_value[:12] if isinstance(hash_value, str) and hash_value else "none"
        short_abi = (self.abi_key or "")[:12] or "none"
        return ("CompiledProblem(name=%s, hash=%s, backend=%s, libraries=%d)"
                % (self.program_name or "problem", short_hash, short_abi, len(self.libraries)))

    def __repr__(self) -> str:
        return "<CompiledProblem %r -> %s>" % (self.program_name, self.so_path)


# CompiledModel (the per-block physics-.so handle) is split into ``_loader_model`` for the
# 500-line cap and re-exported here so ``from pops.codegen.loader import CompiledModel`` is
# unchanged.
from pops.codegen._loader_model import CompiledModel  # noqa: E402,F401  (re-exported at the historical path)
