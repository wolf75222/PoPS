"""pops.codegen.loader : result/wrapper classes for compiled .so artefacts.

``CompiledModel`` packages a model ``.so`` with the metadata needed to wire
it (adder, names, roles, gamma, n_aux, params, caps, abi_key, model_hash).

``CompiledProblem`` packages a program ``.so`` (a compiled time
``pops.time.Program``) plus the metadata to install + reproduce it.

Neither class imports ``pops.dsl`` or ``pops.physics`` at module level.
"""


class CompiledProblem:
    """An advanced, INTERNAL compiled handle: a generated ``problem.so`` (a
    compiled time Program) plus the metadata to install + reproduce it. It is
    produced by the low-level ``pops.codegen.compile_problem(...)`` driver and,
    for the public path, wrapped by ``pops.compile(...)``; wire a runnable
    simulation from it with ``pops.bind(compiled, ...)`` (ADC-523). The concrete
    class stays off the top-level surface -- annotate against the inspectable
    ``pops.CompiledArtifact`` protocol instead. The bound Program drives
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

    def __init__(self, so_path, program, model, abi_key, cxx, std, libraries=None,
                 problem_hash=None, cache_key=None, compile_command=None, generated_sources=None,
                 codegen_env=None, module_manifest=None):
        self.so_path = so_path
        self.program = program          # the pops.time.Program that was lowered
        self.model = model              # the physical model (optional; added as a block in the MVP)
        # The self-describing Module manifest (ADC-585): the operator-first central representation of
        # the resolved model (spaces / params / aux / typed operators / native routes), superseding
        # the legacy flat ModelSpec. None when the model is a bare dsl.Model with no backing Module,
        # or when the handle is built outside compile_problem. Its abi_requirements abi_key slot is
        # bound below from this handle's abi_key (a compile-time fact the manifest builder leaves
        # open).
        self.module_manifest = module_manifest
        if module_manifest is not None and abi_key is not None:
            module_manifest.abi_requirements["abi_key"] = abi_key
        self.program_name = getattr(program, "name", None)
        self.program_hash = program._ir_hash() if hasattr(program, "_ir_hash") else None
        self.abi_key = abi_key          # cache key: header signature | compiler | C++ standard
        self.cxx = cxx
        self.std = std
        # Validated brick libraries (Spec 3 section 21, ADC-464): the LibraryManifests read +
        # ABI-checked from libraries=[...]. Empty when none were passed. Their bricks (and their
        # generated symbols) are exposed to the problem; a compiled library .so was already
        # dlopen'd (and ABI-guarded) by read_library_manifest.
        self.libraries = list(libraries) if libraries else []
        # Compiled-artifact metadata (Spec 5 sec.12.4, #48-49): set by compile_problem. The
        # problem hash is the program-SOURCE hash (the WHAT the .so was built from -- distinct from
        # program_hash, the IR hash of the in-memory Program); cache_key is the (problem_hash|abi)
        # identity the out-of-source cache file name carries; compile_command is the redacted
        # compiler invocation; generated_sources are the .cpp files written for inspection (debug=).
        # None on a route that does not record a value (e.g. an externally constructed handle) -- a
        # documented absence, not a fabricated value (cf. the property accessors below).
        self._problem_hash = problem_hash
        self._cache_key = cache_key
        self._compile_command = compile_command
        self._generated_sources = list(generated_sources) if generated_sources else []
        # Active codegen POPS_* environment snapshot (Spec 5 sec.12.4, #47-48): the resolved
        # CodegenEnv that governed this compile (log level, codegen dir, keep-generated, dump flags,
        # cache dir, profile, autotune, and the UNSAFE jit-backdoor gate). Recorded so the env state
        # is inspectable in inspect(); None for a handle built outside compile_problem (no env was
        # resolved -- a documented absence, not a fabricated default).
        self._codegen_env = codegen_env

    def __fspath__(self):
        return self.so_path

    # --- compiled-artifact metadata (Spec 5 sec.12.4, #48-49) ----------------
    @property
    def codegen_dir(self):
        """Directory the compiled ``.so`` (and any generated source) lives in (sec.12.4, #48).

        The out-of-source cache directory the .so was written to (``os.path.dirname(so_path)``);
        the generated ``.cpp`` -- when ``compile_problem(debug=True)`` wrote one -- sits beside it.
        ``None`` only if the handle carries no ``so_path``."""
        import os
        return os.path.dirname(self.so_path) if self.so_path else None

    @property
    def problem_hash(self):
        """Stable hash of the program SOURCE the ``.so`` was compiled from (sec.12.4, #48).

        The sha256 of the emitted C++ program text -- the cache identity (the WHAT). ``None`` for a
        handle built outside ``compile_problem`` (it records no source hash); use
        :attr:`program_hash` for the in-memory Program's IR hash, which is always available."""
        return self._problem_hash

    @property
    def cache_key(self):
        """The (problem source | abi_key | backend/target) cache key of the ``.so`` (sec.12.4, #48).

        The sha256 the out-of-source build cache keys the artifact on; reproducing it requires the
        same program, headers, compiler and C++ standard. ``None`` for an externally built handle."""
        return self._cache_key

    @property
    def compile_command(self):
        """The REDACTED compiler invocation that built the ``.so`` (sec.12.4, #49).

        A single command string with the ephemeral temp source replaced by the generated-source
        path and any secret-looking token masked (cf. ``_redact_compile_command``). ``None`` on a
        cache HIT (the .so was not rebuilt this call -- a documented absence, never a fabricated
        command) or for an externally constructed handle. Recompile with ``force=True`` to populate
        it."""
        return self._compile_command

    @property
    def generated_sources(self):
        """The generated source files written for inspection (sec.12.4, #49).

        The ``.cpp`` files ``compile_problem(debug=True)`` (or ``POPS_KEEP_GENERATED``) persisted next
        to the ``.so`` (the default keeps the source only in a TemporaryDirectory, so this is empty
        unless one of those was set). A list (possibly empty), never ``None``."""
        return list(self._generated_sources)

    @property
    def codegen_env(self):
        """The resolved codegen ``POPS_*`` environment snapshot that governed this compile (sec.12.4).

        A :class:`pops.codegen.env.CodegenEnv` recording the EFFECTIVE settings (env defaults already
        overridden by any explicit argument): log level, codegen dir, keep-generated, dump-IR /
        dump-CPP, cache dir, profile, autotune level, and the UNSAFE :attr:`CodegenEnv.jit_backdoor`
        gate. Surfaced in :meth:`inspect` so the active env state is never hidden (criterion #47).
        ``None`` for a handle built outside ``compile_problem``."""
        return self._codegen_env

    def runtime_param_routes(self):
        """``(per_block, defaults)`` routing the Program's RUNTIME parameters to the per-PROGRAM-block
        ``set_program_params`` vectors (ADC-510): per_block maps a program block index to its param names
        in within-block index order (matching the ``.so`` metadata + the lowered read), defaults a name to
        its declaration value. Built via the SAME ``program_param_entries`` the codegen emits. No bind."""
        from pops.codegen.program_emit_params import program_param_routes
        return program_param_routes(self.program, self.model)

    # --- operator introspection (Spec 2, S2-5): metadata read from the carried model,
    # no need to load or run the .so.
    def _intro_model(self):
        if self.model is None:
            raise ValueError("this CompiledProblem carries no model; operator introspection "
                             "is unavailable")
        return self.model

    def list_operators(self):
        """Names of the typed operators of the compiled module (registration order)."""
        return self._intro_model().operator_registry().names()

    def list_state_spaces(self):
        """Names of the compiled module's state spaces."""
        return self._intro_model().list_state_spaces()

    def list_field_spaces(self):
        """Names of the compiled module's field spaces."""
        return self._intro_model().list_field_spaces()

    def operator_signature(self, name):
        """The pops.model.Signature of operator ``name`` in the compiled module."""
        return self._intro_model().operator_registry().get(name).signature

    def operator_requirements(self, name):
        """The requirements dict of operator ``name``."""
        return dict(self._intro_model().operator_registry().get(name).requirements)

    def operator_capabilities(self, name):
        """The capabilities dict of operator ``name``."""
        return dict(self._intro_model().operator_registry().get(name).capabilities)

    # --- bind-input + memory introspection (Spec 5 sec.12.2 / 12.3, #44-46) ---
    # These read the carried metadata (the lowered Program + the physical model); they do NOT
    # compile, bind, dlopen or read any runtime array.
    def arguments(self):
        """The runtime inputs this artifact expects at ``System.install`` (Spec 5 sec.12.2, #44-45).

        Returns an :class:`pops.codegen.inspect_compiled.Arguments` listing -- WITHOUT any bind or
        runtime data -- the instances (state space / components / required), params (type / kind /
        required), aux (layout / required), solvers (problem / solver), outputs and the runtime
        layout the artifact expects. Sourced from the carried Program (the blocks it commits, the
        field solves it performs) and the physical model (its state / params / aux). It is DISTINCT
        from :meth:`requirements`-style compile constraints: ``arguments`` lists what you must SUPPLY
        to bind. It allocates and reads nothing."""
        from pops.codegen.inspect_compiled import build_arguments
        return build_arguments(self)

    def manifest(self):
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

    def estimate_memory(self, mesh, *, platform=None, layout=None):
        """A FORMULA-based memory estimate on ``mesh`` (Spec 5 sec.12.3, #46).

        Returns an :class:`pops.codegen.inspect_compiled.MemoryEstimate`: the state /
        field-output / aux / RHS-scratch / state-scratch / scalar-field / Krylov / multigrid /
        AMR-patch / halo / MPI-buffer byte budgets, computed as a FORMULA over the mesh shape and
        the artifact's static cost (``Program.estimate``) + component counts. It NEVER allocates a
        ``MultiFab``; every assumption is in :attr:`MemoryEstimate.assumptions` and the estimate is
        CONSERVATIVE. @p mesh an ``pops.mesh.CartesianMesh`` (or an int / 2-tuple of extents); @p
        platform an optional hint (``"mpi"`` adds the halo-exchange buffer); @p layout an optional
        ``pops.mesh.layouts.AMR`` / ``Uniform`` for an AMR hierarchy estimate (conservative;
        full-refinement worst case)."""
        from pops.codegen.inspect_compiled import build_memory_estimate
        return build_memory_estimate(self, mesh, platform=platform, layout=layout)

    def scratch_plan(self):
        """The scratch-buffer liveness plan of this artifact's time Program (Spec 5 sec.13.11.3, #38).

        Returns a :class:`pops.codegen.scratch_plan.ScratchPlan`: the per-category scratch counts
        (state / rhs / scalar-field), the PROVABLY-reusable buffers (scratch nodes whose SSA live
        ranges are disjoint, so the codegen may share one buffer), the REJECTED reuse (with the
        reason -- a still-live occupant or an aux/field barrier) and the PERSISTENT Krylov / multigrid
        solver buffers. Computed by a liveness analysis over the carried Program IR
        (``Program.scratch_liveness`` / ``buffer_reuse_report``); the step-body reuse is EXACT, the
        persistent solver counts are conservative and labelled so. It NEVER binds, dlopens or
        allocates -- it is inspectable BEFORE ``System.install``. Raises a clear error if this handle
        carries no Program."""
        from pops.codegen.scratch_plan import build_scratch_plan
        return build_scratch_plan(self._require_program("scratch_plan"), model=self.model)

    # --- inspection completeness (Spec 5 sec.12.1, criterion #15) -------------
    # The print(compiled) reports + the codegen/IR dumps. All INERT metadata-reading (they aggregate
    # the carried Program + model + compile artifacts), EXCEPT dump_cpp which REUSES the existing
    # emit_cpp_program codegen. None binds, dlopens, allocates or runs.
    def inspect(self):
        """A printable :class:`pops.codegen.inspect_report.CompiledReport` of this artifact (sec.12.1).

        The ``print(compiled.inspect())`` summary: name, backend, platform, layout, blocks (+ state /
        components), fields (+ solver), program (+ commits), the REQUIRED runtime inputs (states /
        params / aux from :meth:`arguments`), the on-disk artifacts (so_path / abi_key / cache_key)
        and the bind-pending status line. It AGGREGATES the metadata this handle already carries (no
        compile / bind / runtime read); :meth:`~CompiledReport.to_dict` serialises it."""
        from pops.codegen.inspect_report import build_compiled_report
        return build_compiled_report(self)

    def requirements(self):
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

    def inspect_capabilities(self):
        """The descriptor capability rows relevant to THIS compiled artifact (sec.12.1).

        Delegates to the top-level :func:`pops.inspect_capabilities` machinery (the descriptor-sourced
        capability matrix) and SCOPES it to the descriptor categories a compiled time Program can
        select at bind -- the Riemann / reconstruction / limiter / projection bricks and the mesh
        layouts (the solver / field catalogs are bind inputs, kept too). Returns the same printable
        :class:`pops.CapabilityMatrix`. PURE: it imports only the inert authoring catalogs, never
        ``_pops`` (cf. :func:`pops.inspect_capabilities`)."""
        from pops._capabilities import inspect_capabilities, CapabilityMatrix
        matrix = inspect_capabilities()
        scoped = [e for e in matrix if e.category in self._CAPABILITY_CATEGORIES]
        return CapabilityMatrix(scoped)

    def capability_matrix(self):
        """The ADC-549 native route matrix for this compiled artifact.

        Unlike :meth:`inspect_capabilities`, which scopes the descriptor catalog, this reports the
        route support columns ADC-549 requires: feature, layout, backend, platform, MPI, GPU,
        status, limitation and error_message. It is metadata-only: it builds the rich manifest from
        the carried Program/model and never dlopens or binds the ``.so``.
        """
        from pops._capabilities import native_capability_matrix
        manifest = self.manifest()
        layout = "system"
        try:
            layout = self.arguments().layout_runtime.get("layout", "system")
        except Exception:
            pass
        return native_capability_matrix(
            owner=self.program_name or "compiled-problem", layout=layout,
            flags=manifest.supports(), source="manifest")

    # Descriptor categories a compiled time Program selects from at bind (a spatial brick + a layout
    # + a field solver); the capability scope of inspect_capabilities().
    _CAPABILITY_CATEGORIES = ("riemann", "reconstruction", "limiter", "projection", "layout",
                              "solver", "field")

    def dump_ir(self, path=None):
        """Write the serialized Program IR (JSON) -- the SAME serialization ``_ir_hash`` digests.

        EXPOSES the existing codegen: the lowered ``pops.time.Program``'s ``_serialize()`` blob (its
        nodes, commits, block order, optional dt bound) as indented, sort-keyed JSON -- byte-stable
        run to run, the WHAT the ``.so`` was built from. Writes to @p path if given (returns the
        path), else returns the JSON string. Raises a clear error if this handle carries no Program."""
        import json
        program = self._require_program("dump_ir")
        blob = json.dumps(program._serialize(), indent=2, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(blob)
            return path
        return blob

    def dump_cpp(self, target):
        """Write the generated C++ source of the problem ``.so`` (REUSES the existing emit).

        Calls the EXISTING ``Program.emit_cpp_program(model=...)`` codegen (the same source
        ``compile_problem`` compiles) and writes it. @p target is a directory (the source is written
        as ``<program_name>.cpp`` inside it) OR a path ending in ``.cpp`` (written verbatim); the
        parent directory must exist. Returns the written file path. The carried model is passed so a
        Program whose IR names a model source / linear kernel lowers (without it such a Program raises
        the SAME NotImplementedError the compile path raises -- it is not faked). Raises a clear error
        if this handle carries no Program."""
        import os
        program = self._require_program("dump_cpp")
        src = program.emit_cpp_program(model=self.model)
        name = self.program_name or "problem"
        if str(target).endswith(".cpp"):
            out_path = str(target)
            parent = os.path.dirname(out_path) or "."
        else:
            parent = str(target)
            out_path = os.path.join(parent, "%s.cpp" % name)
        if not os.path.isdir(parent):
            raise NotADirectoryError(
                "dump_cpp: the target directory %r does not exist; create it first "
                "(dump_cpp does not allocate or create directories)." % (parent,))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(src)
        return out_path

    def dump_schedule(self, path=None):
        """Write the schedule / commit order of the Program (the block advance order).

        EXPOSES the lowered schedule WITHOUT running it: the committed blocks in the runtime block
        index order (``_block_indices``: the order the Program first declares each block via
        ``P.state``, the order ``install_program`` binds them), each with the IR id of its committed
        State value. A plain, deterministic text listing. Writes to @p path if given (returns the
        path), else returns the string. Raises a clear error if this handle carries no Program."""
        program = self._require_program("dump_schedule")
        commits = program.commits()
        order = program._block_indices() if hasattr(program, "_block_indices") else {}
        ordered = sorted(commits, key=lambda b: order.get(b, len(order)))
        lines = ["schedule for Program %r (block commit order):" % (self.program_name or "problem")]
        for block in ordered:
            state = commits[block]
            lines.append("  %2d  commit %-14s <- %s"
                         % (order.get(block, -1), block, getattr(state, "name", "?")))
        if not ordered:
            lines.append("  (no committed block)")
        text = "\n".join(lines)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def _require_program(self, who):
        """Return the carried Program, or raise a clear error naming what is missing (never fake)."""
        program = self.program
        if program is None:
            raise ValueError(
                "%s: this CompiledProblem carries no Program (the lowered pops.time.Program is "
                "unavailable on this handle), so the IR / C++ / schedule cannot be dumped." % who)
        return program
    def inspect_amr(self, layout=None):
        """STATIC AMR report on this compiled artifact (Spec 5 sec.8.12 / sec.8.4).

        A compiled time ``Program`` carries NO AMR layout descriptor (it lowers a whole-system time
        program, a single-level ``System`` concept today -- ``AmrSystem`` has no ``install_program``
        seam). So this delegates to the top-level :func:`pops.inspect_amr` on an EXPLICIT ``layout``
        argument (an ``pops.mesh.layouts.AMR`` / ``Uniform`` descriptor), and with ``layout=None``
        returns the native AMR envelope report -- never a fabricated hierarchy the artifact does not
        carry. @p layout an optional AMR / Uniform layout descriptor (default: the native envelope).
        """
        from pops import inspect_amr
        return inspect_amr(layout)

    def __str__(self):
        """A short, deterministic, array-free summary (Spec 5 sec.12.1, #40-41).

        Prints the program name, a short program-source/IR hash, a short ABI key and the validated
        library count -- never the ``.so`` contents and never a ``<...object at 0x...>`` repr. The
        hash makes two artifacts of the same program look identical run to run (deterministic)."""
        short_hash = (self._problem_hash or self.program_hash or "")[:12] or "none"
        short_abi = (self.abi_key or "")[:12] or "none"
        return ("CompiledProblem(name=%s, hash=%s, backend=%s, libraries=%d)"
                % (self.program_name or "problem", short_hash, short_abi, len(self.libraries)))

    def __repr__(self):
        return "<CompiledProblem %r -> %s>" % (self.program_name, self.so_path)


# CompiledModel (the per-block physics-.so handle) is split into ``_loader_model`` for the
# 500-line cap and re-exported here so ``from pops.codegen.loader import CompiledModel`` is
# unchanged.
from pops.codegen._loader_model import CompiledModel  # noqa: E402,F401  (re-exported at the historical path)
