"""Compiled artifact wrappers; no physics/dsl imports at module scope."""


class CompiledProblem:
    """Result of ``pops.compile_problem(...)``; install with ``sim.install(compiled, ...)``."""

    def __init__(self, so_path, program, model, abi_key, cxx, std, libraries=None,
                 problem_hash=None, cache_key=None, compile_command=None, generated_sources=None,
                 codegen_env=None, module_hash=None, source_hash=None, problem_identity=None):
        self.so_path = so_path
        self.program = program          # the pops.time.Program that was lowered
        self.model = model              # the canonical pops.model.Module that was lowered
        self.program_name = getattr(program, "name", None)
        self.program_hash = program._ir_hash() if hasattr(program, "_ir_hash") else None
        self.abi_key = abi_key          # cache key: header signature | compiler | C++ standard
        self.cxx = cxx
        self.std = std
        # Validated brick libraries from libraries=[...], ABI-checked before reaching this handle.
        self.libraries = list(libraries) if libraries else []
        # Structured semantic identity plus provenance/cache metadata; None means "not recorded".
        self._problem_hash = problem_hash
        self._module_hash = module_hash
        self._source_hash = source_hash
        self._problem_identity = dict(problem_identity or {})
        self._cache_key = cache_key
        self._compile_command = compile_command
        self._generated_sources = list(generated_sources) if generated_sources else []
        # Active codegen POPS_* environment snapshot; None for externally constructed handles.
        self._codegen_env = codegen_env

    def __fspath__(self):
        return self.so_path

    # --- compiled-artifact metadata (Spec 5 sec.12.4, #48-49) ----------------
    @property
    def codegen_dir(self):
        """Directory containing the compiled ``.so`` and any persisted generated source."""
        import os
        return os.path.dirname(self.so_path) if self.so_path else None

    @property
    def problem_hash(self):
        """Stable semantic hash of Module IR, Program IR, descriptors, layout, backend and libraries."""
        return self._problem_hash

    @property
    def module_hash(self):
        """Stable hash of the carried Module IR, when available."""
        if self._module_hash is not None:
            return self._module_hash
        if self.model is not None and hasattr(self.model, "module_hash"):
            return self.model.module_hash()
        return None

    @property
    def source_hash(self):
        """SHA-256 identity of generated C++ plus codegen source-identity version."""
        return self._source_hash

    @property
    def problem_identity(self):
        """Plain dict identity record: semantic hash inputs plus provenance/source cache guards."""
        return dict(self._problem_identity)

    @property
    def cache_key(self):
        """Structured problem identity plus ABI/backend/target cache key of the ``.so``."""
        return self._cache_key

    @property
    def compile_command(self):
        """Redacted compiler invocation, or ``None`` on cache hit / externally built handles."""
        return self._compile_command

    @property
    def generated_sources(self):
        """Generated source files persisted by debug=True or POPS_KEEP_GENERATED."""
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
        capability matrix) and scopes it to the descriptor categories this compiled problem can
        select at bind -- the Riemann / reconstruction / limiter / projection bricks and the mesh
        layouts (the solver / field catalogs are bind inputs, kept too). Returns the same printable
        :class:`pops.CapabilityMatrix`. PURE: it imports only the inert authoring catalogs, never
        ``_pops`` (cf. :func:`pops.inspect_capabilities`)."""
        from pops._capabilities import inspect_capabilities, CapabilityMatrix
        matrix = inspect_capabilities()
        scoped = [e for e in matrix if e.category in self._CAPABILITY_CATEGORIES]
        return CapabilityMatrix(scoped)

    # Descriptor categories a compiled problem selects from at bind (spatial brick + layout + field
    # solver); the capability scope of inspect_capabilities().
    _CAPABILITY_CATEGORIES = ("riemann", "reconstruction", "limiter", "projection", "layout",
                              "solver", "field")

    def dump_ir(self, path=None):
        """Write the serialized Program IR (JSON) -- the SAME serialization ``_ir_hash`` digests.

        EXPOSES the lowered ``pops.time.Program``'s ``_serialize()`` blob (its nodes, commits, block
        order, optional dt bound) as indented, sort-keyed JSON -- byte-stable run to run. Writes to
        @p path if given (returns the path), else returns the JSON string. Raises a clear error if
        this handle carries no Program."""
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
        ``P.state``, the order ``install_problem`` binds them), each with the IR id of its committed
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
        program, a single-level ``System`` concept today -- ``AmrSystem`` has no ``install_problem``
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


class CompiledModel:
    """Result of ``m.compile(...)``: packages the produced ``.so`` + EVERYTHING
    needed to wire it correctly (dispatch adder, ABI diagnostic,
    reproducibility). Replaces the historical pair (str so_path,
    adder_for(backend)) with a single object.

    The metadata is NOT re-read from the ``.so``: Python already holds
        names/roles/gamma/n_aux/params carried by the compiled model metadata;
    CompiledModel just exposes them for dispatch (add_equation) and
    diagnostics. cf. DSL_MODEL_DESIGN.md section 3.
    """

    def __init__(self, so_path, backend, adder, cons_names, cons_roles, prim_names, n_vars,
                 gamma, n_aux, params, caps, abi_key, model_hash, cxx, std, target="system",
                 hllc=False, roe=False, aux_extra_names=None, wave_speeds=False,
                 elliptic_field_names=None):
        self.has_hllc = bool(hllc)   # HLLC capability emitted (enable_hllc): hllc available beyond 4-var Euler
        self.has_roe = bool(roe)     # ROE hook emitted (enable_roe roles OR m.roe_dissipation provided): roe available beyond 4-var Euler
        self.has_wave_speeds = bool(wave_speeds)  # wave_speeds emitted (explicit pair OR 'p'): hll available
        self.so_path = so_path
        self.backend = backend       # "prototype" | "aot" | "production"
        self.target = target         # "system" | "amr_system": targeted facade (native AMR loader if amr_system)
        self.adder = adder           # method name (Amr)System: add_dynamic_block / add_compiled_block / add_native_block
        self.cons_names = list(cons_names)
        self.cons_roles = list(cons_roles)
        self.prim_names = list(prim_names)
        self.n_vars = int(n_vars)
        self.gamma = gamma           # None = historical default 1.4 on the System side
        self.n_aux = int(n_aux)
        # Names of the NAMED aux fields (aux_field, ADC-70), ORDERED: component index = position
        # AUX_NAMED_BASE + k. The System.add_equation facade builds the name -> component table per
        # block from it, consumed by System.set_aux_field / aux_field. Empty for a model without a named field.
        self.aux_extra_names = list(aux_extra_names) if aux_extra_names else []
        # Names of the model's NAMED elliptic fields (m.elliptic_field, ADC-419 / ADC-428): each is a
        # second-or-further elliptic solve the native loader wires via register_elliptic_field +
        # set_block_elliptic_field after the block is installed. The install seam consults this set to
        # decide whether a bind(solvers={field: ...}) selection names a DECLARED field (route it) or a
        # typo (reject, naming the declared set). Empty for a model with only the default Poisson field.
        self.elliptic_field_names = list(elliptic_field_names) if elliptic_field_names else []
        self.params = dict(params)   # {name: Param}
        self.caps = dict(caps)       # {cpu/mpi/amr/gpu: bool}
        self.abi_key = abi_key       # ABI key mirroring pops_header_signature + compiler/std
        self.model_hash = model_hash  # stable hash formulas+roles+n_aux+params
        self.cxx = cxx
        self.std = std

    @property
    def runtime_param_names(self):
        """Names of the model's RUNTIME parameters (kind='runtime'), SORTED: this is the ORDER of
        the indices on the C++ side (RuntimeParams) AND the order expected by
        System.set_block_params(name, values) (P7-b). Empty if the model has only const params."""
        return sorted(k for k, p in self.params.items() if getattr(p, "kind", "const") == "runtime")

    def runtime_param_values(self):
        """DECLARATION values of the runtime params, parallel to runtime_param_names (default as
        long as no set_block_params has been called)."""
        return [self.params[k].value for k in self.runtime_param_names]

    def check_runtime(self, n=16, state=None, raise_on_error=True, rtol=1e-8, atol=1e-10):
        """RUNTIME re-verification of a CompiledModel ALONE (audit balance, GENERICITY pt 9):
        without the original dsl.Model, the FORMULAS are no longer re-verifiable (symbolic
        check_model), but the .so itself is -- we install it in an EPHEMERAL System (n x n
        periodic, neutral Poisson, minmod+rusanov) and delegate to System.check_model (finite
        state, residual -div F + S finite, positivity by roles, round-trip of THE MODEL
        conversions).

        @p state: dict {conservative variable name: ndarray (n, n)} to control the tested state.
        None -> SMOKE state by ROLES (Density = 1 + gaussian bump, Momentum* = 0,
        Energy = 2.5, other components = 0.5) -- enough to exercise flux/source/conversions;
        provide state= for a precise physical regime. @return the dict from System.check_model.
        """
        import numpy as np  # lazy: only needed at check_runtime call time
        if getattr(self, "target", "system") != "system":
            raise ValueError(
                "CompiledModel.check_runtime: only target='system' is re-verifiable in an "
                "ephemeral System; a target='amr_system' loader is checked installed in its "
                "AmrSystem (AMR test invariants), not in isolation.")
        from pops import System, FiniteVolume, Explicit  # lazy: avoids a top-level runtime import
        from pops.numerics.reconstruction.limiters import Minmod
        from pops.numerics.riemann import Rusanov
        sim = System(n=int(n), L=1.0, periodic=True)
        sim._set_poisson()
        sim._add_equation("check", model=self,
                          spatial=FiniteVolume(limiter=Minmod(), riemann=Rusanov()),
                          time=Explicit())
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        bump = 1.0 + 0.3 * np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        comps = []
        for name, role in zip(self.cons_names, self.cons_roles, strict=True):
            if state is not None and name in state:
                comps.append(np.asarray(state[name], dtype=float).reshape(n, n))
            elif role == "Density":
                comps.append(bump)
            elif role in ("MomentumX", "MomentumY"):
                comps.append(np.zeros((n, n)))
            elif role == "Energy":
                comps.append(2.5 + 0.0 * bump)
            else:
                comps.append(0.5 + 0.0 * bump)
        sim._s.set_state("check", np.stack(comps).ravel())
        return sim.check_model("check", raise_on_error=raise_on_error, rtol=rtol, atol=atol)

    def inspect_amr(self, layout=None):
        """STATIC AMR report on this compiled MODEL (Spec 5 sec.8.12 / sec.8.4).

        A ``CompiledModel`` is a per-block physics ``.so``; ``target='amr_system'`` means it is
        loadable on the native AMR hierarchy (``add_equation``), but the model carries NO layout
        descriptor (the levels / ratio / regrid policy belong to the ``AmrSystem`` it is installed
        in, not to the model). So this delegates to the top-level :func:`pops.inspect_amr` on an
        EXPLICIT ``layout`` argument and returns the native AMR envelope report for ``layout=None``;
        for a target='system' model the report still describes the native AMR envelope (a model is
        AMR-installable only with target='amr_system'). It never fabricates a hierarchy. @p layout
        an optional AMR / Uniform layout descriptor (default: the native envelope).
        """
        from pops import inspect_amr
        return inspect_amr(layout)

    def __repr__(self):
        return ("CompiledModel(backend=%r, target=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "adder=%r, runtime_params=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.target, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.adder, self.runtime_param_names, self.abi_key or "", self.model_hash or ""))
