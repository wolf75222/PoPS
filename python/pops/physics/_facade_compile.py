"""Internal compile mixin for the PDE-model facade (:class:`pops.physics.facade.Model`).

Splits ``Model._model_hash`` + the internal ``Model._compile_for_runtime`` seam out of
``facade.py`` so neither file exceeds the Spec-4 500-line bound. The mixin operates on
``self._m`` (the private :class:`HyperbolicModel`) and ``self.params``; the build engine is
pulled in LAZILY inside ``_compile_for_runtime`` (toolchain/cache/abi/loader names), so
importing ``pops.physics`` never loads it (Spec-4 import-graph rule).
"""
from .aux import aux_total_n_aux, roles_for
from .model import HyperbolicModel


class _FacadeCompileMixin:
    """Model-hash + internal runtime-compile half of the PDE facade (lazy codegen)."""

    def _model_hash(self):
        """Stable hash of the model: formulas (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + NAMED params (m.params). Used to identify/reuse an already-compiled .so (cache key)
        and to trace the run. Delegates to the shared computation HyperbolicModel._model_hash, passing it
        the Param of the facade (otherwise two models differing only by a param would have the same hash)."""
        return self._m._model_hash(params=self.params)

    def _compile_for_runtime(self, so_path=None, include=None, backend=None, target="system",
                             name=None, cxx=None, std=None, require_metadata=False,
                             hoist_reciprocals=False):
        """Internal native-block compilation seam.

        Delegates the GENERATION + compilation to HyperbolicModel.compile (engines unchanged:
        compile_so / compile_aot / compile_native), then packages the .so with the already-known
        metadata (no re-reading of the .so).

        INTERNAL codegen engine (Spec 5 sec.11). The documented PUBLIC compile front door is
        ``pops.compile_problem(model=..., time=..., backend=pops.codegen.Production())`` (with the authoring facade
        ``pops.physics.Model.lower()`` producing the ``pops.model.Module`` it lowers); ``pops.compile``
        and ``pops.compile_library`` take a TYPED backend descriptor. This engine ``backend`` argument
        is the internal lowering vocabulary: it accepts a typed descriptor (lowered via lower_backend)
        AND the legacy token, kept for the codegen path / internal callers / tests.

        - ``backend``: a typed ``pops.codegen.Production()`` / ``AOT()`` / ``JIT()`` descriptor, or the
          internal token "prototype" | "aot" | "production" (cf. HyperbolicModel.compile).
        - ``target``: "system" (default) | "amr_system" (DSL Phase D). "amr_system" requires
          backend=pops.codegen.Production() (the native loader inlines add_compiled_model(AmrSystem&), the only
          .so AMR path; cf. compile_or_jit) -> to be wired via AmrSystem.add_equation. Another backend
          with target="amr_system" raises ValueError (no AMR path outside native).

        NO ``device`` argument: the GPU/MPI/AMR capabilities are checked at wiring time
        (add_equation) / at execution, not frozen at compile time (DSL_MODEL_DESIGN.md point 7).

        ERGONOMICS (does not change the numerics):

        - ``include`` None -> auto-detected (pops_include()); passing include= remains possible;
        - ``so_path`` None -> .so in an out-of-source cache (pops_cache_dir()), file name keyed on
          model_hash (PARAMS INCLUDED) + abi_key (+ backend/target/name). Cache HIT (.so already present)
          -> reuse without recompilation; cache MISS (model/param/toolchain change) ->
          recompilation + storage. Passing so_path= forces that path and recompiles (backward-compat).

        Returns a CompiledModel carrying so_path, backend, target, adder, names/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        # Lazy codegen import (keeps pops.physics codegen-free at module load; Spec-4 rule):
        from pops.codegen.toolchain import (loader_cxx_std,
                                            _native_kokkos_compiler, _default_cxx,
                                            _native_feature_key, pops_include)
        from pops.codegen.cache import _cache_so_path, _record_so_backend
        from pops.codegen.abi import _abi_key_python
        from pops.codegen.compile_emit import _BACKEND_CAPS
        from pops.codegen.loader import CompiledModel
        from pops.codegen.backends import BACKEND_DESCRIPTORS, Production, lower_internal_backend
        # Internal runtime seam: public callers should go through compile_problem(..., backend=Production())
        # and sim.install(...).
        # This lower-level path may already receive a native backend token derived from descriptors.
        if backend is None:
            backend = Production()
        backend = lower_internal_backend(backend)
        auto_reason = None
        if backend == "auto":
            raise TypeError(
                "_compile_for_runtime: backend='auto' was removed; pass a typed backend "
                "descriptor such as Production()")
        if backend not in HyperbolicModel._BACKENDS:
            raise ValueError("compile: unknown backend %r (expected %s + 'auto')"
                             % (backend, sorted(HyperbolicModel._BACKENDS)))
        if target not in ("system", "amr_system"):
            raise ValueError("compile: target 'system' | 'amr_system' (got %r)" % (target,))

        m = self._m
        # effective std: same per-backend default as HyperbolicModel.compile. The native one follows the
        # loader's standard (c++20 under Kokkos, c++23 otherwise, cf. loader_cxx_std); the others stay c++20.
        mode = HyperbolicModel._BACKENDS[backend][0]
        if target == "amr_system" and mode != "native":
            raise ValueError("compile: target='amr_system' only exists for backend=pops.codegen.Production() "
                             "(native AMR path); got backend=%r" % (backend,))
        eff_std = std if std is not None else (loader_cxx_std() if mode == "native" else "c++20")
        # native AND aot (mode "compile") compile the pops headers -> real Kokkos (compiler +
        # kokkos feature-key) so that the cache key MATCHES the produced .so (cf. compile_aot).
        kokkos_like = mode in ("native", "compile")
        eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
        if include is None:  # ergonomics: auto-detection of the pops headers folder
            include = pops_include()

        # Metadata guards BEFORE the cache (a HIT must not mask them; cf.
        # HyperbolicModel._check_require_metadata).
        m._check_require_metadata(require_metadata, backend)

        # PARAMS-INCLUDED model_hash (the one carried by the CompiledModel) AND the ABI key: both also
        # serve as cache keys, so we compute them here to reuse them (key/metadata consistency).
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, eff_std)

        # OUT-OF-SOURCE cache when so_path is omitted: we RESOLVE the keyed path here (with the
        # params-included hash) and pass it explicitly to the engine -- the cache of HyperbolicModel.compile
        # would otherwise use the hash WITHOUT params (the Model facade adds the Param). HIT -> we skip the
        # compilation. Explicit so_path -> forced path, always recompiles (strict backward-compat).
        cache_hit = False
        if so_path is None:
            # kokkos feature-key in the key (cf. compile_native): a SERIAL .so is not reused
            # on a Kokkos module. MUST match the engine's key, otherwise repeated recompilations.
            cache_backend = (backend + ";" + _native_feature_key()) if kokkos_like else backend
            if hoist_reciprocals:  # distinct codegen -> distinct key (cf. HyperbolicModel.compile)
                cache_backend += ";hoist"
            so_path = _cache_so_path(model_hash, abi_key, cache_backend, target, name)
            cache_hit = os.path.exists(so_path)

        if cache_hit:
            out_path = so_path  # .so already compiled for this key: no recompilation
        else:
            # Compilation (engines unchanged): call the strict driver with a typed descriptor, while
            # keeping target as this internal native-loader token.
            from pops.codegen.compile_drivers import _compile_model
            out_path = _compile_model(
                m, so_path=so_path, include=include, backend=BACKEND_DESCRIPTORS[backend](),
                name=name, cxx=cxx, std=std, require_metadata=require_metadata, target=target,
                hoist_reciprocals=hoist_reciprocals)
        # The keyed path (cache HIT) or the path retained by the engine carries the written backend: we
        # record it so a cross-backend reuse of the SAME path in this process is detected.
        _record_so_backend(out_path, backend)

        adder = HyperbolicModel._BACKENDS[backend][1]
        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm = CompiledModel(
            so_path=out_path, backend=backend, adder=adder, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=_BACKEND_CAPS[backend],
            abi_key=abi_key, model_hash=model_hash,
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None
                 or getattr(m, '_roe_jacobian', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs),
            # NAMED elliptic fields the model declares (m.elliptic_field, ADC-419 / ADC-428): the
            # install seam routes a bind(solvers={field: ...}) selection for a DECLARED field and
            # rejects a typo against this set. Empty for the default-Poisson-only model.
            elliptic_field_names=list(m._elliptic_fields))
        # Trace of the 'auto' policy (ADC-63): None if the backend was explicit. Diagnostic,
        # never a silent choice -- cm.backend says what was built, this says WHY.
        cm.backend_auto_reason = auto_reason
        return cm
