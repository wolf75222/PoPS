"""Compile mixin for the PDE-model facade (:class:`pops.physics._facade.Model`).

Splits ``Model._model_hash`` + ``Model.compile`` out of ``facade.py`` so neither
file exceeds the Spec-4 500-line bound. The mixin operates on ``self._m`` (the
private :class:`HyperbolicModel`) and ``self.params``; the build engine is pulled
in LAZILY inside ``compile`` (toolchain/cache/abi/loader names), so importing
``pops.physics`` never loads it (Spec-4 import-graph rule).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .aux import aux_total_n_aux, roles_for
if TYPE_CHECKING:
    from ._model_contract import _FacadeModel
else:
    _FacadeModel = object


class _FacadeCompileMixin(_FacadeModel):
    """Model-hash + compile half of the PDE facade (lazy codegen)."""

    def __pops_artifact_model_metadata__(self) -> dict[str, Any]:
        """Expose the exact metadata protocol of the composed emit model.

        ``compile_problem`` deliberately retains this facade as its single-model
        compile authority because operator resolution and code emission both use
        it.  Artifact introspection must therefore cross the same explicit
        provider boundary instead of probing the private ``_m`` attribute.
        """
        return self._m.__pops_artifact_model_metadata__()

    def __pops_compiler_lowering__(self) -> Any:
        """Provide the explicit compiler-provider boundary for this PDE facade."""
        from pops.codegen import CompilerLowering

        return CompilerLowering(
            emit_model=self,
            source_module=self.module,
            facade=self,
        )

    def _model_hash(self) -> Any:
        """Stable hash of the model: formulas (flux/eig/source/elliptic/primitives/cons_from) + roles +
        n_aux + NAMED params (m.params). Used to identify/reuse an already-compiled .so (cache key)
        and to trace the run. Delegates to the shared computation HyperbolicModel._model_hash, passing it
        the Param of the facade (otherwise two models differing only by a param would have the same hash)."""
        return self._m._model_hash(params=self.params)

    def compile(self, so_path: Any = None, include: Any = None, backend: Any = "production",
                target: str = "system", name: Any = None, cxx: Any = None, std: Any = None,
                require_metadata: bool = False, hoist_reciprocals: bool = False) -> Any:
        """Compiles the model into a CompiledModel (Phase A). Delegates the GENERATION + compilation to
        the native package compiler, then
        packages the .so with the already-known metadata (no re-reading of the .so).

        INTERNAL codegen engine (Spec 5 sec.11). The documented PUBLIC compile front door is
        ``pops.compile(problem, backend=pops.codegen.Production())``; the authoring facade
        ``pops.physics.Model`` is added directly (``case.block(model=m)``) and ``pops.compile``
        captures its operator-first Module internally (ADC-557: no manual ``.lower()`` in the standard
        flow). The public ``pops.resolve`` phase takes a typed backend descriptor. This engine
        ``backend`` argument is private lowering vocabulary retained only inside code generation.

        - ``backend``: the internal ``"production"`` token or typed
          ``pops.codegen.Production()`` descriptor;
        - ``target``: ``"system"`` or ``"amr_system"``.

        NO ``device`` argument: the GPU/MPI/AMR capabilities are checked at wiring time
        (add_equation) / at execution, not frozen at compile time (DSL_MODEL_DESIGN.md point 7).

        ERGONOMICS (does not change the numerics):

        - ``include`` None -> auto-detected (pops_include()); passing include= remains possible;
        - ``so_path`` None -> .so in an out-of-source cache (pops_cache_dir()), file name keyed on
          model_hash (PARAMS INCLUDED) + abi_key (+ backend/target/name). Cache HIT (.so already present)
          -> reuse without recompilation; cache MISS (model/param/toolchain change) ->
          recompilation + storage. Passing so_path= forces that path and recompiles (backward-compat).

        Returns a CompiledModel carrying so_path, backend, target, names/roles/gamma/n_aux/params,
        caps, abi_key, model_hash, cxx, std."""
        import os
        # Lazy codegen import (keeps pops.physics codegen-free at module load; Spec-4 rule):
        from pops.codegen.toolchain import (loader_cxx_std,
                                            _native_kokkos_compiler,
                                            _native_feature_key, pops_include)
        from pops.codegen.cache import (
            _dsl_optflags, _identity_cache_so_path, _platform_cache_key,
            _precision_cache_key, _record_so_backend, _registry_cache_key,
        )
        from pops.codegen.compile_provenance import (
            verify_cached_artifact, write_artifact_sidecar,
        )
        from pops.codegen.abi import _abi_key_python
        from pops.codegen._compile_emit import compiled_capability_flags
        from pops.codegen.loader import CompiledModel
        from pops.codegen._compiled_model_identity import model_compile_identity
        from pops.codegen._backends import lower_backend
        backend = lower_backend(backend)
        if target not in ("system", "amr_system"):
            raise ValueError("compile: target 'system' | 'amr_system' (got %r)" % (target,))

        m = self._m
        eff_std = std if std is not None else loader_cxx_std()
        eff_cxx = _native_kokkos_compiler(cxx)
        if include is None:  # ergonomics: auto-detection of the pops headers folder
            include = pops_include()

        # Metadata guards BEFORE the cache (a HIT must not mask them; cf.
        # HyperbolicModel._check_require_metadata).
        m._check_require_metadata(require_metadata, backend)

        # PARAMS-INCLUDED model_hash (the one carried by the CompiledModel) AND the ABI key: both also
        # serve as cache keys, so we compute them here to reuse them (key/metadata consistency).
        model_hash = self._model_hash()
        abi_key = _abi_key_python(include, eff_cxx, eff_std)
        from pops.identity import artifact_spec_identity
        from pops.identity.semantic import semantic_identity_of

        semantic_identity = semantic_identity_of(model=self)
        feature_key = _native_feature_key()
        spec_identity = artifact_spec_identity(
            semantic_identity,
            target=target,
            backend=backend,
            precision=_precision_cache_key(),
            abi=abi_key,
            toolchain="%s|%s" % (eff_cxx, eff_std),
            routes={"registry": _registry_cache_key(), "features": feature_key},
            components={"model_hash": str(model_hash), "emitted_name": str(name or "")},
            flags=[_platform_cache_key(), *_dsl_optflags(),
                   "hoist_reciprocals=%d" % bool(hoist_reciprocals)],
            libraries=(),
        )

        # OUT-OF-SOURCE cache when so_path is omitted: we RESOLVE the keyed path here (with the
        # params-included hash) and pass it explicitly to the engine -- the cache of HyperbolicModel.compile
        # would otherwise use the hash WITHOUT params (the Model facade adds the Param). HIT -> we skip the
        # compilation. Explicit so_path -> forced path, always recompiles (strict backward-compat).
        cache_requested = so_path is None
        if cache_requested:
            so_path = _identity_cache_so_path(spec_identity)

        if cache_requested and os.path.isfile(so_path):
            binary_identity, final_artifact_identity = verify_cached_artifact(
                so_path, semantic_identity=semantic_identity, spec_identity=spec_identity)
            out_path = so_path
        else:
            # The loader emits the target-specific fixed ABI entry point.
            out_path = m.compile(so_path, include, backend=backend, name=name, cxx=cxx, std=std,
                                 require_metadata=require_metadata, target=target,
                                 hoist_reciprocals=hoist_reciprocals)
            binary_identity, final_artifact_identity = write_artifact_sidecar(
                out_path, semantic_identity=semantic_identity, spec_identity=spec_identity)
        # The keyed path (cache HIT) or the path retained by the engine carries the written backend: we
        # record it so a cross-backend reuse of the SAME path in this process is detected.
        _record_so_backend(out_path, backend)

        cons_roles = roles_for(m.cons_names, m.cons_roles)
        cm: Any = CompiledModel(
            so_path=out_path, backend=backend, target=target,
            cons_names=m.cons_names, cons_roles=cons_roles, prim_names=m.prim_state,
            n_vars=m.n_vars, gamma=m.gamma, n_aux=aux_total_n_aux(m.aux_names, m.aux_extra_names),
            params=self.params, caps=compiled_capability_flags(backend),
            abi_key=abi_key, model_hash=model_hash,
            definition_identity=model_compile_identity(self),
            cxx=eff_cxx, std=eff_std, hllc=m._hllc,
            roe=(m._roe or getattr(m, '_roe_rows', None) is not None
                 or getattr(m, '_roe_jacobian', None) is not None),
            aux_extra_names=m.aux_extra_names,
            wave_speeds=(m._wave_speeds is not None or m._ws_jacobian is not None
                         or "p" in m.prim_defs),
            # NAMED elliptic fields the model declares (m.elliptic_field, ADC-419 / ADC-428): the
            # detached model preserves the declaration inventory while the resolved simulation plan
            # owns the field discretization and provider. Empty for the default-Poisson-only model.
            elliptic_field_names=list(m._elliptic_fields))
        cm.semantic_identity = semantic_identity
        cm.artifact_spec_identity = spec_identity
        cm.binary_identity = binary_identity
        cm.artifact_identity = final_artifact_identity
        # Exact ABI order of only the RuntimeParamRef nodes actually read by emitted formulas.
        # BindSchema routes qualified values into this local vector; declarations that are never
        # read do not create native slots.
        cm._runtime_param_names = [node.name for node in m.assign_runtime_indices()]
        return cm
