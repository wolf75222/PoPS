"""Compiled per-block model handle (ADC-619 split).

:class:`CompiledModel` -- the result of ``m.compile(...)`` -- packages a per-block
physics ``.so`` with the metadata needed to wire it (names, roles,
gamma, n_aux, params, caps, abi_key, model_hash) plus its runtime-param routing,
runtime re-verification, static AMR report and route matrix. Split out of
``pops.codegen.loader`` for the 500-line cap; ``pops.codegen.loader`` re-exports it
so the historical ``from pops.codegen.loader import CompiledModel`` path is
unchanged. It imports neither ``pops.dsl`` nor ``pops.physics`` at module level.
"""

from __future__ import annotations

from typing import Any


class CompiledModel:
    """Result of ``m.compile(...)``: packages the produced ``.so`` + EVERYTHING
    needed to wire it correctly (fixed native ABI, metadata and reproducibility).

    The metadata is NOT re-read from the ``.so``: Python already holds
    names/roles/gamma/n_aux/params (the HyperbolicModel carries them);
    CompiledModel just exposes them for dispatch (add_equation) and
    diagnostics. cf. DSL_MODEL_DESIGN.md section 3.
    """

    def __init__(self, so_path: Any, backend: Any, cons_names: Any, cons_roles: Any,
                 prim_names: Any, n_vars: Any, gamma: Any, n_aux: Any, params: Any, caps: Any,
                 abi_key: Any, model_hash: Any, cxx: Any, std: Any, target: Any = "system",
                 hllc: Any = False, roe: Any = False, aux_extra_names: Any = None,
                 wave_speeds: Any = False, elliptic_field_names: Any = None,
                 bind_schema: Any = None, definition_identity: Any = None,
                 state_spaces: Any = ("U",)) -> None:
        self.has_hllc = bool(hllc)   # HLLC capability emitted (enable_hllc): hllc available beyond 4-var Euler
        self.has_roe = bool(roe)     # ROE hook emitted (enable_roe roles OR m.roe_dissipation provided): roe available beyond 4-var Euler
        self.has_wave_speeds = bool(wave_speeds)  # wave_speeds emitted (explicit pair OR 'p'): hll available
        self.so_path = so_path
        if backend != "production":
            raise ValueError("CompiledModel backend must be the native production route")
        self.backend = backend
        self.target = target         # "system" | "amr_system": targeted facade (native AMR loader if amr_system)
        self.cons_names = list(cons_names)
        self.state_spaces = list(state_spaces)
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
        # set_block_elliptic_field after the block is installed. The names remain detached compiled
        # model evidence; the resolved simulation plan owns every solver/provider choice. Empty for a
        # model with only the default Poisson field.
        self.elliptic_field_names = list(elliptic_field_names) if elliptic_field_names else []
        # Compiler-owned declarations remain live until orchestration has finished attaching the
        # bind schema.  ``_seal`` replaces this mapping by registry-free CompiledParameter values;
        # no public artifact retains ParamHandle/Expr/OwnerPath authority through ``params``.
        self.params = dict(params)
        self.caps = dict(caps)       # {cpu/mpi/amr/gpu: bool}
        self.abi_key = abi_key       # ABI key mirroring pops_header_signature + compiler/std
        self.model_hash = model_hash  # stable hash formulas+roles+n_aux+params
        # Authenticated structural preimage of model.compile(). Public orchestration requires this
        # to match the frozen compiler input; low-level test/runtime handles may leave it absent.
        self.definition_identity = definition_identity
        self.cxx = cxx
        self.std = std
        # Set directly only by advanced constructors. The public pops.compile route replaces this
        # with the one BindSchema captured from the complete frozen Problem (all block instances).
        self.bind_schema = bind_schema
        self.install_plan = None
        self.semantic_identity = None
        self.artifact_spec_identity = None
        self.binary_identity = None
        self.artifact_identity = None

    def _seal(self) -> None:
        """Freeze a public per-block artifact after orchestration attaches metadata."""
        stored = object.__getattribute__(self, "__dict__")
        if stored.get("_sealed", False):
            return
        from pops.codegen._compiled_parameter import compiled_parameters
        from pops.codegen._artifact_freeze import seal_attributes
        from pops.codegen._compiled_model_boundary import seal_compiled_model

        object.__setattr__(self, "params", compiled_parameters(stored["params"]))
        seal_compiled_model(self)
        seal_attributes(self)

    @property
    def authoring_snapshot(self) -> Any:
        """Complete immutable authoring identity, or ``None`` on a low-level handle."""
        return getattr(self, "_problem_snapshot", None)

    def __setattr__(self, name: Any, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "CompiledModel is immutable after pops.compile: cannot set %r; "
                "assemble a new Problem and recompile" % name
            )
        object.__setattr__(self, name, value)

    @property
    def capabilities(self) -> Any:
        """Typed capability handles of this compiled model (ADC-552): ``compiled.capabilities.
        wave_speeds`` returns the artifact's :class:`~pops.numerics.riemann.waves.WaveSpeedProvider`.
        Derived from the carried authoring model when present, else from ``has_wave_speeds`` (a
        generic signed-pair provider); raises a precise error when the artifact declares no wave
        speeds (no silent None)."""
        from pops.numerics.riemann.waves import _CapabilityHandles  # lazy: loader <-> numerics edge
        return _CapabilityHandles(self)

    def __pops_artifact_model_metadata__(self) -> dict[str, Any]:
        """Exact report data; consumers never probe optional model attributes."""
        return {
            "schema_version": 1,
            "state_spaces": tuple(self.state_spaces),
            "cons_names": tuple(self.cons_names),
            "n_vars": self.n_vars,
            "params": dict(self.params),
            "aux_names": tuple(self.aux_extra_names),
            "n_aux": self.n_aux,
            "capabilities": dict(self.caps),
        }

    @property
    def runtime_param_names(self) -> list:
        """Runtime-carrier local IDs in the exact emitted within-model slot order."""
        captured = getattr(self, "_runtime_param_names", None)
        if captured is not None:
            return list(captured)
        result = []
        for name, declaration in self.params.items():
            kind = getattr(declaration, "kind", None)
            kind = getattr(kind, "value", kind)
            phase = getattr(declaration, "phase", None)
            phase = getattr(phase, "value", phase)
            if kind == "runtime" or (kind == "derived" and phase == "bind"):
                result.append(name)
        return sorted(result)

    def runtime_param_values(self) -> list:
        """DECLARATION values of the runtime params, parallel to runtime_param_names (default as
        before the canonical BindSchema vector is injected at package installation."""
        return [
            self.params[name].default
            if getattr(self.params[name], "has_default", False)
            else None
            for name in self.runtime_param_names
        ]

    def check_runtime(self, n: Any = 16, state: Any = None, raise_on_error: Any = True,
                      rtol: Any = 1e-8, atol: Any = 1e-10) -> Any:
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
        from pops.runtime._engine_descriptors import Explicit, Spatial
        from pops.runtime._system import System  # advanced seam (ADC-545: off the public surface)
        from pops.numerics.reconstruction.limiters import Minmod
        from pops.numerics.riemann import Rusanov
        sim = System(n=int(n), L=1.0, periodic=True)
        sim.set_poisson()
        sim.add_equation("check", model=self,
                         spatial=Spatial(limiter=Minmod(), flux=Rusanov()),
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

    def arguments(self) -> Any:
        """The runtime inputs this AMR-route artifact expects at bind (Spec 5 sec.12.2, ADC-515).

        The AMR route of ``pops.compile(problem, layout=<structured AMR descriptor>)`` returns the first block's
        ``CompiledModel`` (there is no whole-system ``CompiledProblem`` on AMR: each block is a native
        ``add_native_block`` loader). So the ``arguments()`` seam ``CompiledProblem`` exposes on the
        Uniform route lives HERE too, built from the SAME :func:`~pops.codegen.inspect_compiled.
        build_arguments` via the model-as-handle path (the handle IS its own physical model). It lists
        -- WITHOUT any bind or runtime read -- the block instance (state space / components /
        required), the model's declared params (type / kind / required), its named aux (layout /
        required) and the runtime layout the artifact targets (``layout='amr'`` for this handle). It
        allocates and reads nothing."""
        from pops.codegen.inspect_compiled import build_component_arguments
        return build_component_arguments(self)

    def inspect(self) -> Any:
        """Return the same inert compiled-artifact report as a whole-system handle."""
        from pops.codegen.inspect_report import build_compiled_report

        return build_compiled_report(self)

    def requirements(self) -> Any:
        """Return compile-time requirements for every model carried by the InstallPlan."""
        from pops.codegen.inspect_report import build_requirements

        return build_requirements(self)

    def manifest(self) -> Any:
        """Return the rich manifest consumed by the pre-bind refusal gates."""
        from pops.external.artifact_manifest import build_compiled_manifest

        return build_compiled_manifest(self)

    def estimate_memory(self, mesh: Any, *, platform: Any = None, layout: Any = None) -> Any:
        """A FORMULA-based memory estimate for this AMR-route artifact on ``mesh`` (sec.12.3, ADC-515).

        The AMR counterpart of ``CompiledProblem.estimate_memory``: a pure FORMULA over the mesh shape
        and the model's component counts (state / aux / halo, plus the conservative AMR patch budget),
        via the SAME :func:`~pops.codegen.inspect_compiled.build_memory_estimate` with no Program (the
        no-Program branch skips Program-only scratch and solver categories). @p layout defaults to the
        AMR layout carried by the immutable ``InstallPlan``, so a
        bare ``estimate_memory(mesh)`` auto-reports the AMR hierarchy budget (``layout='amr'``,
        conservative full-refinement worst case); an explicit @p layout / @p platform still wins. It
        NEVER allocates a ``MultiFab``; every assumption is in ``MemoryEstimate.assumptions``."""
        from pops.codegen.inspect_compiled import build_memory_estimate
        return build_memory_estimate(self, mesh, platform=platform,
                                     layout=layout or (
                                         self.install_plan.layout
                                         if self.install_plan is not None else None))

    def __repr__(self) -> str:
        return ("CompiledModel(backend=%r, target=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "runtime_params=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.target, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.runtime_param_names, self.abi_key or "", self.model_hash or ""))
