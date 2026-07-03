"""Compiled per-block model handle (ADC-619 split).

:class:`CompiledModel` -- the result of ``m.compile(...)`` -- packages a per-block
physics ``.so`` with the metadata needed to wire it (dispatch adder, names, roles,
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
    needed to wire it correctly (dispatch adder, ABI diagnostic,
    reproducibility). Replaces the historical pair (str so_path,
    adder_for(backend)) with a single object.

    The metadata is NOT re-read from the ``.so``: Python already holds
    names/roles/gamma/n_aux/params (the HyperbolicModel carries them);
    CompiledModel just exposes them for dispatch (add_equation) and
    diagnostics. cf. DSL_MODEL_DESIGN.md section 3.
    """

    def __init__(self, so_path: Any, backend: Any, adder: Any, cons_names: Any, cons_roles: Any,
                 prim_names: Any, n_vars: Any, gamma: Any, n_aux: Any, params: Any, caps: Any,
                 abi_key: Any, model_hash: Any, cxx: Any, std: Any, target: Any = "system",
                 hllc: Any = False, roe: Any = False, aux_extra_names: Any = None,
                 wave_speeds: Any = False, elliptic_field_names: Any = None) -> None:
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
    def capabilities(self) -> Any:
        """Typed capability handles of this compiled model (ADC-552): ``compiled.capabilities.
        wave_speeds`` returns the artifact's :class:`~pops.numerics.riemann.waves.WaveSpeedProvider`.
        Derived from the carried authoring model when present, else from ``has_wave_speeds`` (a
        generic signed-pair provider); raises a precise error when the artifact declares no wave
        speeds (no silent None)."""
        from pops.numerics.riemann.waves import _CapabilityHandles  # lazy: loader <-> numerics edge
        return _CapabilityHandles(self)

    @property
    def runtime_param_names(self) -> list:
        """Names of the model's RUNTIME parameters (kind='runtime'), SORTED: this is the ORDER of
        the indices on the C++ side (RuntimeParams) AND the order expected by
        System.set_block_params(name, values) (P7-b). Empty if the model has only const params."""
        return sorted(k for k, p in self.params.items() if getattr(p, "kind", "const") == "runtime")

    def runtime_param_values(self) -> list:
        """DECLARATION values of the runtime params, parallel to runtime_param_names (default as
        long as no set_block_params has been called)."""
        return [self.params[k].value for k in self.runtime_param_names]

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
        from pops import FiniteVolume, Explicit  # lazy: avoids a top-level runtime import
        from pops.runtime.system import System  # advanced seam (ADC-545: off the public surface)
        from pops.numerics.reconstruction.limiters import Minmod
        from pops.numerics.riemann import Rusanov
        sim = System(n=int(n), L=1.0, periodic=True)
        sim.set_poisson()
        sim.add_equation("check", model=self,
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

    def inspect_amr(self, layout: Any = None) -> Any:
        """STATIC AMR report on this compiled MODEL (Spec 5 sec.8.12 / sec.8.4).

        A ``CompiledModel`` produced off the AMR route (``pops.compile(problem, layout=AMR(...))``)
        carries the compile-time layout on ``self._layout`` (ADC-555: the refine/regrid/patches
        tags must appear in ``compiled.inspect_amr()``, not just ``layout.inspect()``), so a bare
        call with no argument reports THAT layout when one is attached, rather than the generic
        native envelope. An explicit @p layout argument always wins (it overrides the carried
        one); a handle with no ``_layout`` (a ``target='system'`` model, or a stub built outside
        ``pops.compile``) falls back to the native envelope, same as before. Delegates to the
        top-level :func:`pops.inspect_amr`; never fabricates a hierarchy.
        """
        from pops import inspect_amr
        if layout is None:
            layout = getattr(self, "_layout", None)
        return inspect_amr(layout)

    def capability_matrix(self) -> Any:
        """The ADC-549 native route matrix for this compiled model handle."""
        from pops._capabilities import native_capability_matrix
        flags = {
            "supports_uniform": bool(self.caps.get("cpu", False)),
            "supports_amr": bool(self.caps.get("amr", False)
                                 and getattr(self, "target", "system") == "amr_system"),
            "supports_mpi": bool(self.caps.get("mpi", False)),
            "supports_gpu": bool(self.caps.get("gpu", False)),
            "supports_stride": bool(getattr(self, "backend", None) == "production"),
            "supports_named_fields": True,
            "supports_partial_imex_mask": False,
            "supports_custom_communicator": False,
        }
        return native_capability_matrix(
            owner=getattr(self, "so_path", None) or "compiled-model",
            layout="amr" if getattr(self, "target", "system") == "amr_system" else "system",
            flags=flags, source="manifest")

    def __repr__(self) -> str:
        return ("CompiledModel(backend=%r, target=%r, so_path=%r, n_vars=%d, gamma=%r, n_aux=%d, "
                "adder=%r, runtime_params=%r, abi_key=%.12s..., model_hash=%.12s...)"
                % (self.backend, self.target, self.so_path, self.n_vars, self.gamma, self.n_aux,
                   self.adder, self.runtime_param_names, self.abi_key or "", self.model_hash or ""))
