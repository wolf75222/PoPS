"""AmrSystem compiled-problem install mixin (Spec 6 sec.11, epic ADC-511 / ADC-508).

Extracted from :mod:`pops.runtime.amr_system` to keep that module under the Spec-4 36.3
500-line budget. Holds the compiled-artifact tail of ``_install_compiled``: native artifact attach,
runtime params and global cadence. Mixed in via inheritance; operates on ``self._s`` through the
native binding and on the ``_install_*`` helpers of the host class. Mirror of the System routes in
:mod:`pops.runtime._system_unified_install`.
"""


class _AmrSystemProgram:
    """Compiled-problem attach / params / cadence methods of AmrSystem (ADC-508)."""

    def _install_problem_so(self, so_path):
        """Install a combined compiled-problem shared object through the native AMR runtime.

        This wrapper is the private compiled-problem loader used by ``sim.install``. The generated
        shared object may still export the historical C ABI symbol it was built around, but the
        Python/native binding seam is a compiled-problem attach, not a public Program route.
        """
        return self._s.install_problem(so_path)

    def _finish_problem_install(self, compiled, so_path, params, cadence):
        """Steps 5/5b/6 of ``_install_compiled`` for a compiled problem artifact (ADC-508).

        Runs AFTER the field solvers, blocks, aux inputs and initial state are wired:

          - (5) attach the compiled problem on the AMR hierarchy (binds blocks by name + runs native
            requirement validation: block instance / solver). The .so must
            export pops_install_program_amr (target='amr_system'); a target='system' .so is rejected
            at the C++ loader with an actionable message. NATIVE mode (so_path is None) has no
            compiled artifact; the step-2 blocks drive the native AMR loop, so a non-empty params=
            raises (the native AMR block loader does not transport runtime params, and AmrSystem has
            no set_block_params).
          - (5b) COMPILED-PROBLEM RUNTIME PARAMS (parity ADC-510): route params to the per-block
            runtime parameter table seeded by the native attach. A name declared by no generated
            kernel raises (no silent drop).
          - (6) COMPILED-PROBLEM CADENCE (substeps / stride): the artifact is one whole-system
            closure, so its macro-step cadence is GLOBAL. Apply it AFTER attach. A native AMR install
            has no compiled artifact -- set substeps / stride on the native time policy instead.
        """
        if so_path is not None:
            self._install_problem_so(so_path)
            if params:
                self._install_problem_params(compiled, params)
        elif params:
            raise ValueError(
                "sim.install: runtime params (params=%s) are not wired on a NATIVE AMR install (the AMR "
                "native .so loader does not transport runtime params, and AmrSystem has no "
                "set_block_params); pass a compiled problem artifact "
                "(pops.compile_problem(..., layout=AMR(...), backend=Production())) "
                "to use params=, set them as const on the native model, or use System."
                % sorted(params))

        if cadence is not None:
            if so_path is None:
                raise ValueError(
                    "sim.install(cadence=): a cadence applies to a compiled problem artifact; a native "
                    "AMR install (compiled=None) has no compiled artifact -- set substeps / stride on "
                    "the native time policy instead.")
            self._install_cadence(cadence)

    def _install_problem_params(self, compiled, params):
        """Route flat {param_name: value} to native runtime params for a compiled AMR problem.

        Reads the compiled handle's declared routing (runtime_param_routes), builds each block's
        complete value vector (declaration defaults for unspecified names), and pushes it to the
        AMR-owned per-block RuntimeParams the generated kernels read. A name declared by no generated
        kernel raises (no silent drop).
        """
        from pops.runtime._install_param_routing import route_program_params
        routes_fn = getattr(compiled, "runtime_param_routes", None)
        routes, defaults = routes_fn() if callable(routes_fn) else ({}, {})
        per_block, unknown = route_program_params(routes, defaults, params)
        for blk, values in per_block.items():
            self.set_program_params(blk, values)
        if unknown:
            raise ValueError(
                "sim.install: params %s declared by no runtime parameter of the compiled problem "
                "(a runtime param must be read by generated kernels and declared as a runtime param)"
                % (unknown,))

    def _set_problem_cadence(self, substeps, stride):
        """Private native cadence attach used by the compiled-problem install seam."""
        return self._s.set_program_cadence(substeps, stride)

    def _install_cadence(self, cadence):
        """Apply a compiled-problem macro-step cadence to the installed AMR artifact.

        ``substeps=n`` re-runs the whole artifact over ``eff_dt/n``; ``stride=M`` runs it once per
        M macro-steps. A numeric ``cadence.cfl`` is pinned so a bare run() with no explicit cfl= uses
        it (step_cfl routes the per-block CFL dt through the installed artifact).
        """
        from pops.runtime._compiled_cadence import CompiledProgramCadence
        if not isinstance(cadence, CompiledProgramCadence):
            raise TypeError("sim.install(cadence=): expected an internal CompiledProgramCadence "
                            "(substeps=, stride=), got %r" % type(cadence).__name__)
        if isinstance(cadence.cfl, (int, float)):
            self._program_cadence_cfl = float(cadence.cfl)
        elif cadence.cfl == "program":
            self._program_cadence_cfl = "program"
        self._set_problem_cadence(cadence.substeps, cadence.stride)
