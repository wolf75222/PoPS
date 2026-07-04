"""AmrSystem compiled-Program install mixin (Spec 6 sec.11, epic ADC-511 / ADC-508).

Extracted from :mod:`pops.runtime.amr_system` to keep that module under the Spec-4 36.3
500-line budget. Holds the COMPILED time-Program tail of ``_install_compiled``: the
``install_program`` step on the AMR hierarchy plus its runtime params (``set_program_params``)
and global cadence (``set_program_cadence``). Mixed in via inheritance; operates on ``self._s``
through the native binding and on the ``_install_*`` helpers of the host class. Mirror of the
System routes in :mod:`pops.runtime._system_unified_install`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _AmrSystemProgram(_AmrSystem):
    """COMPILED time-Program install / params / cadence methods of AmrSystem (ADC-508)."""

    def _finish_program_install(self, compiled: Any, so_path: Any, params: Any,
                                cadence: Any) -> Any:
        """Steps 5/5b/6 of ``_install_compiled`` for a COMPILED time Program (ADC-508).

        Runs AFTER the field solvers, blocks, aux inputs and initial state are wired:

          - (5) install the compiled time Program on the AMR hierarchy (binds blocks by name +
            runs the section-24 .so requirement validation: block instance / solver). The .so must
            export pops_install_program_amr (target='amr_system'); a target='system' .so is rejected
            at the C++ loader with an actionable message. NATIVE mode (so_path is None) has no Program
            -- the step-2 blocks drive the native AMR loop, and the native per-block runtime params were
            already routed to set_block_params in step 4b (ADC-514), so @p params here holds only the
            names NOT consumed by an instance (empty on a native install, since step 4b rejected any).
          - (5b) COMPILED-PROGRAM RUNTIME PARAMS (parity ADC-510): route the remaining params to the
            per-PROGRAM-block set_program_params, AFTER install_program seeded each block's declaration
            defaults. A name declared by no Program kernel raises (no silent drop).
          - (6) PROGRAM CADENCE (substeps / stride): a compiled Program is ONE whole-system closure, so
            its macro-step cadence is GLOBAL. Apply it AFTER install_program. A native AMR install has no
            Program -- set substeps / stride on the native time policy instead.
        """
        if so_path is not None:
            self.install_program(so_path)
            if params:
                self._install_program_params(compiled, params)

        if cadence is not None:
            if so_path is None:
                raise ValueError(
                    "pops.bind(cadence=): a cadence applies to a compiled time Program; a native AMR "
                    "install (compiled=None) has no Program -- set substeps / stride on the native time "
                    "policy (pops.Explicit(substeps=, stride=)) instead.")
            self._install_cadence(cadence)

    def _install_program_params(self, compiled: Any, params: Any) -> Any:
        """Route flat {param_name: value} to set_program_params per PROGRAM block (ADC-508, AMR mirror
        of System._install_program_params): read the compiled handle's declared routing
        (runtime_param_routes), build each block's COMPLETE value vector (declaration defaults for
        unspecified names) and push it to the AMR-owned per-block RuntimeParams the Program kernels read.
        A name declared by no Program kernel raises (no silent drop)."""
        from pops.runtime._install_param_routing import route_program_params
        routes_fn = getattr(compiled, "runtime_param_routes", None)
        routing: Any = routes_fn() if callable(routes_fn) else ({}, {})
        routes, defaults = routing
        per_block, unknown = route_program_params(routes, defaults, params)
        for blk, values in per_block.items():
            self.set_program_params(blk, values)
        if unknown:
            raise ValueError(
                "pops.bind: params %s declared by no runtime parameter of the compiled Program "
                "(a runtime param must be read by the Program's source / linear-source kernels and "
                "declared dsl.Param(..., kind='runtime'))" % (unknown,))

    def _install_cadence(self, cadence: Any) -> Any:
        """Apply a CompiledTime macro-step cadence to the installed AMR program (set_program_cadence,
        AMR mirror of System._install_cadence). substeps=n re-runs the whole program over eff_dt/n;
        stride=M runs it once per M macro-steps. A NUMERIC cadence.cfl is pinned so a bare run() with no
        explicit cfl= uses it (step_cfl routes the per-block CFL dt through the installed program)."""
        from pops.time.program import CompiledTime
        if not isinstance(cadence, CompiledTime):
            raise TypeError("pops.bind(cadence=): expected a pops.time.CompiledTime(substeps=, stride=), "
                            "got %r" % type(cadence).__name__)
        if isinstance(cadence.cfl, (int, float)):
            self._program_cadence_cfl = float(cadence.cfl)
        self.set_program_cadence(cadence.substeps, cadence.stride)
