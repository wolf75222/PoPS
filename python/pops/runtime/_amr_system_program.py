"""AmrSystem compiled-Program install mixin (Spec 6 sec.11, epic ADC-511 / ADC-508).

Extracted from :mod:`pops.runtime.amr_system` to keep that module under the Spec-4 36.3
500-line budget. Holds the COMPILED time-Program tail of ``_install_compiled``: the
``install_program`` step on the AMR hierarchy plus its runtime params (``set_program_params``)
and typed step-transaction contract. Mixed in via inheritance; operates on ``self._s``
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
    """COMPILED time-Program install and typed transaction methods of AmrSystem."""

    def _finish_program_install(self, compiled: Any, so_path: Any, schema: Any,
                                params: Any) -> Any:
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
          - (6) attach the exact typed StepTransactionPlan authored by the installed Program.
        """
        if so_path is not None:
            self.install_program(so_path)
            # (5a) HISTORY-PERSISTENCE POLICIES (ADC-631, parity with the uniform step-5a): the compiled
            # Program records a per-ring persistence policy (Dense / Interval / Revolve) on
            # program._history_persistence. Attach the name -> policy map so the v3 checkpoint stores
            # only the policy-selected slots and the restart replays the gaps. Absent -> Dense.
            program = getattr(compiled, "program", None)
            program = getattr(program, "program", program)
            persistence = getattr(program, "_history_persistence", None) if program else None
            set_persistence = getattr(self, "set_history_persistence", None)
            if persistence and set_persistence is not None:
                set_persistence(
                    {name: policy for name, (_depth, policy) in persistence.items()})
            self._install_program_params(compiled, schema, params)
            component = getattr(compiled, "program", None)
            authored = getattr(component, "program", component)
            self._step_strategy = getattr(authored, "_step_strategy", None)
            self._step_transaction_plan = (
                authored.transaction_plan() if authored is not None else None)
            if authored is not None:
                self._temporal_restart_state.configure_program(
                    authored.temporal_manifest(),
                    time=self._s.time(), macro_step=self._s.macro_step())

    def _install_program_params(self, compiled: Any, schema: Any, params: Any) -> None:
        """Install complete owner-qualified Program vectors from BindSchema."""
        from pops.runtime._install_param_routing import route_program_params
        per_block = route_program_params(compiled, schema, params)
        for blk, values in per_block.items():
            self.set_program_params(blk, values)
