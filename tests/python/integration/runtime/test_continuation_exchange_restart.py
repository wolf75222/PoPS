"""Declared diffusion refinement matrix with accepted mailbox restart and rollback parity."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_public_diffusion_matrix import (
    REFINEMENTS, _bind, _build, _centers, _record,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_full_exchange_continuation_restart_and_failed_publication(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, monkeypatch, method):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.runtime import _continuation_transitions as transitions
    rows = []
    for n in REFINEMENTS:
        x, y = _centers(n)
        initial = 2+.3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        dt = .15/(.4*n*n+1.1*n)
        case, layout, _ = _build("constant", n, dt, method=method, transport=(.7, -.4))
        runtime, artifact = _bind(case, layout, initial, {})
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
        accepted = runtime._executor._checkpoint_program_exchanges()
        records = runtime._executor._program_exchange_records()
        assert len(records) == 8*n*n*(1 if method == "euler" else 2)
        checkpoint = runtime.checkpoint(tmp_path / (method+"-"+str(n)))
        pops.run(runtime, t_end=2*dt, max_steps=1, console=False)
        final_state = np.asarray(runtime.state_global("heat")).copy()
        final_mailbox = runtime._executor._checkpoint_program_exchanges()
        restored = pops.bind(artifact, initial_state={"heat": np.ascontiguousarray(initial[None])},
            resources={"execution_context": artifact_execution_context(artifact)})
        before = (np.asarray(restored.state_global("heat")).copy(),
                  restored._executor._checkpoint_program_exchanges(),
                  restored.continuation_transition_report())
        original = transitions.completed_restart_receipt
        def reject_receipt(owner):
            original(owner)
            raise RuntimeError("injected continuation receipt rejection")
        with monkeypatch.context() as patch:
            patch.setattr(transitions, "completed_restart_receipt", reject_receipt)
            with pytest.raises(RuntimeError, match="injected continuation receipt rejection"):
                restored.restart(checkpoint)
        np.testing.assert_array_equal(restored.state_global("heat"), before[0])
        assert restored.time() == 0 and restored.macro_step() == 0
        assert restored._executor._checkpoint_program_exchanges() == before[1]
        assert restored.continuation_transition_report() == before[2]
        restored.restart(checkpoint)
        assert restored._executor._checkpoint_program_exchanges() == accepted
        assert restored._executor._program_exchange_records() == records
        receipt = restored.continuation_transition_report()
        assert receipt["transition"] == "restart"
        assert next(row for row in receipt["objects"] if row["kind"] == "accepted_exchanges")["action"] == "preserve"
        pops.run(restored, t_end=2*dt, max_steps=1, console=False)
        np.testing.assert_array_equal(restored.state_global("heat"), final_state)
        assert restored._executor._checkpoint_program_exchanges() == final_mailbox
        rows.append({"n": n, "method": method, "accepted_records": len(records),
                     "accepted_mailbox_bytes": len(accepted), "rollback_exact": True,
                     "restart_exact": True, "continuation_exact": True,
                     "artifact": artifact.artifact_identity.token})
    _record("continuation-exchange-restart-"+method, rows)
