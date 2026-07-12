#!/usr/bin/env python3
"""The public blackboard Model resolves with no manual lowering call.

The standard flow adds ``pops.physics.Model`` directly (``problem.add_block(name, model=m)``)
and ``pops.compile`` captures its operator-first :class:`pops.model.Module` internally -- the user
never calls ``m.to_module()`` / ``m.lower()``. This pins, at the pure metadata level (no compile /
no ``.so``), that the compile-side resolver hands the emit model a Module-backed physics model whose
operator-first Module (and its stable hash) is reachable for the trace.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
import sys

import pytest

pytest.importorskip("pops")
from pops import model as model_pkg  # noqa: E402
from pops.physics import Model  # noqa: E402
from pops.codegen._phases import _resolve_problem_model  # noqa: E402


def _facade_model(name="gas"):
    m = Model(name)
    state = m.state("U", components=("rho", "mx"))
    rho, mx = state
    m.flux("F", on=state, x=(mx, mx * mx / rho), y=(rho, mx))
    return m


def test_resolve_facade_model_exposes_operator_first_module():
    m = _facade_model()
    resolved = _resolve_problem_model(m)
    # The resolved model the emitter consumes carries the operator-first Module (the canonical IR
    # authority the compile pipeline captures) -- no manual to_module() was needed.
    assert hasattr(resolved, "module"), "the resolved model exposes its operator-first Module"
    assert isinstance(resolved.module, model_pkg.Module)
    assert resolved.module.module_hash(), "the Module carries a stable hash for drift detection"


def test_facade_model_never_requires_manual_to_module():
    # The public Model is added AS-IS; the standard flow does not call to_module / lower. The model
    # exposes the Module view (advanced/inspection), but the user is not required to invoke it.
    m = _facade_model()
    assert hasattr(m, "module"), "the facade exposes the Module view for inspection"
    # Resolving does not raise and does not require the user to have lowered anything.
    resolved = _resolve_problem_model(m)
    assert resolved is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
