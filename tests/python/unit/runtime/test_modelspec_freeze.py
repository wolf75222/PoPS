"""The native ModelSpec participates atomically in Problem.freeze()."""
from __future__ import annotations

import pytest

from pops.runtime import ModelSpec


_VALUES = {
    "transport": "exb",
    "source": "none",
    "elliptic": "charge",
    "B0": 2.0,
    "gamma": 1.2,
    "cs2": 3.0,
    "vacuum_floor": 1.0e-9,
    "qom": -1.0,
    "q": 4.0,
    "alpha": 5.0,
    "n0": 6.0,
    "sign": -1.0,
    "four_pi_G": 7.0,
    "rho0": 8.0,
}


def test_modelspec_freeze_guards_every_python_property():
    spec = ModelSpec()
    for name, value in _VALUES.items():
        setattr(spec, name, value)
    assert spec.frozen is False

    spec.freeze()
    spec.freeze()
    assert spec.frozen is True
    for name, value in _VALUES.items():
        with pytest.raises(RuntimeError, match="ModelSpec is frozen.*%s" % name):
            setattr(spec, name, value)


def test_private_transaction_hooks_reject_calls_without_coordinator_capability():
    spec = ModelSpec()
    with pytest.raises(TypeError):
        spec._pops_freeze_snapshot()
    with pytest.raises(RuntimeError, match="private PoPS transaction capability"):
        spec._pops_freeze_snapshot(object())

    spec.freeze()
    with pytest.raises(TypeError):
        spec._pops_freeze_restore(False)
    with pytest.raises(RuntimeError, match="private PoPS transaction capability"):
        spec._pops_freeze_restore(object(), False)

    assert spec.frozen is True
    with pytest.raises(RuntimeError, match="ModelSpec is frozen"):
        spec.transport = "isothermal"


def test_problem_freeze_rolls_modelspec_back_when_a_later_descriptor_fails():
    import pops
    from pops.descriptors import BrickDescriptor

    class FailingSpatial(BrickDescriptor):
        def freeze(self):
            super().freeze()
            raise RuntimeError("spatial freeze failed")

    spec = ModelSpec()
    spec.transport = "exb"
    spec.source = "none"
    spec.elliptic = "charge"
    problem = pops.Problem(name="native-rollback")
    problem.add_block("u", spec, spatial=FailingSpatial("fv", "native"))

    with pytest.raises(RuntimeError, match="spatial freeze failed"):
        problem.freeze()

    assert spec.frozen is False
    spec.transport = "isothermal"
    assert spec.transport == "isothermal"
