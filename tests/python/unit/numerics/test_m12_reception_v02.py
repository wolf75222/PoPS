"""Negative reception test only: passing this is NOT native model qualification."""
import numpy as np
from examples.api_v02.models_M12 import build_m12, reference_rk4


def test_m12_reception_rejects_translation_even_when_global_invariants_hold():
    spec=build_m12(cells=(8,8))
    labels=("five","ten","EM")
    initial=tuple(spec.initial_values[spec.subjects[label]] for label in labels)
    expected=reference_rk4(initial,(2*np.pi/8,2*np.pi/8),spec.dt,2)
    shifted={label:np.roll(value,1,axis=-1) for label,value in zip(labels,expected,strict=True)}
    metrics=spec.invariants(shifted)
    assert metrics["Gauss_defect"]<2.e-11 and metrics["divB_defect"]<2.e-11
    assert metrics["mass_defect"]<2.e-12 and metrics["total_energy_defect"]<2.e-10
    assert abs(metrics["field_energy_exchange"])>1.e-9
    assert max(metrics["full_state_reference_errors"])>1.e-3
    assert not metrics["passed"]
