"""ADC-615 embedded-boundary / cut-cell thresholds end to end.

The typed pops.mesh.masks.CutCell(kappa_min, face_open_eps, cut_theta_min) lowers through a
generic LevelSet to the native EbThresholds, surfaced in the System effective-options report; the
same cut_theta_min feeds both the EB transport and the elliptic wall. Descriptor refuses
out-of-domain values structurally. Kokkos-gated for the runtime tier; the descriptor tier is pure.
"""
import pytest

from pops.mesh.masks import CutCell, transport_mask_thresholds


# --- descriptor tier (pure Python) -------------------------------------------

def test_cutcell_default_thresholds_are_zero_native_default():
    th = CutCell().thresholds()
    assert th == {"kappa_min": 0.0, "face_open_eps": 0.0, "cut_theta_min": 0.0}


def test_cutcell_thresholds_carry_configured_values():
    th = CutCell(kappa_min=0.05, face_open_eps=1e-5, cut_theta_min=5e-3).thresholds()
    assert th["kappa_min"] == pytest.approx(0.05)
    assert th["face_open_eps"] == pytest.approx(1e-5)
    assert th["cut_theta_min"] == pytest.approx(5e-3)


def test_cutcell_refuses_out_of_domain():
    with pytest.raises(ValueError):
        CutCell(kappa_min=0.0)
    with pytest.raises(ValueError):
        CutCell(kappa_min=2.0)
    with pytest.raises(ValueError):
        CutCell(cut_theta_min=1.5)
    with pytest.raises(ValueError):
        CutCell(face_open_eps=-1.0)
    with pytest.raises(ValueError):
        CutCell(face_open_eps=1.01)
    with pytest.raises(ValueError, match="finite"):
        CutCell(kappa_min=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        CutCell(face_open_eps=float("inf"))
    with pytest.raises(TypeError, match="real number"):
        CutCell(kappa_min=True)


def test_transport_mask_thresholds_require_typed_mask():
    from pops.mesh.masks import NoMask, Staircase
    with pytest.raises(TypeError, match="TransportMask"):
        transport_mask_thresholds("cutcell")
    assert transport_mask_thresholds(NoMask()) == {}
    assert transport_mask_thresholds(Staircase()) == {}


# --- runtime tier (needs _pops) ----------------------------------------------

pops = pytest.importorskip("pops")
from pops.runtime._engine_descriptors import (  # noqa: E402
    ChargeDensity, FluidState, IsothermalFlux, Model, NoSource, Spatial,
)
from pops.runtime._system import System  # noqa: E402  # ADC-545 advanced runtime seam


def _sim():
    sim = System(n=16, L=1.0, periodicity=(False, False))
    sim.add_equation("ion", Model(FluidState.isothermal(cs2=0.7), IsothermalFlux(),
                                  NoSource(), ChargeDensity(charge=1.0)),
                     # The native embedded-boundary facade currently provides a geometry-aware
                     # first-order reconstruction. Higher-order neighbor stencils are rejected
                     # instead of reading inactive cells.
                     spatial=Spatial(none=True))
    return sim


def _install_half_space(sim, mask: CutCell) -> None:
    from pops.analytic import coordinates
    from pops.domain import CartesianDomain
    from pops.mesh.geometry import LevelSet
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    frame = CartesianDomain("test-cutcell-thresholds", (0.0, 0.0), (1.0, 1.0)).frame()
    level_set = LevelSet(coordinates(frame)[0] - 0.5)
    ((opcodes, literals),) = lower_analytic_components(
        (level_set.expression.to_data(),), frame_id=frame.canonical_id
    )
    thresholds = mask.thresholds()
    sim._s._set_analytic_level_set(
        list(opcodes), list(literals), mask.lower(),
        thresholds["kappa_min"], thresholds["face_open_eps"], thresholds["cut_theta_min"],
    )


def test_default_eb_report_is_native_defaults():
    sim = _sim()
    _install_half_space(sim, CutCell())
    eb = sim.inspect().to_dict()["options"]["eb"]
    assert eb["enabled"] is True
    assert eb["geometry_mode"] == "cutcell"
    assert eb["kappa_min"] == pytest.approx(1e-2)
    assert eb["face_open_eps"] == pytest.approx(1e-6)
    assert eb["cut_theta_min"] == pytest.approx(1e-3)


def test_typed_cutcell_thresholds_reach_the_report():
    sim = _sim()
    _install_half_space(sim, CutCell(kappa_min=0.05, face_open_eps=1e-5, cut_theta_min=5e-3))
    eb = sim.inspect().to_dict()["options"]["eb"]
    assert eb["kappa_min"] == pytest.approx(0.05)
    assert eb["face_open_eps"] == pytest.approx(1e-5)
    assert eb["cut_theta_min"] == pytest.approx(5e-3)


def test_level_set_thresholds_refuse_out_of_domain_before_native_lowering():
    """Typed CutCell validation rejects invalid threshold data before any LevelSet lowering."""
    with pytest.raises(ValueError):
        CutCell(kappa_min=2.0)
    with pytest.raises(ValueError):
        CutCell(cut_theta_min=-1.0)


def main():
    test_cutcell_default_thresholds_are_zero_native_default()
    test_cutcell_thresholds_carry_configured_values()
    test_cutcell_refuses_out_of_domain()
    test_transport_mask_thresholds_require_typed_mask()
    test_default_eb_report_is_native_defaults()
    test_typed_cutcell_thresholds_reach_the_report()
    test_level_set_thresholds_refuse_out_of_domain_before_native_lowering()
    print("OK  ADC-615 cut-cell thresholds")


if __name__ == "__main__":
    main()
