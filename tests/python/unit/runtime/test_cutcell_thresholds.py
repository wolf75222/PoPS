"""ADC-615 embedded-boundary / cut-cell thresholds end to end.

The typed pops.mesh.masks.CutCell(kappa_min, face_open_eps, cut_theta_min) lowers through
set_disc_domain to the native EbThresholds, surfaced in the System effective-options report; the
same cut_theta_min feeds both the EB transport and the elliptic wall. Descriptor refuses
out-of-domain values structurally. Kokkos-gated for the runtime tier; the descriptor tier is pure.
"""
import pytest

from pops.mesh.geometry import DiscDomain
from pops.mesh.masks import CutCell, disc_mode_thresholds


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


def test_disc_mode_thresholds_require_typed_mask():
    from pops.mesh.masks import NoMask, Staircase
    with pytest.raises(TypeError, match="TransportMask"):
        disc_mode_thresholds("cutcell")
    assert disc_mode_thresholds(NoMask()) == {}
    assert disc_mode_thresholds(Staircase()) == {}


# --- runtime tier (needs _pops) ----------------------------------------------

pops = pytest.importorskip("pops")
from pops.runtime.bricks import (
    ChargeDensity, FluidState, IsothermalFlux, Model, NoSource, Spatial,
)
from pops.runtime._system import System  # ADC-545 advanced runtime seam


def _sim():
    sim = System(n=16, L=1.0, periodic=False)
    sim.add_block("ion", Model(FluidState.isothermal(cs2=0.7), IsothermalFlux(),
                               NoSource(), ChargeDensity(charge=1.0)),
                  spatial=Spatial())
    return sim


def test_default_eb_report_is_native_defaults():
    sim = _sim()
    sim.set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3, mode=CutCell()))
    eb = sim.inspect().to_dict()["options"]["eb"]
    assert eb["enabled"] is True
    assert eb["geometry_mode"] == "cutcell"
    assert eb["kappa_min"] == pytest.approx(1e-2)
    assert eb["face_open_eps"] == pytest.approx(1e-6)
    assert eb["cut_theta_min"] == pytest.approx(1e-3)


def test_typed_cutcell_thresholds_reach_the_report():
    sim = _sim()
    sim.set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3,
                                   mode=CutCell(kappa_min=0.05, face_open_eps=1e-5,
                                                cut_theta_min=5e-3)))
    eb = sim.inspect().to_dict()["options"]["eb"]
    assert eb["kappa_min"] == pytest.approx(0.05)
    assert eb["face_open_eps"] == pytest.approx(1e-5)
    assert eb["cut_theta_min"] == pytest.approx(5e-3)


def test_native_set_disc_domain_refuses_out_of_domain():
    """Out-of-domain thresholds refuse BEFORE the native call: set_disc_domain carries no loose
    threshold kwargs; the typed CutCell descriptor validates at construction, so the inline
    route can never reach C++ with a bad value (validate-early by design)."""
    with pytest.raises(ValueError):
        _sim().set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3,
                                          mode=CutCell(kappa_min=2.0)))
    with pytest.raises(ValueError):
        _sim().set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3,
                                          mode=CutCell(cut_theta_min=-1.0)))


def main():
    test_cutcell_default_thresholds_are_zero_native_default()
    test_cutcell_thresholds_carry_configured_values()
    test_cutcell_refuses_out_of_domain()
    test_disc_mode_thresholds_require_typed_mask()
    test_default_eb_report_is_native_defaults()
    test_typed_cutcell_thresholds_reach_the_report()
    test_native_set_disc_domain_refuses_out_of_domain()
    print("OK  ADC-615 cut-cell thresholds")


if __name__ == "__main__":
    main()
