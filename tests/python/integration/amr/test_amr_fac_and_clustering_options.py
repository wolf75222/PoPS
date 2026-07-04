"""ADC-614 (FAC Poisson AMR options) + ADC-616 (Berger-Rigoutsos clustering params) end to end.

Both surface in the AmrSystem effective-options report (sim.inspect()) and refuse out-of-domain
values structurally. Kokkos-gated (self-skips without _pops); a small Serial AmrSystem is enough.
"""
import numpy as np
import pytest
from pops.runtime.system import AmrSystem

pops = pytest.importorskip("pops")


def _model():
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                      source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def _built(**cfg):
    sim = AmrSystem(n=32, L=1.0, periodic=True, regrid_every=2, coarse_max_grid=16, **cfg)
    sim.add_block("ne", model=_model(), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_refinement(threshold=0.5)
    ne = np.ones((32, 32))
    ne[10:22, 10:22] = 5.0
    sim.set_density("ne", ne)
    for _ in range(3):
        sim.step_cfl(0.4)
    return sim


# --- ADC-616 clustering ------------------------------------------------------

def test_clustering_default_reported_bit_identically():
    amr = AmrSystem(n=16, L=1.0, periodic=True)
    opts = amr.inspect().to_dict()["options"]["amr"]
    assert opts["cluster_min_efficiency"] == pytest.approx(0.7)
    assert opts["cluster_min_box_size"] == 1
    assert opts["cluster_max_box_size"] == 32


def test_clustering_override_visible_in_report():
    amr = AmrSystem(n=32, L=1.0, periodic=True, cluster_min_efficiency=0.9,
                    cluster_min_box_size=2, cluster_max_box_size=16)
    opts = amr.inspect().to_dict()["options"]["amr"]
    assert opts["cluster_min_efficiency"] == pytest.approx(0.9)
    assert opts["cluster_min_box_size"] == 2
    assert opts["cluster_max_box_size"] == 16


def test_clustering_descriptor_refuses_out_of_domain():
    from pops.mesh.amr import PatchClustering
    with pytest.raises(ValueError):
        PatchClustering(min_efficiency=0.0)
    with pytest.raises(ValueError):
        PatchClustering(min_efficiency=1.5)
    with pytest.raises(ValueError):
        PatchClustering(min_box_size=0)
    with pytest.raises(ValueError):
        PatchClustering(min_box_size=64, max_box_size=32)


# --- ADC-614 FAC options -----------------------------------------------------

def test_fac_options_default_reported_on_schur_stage():
    """A default condensed Schur AMR stage reports the kFAC* defaults in its effective report."""
    amr = AmrSystem(n=16, L=1.0, periodic=True)
    amr.add_block("ion", model=pops.Model(state=pops.Compressible(gamma=1.4),
                                          transport=pops.EulerFlux(),
                                          source=pops.NoSource(),
                                          elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.Spatial(), time=pops.Explicit())
    amr.set_magnetic_field(np.ones((16, 16)))
    amr.set_source_stage("ion", "electrostatic_lorentz", 0.5, 1.0)
    stages = amr.inspect().to_dict()["options"]["source_stages"]
    assert stages, "the condensed Schur stage must appear in the report"
    st = stages[0]
    assert st["effective_fac_max_iters"] == 30
    assert st["effective_fac_fine_sweeps"] == 400
    assert st["effective_fac_tol"] == pytest.approx(1e-9)
    assert st["effective_fac_coarse_rel_tol"] == pytest.approx(1e-12)
    assert st["effective_fac_coarse_cycles"] == 100


def test_fac_options_override_visible_and_refused_out_of_domain():
    amr = AmrSystem(n=16, L=1.0, periodic=True)
    amr.add_block("ion", model=pops.Model(state=pops.Compressible(gamma=1.4),
                                          transport=pops.EulerFlux(),
                                          source=pops.NoSource(),
                                          elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.Spatial(), time=pops.Explicit())
    amr.set_magnetic_field(np.ones((16, 16)))
    amr.set_source_stage("ion", "electrostatic_lorentz", 0.5, 1.0,
                         fac_max_iters=12, fac_fine_sweeps=200, fac_tol=1e-7,
                         fac_coarse_rel_tol=1e-10, fac_coarse_cycles=40, fac_verbose=True)
    st = amr.inspect().to_dict()["options"]["source_stages"][0]
    assert st["effective_fac_max_iters"] == 12
    assert st["effective_fac_fine_sweeps"] == 200
    assert st["effective_fac_tol"] == pytest.approx(1e-7)
    assert st["effective_fac_coarse_rel_tol"] == pytest.approx(1e-10)
    assert st["effective_fac_coarse_cycles"] == 40
    assert st["fac_verbose"] is True

    bad = AmrSystem(n=16, L=1.0, periodic=True)
    bad.add_block("ion", model=pops.Model(state=pops.Compressible(gamma=1.4),
                                          transport=pops.EulerFlux(), source=pops.NoSource(),
                                          elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.Spatial(), time=pops.Explicit())
    bad.set_magnetic_field(np.ones((16, 16)))
    with pytest.raises((RuntimeError, ValueError)):
        bad.set_source_stage("ion", "electrostatic_lorentz", 0.5, 1.0, fac_tol=2.0)


def test_condensed_schur_descriptor_refuses_out_of_domain_fac():
    from pops.runtime._bricks_time import CondensedSchur
    with pytest.raises(ValueError):
        CondensedSchur(fac_tol=2.0)
    with pytest.raises(ValueError):
        CondensedSchur(fac_max_iters=0)
    with pytest.raises(ValueError):
        CondensedSchur(fac_coarse_rel_tol=1.5)


def main():
    test_clustering_default_reported_bit_identically()
    test_clustering_override_visible_in_report()
    test_clustering_descriptor_refuses_out_of_domain()
    test_fac_options_default_reported_on_schur_stage()
    test_fac_options_override_visible_and_refused_out_of_domain()
    test_condensed_schur_descriptor_refuses_out_of_domain_fac()
    print("OK  ADC-614 + ADC-616")


if __name__ == "__main__":
    main()
