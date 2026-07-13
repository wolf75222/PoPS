"""ADC-616 Berger-Rigoutsos clustering parameters end to end.

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
    sim.block("ne", model=_model(), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
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

def main():
    test_clustering_default_reported_bit_identically()
    test_clustering_override_visible_in_report()
    test_clustering_descriptor_refuses_out_of_domain()
    print("OK ADC-616")


if __name__ == "__main__":
    main()
