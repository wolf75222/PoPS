"""ADC-616 Berger-Rigoutsos clustering parameters end to end.

Both surface in the AmrSystem effective-options report (sim.inspect()) and refuse out-of-domain
values structurally. Kokkos-gated (self-skips without _pops); a small Serial AmrSystem is enough.
"""
import pytest
from pops.runtime._system import AmrSystem

pops = pytest.importorskip("pops")


# --- ADC-616 clustering ------------------------------------------------------

def test_clustering_default_reported_bit_identically():
    amr = AmrSystem(n=16, L=1.0, periodicity=(True, True))
    opts = amr.inspect().to_dict()["options"]["amr"]
    assert opts["cluster_min_efficiency"] == pytest.approx(0.7)
    assert opts["cluster_min_box_size"] == 1
    assert opts["cluster_max_box_size"] == 32


def test_clustering_override_visible_in_report():
    amr = AmrSystem(n=32, L=1.0, periodicity=(True, True), cluster_min_efficiency=0.9,
                    cluster_min_box_size=2, cluster_max_box_size=16)
    opts = amr.inspect().to_dict()["options"]["amr"]
    assert opts["cluster_min_efficiency"] == pytest.approx(0.9)
    assert opts["cluster_min_box_size"] == 2
    assert opts["cluster_max_box_size"] == 16


def test_clustering_descriptor_refuses_out_of_domain():
    from pops.mesh._amr import PatchClustering
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
