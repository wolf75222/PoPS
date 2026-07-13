"""ADC-604 structured fallback/degraded-route diagnostics."""

import pytest
from pops.runtime.system import System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")


def _keys(report):
    return {row["key"]: row for row in report["entries"]}


def test_fallback_diagnostics_report_has_explicit_policies():
    pops.reset_fallback_diagnostics()
    report = pops.fallback_diagnostics_report()
    rows = _keys(report)

    assert report["schema_version"] == 1
    assert "elliptic.fft.direct_dft" in rows
    assert rows["elliptic.fft.direct_dft"]["policy"] == "allowed_with_counter"
    assert rows["elliptic.fft.direct_dft"]["performance_degraded"] is True
    assert rows["linalg.dense_eig.gershgorin"]["default_action"] == "return_bound_not_spectrum"
    assert rows["spatial.positivity.order1_face"]["policy"] == "explicit_opt_in"
    assert rows["runtime.limiter_unknown_muscl_ghost"]["policy"] == "refuse_final_route"


def test_runtime_inspect_includes_fallback_diagnostics_and_configured_opt_in():
    sim = System(n=8, L=1.0, periodic=True)
    model = pops.Model(
        pops.FluidState.isothermal(cs2=0.7),
        pops.IsothermalFlux(),
        pops.NoSource(),
        pops.ChargeDensity(),
    )
    sim.block("ion", model, spatial=pops.Spatial(positivity_floor=1e-12))

    report = sim.inspect().to_dict()
    fallbacks = report["diagnostics"]["fallbacks"]
    rows = _keys(fallbacks)
    configured = fallbacks["configured"]

    assert rows["spatial.positivity.order1_face"]["default_action"].startswith("disabled")
    assert any(row["key"] == "spatial.positivity.order1_face" and row["block"] == "ion"
               for row in configured)
    assert "fallbacks=" in str(sim.inspect())
