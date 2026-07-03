"""ADC-591 structured runtime inspection reports."""

import json

import pytest
from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR, Uniform  # noqa: E402


def test_system_inspect_is_structured_and_array_free():
    sim = System(n=8, L=1.0, periodic=True)
    rep = sim.inspect()
    d = rep.to_dict()
    assert d["schema_version"] == 1
    assert d["report_type"] == "runtime_inspection"
    assert d["runtime"] == "system"
    assert d["runtime_environment"]["dimension"] == 2
    assert d["runtime_environment"]["precision"] == "double"
    assert d["capabilities"]["schema_version"] >= 1
    assert any(row["route_id"] == "parallel:custom_communicator"
               for row in d["capabilities"]["routes"])
    assert d["profile"]["source"] in ("snapshot", "text")
    assert d["history"] == []
    assert d["cache"] == []
    assert d["diagnostics"]["fallbacks"]["schema_version"] == 1
    assert any(row["key"] == "elliptic.fft.direct_dft"
               for row in d["diagnostics"]["fallbacks"]["entries"])
    assert d["options"]["defaults"]["newton"]["max_iters"] == 2
    assert d["options"]["poisson"]["solver"] == "geometric_mg"
    assert "array(" not in str(rep)
    assert json.loads(rep.to_json())["runtime"] == "system"


def test_amr_system_inspect_composes_amr_snapshot():
    sim = AmrSystem(n=8, L=1.0, periodic=True)
    rep = sim.inspect()
    d = rep.to_dict()
    assert d["runtime"] == "amr_system"
    assert d["amr"] is not None
    assert d["amr"]["max_levels"] == 2
    assert d["amr"]["ratio"] == 2
    assert d["runtime_environment"]["amr_refinement_ratio"] == 2
    assert d["options"]["defaults"]["amr"]["refinement_ratio"] == 2
    assert d["options"]["amr"]["disabled"] is True
    assert any(row["feature"] == "amr:refinement_ratio" and row["status"] == "partial"
               for row in d["limitations"])
    view_rep = sim.amr.inspect()
    view_d = view_rep.to_dict()
    # ADC-589: sim.amr.inspect() is the unified 4-part RuntimeInspection; the
    # hierarchy snapshot (which used to BE the whole report) is one component.
    assert set(view_d) == {"hierarchy", "patches", "regrid", "limitations"}
    assert view_d["hierarchy"]["ratio"] == 2
    assert "array(" not in str(rep)


def test_layout_inspect_reports_native_routes_and_limitations():
    uniform = Uniform(CartesianMesh(n=8))
    uniform_info = uniform.inspect()
    assert uniform_info["schema_version"] == 1
    assert uniform_info["report_type"] == "layout_inspection"
    assert uniform_info["capabilities"]["layout"] == "uniform"
    assert uniform_info["available"]["status"] == "yes"
    assert any(row["route_id"] == "layout:Uniform"
               for row in uniform_info["native_capabilities"]["routes"])
    assert uniform_info["amr_report"]["layout"] == "uniform"
    assert any(row["route_id"] == "mesh:2d_storage_arithmetic" and row["status"] == "partial"
               for row in uniform_info["native_capabilities"]["routes"])

    amr = AMR(CartesianMesh(n=8), max_levels=2, ratio=2)
    amr_info = amr.inspect()
    assert amr_info["capabilities"]["layout"] == "amr"
    routes = {row["route_id"]: row for row in amr_info["native_capabilities"]["routes"]}
    assert routes["amr:refinement_ratio"]["status"] == "partial"
    assert routes["amr:refinement_ratio"]["reason"]
    assert any(row["feature"] == "amr:refinement_ratio" for row in amr_info["limitations"])
