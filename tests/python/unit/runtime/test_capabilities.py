"""Coherence contract for ``pops.runtime.doctor.capabilities()`` (ADC-297).

The runtime doctor capability report is the source of truth for what the runtime can dispatch
(Riemann fluxes, time methods, stability bounds, Poisson, geometry, Schur, DSL backends,
IO, AMR layout, aux). It is a hand-written dict, so it can silently drift from the gates it
claims to mirror. These checks pin the capability surface to facts that are verified
elsewhere in the suite, so that changing a capability forces either a test update or a
documentation update:

  T1 - the published top-level keys stay present (the doc and the limitations pages key off
       them; a vanished key means a stale reference).
  T2 - the Riemann surface matches the dispatch gates: all four public providers have one
       capability-gated Cartesian and AMR route; the retired polar System is absent.
  T3 - backends_dsl MPI/AMR flags agree (truthiness) with the _BACKEND_CAPS table that
       actually drives backend selection; catches drift between the two tables.
  T4 - no capability family advertises the retired ``system_polar`` runtime.
  T5 - the AMR Schur stage is advertised as implemented (Phase 4a), not "to be done";
       guards the ALGORITHMS.md section 25 "the implementation does not exist" regression.
  T6 - the exact selected native spatial rank is published as a structured scalar and agrees
       with the loaded 1D/2D/3D specialization.
  T7 - the AMR regrid variable is advertised as selectable by name/role (ADC-296 / ADR-0001
       Decision 5), on native and compiled runtime blocks through one prepared graph; guards
       the "regrid is component-0 only" doc regression now that a selector exists.

The test reads only ``capabilities()`` and the backend table.  It explicitly selects the native
specialization requested by the test environment before importing runtime modules; it does not
build or run a model.
"""
import os

from pops._native_selector import select_native_dimension
from pops.codegen._compile import _BACKEND_CAPS
from pops.runtime.doctor import capabilities


_TEST_NATIVE_DIMENSION = os.environ.get("POPS_NATIVE_DIM")
if _TEST_NATIVE_DIMENSION not in {"1", "2", "3"}:
    raise RuntimeError("test_capabilities requires POPS_NATIVE_DIM=1, 2, or 3")
select_native_dimension(int(_TEST_NATIVE_DIMENSION))

EXPECTED_TOP_KEYS = {
    "dimension", "riemann", "time", "stability_policy", "poisson", "geometry", "schur",
    "backends_dsl", "io", "amr_layout", "aux", "regrid", "precision",
    "runtime_environment",
}


def test_top_level_keys_present():
    caps = capabilities()
    missing = EXPECTED_TOP_KEYS - set(caps)
    assert not missing, "capabilities() lost published top-level key(s): %s" % sorted(missing)


def test_riemann_surface_matches_dispatch():
    # ADC-752: each provider has one capability-gated route on every executable geometry.
    riemann = capabilities()["riemann"]
    expected = ["rusanov", "hll", "hllc", "roe"]
    assert riemann["system_cartesian"] == expected, riemann["system_cartesian"]
    assert riemann["amr"] == expected, riemann["amr"]


def test_backends_dsl_flags_match_backend_caps():
    caps_b = capabilities()["backends_dsl"]
    assert set(_BACKEND_CAPS) == {"production"}
    assert set(caps_b) == {"default", "production"}
    ref = _BACKEND_CAPS["production"]
    got = caps_b["production"]
    assert bool(got["mpi"]) == bool(ref["mpi"])
    assert bool(got["amr"]) == bool(ref["amr"])


def test_retired_polar_system_is_absent_from_runtime_capabilities():
    caps = capabilities()
    for family in ("riemann", "time", "stability_policy", "poisson", "geometry", "schur"):
        assert "system_polar" not in caps[family], family


def test_amr_condensed_program_advertised_implemented():
    amr_schur = capabilities()["schur"]["amr"]
    for fragment in ("CompositeTensorFAC", "gather-all-levels", "reconstruct-all-levels"):
        assert fragment in amr_schur, \
            "schur.amr must advertise the implemented hierarchy route, got: %r" % amr_schur
    for stale_limit in ("no native", "not implemented", "the implementation does not"):
        assert stale_limit not in amr_schur


def test_dimension_matches_selected_native_specialization():
    from pops import _pops

    # The rank is a separate structured scalar, not geometry metadata. It must be the immutable
    # 1D/2D/3D specialization selected before the runtime is imported.
    caps = capabilities()
    dim = caps["dimension"]
    assert dim == _pops.__native_dimension__
    # bool is a subclass of int in Python; pin to a plain int so True / 2.0 / "2" cannot pass.
    assert isinstance(dim, int) and not isinstance(dim, bool), \
        "capabilities()['dimension'] should be a plain int scalar, got %r" % (dim,)
    assert "dimension" not in caps["geometry"], \
        "the dimension invariant must stay a separate top-level key, not nested under geometry"


def test_runtime_environment_and_precision_facts():
    from pops import _pops

    caps = capabilities()
    precision = caps["precision"]
    assert precision["real"] == "double"
    assert precision["real_bytes"] == 8
    assert precision["supports_single_precision"] is False
    assert precision["supports_mixed_precision"] is False
    env = caps["runtime_environment"]
    assert env["dimension"] == _pops.__native_dimension__
    assert env["amr_refinement_ratio"] is None
    assert env["amr_refinement_ratio_selection"] == "hierarchy_exact_rank"
    assert env["amr_refinement_ratio_rank"] == _pops.__native_dimension__
    assert env["precision"] == "double"
    assert env["supports_custom_communicator"] is False
    assert env["communicator"] in ("serial", "MPI_COMM_WORLD", "unknown")


def test_regrid_prepared_graph_contract_advertised():
    # ADC-672: one resolved graph owns state/field selection and logical composition. The retired
    # scalar selector must not remain advertised as a parallel runtime route.
    regrid = capabilities()["regrid"]
    assert "variable_selector" not in regrid
    assert regrid["authority"] == \
        "resolved AMRTagging graph installed as one prepared native program"
    assert regrid["state_and_field_leaves"] == \
        "block-qualified exact Handle identities"
    assert regrid["logical_nodes"] == \
        "prepared not/any/all bytecode; no scalar threshold fallback"


def test_geometry_advertises_ranked_rectangular_cells():
    geometry = capabilities()["geometry"]["system_cartesian"]
    assert "ranked n cells" in geometry
    assert "rectangular allowed" in geometry
    assert "square n x n" not in geometry
    amr = capabilities()["geometry"]["amr"]
    assert "exact native-rank" in amr
    assert "process-global AMR ratio" in amr


if __name__ == "__main__":
    test_top_level_keys_present()
    test_riemann_surface_matches_dispatch()
    test_backends_dsl_flags_match_backend_caps()
    test_retired_polar_system_is_absent_from_runtime_capabilities()
    test_dimension_matches_selected_native_specialization()
    test_runtime_environment_and_precision_facts()
    test_regrid_prepared_graph_contract_advertised()
    test_geometry_advertises_ranked_rectangular_cells()
    print("test_capabilities : OK (top keys, riemann surface, backends_dsl, polar retirement, "
          "native dimension and prepared regrid graph)")
