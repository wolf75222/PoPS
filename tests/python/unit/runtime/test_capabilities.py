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

The test is pure Python: it only reads ``capabilities()`` and the backend table, so it
needs the _pops extension to import but does not build or run any model.
"""
from pops.codegen._compile import _BACKEND_CAPS
from pops.physics.aux import AUX_CANONICAL_NAMES, AUX_NAMED_MAX, aux_layout
from pops.runtime.doctor import capabilities

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
    assert env["amr_refinement_ratio"] == 2
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


def test_aux_named_surface_and_limit_parity():
    # ADC-291: named aux is advertised on Cartesian System AND AMR (single + multi block),
    # no longer "cartesian System only". The remaining compile-time limit (kAuxMaxExtra) is published
    # as an introspectable scalar and MUST match BOTH the C++ source (_pops.__aux_max_extra__) and the
    # DSL mirror (AUX_NAMED_MAX) -- this pins the hand-maintained Python<->C++ mirror so it cannot
    # silently drift (the historical #51-class risk the issue calls out).
    from pops import _pops
    layout = aux_layout(_pops.__native_dimension__)
    named = capabilities()["aux"]["named"]
    assert set(named["backends"]) >= {
        "system_cartesian", "amr_single_block", "amr_multi_block",
    }, named["backends"]
    assert "system_polar" not in named["backends"]
    # the limit is the SINGLE C++ source, mirrored by the DSL constant.
    assert named["limit"] == _pops.__aux_max_extra__ == AUX_NAMED_MAX, \
        "aux named limit drift: caps=%r, C++=%r, dsl=%r" % (
            named["limit"], _pops.__aux_max_extra__, AUX_NAMED_MAX)
    # the aux ghost width is explicit (the configurable-radius mechanism is a documented follow-up).
    assert named["halo_radius"] == 1, named["halo_radius"]
    # the other mirrored aux constants stay coherent C++ <-> DSL.
    assert _pops.__aux_named_base__ == layout.named_base, "ranked aux named base drift"
    assert _pops.__aux_base_comps__ == layout.base_components, "ranked aux base width drift"
    assert _pops.__aux_max_comps__ == _pops.__aux_named_base__ + _pops.__aux_max_extra__
    # The selected C++ rank and the Python rank-qualified authority expose the same table.
    assert dict(_pops.__aux_canonical__) == dict(layout.canonical), \
        "C++ aux_names table != Python ranked aux layout: %r vs %r" % (
            dict(_pops.__aux_canonical__), dict(layout.canonical))
    assert set(layout.canonical).issubset(AUX_CANONICAL_NAMES)
    # no stale "cartesian System only" claim survives in the aux surface.
    blob = repr(capabilities()["aux"]).lower()
    assert "cartesian system only" not in blob, "stale 'cartesian System only' aux claim"


if __name__ == "__main__":
    test_top_level_keys_present()
    test_riemann_surface_matches_dispatch()
    test_backends_dsl_flags_match_backend_caps()
    test_retired_polar_system_is_absent_from_runtime_capabilities()
    test_dimension_matches_selected_native_specialization()
    test_runtime_environment_and_precision_facts()
    test_regrid_prepared_graph_contract_advertised()
    test_aux_named_surface_and_limit_parity()
    print("test_capabilities : OK (top keys, riemann surface, backends_dsl, polar retirement, "
          "native dimension, prepared regrid graph, aux named surface + limit parity)")
