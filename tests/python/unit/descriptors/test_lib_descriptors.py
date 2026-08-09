"""Spec 3 pops.lib: a catalog of typed brick descriptors and IR macros.

pops.lib never computes in Python. A descriptor names a brick (native C++ id,
generated, macro or external) and carries its requirements / capabilities; the
codegen and runtime consume it. These tests check that the descriptors are
lightweight metadata that lower to native ids -- not numerical code.
"""
import pytest

import types as _t
_num = pytest.importorskip("pops.numerics")
_desc = pytest.importorskip("pops.descriptors")
# Spec 5: the catalogs moved out of pops.lib. This alias maps the old pops.lib attribute surface
# onto the new homes so the Spec-3 descriptor tests keep exercising the real (relocated) descriptors:
# the solver descriptors are the ONE public home pops.solvers (the pops.lib.solvers shim was
# removed, no back-compat alias); the solver-generation DSL is internal/experimental under
# pops.codegen.solvers (criterion 19); the spatial brick catalog under pops.numerics.spatial and
# the field brick catalog under pops.fields.catalog (criterion 7).
_solv = pytest.importorskip("pops.solvers")
_cs = pytest.importorskip("pops.codegen.solvers")
_flds = pytest.importorskip("pops.fields")
lib = _t.SimpleNamespace(
    riemann=_num.riemann.riemann, reconstruction=_num.reconstruction.reconstruction,
    limiters=_num.limiters, projections=_num.projections.projections,
    BrickDescriptor=_desc.BrickDescriptor, external=_desc.external,
    load_cpp_library=_desc.load_cpp_library,
    _register_manifest=_desc._register_manifest,
    _clear_external_catalog=_desc._clear_external_catalog,
    solvers=_solv.solvers, preconditioners=_solv.preconditioners, solver=_cs.solver,
    build_solver_ir=_cs.build_solver_ir, generate_solver_cpp=_cs.generate_solver_cpp,
    SolverContext=_cs.SolverContext, SolverIR=_cs.SolverIR,
    spatial=_num.spatial, fields=_flds.catalog,
)


def test_riemann_hllc_is_a_native_descriptor():
    d = lib.riemann.HLLC()
    assert d.brick_type == "native"
    assert d.available().ok
    assert d.native_id == "pops::HLLCFlux"   # the EXACT C++ symbol (namespace pops)
    assert d.scheme == "hllc"               # the runtime scheme string


def test_riemann_native_ids_are_exact():
    # Guard against the wrong-namespace overclaim: ids must be the real pops:: symbols.
    assert lib.riemann.Rusanov().native_id == "pops::RusanovFlux"
    assert lib.riemann.HLL().native_id == "pops::HLLFlux"
    assert lib.riemann.Roe().native_id == "pops::RoeFlux"


def test_riemann_recovery_is_the_exact_prepared_native_policy():
    descriptor = lib.riemann.Recovery(
        primary=lib.riemann.Roe(),
        fallbacks=(lib.riemann.HLL(), lib.riemann.Rusanov()),
    )

    assert descriptor.brick_type == "native"
    assert descriptor.scheme == "roe_hll_rusanov_recovery"
    assert descriptor.native_id == (
        "pops::PreparedRiemannRecoveryPolicy<pops::RoeFlux,pops::HLLFlux,"
        "pops::RusanovFlux,pops::RejectRiemannRecovery>"
    )
    assert descriptor.options["recovery_order"] == (
        "roe", "hll", "rusanov", "reject"
    )
    assert set(descriptor.requirements["capabilities"]) >= {
        "physical_flux", "provider_pack", "stability_bound", "wave_speeds",
        "roe_dissipation",
    }


@pytest.mark.parametrize(
    ("primary", "fallbacks", "message"),
    (
        ("roe", (lib.riemann.HLL(), lib.riemann.Rusanov()), "typed built-in"),
        (lib.riemann.Roe(), [lib.riemann.HLL(), lib.riemann.Rusanov()], "requires a tuple"),
        (
            lib.riemann.Roe(),
            (lib.riemann.HLL(), lib.riemann.HLL()),
            "candidates must be unique",
        ),
        (
            lib.riemann.HLL(),
            (lib.riemann.Roe(), lib.riemann.Rusanov()),
            "supports exactly primary=Roe()",
        ),
        (
            lib.riemann.Roe(),
            (lib.riemann.HLL(waves=_num.riemann.waves.ExplicitPair()),
             lib.riemann.Rusanov()),
            "carries candidate options",
        ),
    ),
)
def test_riemann_recovery_refuses_non_exact_sequences(primary, fallbacks, message):
    with pytest.raises((TypeError, ValueError), match=message):
        lib.riemann.Recovery(primary=primary, fallbacks=fallbacks)


def test_riemann_recovery_refuses_external_and_forged_native_candidates():
    external = lib.BrickDescriptor(
        "acme.roe", "external_cpp", category="riemann", native_id="acme_roe",
        scheme="roe",
    )
    with pytest.raises(ValueError, match="refuses external/non-native"):
        lib.riemann.Recovery(
            primary=external,
            fallbacks=(lib.riemann.HLL(), lib.riemann.Rusanov()),
        )

    forged = lib.BrickDescriptor(
        "roe", "native", category="riemann", native_id="pops::RoeFlux", scheme="roe",
        requirements={"capabilities": []},
    )
    with pytest.raises(ValueError, match="not the catalog-authenticated"):
        lib.riemann.Recovery(
            primary=forged,
            fallbacks=(lib.riemann.HLL(), lib.riemann.Rusanov()),
        )


def test_riemann_recovery_public_validation_refuses_polar_without_substitution():
    descriptor = lib.riemann.Recovery(
        primary=lib.riemann.Roe(),
        fallbacks=(lib.riemann.HLL(), lib.riemann.Rusanov()),
    )

    availability = lib.riemann.available(descriptor, {"layout": "polar"})
    assert availability.ok is False
    assert "catalog polar_ok=false" in availability.reason
    assert "no fallback or candidate substitution" in availability.reason
    with pytest.raises(ValueError, match="unavailable on annular polar geometry"):
        lib.riemann.validate(descriptor, {"layout": "polar"})


def test_reconstruction_weno5z_is_native():
    d = lib.reconstruction.WENO5Z()
    assert d.brick_type == "native"
    assert d.native_id == "pops::Weno5"      # pops::Weno5 IS the WENO5-Z reconstruction
    assert d.scheme == "weno5"


def test_unwired_placeholder_bricks_are_absent_from_final_catalogs():
    for catalog, name in (
        (lib.fields, "Poisson"),
        (lib.preconditioners, "Jacobi"),
    ):
        assert not hasattr(catalog, name)

    newton = lib.solvers.Newton()
    assert newton.available().ok is True
    assert newton.native_id == "pops::FieldNewtonSolver"


def test_mc_limiter_is_an_executable_native_descriptor():
    descriptor = lib.limiters.MC()
    assert descriptor.available().ok is True
    assert descriptor.native_id == "pops::MC"
    assert descriptor.scheme == "mc"


def test_available_native_ids_exist_and_are_namespaced():
    for d in (lib.fields.GeometricMG(), lib.solvers.CG(max_iter=200),
              lib.solvers.GMRES(max_iter=200),
              lib.projections.positivity()):
        assert d.available().ok
        assert d.native_id.startswith("pops::")


def test_riemann_descriptors_compute_nothing():
    # A descriptor exposes metadata only -- no eval / compile / __call__ numeric path.
    d = lib.riemann.Rusanov()
    assert not hasattr(d, "eval")
    assert not hasattr(d, "compile")
    assert d.scheme == "rusanov"
    # frozen-ish: the same descriptor twice compares equal (value type)
    assert lib.riemann.Rusanov() == lib.riemann.Rusanov()


def test_brick_identity_includes_requirements_capabilities_and_nested_options():
    base = lib.BrickDescriptor(
        "acme.flux", "external_cpp", category="riemann", native_id="acme_flux",
        scheme="user", requirements={"capabilities": []},
        capabilities={"provides": ["cpu"]},
        options={"method": {"weights": [1, 2], "limiter": lib.limiters.Minmod()}},
    )
    same = lib.BrickDescriptor(
        "acme.flux", "external_cpp", category="riemann", native_id="acme_flux",
        scheme="user", requirements={"capabilities": []},
        capabilities={"provides": ["cpu"]},
        options={"method": {"weights": [1, 2], "limiter": lib.limiters.Minmod()}},
    )
    needs_waves = lib.BrickDescriptor(
        "acme.flux", "external_cpp", category="riemann", native_id="acme_flux",
        scheme="user", requirements={"capabilities": ["wave_speeds"]},
        capabilities={"provides": ["cpu"]},
        options={"method": {"weights": [1, 2], "limiter": lib.limiters.Minmod()}},
    )
    different_capability = lib.BrickDescriptor(
        "acme.flux", "external_cpp", category="riemann", native_id="acme_flux",
        scheme="user", requirements={"capabilities": []},
        capabilities={"provides": ["gpu"]},
        options={"method": {"weights": [1, 2], "limiter": lib.limiters.Minmod()}},
    )

    assert base == same
    assert hash(base) == hash(same)
    assert base != needs_waves
    assert base != different_capability
    assert len({base, same, needs_waves, different_capability}) == 3


def test_field_solver_descriptor_carries_options():
    d = lib.fields.GeometricMG(tolerance=1e-10, max_iters=200)
    assert d.brick_type == "native"
    assert d.options["tolerance"] == 1e-10
    assert d.options["max_iters"] == 200


def test_solver_descriptors():
    assert lib.solvers.BiCGStab(max_iter=200).scheme == "bicgstab"
    assert lib.solvers.GMRES(max_iter=200).scheme == "gmres"
    assert lib.solvers.CG(max_iter=200).scheme == "cg"


def test_user_riemann_is_external():
    # A User brick must be loaded first (ADC-463); registering its manifest then makes
    # riemann.User(id) surface an external_cpp descriptor.
    import json
    # ADC-611 : le schema strict versionne exige schema_version + chaque champ d'entree.
    from pops.runtime._engine_descriptors import abi_key
    lib._register_manifest(json.dumps(
        {"schema_version": _desc.BRICK_MANIFEST_SCHEMA_VERSION, "abi_key": abi_key(),
         "annotations": {},
         "bricks": [{"id": "my_hllc_variant", "category": "riemann",
                     "requirements": "", "capabilities": "",
                     "native_id": "my_hllc_variant", "supported_layouts": "",
                     "supported_platforms": "", "params": "", "options": "",
                     "exported_symbols": ""}]}))
    try:
        d = lib.riemann.User("my_hllc_variant")
        assert d.brick_type == "external_cpp"
        assert d.native_id == "my_hllc_variant"
    finally:
        lib._clear_external_catalog()


def test_descriptor_requirements_present():
    # HLLC requires the model HLLC capabilities; Rusanov only needs a max wave speed.
    assert "hllc_star_state" in lib.riemann.HLLC().requirements.get("capabilities", [])
    assert lib.riemann.Rusanov().requirements.get("capabilities") == [
        "physical_flux", "provider_pack", "stability_bound"]
