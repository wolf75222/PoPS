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


def test_reconstruction_weno5z_is_native():
    d = lib.reconstruction.WENO5Z()
    assert d.brick_type == "native"
    assert d.native_id == "pops::Weno5"      # pops::Weno5 IS the WENO5-Z reconstruction
    assert d.scheme == "weno5"


def test_catalogued_but_unwired_bricks_are_marked_unavailable():
    # No native symbol is fabricated: planned bricks refuse via availability(), empty id.
    for d in (lib.fields.Poisson(), lib.solvers.Newton(),
              lib.preconditioners.Jacobi(), lib.limiters.MC()):
        assert d.available().ok is False
        assert d.native_id == ""


def test_available_native_ids_exist_and_are_namespaced():
    for d in (lib.fields.GeometricMG(), lib.solvers.CG(max_iter=200),
              lib.solvers.GMRES(max_iter=200),
              lib.solvers.Schur(), lib.projections.positivity()):
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
    # ADC-544 : le schema passe a la v2 (les champs v2 sont optionnels; native_id defaut = id).
    lib._register_manifest(json.dumps(
        {"schema_version": 2,
         "bricks": [{"id": "my_hllc_variant", "category": "riemann",
                     "requirements": "", "capabilities": ""}]}))
    try:
        d = lib.riemann.User("my_hllc_variant")
        assert d.brick_type == "external_cpp"
        assert d.native_id == "my_hllc_variant"
    finally:
        lib._clear_external_catalog()


def test_descriptor_requirements_present():
    # HLLC requires the model HLLC capabilities; Rusanov only needs a max wave speed.
    assert "hllc_star_state" in lib.riemann.HLLC().requirements.get("capabilities", [])
    assert lib.riemann.Rusanov().requirements.get("capabilities") == ["max_wave_speed"]
