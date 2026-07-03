"""Spec 5 sec.8.15 / criterion 22: the typed compile-backend + execution-platform descriptors.

These checks pin the typed backend/platform surface added under epic ADC-479:

  - the backend descriptors (Production / AOT / JIT) lower to the legacy backend string
    ("production" / "aot" / "prototype") the compile drivers already key on, and expose the same
    token via ``.scheme``;
  - ``lower_backend`` is ADDITIVE and TRANSPARENT: a typed descriptor lowers to its string, while a
    plain string / None / any other value passes through unchanged so the compile driver's existing
    ``backend not in _BACKENDS`` guard stays the single source of the unknown-backend ValueError;
  - the consumer (``compile_problem`` / ``compile_model``) accepts BOTH a string and a typed
    backend -- a typed AOT() hits the SAME production-only guard as the string "aot", proving the
    lowering runs before the guard;
  - the higher-level AUTHORING FACADES the docs teach (the PDE ``Model`` facade, the ``HybridModel``
    composer, the ``CoupledSource`` compiler) accept a typed backend too -- each validated/stored
    ``backend`` itself before reaching a driver, so each wires the same additive lowering;
  - ``Production(platform=KokkosOpenMP())`` records the platform (inert) and refuses a string;
  - the platform descriptors (KokkosSerial / KokkosOpenMP / KokkosCuda / KokkosHIP / MPI) declare
    host/gpu/mpi capabilities and answer ``available()`` with an EXPLAINABLE Availability that names
    a missing build flag, never a bare bool.

Pure Python: it imports the inert authoring/codegen packages (the compiled _pops loads as a side
effect of ``import pops`` -- platform availability reads its build flags -- but no model is built
or run, and no compiler is invoked).
"""

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import AOT, JIT, Production, lower_backend  # noqa: E402
from pops.codegen.backends import BACKEND_DESCRIPTORS, _Backend  # noqa: E402
from pops.descriptors import Availability, Descriptor  # noqa: E402
from pops.runtime.platforms import (  # noqa: E402
    KokkosCuda, KokkosHIP, KokkosOpenMP, KokkosSerial, MPI)


# --- backend descriptors lower to the legacy string -------------------------------------------
def test_backend_descriptors_lower_to_legacy_string():
    assert Production().lower() == "production"
    assert AOT().lower() == "aot"
    assert JIT().lower() == "prototype"


def test_backend_scheme_matches_lower():
    for cls in (Production, AOT, JIT):
        desc = cls()
        assert desc.scheme == desc.lower()


def test_backend_descriptor_is_inert_typed_descriptor():
    prod = Production()
    assert isinstance(prod, Descriptor)
    assert isinstance(prod, _Backend)
    assert prod.category == "backend"
    # capabilities come from the honest native backend table (cpu/mpi/amr/gpu).
    caps = prod.capabilities().to_dict()
    assert caps["cpu"] is True and caps["mpi"] is True and caps["amr"] is True
    assert JIT().capabilities().to_dict()["mpi"] is False
    # inspect is a plain dict carrying the platform slot.
    record = prod.inspect()
    assert record["category"] == "backend"
    assert record["options"]["backend"] == "production"
    assert record["platform"] is None


def test_backend_registry_maps_token_to_class():
    assert BACKEND_DESCRIPTORS["production"] is Production
    assert BACKEND_DESCRIPTORS["aot"] is AOT
    assert BACKEND_DESCRIPTORS["prototype"] is JIT


# --- lower_backend is additive (string AND typed) ---------------------------------------------
def test_lower_backend_passes_string_through():
    for token in ("production", "aot", "prototype", "auto"):
        assert lower_backend(token) == token


def test_lower_backend_lowers_typed():
    assert lower_backend(Production()) == "production"
    assert lower_backend(AOT()) == "aot"
    assert lower_backend(JIT()) == "prototype"


def test_lower_backend_passes_non_descriptor_through():
    # lower_backend is a TRANSPARENT coercion: it lowers a typed descriptor and returns anything
    # else (None, a wrong type, an unknown string) UNCHANGED, so the compile entry point's existing
    # `backend not in _BACKENDS` guard stays the single source of the unknown-backend ValueError.
    # A guardrail such as test_dsl_compile_facade passes backend=None expecting that ValueError, so
    # lower_backend must NOT pre-empt it with a TypeError of its own.
    assert lower_backend(None) is None
    assert lower_backend(123) == 123
    assert lower_backend("nope") == "nope"


# --- the consumer accepts BOTH a string and a typed backend -----------------------------------
def _guard_error(backend):
    """Run compile_problem far enough to hit the backend guard; return the ValueError text."""
    from pops.codegen.compile_drivers import compile_problem
    try:
        compile_problem(model=None, time=None, backend=backend)
    except ValueError as err:
        return str(err)
    raise AssertionError("compile_problem(backend=%r) did not raise" % (backend,))


def test_compile_problem_accepts_string_and_typed_production():
    # Both reach PAST the production-only guard and fail identically at the time= check.
    string_msg = _guard_error("production")
    typed_msg = _guard_error(Production())
    assert "time must be" in string_msg
    assert string_msg == typed_msg


def test_compile_problem_typed_non_production_hits_same_guard_as_string():
    # A typed AOT() lowers to "aot" BEFORE the production-only guard -> the SAME message the
    # string "aot" produces (proving the additive lowering runs first).
    string_msg = _guard_error("aot")
    typed_msg = _guard_error(AOT())
    assert string_msg == "compiled time programs require backend='production'"
    assert typed_msg == string_msg


def test_compile_problem_accepts_typed_backend_with_platform():
    # Recording a platform must not change the lowered backend string.
    msg = _guard_error(Production(platform=KokkosOpenMP()))
    assert "time must be" in msg


def test_compile_model_lowers_typed_backend_past_unknown_guard():
    # compile_model's unknown-backend guard sees the LOWERED string, so a typed JIT() is not
    # rejected as unknown; it proceeds and trips later on the fake model (no .name attribute).
    from pops.codegen.compile_drivers import compile_model

    class _FakeModel:
        def _check_require_metadata(self, *a, **k):
            pass

    with pytest.raises(AttributeError):
        compile_model(_FakeModel(), backend=JIT())
    # A genuinely unknown string still raises the unknown-backend ValueError (additive, not lossy).
    with pytest.raises(ValueError, match="backend 'nope' unknown"):
        compile_model(_FakeModel(), backend="nope")


# --- the authoring FACADES accept a typed backend too (Spec 5 sec.8.15) ------------------------
# The tests above pin the DRIVERS (compile_problem / compile_model). The higher-level authoring
# facades the docs teach -- the PDE Model facade, the hybrid composer, the coupled-source compiler
# -- each validated/stored `backend` ITSELF before (or instead of) reaching a driver, so each needs
# the same additive `lower_backend` coercion. Compile-free: every assertion trips a validation guard
# (or stores the lowered token) BEFORE any toolchain/compiler is invoked.

def test_facade_model_compile_lowers_typed_backend():
    # pops.physics.facade.Model.compile validated `backend not in _BACKENDS` itself; a typed
    # Production() must lower so it is NOT rejected as unknown. A deliberately bogus target pushes
    # PAST the (now-passing) backend guard and trips the target guard -- proving the coercion ran
    # without reaching any heavy Kokkos compile.
    from pops.physics.facade import Model
    with pytest.raises(ValueError) as excinfo:
        Model("t").compile(backend=Production(), target="__bogus__")
    msg = str(excinfo.value)
    assert "__bogus__" in msg and "target" in msg
    assert "unknown backend" not in msg


def test_hybrid_compile_lowers_typed_backend():
    # HybridModel.compile validated `backend not in (...)` itself; same proof via a bogus target.
    # The composite is stitched from inert native-brick descriptors (no _pops, no compile invoked).
    from pops.physics import HybridModel
    from pops.physics.bricks import NativeBrick
    hyp = NativeBrick("pops::Dummy", "hyperbolic", var_names=["rho", "mx", "E"], n_vars=3, gamma=1.4)
    src = NativeBrick("pops::NoSource", "source")
    ell = NativeBrick("pops::ZeroRhs", "elliptic")
    with pytest.raises(ValueError) as excinfo:
        HybridModel(hyp, src, ell).compile(backend=Production(), target="__bogus__")
    msg = str(excinfo.value)
    assert "__bogus__" in msg and "target" in msg
    assert "got Production" not in msg


def test_coupled_source_compile_lowers_typed_backend():
    # CoupledSource.compile STORES backend on the compiled handle (introspection / API parity); a
    # typed Production() must be lowered to its string so the handle stays string-typed. Pure Python
    # (the coupling is interpreted as bytecode; no .so is produced).
    from pops.physics import CoupledSource
    src = CoupledSource("ion")
    ne = src.block("e").role("density")
    src.add("e", role="density", expr=ne)
    compiled = src.compile(backend=Production())
    assert compiled.backend == "production"


def test_hyperbolic_model_compile_already_lowers_typed_backend():
    # The thin authoring delegator HyperbolicModel.compile forwards to compile_model, which already
    # lowers (additive) -- pin that a typed backend reaches PAST the backend guard, so the delegation
    # path stays coercing (regression guard; no change was needed in the delegator itself).
    from pops.physics.model import HyperbolicModel
    with pytest.raises(ValueError) as excinfo:
        HyperbolicModel("z").compile(backend=Production(), target="__bogus__")
    msg = str(excinfo.value)
    assert "__bogus__" in msg
    assert "unknown backend" not in msg


# --- Production(platform=...) records the platform / refuses a string --------------------------
def test_production_records_platform():
    prod = Production(platform=KokkosOpenMP())
    assert prod.platform is not None
    assert prod.options()["platform"] == "KokkosOpenMP"
    assert prod.inspect()["platform"]["options"]["device"] == "openmp"
    # The platform never changes the backend token.
    assert prod.lower() == "production"


def test_backend_refuses_string_platform():
    with pytest.raises(TypeError, match="platform must be a typed"):
        Production(platform="openmp")


# --- platform descriptors: capabilities + explainable availability ----------------------------
def test_platform_descriptors_declare_capabilities():
    assert KokkosSerial().capabilities().to_dict() == {"host": True, "gpu": False, "mpi": False}
    assert KokkosOpenMP().capabilities().to_dict()["host"] is True
    assert KokkosCuda().capabilities().to_dict()["gpu"] is True
    assert KokkosHIP().capabilities().to_dict()["gpu"] is True
    assert MPI().capabilities().to_dict()["mpi"] is True
    for cls in (KokkosSerial, KokkosOpenMP, KokkosCuda, KokkosHIP, MPI):
        desc = cls()
        assert desc.category == "platform"
        assert isinstance(desc, Descriptor)
        assert desc.options()["device"]


def test_platform_available_is_explainable():
    # available() always returns an Availability (never a bare bool), with a reason.
    for cls in (KokkosSerial, KokkosOpenMP, KokkosCuda, KokkosHIP, MPI):
        status = cls().available()
        assert isinstance(status, Availability)
        assert status.status in ("yes", "no", "partial")
        assert status.reason


def test_serial_platform_always_available():
    assert KokkosSerial().available().ok


def test_unavailable_platform_explains_missing_build_flag():
    # On a build that lacks a flag, available() is "no"/"partial" and NAMES the missing flag +
    # an alternative. The exact verdict depends on the loaded _pops build, so assert the contract,
    # not a fixed verdict: a non-yes status must carry a reason and either missing or alternatives.
    has_mpi = getattr(pops._pops, "__has_mpi__", None)
    mpi_status = MPI().available()
    if has_mpi is False:
        assert mpi_status.status == "no"
        assert "MPI" in mpi_status.reason
        assert mpi_status.missing  # names the build flag
        assert mpi_status.alternatives
    # A GPU device on a non-GPU build is never a false "yes".
    for cls in (KokkosCuda, KokkosHIP):
        status = cls().available()
        if not status.ok:
            assert status.reason
            assert status.missing or status.alternatives


def test_platform_lower_is_inert_metadata():
    record = MPI().lower().to_dict()
    assert record["category"] == "platform"
    assert record["device"] == "mpi"
    assert record["capabilities"]["mpi"] is True


# --- ADC-540: layout / backend / platform are THREE orthogonal, separately-typed axes ----------
def test_layout_backend_platform_are_distinct_typed_axes():
    # The mesh STRUCTURE (Uniform / AMR), the compile ENGINE (Production / AOT / JIT) and the
    # execution DEVICE (KokkosOpenMP / MPI / ...) are three different descriptor categories: none
    # is a substitute for another (a layout is not a target string, a platform is not a backend).
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    uniform = Uniform(CartesianMesh(n=64))
    amr = AMR(base=CartesianMesh(n=64))
    assert uniform.category == "layout" and amr.category == "layout"
    assert Production().category == "backend"
    assert KokkosOpenMP().category == "platform"
    # Recording a platform on a backend does NOT change the backend token, and neither the backend
    # nor the platform carries a layout: the three stay separable.
    prod = Production(platform=KokkosOpenMP())
    assert prod.lower() == "production"
    assert prod.options().get("platform") == "KokkosOpenMP"
    assert "layout" not in prod.options()


def test_layout_descriptors_expose_no_target_string():
    # Spec 5 sec.5.10: layout=AMR(...) REPLACES the old target="amr_system" string. A layout
    # descriptor selects the structure via its capabilities()["layout"] token, NOT a target string
    # on its options -- so no "amr_system" / "system" target string leaks onto the public surface.
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    for layout, kind in ((Uniform(CartesianMesh(n=64)), "uniform"), (AMR(base=CartesianMesh(n=64)), "amr")):
        assert layout.capabilities().to_dict()["layout"] == kind
        opts = layout.options()
        assert "target" not in opts, opts
        for value in opts.values():
            assert value not in ("system", "amr_system"), opts


def test_unavailable_platform_refuses_before_compile_with_reason():
    # An unavailable device is refused through the EXPLAINABLE Availability BEFORE any compile: the
    # status is not "yes", it carries a reason, and it names the missing build flag or an alternative
    # (never a bare bool, never a silent fallback). MPI on a non-MPI build is the deterministic case.
    has_mpi = getattr(pops._pops, "__has_mpi__", None)
    status = MPI().available()
    assert isinstance(status, Availability)
    if has_mpi is False:
        assert status.status == "no"
        assert status.reason and status.missing and status.alternatives
        # A backend targeting an unavailable platform surfaces the SAME refusal (no compile).
        back = Production(platform=MPI()).available()
        assert not back.ok and "MPI" in back.reason


def test_backend_string_is_refused_on_the_typed_platform_slot():
    # backend="production" / platform="cuda" (bare strings) are the Spec 5 sec.7 anti-pattern on the
    # typed platform slot: Production(platform="cuda") is refused naming the typed alternative.
    with pytest.raises(TypeError, match="platform must be a typed"):
        Production(platform="cuda")
    with pytest.raises(TypeError, match="platform must be a typed"):
        AOT(platform="openmp")


def test_optimization_is_separate_from_backend_and_platform():
    # ADC-540: the codegen Optimization policy is a FOURTH axis (which transforms + math mode); it is
    # neither the backend nor the platform, and its category is distinct.
    from pops.codegen import Optimization
    opt = Optimization()
    assert opt.category == "optimization"
    assert opt.category != Production().category
    assert opt.category != KokkosOpenMP().category
