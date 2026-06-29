"""Spec 5 sec.8.15 / criterion 22: the typed compile-backend + execution-platform descriptors.

These checks pin the typed backend/platform surface added under epic ADC-479:

  - the backend descriptors (Production / AOT / JIT) lower to the legacy backend string
    ("production" / "aot" / "prototype") the compile drivers already key on, and expose the same
    token via ``.scheme``;
  - ``lower_backend`` is strict: public route choices must be typed descriptors, not strings;
  - ``compile_problem`` and ``compile_model`` both reject string route selectors;
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
import os
import sys

import pytest

pops = pytest.importorskip("pops")

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from pops.codegen import AOT, JIT, Production, lower_backend  # noqa: E402
from pops.codegen.backends import BACKEND_DESCRIPTORS, _Backend  # noqa: E402
from pops.codegen.backends import lower_problem_backend  # noqa: E402
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
    caps = prod.capabilities()
    assert caps["cpu"] is True and caps["mpi"] is True and caps["amr"] is True
    assert JIT().capabilities()["mpi"] is False
    # inspect is a plain dict carrying the platform slot.
    record = prod.inspect()
    assert record["category"] == "backend"
    assert record["options"]["backend"] == "production"
    assert record["platform"] is None


def test_backend_registry_maps_token_to_class():
    assert BACKEND_DESCRIPTORS["production"] is Production
    assert BACKEND_DESCRIPTORS["aot"] is AOT
    assert BACKEND_DESCRIPTORS["prototype"] is JIT


def test_lower_backend_lowers_typed():
    assert lower_backend(Production()) == "production"
    assert lower_backend(AOT()) == "aot"
    assert lower_backend(JIT()) == "prototype"
    assert lower_backend(None) == "production"


def test_lower_backend_rejects_public_strings_and_wrong_types():
    for token in ("production", "aot", "prototype", "auto", "nope"):
        with pytest.raises(TypeError, match="typed"):
            lower_backend(token)
    with pytest.raises(TypeError, match="Production"):
        lower_backend(123)


# --- compile_problem is strict: public route choices are typed -------------------------------
def _guard_error(backend):
    """Run compile_problem far enough to hit the backend guard; return (type, message)."""
    from pops.codegen.compile_drivers import compile_problem
    try:
        compile_problem(model=None, time=None, backend=backend)
    except Exception as err:  # noqa: BLE001 -- the exception type is part of the assertion
        return type(err), str(err)
    raise AssertionError("compile_problem(backend=%r) did not raise" % (backend,))


def test_compile_problem_rejects_string_backend():
    typ, msg = _guard_error("production")
    assert typ is TypeError
    assert "typed" in msg and "Production()" in msg
    with pytest.raises(TypeError):
        lower_problem_backend("production")


def test_compile_problem_typed_non_production_hits_guard():
    typ, msg = _guard_error(AOT())
    assert typ is ValueError
    assert msg == "compile_problem: compiled problems require backend=Production()"


def test_compile_problem_accepts_typed_backend_with_platform():
    # Recording a platform must not change the lowered backend string.
    typ, msg = _guard_error(Production(platform=KokkosOpenMP()))
    assert typ is ValueError
    assert "time must be" in msg


def test_compile_model_lowers_typed_backend_past_unknown_guard():
    # compile_model's unknown-backend guard sees the LOWERED string, so a typed JIT() is not
    # rejected as unknown; it proceeds and trips later on the fake model (no .name attribute).
    from pops.codegen.compile_drivers import compile_model

    class _FakeModel:
        def _check_require_metadata(self, *a, **k):
            pass

    with pytest.raises(AttributeError):
        compile_model(_FakeModel(), backend=JIT(), include=INCLUDE)
    # A string selector is rejected before the backend table.
    with pytest.raises(TypeError, match="typed"):
        compile_model(_FakeModel(), backend="nope", include=INCLUDE)


# --- the remaining lower-level compilers accept a typed backend too (Spec 5 sec.8.15) -----------
# The tests above pin the DRIVERS (compile_problem / compile_model). The higher-level authoring
# helpers that still compile lower-level artifacts -- the hybrid composer and the coupled-source
# compiler -- each validated/stored `backend` ITSELF before (or instead of) reaching a driver, so each
# needs the same additive `lower_backend` coercion. The physics authoring surface is different: public
# `.compile(...)` is removed; complete problems lower to Module and compile through compile_problem.

def test_physics_model_compile_removed_from_public_api():
    # Clean break: the physics authoring model no longer exposes `.compile(...)`; complete problems are
    # compiled through compile_problem after lowering to a pops.model.Module.
    import pops.physics as physics
    from pops import model
    m = physics.Model("t")
    assert not hasattr(m, "compile")
    assert isinstance(m.to_module(), model.Module)


def test_hybrid_compile_lowers_typed_backend():
    # HybridModel.compile validated `backend not in (...)` itself; same proof via a bogus target.
    # The composite is stitched from inert native-brick descriptors (no _pops, no compile invoked).
    from pops.physics.bricks import NativeBrick
    from pops.physics.hybrid import HybridModel
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
    assert KokkosSerial().capabilities() == {"host": True, "gpu": False, "mpi": False}
    assert KokkosOpenMP().capabilities()["host"] is True
    assert KokkosCuda().capabilities()["gpu"] is True
    assert KokkosHIP().capabilities()["gpu"] is True
    assert MPI().capabilities()["mpi"] is True
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
    record = MPI().lower()
    assert record["category"] == "platform"
    assert record["device"] == "mpi"
    assert record["capabilities"]["mpi"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
