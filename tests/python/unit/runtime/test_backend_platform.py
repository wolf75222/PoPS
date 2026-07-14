"""Final typed compiler/backend and execution-platform contracts.

PoPS has one compiler route: :class:`Production`. Layout, platform and optimization remain
orthogonal typed axes; an unsupported route is rejected before compilation and never falls back.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("pops")

from pops.codegen._backends import (  # noqa: E402
    BACKEND_DESCRIPTORS,
    Production,
    _Backend,
    lower_backend,
)
from pops.descriptors import Availability, Descriptor  # noqa: E402
from pops.runtime.platforms import (  # noqa: E402
    KokkosCuda,
    KokkosHIP,
    KokkosOpenMP,
    KokkosSerial,
    MPI,
)


def _native_module_or_skip():
    try:
        return importlib.import_module("pops._pops")
    except ImportError:
        pytest.skip("installed pops._pops is required for platform availability")


def test_production_is_the_only_backend_descriptor() -> None:
    descriptor = Production()
    assert descriptor.lower() == descriptor.scheme == "production"
    assert descriptor.tier == "production"
    assert isinstance(descriptor, (Descriptor, _Backend))
    assert descriptor.category == "backend"
    assert BACKEND_DESCRIPTORS == {"production": Production}
    assert descriptor.capabilities().to_dict()["mpi"] is True
    assert descriptor.inspect()["platform"] is None


def test_lower_backend_accepts_only_authenticated_production() -> None:
    assert lower_backend(Production()) == "production"
    # The internal token remains accepted at the private compiler seam.
    assert lower_backend("production") == "production"
    for unsupported in ("aot", "prototype", "jit", "auto", None, 123):
        with pytest.raises(TypeError, match="Production"):
            lower_backend(unsupported)


def test_compile_problem_accepts_typed_and_private_production_tokens() -> None:
    from pops.codegen._compile_drivers import compile_problem

    messages = []
    for backend in ("production", Production()):
        with pytest.raises(ValueError, match="time must be") as error:
            compile_problem(model=None, time=None, backend=backend)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_compile_problem_refuses_retired_routes_before_model_lowering() -> None:
    from pops.codegen._compile_drivers import compile_problem

    for backend in ("aot", "prototype", "jit"):
        with pytest.raises(TypeError, match="not part of the final runtime"):
            compile_problem(model=None, time=None, backend=backend)


def test_authoring_model_accepts_typed_production() -> None:
    from pops.physics._facade import Model

    model = Model("typed_backend")
    (state,) = model.conservative_vars("state")
    model.flux(x=[state], y=[state])
    model.eigenvalues(x=[state], y=[state])
    model.primitive_vars(state)
    model.conservative_from([state])
    with pytest.raises(ValueError, match="target"):
        model.compile(backend=Production(), target="__bogus__")


def test_production_records_platform_and_refuses_string() -> None:
    platform = KokkosOpenMP()
    descriptor = Production(platform=platform)
    assert descriptor.platform is platform
    assert descriptor.options()["platform"] == "KokkosOpenMP"
    assert descriptor.inspect()["platform"]["options"]["device"] == "openmp"
    assert descriptor.lower() == "production"
    with pytest.raises(TypeError, match="platform must be a typed"):
        Production(platform="openmp")


def test_platform_descriptors_declare_capabilities() -> None:
    assert KokkosSerial().capabilities().to_dict() == {
        "host": True, "gpu": False, "mpi": False,
    }
    assert KokkosOpenMP().capabilities().to_dict()["host"] is True
    assert KokkosCuda().capabilities().to_dict()["gpu"] is True
    assert KokkosHIP().capabilities().to_dict()["gpu"] is True
    assert MPI().capabilities().to_dict()["mpi"] is True
    for cls in (KokkosSerial, KokkosOpenMP, KokkosCuda, KokkosHIP, MPI):
        descriptor = cls()
        assert isinstance(descriptor, Descriptor)
        assert descriptor.category == "platform"
        assert descriptor.options()["device"]


def test_platform_availability_is_explainable_without_fallback() -> None:
    native = _native_module_or_skip()
    assert KokkosSerial().available().ok
    has_mpi = getattr(native, "__has_mpi__", None)
    status = MPI().available()
    if has_mpi is False:
        assert status.status == "no"
        assert status.missing and status.alternatives
        routed = Production(platform=MPI()).available()
        assert not routed.ok and "MPI" in routed.reason
    for cls in (KokkosCuda, KokkosHIP):
        status = cls().available()
        if not status.ok:
            assert status.missing or status.alternatives


def test_layout_backend_platform_and_optimization_are_distinct_axes() -> None:
    from pops.codegen.optimization import Optimization
    from pops.layouts import Uniform
    from pops.mesh.cartesian import CartesianMesh
    from tests.python.support.layout_plan import final_amr_layout

    mesh = CartesianMesh(n=64)
    uniform = Uniform(mesh)
    amr = final_amr_layout(mesh)
    assert uniform.category == amr.category == "layout"
    assert Production().category == "backend"
    assert KokkosOpenMP().category == "platform"
    assert Optimization().category == "optimization"
    assert uniform.capabilities().to_dict()["layout"] == "uniform"
    assert amr.capabilities().to_dict()["layout"] == "amr"
    for layout in (uniform, amr):
        assert "target" not in layout.options()
