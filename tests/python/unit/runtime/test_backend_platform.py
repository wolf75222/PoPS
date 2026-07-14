"""Final compiler selection and exact execution-platform evidence contracts.

PoPS has one compiler route: :class:`Production`. The selected platform is an exact
:class:`PlatformManifest` derived from authenticated compiled components.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen._backends import (  # noqa: E402
    BACKEND_DESCRIPTORS,
    Production,
    _Backend,
    lower_backend,
)
from pops._platform_contracts import PlatformManifest, proven_serial_manifest  # noqa: E402
from pops.descriptors import Descriptor  # noqa: E402


def test_production_is_the_only_backend_descriptor() -> None:
    descriptor = Production()
    assert descriptor.lower() == descriptor.scheme == "production"
    assert descriptor.tier == "production"
    assert isinstance(descriptor, (Descriptor, _Backend))
    assert descriptor.category == "backend"
    assert BACKEND_DESCRIPTORS == {"production": Production}
    assert descriptor.capabilities().to_dict()["tier"] == "production"
    assert descriptor.options() == {"backend": "production"}
    assert "platform" not in descriptor.inspect()


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


def test_production_rejects_platform_instead_of_ignoring_it() -> None:
    platform = proven_serial_manifest(
        backend="production", target="system", abi="pops-test-abi")

    with pytest.raises(TypeError, match="platform"):
        Production(platform=platform)


def test_platform_manifest_is_private_artifact_evidence_not_a_resolve_authority() -> None:
    from inspect import signature

    import pops
    from pops.codegen._phases import resolve

    platform = proven_serial_manifest(
        backend="production", target="system", abi="pops-test-abi")

    assert "backend" in signature(resolve).parameters
    assert "platform" not in signature(resolve).parameters
    assert "backend" in signature(pops.resolve).parameters
    assert "platform" not in signature(pops.resolve).parameters
    assert type(platform) is PlatformManifest
    assert platform.backend.require("platform.backend") == "production"
    assert platform.precision.storage.require("platform.precision.storage") == "float64"
    assert platform.device.require("platform.device") == "host"
    assert platform.communicator.require("platform.communicator") == "serial"


def test_layout_and_backend_are_distinct_axes() -> None:
    from pops.layouts import Uniform
    from tests.python.support.layout_plan import cartesian_grid, final_amr_layout

    mesh = cartesian_grid(n=64)
    uniform = Uniform(mesh)
    amr = final_amr_layout(mesh)
    assert uniform.category == amr.category == "layout"
    assert Production().category == "backend"
    assert uniform.capabilities().to_dict()["layout"] == "uniform"
    assert amr.capabilities().to_dict()["layout"] == "amr"
    for layout in (uniform, amr):
        assert "target" not in layout.options()
