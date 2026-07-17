"""The private native boundary accepts one exact, deterministic spatial-lowering protocol."""
from __future__ import annotations

import pytest

from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLL, Rusanov
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._engine_descriptors import Spatial
from pops.runtime._system_unified_install import _SystemUnifiedInstall


_LOWERERS = (
    _SystemUnifiedInstall._lower_spatial,
    _AmrSystemEquation._lower_spatial,
)


class _StableProvider:
    def __init__(self) -> None:
        self.calls = 0

    def runtime_spatial(self) -> Spatial:
        self.calls += 1
        return Spatial(limiter=Minmod(), flux=Rusanov())


class _ChangingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def runtime_spatial(self) -> Spatial:
        self.calls += 1
        flux = Rusanov() if self.calls == 1 else HLL()
        return Spatial(limiter=Minmod(), flux=flux)


class _SpatialSubclass(Spatial):
    pass


@pytest.mark.parametrize("lower", _LOWERERS)
def test_lowering_calls_the_protocol_twice_and_returns_exact_spatial(lower) -> None:
    provider = _StableProvider()
    result = lower(None, provider)

    assert type(result) is Spatial
    assert provider.calls == 2


@pytest.mark.parametrize("lower", _LOWERERS)
def test_lowering_refuses_nondeterministic_or_nonexact_results(lower) -> None:
    with pytest.raises(ValueError, match="deterministic"):
        lower(None, _ChangingProvider())

    class Opaque:
        def runtime_spatial(self):
            return object()

    with pytest.raises(TypeError, match="exact.*Spatial|exact private Spatial"):
        lower(None, Opaque())

    class SubclassProvider:
        def runtime_spatial(self):
            return _SpatialSubclass()

    with pytest.raises(TypeError, match="exact.*Spatial|exact private Spatial"):
        lower(None, SubclassProvider())


def test_no_token_or_legacy_finite_volume_engine_constructor_remains() -> None:
    import pops.runtime._engine_descriptors as engine

    assert not hasattr(Spatial, "_from_tokens")
    assert not hasattr(engine, "FiniteVolume")
