"""Typed-only boundary selectors for the low-level Poisson runtime seam."""

import sys

import pytest

import pops
from pops.runtime._engine_descriptors import Dirichlet, Neumann, Periodic
from pops.runtime._system_install import _lower_bc

try:
    import pops._pops  # noqa: F401
    from pops.runtime._system import (  # ADC-545 advanced runtime seam
        AmrSystem,
        AmrSystemConfig,
        System,
        SystemConfig,
    )

    _HAVE_ENGINE = True
except Exception:  # pragma: no cover - exercised only without a built extension
    _HAVE_ENGINE = False
requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled _pops extension not importable"
)


def _system_config_2d():
    config = SystemConfig()
    config.shape = (8, 8)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (False, False)
    config.boxes = (((0, 0), (8, 8)),)
    return config


def _amr_config_2d():
    config = AmrSystemConfig()
    config.shape = (8, 8)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (False, False)
    config.boxes = (((0, 0), (8, 8)),)
    config.regrid_every = 0
    return config


def test_bc_lowers_to_private_native_tokens():
    assert _lower_bc(Dirichlet()) == "dirichlet"
    assert _lower_bc(Neumann()) == "neumann"
    assert _lower_bc(Periodic()) == "periodic"


def test_bc_strings_and_bad_types_are_rejected():
    for value in ("auto", "dirichlet", "neumann", "periodic", "bogus", 12345):
        with pytest.raises(TypeError):
            _lower_bc(value)


@requires_engine
def test_set_poisson_rejects_string_bc():
    with pytest.raises(TypeError, match="string selectors"):
        System(_system_config_2d()).set_poisson(bc="dirichlet")


@requires_engine
def test_amr_set_poisson_uses_the_same_typed_contract():
    system = AmrSystem(_amr_config_2d())
    with pytest.raises(TypeError, match="string selectors"):
        system.set_poisson(bc="dirichlet")
    with pytest.raises(TypeError, match="unexpected keyword argument 'wall'"):
        system.set_poisson(wall="circle")
    system.set_poisson(bc=Dirichlet())


@requires_engine
def test_set_poisson_typed_bc_executes():
    system = System(_system_config_2d())
    system.set_poisson(bc=Dirichlet())
    assert system.poisson_solver() == "cartesian_cg"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
