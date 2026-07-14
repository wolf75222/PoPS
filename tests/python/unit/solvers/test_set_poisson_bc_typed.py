"""Typed-only boundary selectors for the low-level Poisson runtime seam."""
import sys

import pytest

import pops
from pops.runtime._engine_descriptors import Dirichlet, Neumann, Periodic
from pops.runtime._system_install import _lower_bc

try:
    import pops._pops  # noqa: F401
    from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
    _HAVE_ENGINE = True
except Exception:  # pragma: no cover - exercised only without a built extension
    _HAVE_ENGINE = False
requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled _pops extension not importable")


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
        System(n=8, L=1.0, periodic=False).set_poisson(bc="dirichlet")


@requires_engine
def test_amr_set_poisson_uses_the_same_typed_contract():
    from pops.mesh.geometry import Disc
    system = AmrSystem(n=8, L=1.0, periodic=False, regrid_every=0)
    with pytest.raises(TypeError, match="string selectors"):
        system.set_poisson(bc="dirichlet")
    with pytest.raises(TypeError, match="string selectors"):
        system.set_poisson(wall="circle")
    system.set_poisson(bc=Dirichlet(), wall=Disc(radius=0.4))


@requires_engine
def test_set_poisson_typed_bc_executes():
    from pops.mesh.geometry import Disc
    system = System(n=8, L=1.0, periodic=False)
    system.set_poisson(bc=Dirichlet(), wall=Disc(radius=0.4))
    assert system.poisson_solver() == "geometric_mg"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
