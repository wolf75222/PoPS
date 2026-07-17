"""Low-level multi-rank global accessors and strict checkpoint refusal.

CONTEXTE. Le System construit UNE box couvrant tout le domaine (mono-box ; cf. system.cpp ctor :
ba = {index_domain}, dm round-robin -> box 0 sur le rang 0). Sous MPI np>1, les accesseurs
non-globaux (density / get_state / potential) lisent fab(0) : valides sur le rang proprietaire, mais
HORS BORNES sur un rang sans box. Les variantes _global rassemblent le champ par all_reduce_sum.
La publication scientifique est couverte exclusivement par ConsumerGraph ; ce fichier ne teste que
le seam natif et le refus d'un checkpoint direct sans identite compilee.

CE TEST tourne en MONO-RANG (la batterie pytest n'a pas de harnais MPI ; le cas np>1 -- gather
bit-identique a np=1/2/4 et aller-retour checkpoint/restart -- est couvert par le test C++
tests/cpp/integration/mpi/test_mpi_system_io_gather.cpp, lance sous mpirun par le preset mpi/ci-mpi). Il verrouille
l'invariant CENTRAL :

  T1 - EQUIVALENCE GLOBAL == LOCAL en mono-rang : state_global == get_state, density_global ==
       density, potential_global == potential, BIT-IDENTIQUE (all_reduce = identite, box = domaine
       complet). C'est la garantie que la facade IO multi-rangs n'a RIEN change au mono-rang.
  T2 - CHECKPOINT direct refuse sans ExecutionContext installe par pops.bind. Le round-trip
       authentifie et l'identite du Program compile sont couverts par les tests du lifecycle public
       et le test C++ MPI ; ce test bas niveau ne fabrique jamais une fausse autorite.
  T3 - my_rank / n_ranks exposes (0 / 1 en serie).
"""
from pops.numerics.reconstruction.limiters import Minmod
import os
import tempfile

import numpy as np
import pytest

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import System  # ADC-545 advanced runtime seam


def _build(n=16):
    sim = System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.add_equation("ions",
                  engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                            transport=engine.IsothermalFlux(),
                            source=engine.PotentialForce(charge=1.0),
                            elliptic=engine.ChargeDensity(charge=1.0)),
                  spatial=engine.Spatial(limiter=Minmod()), time=engine.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


def test_io_global_equals_local_mono_rank():
    """T1 : en mono-rang, les accesseurs GLOBAUX rendent EXACTEMENT les accesseurs locaux."""
    sim = _build()
    for _ in range(4):
        sim.step(2e-3)
    assert np.array_equal(np.asarray(sim.state_global("ions")),
                          np.asarray(sim.get_state("ions"))), "state_global != get_state (mono-rang)"
    assert np.array_equal(np.asarray(sim.density_global("ions")),
                          np.asarray(sim.density("ions"))), "density_global != density (mono-rang)"
    assert np.array_equal(np.asarray(sim.potential_global()),
                          np.asarray(sim.potential())), "potential_global != potential (mono-rang)"


def test_io_checkpoint_requires_installed_execution_context():
    """T2 : le chemin direct ne publie pas sans l'autorite installee par pops.bind."""
    tmp = tempfile.mkdtemp()
    sim = _build()
    for _ in range(3):
        sim.step(2e-3)
    checkpoint = os.path.join(tmp, "chk")
    with pytest.raises(
        ValueError, match="authenticated ExecutionContext installed by pops.bind"
    ):
        sim.checkpoint(checkpoint)
    assert not os.path.exists(checkpoint + ".npz")


def test_mpi_helpers_exposed():
    """T3 : my_rank / n_ranks exposes au module (0 / 1 en serie)."""
    from pops import _pops
    assert _pops.my_rank() == 0
    assert _pops.n_ranks() >= 1


if __name__ == "__main__":
    test_io_global_equals_local_mono_rank()
    print("OK T1 : global == local (mono-rang)")
    test_io_checkpoint_requires_installed_execution_context()
    print("OK T2 : checkpoint direct refuse sans ExecutionContext installe")
    test_mpi_helpers_exposed()
    print("OK T3 : my_rank/n_ranks exposes")
    print("test_io_multirank : OK")
