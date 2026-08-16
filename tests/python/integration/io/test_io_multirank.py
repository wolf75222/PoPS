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
from functools import cache
import os
import tempfile

import numpy as np
import pytest

import pops
import pops.runtime._engine_descriptors as engine
from pops.numerics.reconstruction.limiters import Minmod
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import System, SystemConfig  # ADC-545 advanced runtime seam
from tests.python.integration._final_field_program import (
    density_advection_model,
    forward_euler_program,
    resolve_periodic_field_program,
)
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_native_missing = missing_native_compile_requirement(repo_include(), default_cxx())
if _native_missing:
    require_native_or_skip("test_io_multirank: %s" % _native_missing)


def _system_config_2d(n, *, length=1.0, periodicity=(True, True)):
    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (float(length), float(length))
    config.periodicity = tuple(periodicity)
    config.boxes = (((0, 0), (n, n)),)
    return config


@cache
def _density_artifact(n):
    model = density_advection_model("io-multirank-density-%d" % n, speed=0.0)
    resolved = resolve_periodic_field_program(
        model,
        forward_euler_program,
        name="io-multirank-%d" % n,
        block_name="ions",
        target="system",
        n=n,
        cxx=default_cxx(),
        include=repo_include(),
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    return artifact


def _build(n=16):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    density = 1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))
    artifact = _density_artifact(n)
    context = artifact_execution_context(artifact)
    sim = System(_system_config_2d(n))
    # Install only the native lane.  Deliberately do not publish the Python ExecutionContext: T2
    # proves that a direct low-level System cannot checkpoint without the final pops.bind authority.
    sim._s._prepare_boundary_execution_lane(
        context.communicator.handle,
        context.identity.token,
    )
    (state_identity,) = artifact.plan.blocks[0].state_identities
    sim._s._install_block_state_route("ions", state_identity)
    sim.add_equation(
        "ions",
        artifact.blocks[0].model,
        spatial=engine.Spatial(limiter=Minmod()),
        time=engine.Explicit(),
    )
    if sim._pending_native_packages:
        sim._s._finalize_native_packages()
        sim._pending_native_packages = 0
    sim.set_poisson(rhs="charge_density", solver="cartesian_cg", bc=Periodic())
    sim.set_density("ions", density.ravel())
    install_forward_euler_program(sim)
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
