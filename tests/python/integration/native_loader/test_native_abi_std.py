"""Garde-fou de REGRESSION : la norme C++ du modele NATIF (backend="production") doit suivre celle du
LOADER (module _pops), sinon add_native_block rejette le bloc avec "incompatible ABI".

CONTEXTE (regression observee sur GH200). Le module _pops est compile en C++20 sous Kokkos (CUDA 12.x
n'offre pas -std=c++23 ; cf. POPS_CXX_STD dans CMakeLists.txt), en C++23 sinon. Avant le fix, le DSL
backend="production" figeait le std du modele natif a "c++23" en dur. Sous Kokkos cela donnait :
loader C++20 (__cplusplus=202002L) vs modele C++23 (__cplusplus!=202002L) -> les cles d'ABI (qui
encodent __cplusplus) divergeaient -> add_native_block levait "incompatible ABI" -> AUCUN cas ne
pouvait tourner en natif sur GH200. Le fix derive le std du modele natif de la norme reelle du loader
(pops.dsl.loader_cxx_std() / pops._pops.__cxx_std__), donc les cles concordent SUR TOUTE toolchain.

Ce test :
  1) verifie l'INVARIANT de norme : loader_cxx_std() == norme reellement bakee par le module
     (pops._pops.__cxx_std__), avec fallback sur le std encode dans abi_key() ;
  2) bout-en-bout : un modele trivial compile par le lifecycle public puis branche par
     ``System.add_equation`` (le dispatcher public authentifie le package de production) se charge
     SANS erreur d'ABI -- c'est exactement ce qui cassait sous Kokkos. Le test echouerait
     sous Kokkos avec l'ancien defaut c++23 (mismatch __cplusplus), il passe avec le std aligne.

S'auto-saute explicitement sur une machine locale sans toolchain native. Dans le job CI Kokkos
(OpenMP), ou loader != c++23, toute capacite native manquante est un echec de release.
"""
import numpy as np

import pops
from pops.codegen.toolchain import loader_cxx_std
from test_dsl_coupled import GAMMA, build_euler, compile_euler_artifact

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)
from pops.runtime._system import (  # runtime facade used by the low-level ABI test
    System,
    SystemConfig,
)
from pops.runtime._engine_descriptors import Explicit, Spatial
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
from tests.python.support.native_execution_context import artifact_execution_context

INCLUDE = repo_include()


def _system_config_2d(n):
    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    return config


def _expected_std_from_module():
    """Norme attendue du loader, lue DIRECTEMENT du module (independamment de loader_cxx_std), pour
    constituer une reference croisee : __cxx_std__ (entier 20/23) sinon le std encode dans abi_key()."""
    n = getattr(pops._pops, "__cxx_std__", None)
    if isinstance(n, int) and n in (20, 23):
        return "c++%d" % n
    key = pops._pops.abi_key()
    for tok in str(key).split(";"):
        if tok.startswith("std="):
            val = tok[len("std="):].rstrip("Ll")
            if val.isdigit():
                return "c++23" if int(val) > 202002 else "c++20"
    raise AssertionError("impossible de deduire la norme du loader (ni __cxx_std__ ni abi_key std=)")


def check_std_invariant():
    """La norme retournee par loader_cxx_std() DOIT coincider avec la norme reelle du module charge."""
    got = loader_cxx_std()
    assert got in ("c++20", "c++23"), "loader_cxx_std() = %r (attendu c++20|c++23)" % got
    expected = _expected_std_from_module()
    assert got == expected, (
        "loader_cxx_std()=%r != norme du module %r : le modele natif serait compile avec un std "
        "different du loader -> __cplusplus divergent -> cle d'ABI incompatible" % (got, expected))
    print("OK  invariant de norme : loader_cxx_std()=%s == module _pops (%s)" % (got, expected))
    return got


def check_native_loads_without_abi_error(expected_std, cxx):
    """Compile by the public lifecycle, then exercise the detached package ABI seam."""
    n = 16
    model = build_euler("euler_abistd")
    artifact = compile_euler_artifact(model, cells=n, cxx=cxx)
    assert len(artifact.blocks) == 1
    component = artifact.blocks[0].model
    assert component.std == expected_std

    sys = System(_system_config_2d(n))
    context = artifact_execution_context(artifact)
    sys._execution_context = context
    sys._s._prepare_boundary_execution_lane(
        context.communicator.handle,
        context.identity.token,
    )
    (state_identity,) = artifact.plan.blocks[0].state_identities
    sys._s._install_block_state_route("gas", state_identity)
    # Si le std du modele != std du loader, le dispatcher leve RuntimeError("incompatible ABI").
    try:
        sys.add_equation(
            "gas", component,
            spatial=Spatial(limiter=Minmod(), flux=Rusanov()),
            time=Explicit(),
        )
    except RuntimeError as ex:
        if "incompatible ABI" in str(ex):
            raise AssertionError(
                "REGRESSION : add_native_block rejette le modele production (std du modele != "
                "std du loader %s). C'est exactement le bug GH200 sous Kokkos. Detail : %s"
                % (expected_std, ex)) from ex
        raise
    if sys._pending_native_packages:
        sys._s._finalize_native_packages()
        sys._pending_native_packages = 0

    # Sanity end-to-end : un etat trivial + eval_rhs renvoie un residu fini (le bloc tourne vraiment).
    U = np.zeros((4, n, n))
    U[0] = 1.0
    U[3] = 1.0 / (GAMMA - 1.0)
    sys.set_state("gas", U.reshape(-1).tolist())
    R = np.array(sys.eval_rhs("gas"))
    assert R.size == 4 * n * n and np.all(np.isfinite(R)), "eval_rhs du bloc natif non fini"
    print("OK  production + add_equation : charge SANS erreur d'ABI (std modele = loader %s)"
          % expected_std)


def main():
    cxx = default_cxx()
    missing = missing_native_compile_requirement(INCLUDE, cxx)
    if missing is not None:
        require_native_or_skip(missing)
    assert cxx is not None

    expected_std = check_std_invariant()
    check_native_loads_without_abi_error(expected_std, cxx)
    print("test_native_abi_std : tout est vert")


if __name__ == "__main__":
    main()
