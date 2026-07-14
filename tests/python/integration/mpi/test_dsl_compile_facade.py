"""Facade de compilation par intention pour le package natif ``production``.

``HyperbolicModel.compile(backend="production")`` produit le package charge par ``add_native_block``
et preserve de bout en bout noms, ``VariableRole``, gamma, n_aux et B_z.

Deux niveaux :
(1) GARDE-FOUS pur-Python (aucun compilateur requis) : backend inconnu, mapping production -> adder,
    et erreur EXPLICITE quand ``require_metadata`` est demande sur un modele sans roles/gamma.
(2) BOUT EN BOUT (saute si aucun compilateur C++ / en-tetes pops) : le package branche via
    ``add_equation`` expose les BONS noms/roles/gamma et lit le canal aux etendu B_z.

Lance avec python3, meme PYTHONPATH que les autres tests DSL.
"""
import os
import shutil
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.math import sqrt
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import HLLC, Rusanov
from pops.numerics.variables import Conservative, Primitive
from pops.physics.aux import roles_for
from pops.physics._model import HyperbolicModel

from tests.python.support.requirements import repo_include
from pops.runtime._system import System  # ADC-545 advanced runtime seam
# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_compile_cache_backend).
POPS_PROCESS_TIMEOUT = 900
INCLUDE = repo_include()
GAMMA = 1.6667  # gamma NON STANDARD (5/3), distinct du defaut historique 1.4


def build_meta_euler():
    """Euler aux roles canoniques + gamma 5/3 : fournit des metadonnees UTILES (roles non 'Custom',
    gamma explicite). Sert a prouver que la facade les transporte et que require_metadata passe."""
    e = HyperbolicModel("euler_facade")
    rho, rhou, rhov, E = e.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    e.set_gamma(GAMMA)  # gamma explicite -> transporte via pops_compiled_gamma
    e.enable_hllc()  # ADC-590 : riemann='hllc' generique exige la capability emise
    return e


def build_bare_scalar():
    """Transport scalaire 'q' SANS role canonique ni gamma : metadonnees PAUVRES, declenche les
    erreurs require_metadata (le System retomberait sur le fallback custom / 1.4)."""
    e = HyperbolicModel("bare_q")
    (q,) = e.conservative_vars("q")
    e.set_flux(x=[q], y=[q])
    e.set_eigenvalues(x=[q], y=[q])
    e.set_primitive_state(q)
    e.set_conservative_from([q])
    return e


def build_bz_scalar():
    """Scalaire sans flux, source magnetisee S = B_z * n (lit aux('B_z')) : exerce le canal aux
    etendu (n_aux=4) a travers la facade."""
    m = HyperbolicModel("bz_facade")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn
    m.set_flux(x=[zero], y=[zero])
    m.set_eigenvalues(x=[zero], y=[zero])
    m.set_primitive_state(nn)
    m.set_conservative_from([nn])
    bz = m.aux("B_z")
    m.set_source([bz * nn])
    return m


def test_guardrails():
    """(1) Garde-fous pur-Python : ne compilent rien, tournent toujours."""
    e = build_meta_euler()
    bare = build_bare_scalar()

    # backend inconnu -> ValueError explicite
    for bad in ("jit", "compile", "gpu", "", None):
        try:
            e.compile("x.so", INCLUDE, backend=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("backend %r aurait du lever" % (bad,))
    try:
        HyperbolicModel.adder_for("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("adder_for(backend inconnu) aurait du lever")
    print("OK  backend inconnu rejete (compile + adder_for)")

    # mapping backend -> adder System (couplage compilation/execution)
    assert HyperbolicModel.adder_for("production") == "add_native_block"
    print("OK  adder_for : production->add_native_block")

    # require_metadata sur un modele PAUVRE (pas de roles, pas de gamma) : erreur listant le manque
    try:
        bare.compile("x.so", INCLUDE, backend="production", require_metadata=True)
    except ValueError as ex:
        msg = str(ex)
        assert "roles" in msg and "gamma" in msg, "le message devrait lister roles ET gamma : %r" % msg
    else:
        raise AssertionError("modele sans roles/gamma + require_metadata aurait du lever")
    print("OK  require_metadata sur modele pauvre rejete (roles + gamma manquants signales)")

    # le modele RICHE (roles canoniques + gamma) ne doit PAS echouer la verification de metadonnees :
    # on isole le pre-check en passant un backend pauvre cote compilation (mais on ne compile pas ici,
    # on verifie juste que la verification de metadonnees ne leve pas avant l'appel au moteur). On le
    # prouve via le chemin bout-en-bout ci-dessous ; ici on s'assure juste que bare != e.
    assert all(r == "Custom" for r in roles_for(bare.cons_names)), "scalaire q devrait etre Custom"
    assert roles_for(e.cons_names) == ["Density", "MomentumX", "MomentumY", "Energy"]
    print("OK  garde-fous pur-Python verts")


def test_end_to_end():
    """(2) Bout en bout : compile(backend=...) -> .so -> adder -> metadonnees + B_z preserves."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> bout-en-bout saute")
        return

    e = build_meta_euler()
    n, L = 16, 1.0
    tmp = tempfile.mkdtemp()
    try:
        compiled = e.compile(
            os.path.join(tmp, "facade_production.so"),
            INCLUDE,
            backend="production",
            require_metadata=True,
        )
        assert compiled.adder == "add_native_block"
        s = System(n=n, L=L, periodic=True)
        s.add_equation(
            "gas",
            compiled,
            spatial=engine.Spatial(limiter=Minmod(), flux=HLLC(), recon=Primitive()),
            time=engine.Explicit(),
        )
        # noms/roles DU MODELE (pas le fallback u0.. / custom)
        assert s.variable_names("gas") == ["rho", "rho_u", "rho_v", "E"]
        assert s.variable_roles("gas") == ["density", "momentum_x", "momentum_y", "energy"]
        assert s.variable_roles("gas", "primitive") == \
            ["density", "velocity_x", "velocity_y", "pressure"]
        assert abs(s.block_gamma("gas") - GAMMA) < 1e-12
        print("OK  production : noms/roles/gamma propages via add_equation -> add_native_block")

        # --- n_aux / B_z preserves a travers la facade (canal aux etendu) ---
        m = build_bz_scalar()
        c = 0.7
        so_bz = m.compile(os.path.join(tmp, "facade_bz.so"), INCLUDE, backend="production")
        sb = System(n=n, L=L, periodic=True)
        sb.add_equation(
            "bz",
            so_bz,
            spatial=engine.Spatial(
                limiter=FirstOrder(), flux=Rusanov(), recon=Conservative()
            ),
            time=engine.Explicit(),
        )
        sb.set_poisson(rhs="charge_density", solver="geometric_mg")
        sb.set_density("bz", np.ones((n, n)))
        sb.set_magnetic_field(c * np.ones((n, n)))  # peuple le canal B_z partage (n_aux=4)
        sb.solve_fields()
        R = np.array(sb.eval_rhs("bz"))
        err = float(np.max(np.abs(R - c)))  # flux nul -> R = S = B_z n = c
        assert err < 1e-12, "B_z non lu a travers la facade (ecart %.2e)" % err
        print("OK  production : n_aux/B_z preserves (max|R - B_z| = %.2e)" % err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_guardrails()
    test_end_to_end()
    print("test_dsl_compile_facade : tout est vert")


if __name__ == "__main__":
    main()
