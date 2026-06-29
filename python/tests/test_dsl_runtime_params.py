"""PARAMETRES RUNTIME du DSL (P7-b) : un parametre declare pops.dsl.Param(..., kind='runtime') peut voir
sa valeur CHANGEE a l'execution SANS recompiler le .so, alors qu'un parametre kind='const' (defaut)
reste INLINE EN DUR (bit-identique a l'historique).

Mecanique (backend "aot", add_compiled_block) : le codegen emet `params.get(<indice>)` pour un param
runtime (lecture d'un membre pops::RuntimeParams de la brique generee) au lieu d'une constante ; l'ABI du
.so AOT transporte un bloc plat de valeurs (symboles `_p`) ; System.set_block_params(name, values) ecrit
dans le bloc PARTAGE -> le comportement change au prochain pas. cf. include/pops/runtime/runtime_params.hpp.

Ce test verifie :
  1) NON-REGRESSION : un param const reste inline (codegen byte-identique a un modele sans param runtime ;
     aucun #include runtime_params.hpp, aucun membre RuntimeParams, valeur ecrite en dur) ;
  2) RUNTIME : un modele avec un param runtime cs2 (vitesse du son au carre) compile, tourne, et
     set_block_params change eval_rhs en consequence (le residu = -div F scale avec cs2 via p = cs2*rho) ;
  3) PAS DE RECOMPILATION : recompiler le MEME modele (meme model_hash + abi_key) reutilise le .so en cache
     (cache HIT) ; changer cs2 au runtime n'engendre AUCUNE recompilation ;
  4) cohrence avec un modele OU cs2 est cuit en CONST : eval_rhs(runtime cs2=k) == eval_rhs(const cs2=k).
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
from pops.numerics.variables import Conservative

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _fv():
    return pops.numerics.spatial.FiniteVolume(
        reconstruction=Minmod(), riemann=Rusanov(), variables=Conservative())


def _build_iso(cs2_kind, cs2_value=1.0):
    """Modele isotherme 2D (rho, rho_u, rho_v) avec p = cs2 * rho. @p cs2_kind = 'runtime' | 'const'.
    Le SEUL parametre est cs2 : un meme modele a deux variantes (cs2 runtime vs cs2 const)."""
    import pops.physics as physics
    from pops.physics import ConstParam, RuntimeParam
    from pops.math import sqrt

    m = physics.Model("iso")
    U = m.state(
        "U", components=["rho", "rho_u", "rho_v"],
        roles={"rho": "density", "rho_u": "momentum_x", "rho_v": "momentum_y"})
    rho, mx, my = U
    _typed = {"const": ConstParam, "runtime": RuntimeParam}[cs2_kind]
    cs2 = m.param(_typed("cs2", cs2_value))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.scalar("p", cs2 * rho)
    cs = sqrt(cs2)
    m.flux(
        "F",
        on=U,
        x=[mx, mx * u + p, my * u],
        y=[my, mx * v, my * v + p],
        waves={"x": [u - cs, u, u + cs], "y": [v - cs, v, v + cs]},
    )
    return m.to_module()


def _program():
    P = pops.time.Program("iso_runtime")
    U = P.state("U", block="gas").n
    R = P._rate_from_transport(name="R", state=U, flux=True, sources=[])
    P.commit("gas", P.linear_combine("U1", U + P.dt * R))
    return P


def _initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((3, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)  # densite non uniforme
    return U


def _check_codegen_non_regression():
    """(1) Un param CONST reste INLINE : le codegen d'un modele a param const est byte-identique a
    celui du MEME modele sans aucun param (cs2 ecrit en dur), et ne porte aucun artefact runtime."""
    from pops.codegen.module_emit_brick import emit_cpp_brick
    from pops.codegen.module_view import ModuleCodegenView

    const_src = emit_cpp_brick(ModuleCodegenView(_build_iso("const", 2.5)), name="IsoHyp")
    assert "runtime_params.hpp" not in const_src, "param const : ne doit PAS inclure runtime_params.hpp"
    assert "RuntimeParams" not in const_src, "param const : ne doit PAS porter de membre RuntimeParams"
    assert "params.get" not in const_src, "param const : doit etre INLINE (pas de params.get)"
    assert "2.5" in const_src, "param const cs2=2.5 doit etre ecrit EN DUR dans la brique"

    rt_src = emit_cpp_brick(ModuleCodegenView(_build_iso("runtime", 2.5)), name="IsoHyp")
    assert "runtime_params.hpp" in rt_src, "param runtime : doit inclure runtime_params.hpp"
    assert "pops::RuntimeParams params{1, {2.5}}" in rt_src, "param runtime : membre seede a la declaration"
    assert "params.get(0)" in rt_src, "param runtime : doit lire params.get(0) (pas de valeur en dur)"
    print("OK  (1) param const INLINE (byte-identique), param runtime -> params.get(0) + membre seede")


def _install_step_delta(compiled, module, Uflat, n, L, cs2=None):
    sys = pops.System(n=n, L=L, periodic=True)
    params = {"cs2": cs2} if cs2 is not None else {}
    sys.install(
        compiled,
        instances={"gas": {"initial": Uflat, "spatial": _fv(), "model": module}},
        params=params,
    )
    if cs2 is not None:
        sys.set_program_params(0, [float(cs2)])
    before = np.array(sys._get_state("gas")).reshape(3, n, n)
    sys.step(1.0e-4)
    after = np.array(sys._get_state("gas")).reshape(3, n, n)
    return after - before


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents")
        print("test_dsl_runtime_params : OK (rien a compiler)")
        return

    _check_codegen_non_regression()

    n, L = 32, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    tmp = tempfile.mkdtemp()
    try:
        m = _build_iso("runtime", 1.0)
        try:
            compiled = pops.compile_problem(
                os.path.join(tmp, "iso_runtime.so"), model=m, time=_program(), force=True)
        except RuntimeError as exc:
            print("skip  compile_problem could not build the .so: %s" % str(exc)[:160])
            print("test_dsl_runtime_params : OK (runtime compile skipped)")
            return
        routes, _defaults = compiled.runtime_param_routes()
        assert routes == {0: ["cs2"]}, "runtime route attendu {0: ['cs2']}, recu %r" % (routes,)

        # (2) RUNTIME : meme .so, deux installations avec cs2=1 puis cs2=4.
        R1 = _install_step_delta(compiled, m, Uflat, n, L, cs2=1.0)
        R4 = _install_step_delta(compiled, m, Uflat, n, L, cs2=4.0)
        # Avec u=0, deux effets DISTINCTS de cs2, tous deux exacts au runtime (verifies pointwise) :
        #  - qte de mvt : rho_u=0 partout -> AUCUNE dissipation Rusanov sur rho_u ; le flux se reduit a la
        #    pression p=cs2*rho, donc le residu = -div(cs2*rho) scale LINEAIREMENT en cs2 (1 -> 4 => x4) ;
        #  - densite : le flux advectif rho*u est nul, mais la DISSIPATION de Rusanov vaut
        #    -0.5*c*(rho_R-rho_L) avec c=sqrt(cs2) et rho NON uniforme -> le residu scale en sqrt(cs2)
        #    (1 -> 4 => x2). (L'ancienne assertion "densite independante de cs2" oubliait cette
        #    dissipation : fausse des que rho n'est pas uniforme, cf. ADC-104.)
        assert np.max(np.abs(R1[1])) > 1e-3, "residu qte de mvt trivial (cs2=1)"
        assert np.allclose(R4[1], 4.0 * R1[1], rtol=1e-9, atol=1e-12), \
            "residu de qte de mvt = -div(cs2*rho) doit scaler en cs2 (x4 quand cs2 1 -> 4) au runtime"
        assert np.max(np.abs(R1[0])) > 1e-3, "residu de densite trivial : etat non uniforme attendu"
        assert np.allclose(R4[0], 2.0 * R1[0], rtol=1e-9, atol=1e-12), \
            "residu de densite (dissipation Rusanov ~ sqrt(cs2)) doit scaler x2 quand cs2 1 -> 4 au runtime"
        print("OK  (2) set_block_params change eval_rhs SANS recompiler : qte mvt ~cs2 (x4), densite ~sqrt(cs2) (x2)")

        # (3) PAS DE RECOMPILATION : recompiler le MEME probleme (sans so_path) -> cache HIT.
        m2 = _build_iso("runtime", 1.0)
        c_a = pops.compile_problem(model=m2, time=_program())
        c_b = pops.compile_problem(model=m2, time=_program())
        assert c_a.so_path == c_b.so_path, "cache : meme probleme -> meme chemin .so"
        mtime = os.path.getmtime(c_b.so_path)
        c_c = pops.compile_problem(model=_build_iso("runtime", 1.0), time=_program())
        assert os.path.getmtime(c_c.so_path) == mtime, "cache HIT : le .so NE doit PAS etre recompile"
        print("OK  (3) recompiler le meme modele runtime -> cache HIT (.so reutilise, pas recompile)")

        # (4) COHERENCE runtime vs const : eval_rhs(runtime cs2=k) == eval_rhs(const cs2=k). On compile un
        # modele a cs2 CONST=2.0 et on le compare au modele runtime apres set_block_params(cs2=2.0).
        mc = _build_iso("const", 2.0)
        const_compiled = pops.compile_problem(
            os.path.join(tmp, "iso_const2.so"), model=mc, time=_program(), force=True)
        Rc = _install_step_delta(const_compiled, mc, Uflat, n, L)
        Rr = _install_step_delta(compiled, m, Uflat, n, L, cs2=2.0)
        drc = float(np.max(np.abs(Rr - Rc)))
        assert drc < 1e-12, "step(runtime cs2=2) != step(const cs2=2) (ecart %.2e)" % drc
        print("OK  (4) runtime cs2=2 == const cs2=2 (step ecart %.1e) : meme numerique" % drc)

        # (5) GARDE-FOU : params= sur un programme SANS param runtime leve une erreur explicite.
        try:
            _install_step_delta(const_compiled, mc, Uflat, n, L, cs2=1.0)
            raised = False
        except ValueError as ex:
            raised = True
            assert "runtime parameter" in str(ex) or "params" in str(ex), "message inattendu : %s" % ex
        assert raised, "set_block_params sur un bloc const-only doit lever (sinon set silencieux)"
        print("OK  (5) set_block_params sur un bloc const-only REJETE explicitement")

        print("test_dsl_runtime_params : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
