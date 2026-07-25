"""Final Model/parameter/package contract through the public lifecycle.

Deux niveaux :
(1) PUR-PYTHON (aucun compilateur requis) : Param nomme + runtime supporte (P7-b), flux vs flux_value distincts,
    etat/roles et axes physiques types,
    FiniteVolume(riemann=), et les erreurs explicites (backend/target inconnus, hllc sans pression,
    remplacement de noms interdit sur un package natif).
(2) BOUT EN BOUT (saute si pas de compilateur / en-tetes) :
    ``Case -> validate -> resolve -> compile -> bind -> run``.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
from pops.numerics.reconstruction import WENO5
import os
import shutil

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.codegen.loader import CompiledModel
from pops.math import ddt, div, sqrt
from pops.physics import Density, Energy, Model, Momentum
from pops.params import ConstParam, RuntimeParam

from tests.python.support.initial_states import euler_bubble_state
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
from tests.python.support.requirements import repo_include
from test_dsl_coupled import compile_euler_artifact
from pops.runtime._system import System  # ADC-545 advanced runtime seam
INCLUDE = repo_include()
GAMMA = 1.6667


def build_euler(name="euler_pa"):
    """Euler 2D through the final board model and typed frame/role descriptors."""
    m = Model(name, frame=FRAME)
    U = m.state("U", components=["rho", "rho_u", "rho_v", "E"], roles={
        "rho": Density(), "rho_u": Momentum(axis=X_AXIS), "rho_v": Momentum(axis=Y_AXIS),
        "E": Energy(),
    })
    rho, rhou, rhov, E = U
    g = m.value(m.param(ConstParam("gamma", GAMMA)))
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.scalar("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = m.scalar("H", (E + p) / rho)
    c = m.scalar("c", sqrt(g * p / rho))
    F = m.flux("transport", frame=FRAME, state=U, components={
        X_AXIS: [rhou, rhou * u + p, rhou * v, rho * H * u],
        Y_AXIS: [rhov, rhov * u, rhov * v + p, rho * H * v],
    }, waves={
        X_AXIS: [u - c, u, u, u + c],
        Y_AXIS: [v - c, v, v, v + c],
    })
    # ``sound_speed`` is a two-state Riemann hook, not a one-state board formula.  Let the
    # HLLC capability derive it from the typed density/momentum/energy roles; pressure remains
    # the supported one-state formula override.
    m.riemann(HLLC(), pressure=p)
    m.rate("transport", equation=ddt(U) == -div(F))
    return m


def build_euler_predef(name="euler_predef"):
    """The old primitive-layout spelling has one final board representation."""
    return build_euler(name)


def initial_state(n):
    return euler_bubble_state(n, GAMMA)


def expect_raises(exc, fn, label):
    try:
        fn()
    except exc:
        print("OK  %s : %s levee" % (label, exc.__name__))
        return
    raise AssertionError("%s : %s attendue, non levee" % (label, exc.__name__))


def _bind_final_artifact(artifact, **inputs):
    """Bind through the communicator explicitly authenticated by the final artifact."""
    communicator = artifact.platform_manifest.communicator.require(
        "DSL phase-A artifact communicator")
    if communicator == "serial":
        return pops.bind(artifact, **inputs)
    if communicator == "MPI_COMM_WORLD":
        return pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
            **inputs,
        )
    raise RuntimeError("unsupported DSL phase-A communicator %r" % communicator)


def pure_python_checks():
    # Declarations explicites + identites de handles ; runtime supporte (P7-b).
    m = build_euler()
    materialized_module = m.module
    g = materialized_module.params()["gamma"]
    assert isinstance(g, ConstParam) and g.name == "gamma" and abs(g.value - GAMMA) < 1e-12
    # P7-b : les parametres runtime sont desormais implementes (cf. test_dsl_runtime_params). L'ancienne
    # assertion "runtime rejete -> NotImplementedError" etait perimee depuis l'arrivee de la feature et
    # echouait en silence (CI auto-decouverte avalant l'echec, cf. ADC-104).
    kp = m.param(RuntimeParam("kappa", default=1.0))
    assert kp.param_kind == "runtime" and kp.local_id == "kappa"
    assert m.value(kp) is not kp
    assert m.module is materialized_module
    print("OK  declarations explicites + handles distincts des Expr")

    # The final host oracle is typed by an axis, distinct from flux declaration.
    assert m.flux is not m.flux_value, "flux et flux_value doivent etre distincts"
    assert "transport" in m.fluxes, "m.flux(...) a bien declare le flux transport"
    print("OK  m.flux (declarateur) != m.flux_value (oracle axe type)")

    state = m.module.state_spaces()["U"]
    assert state.components == ("rho", "rho_u", "rho_v", "E")
    assert state.roles["rho"] == "Density" and state.roles["E"] == "Energy"
    print("OK  etat final : composantes et roles physiques types")

    # FiniteVolume : riemann (PAS flux) -> Spatial.flux ; variables -> recon
    fv = engine.Spatial(limiter=Minmod(), flux=HLLC(), recon=Primitive())
    assert fv.flux == "hllc" and fv.limiter == "minmod" and fv.recon == "primitive", \
        "FiniteVolume(riemann=) -> Spatial.flux"
    print("OK  FiniteVolume(limiter=, riemann=, variables=) remappe sur Spatial")

    # A physics model is an authoring object, never an alternate compiler facade.
    assert not hasattr(m, "compile")

    # add_equation : erreurs sur un CompiledModel FACTICE (pas de .so reel necessaire, les gardes
    # levent AVANT la frontiere C++).
    sys = System(n=16, periodicity=(True, True))
    fake = CompiledModel(so_path="/inexistant.so", backend="production",
                         cons_names=["rho", "rho_u", "rho_v", "E"],
                         cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
                         prim_names=["rho", "u", "v"],  # PAS de 'p' -> hllc/roe doit lever
                         n_vars=4, gamma=GAMMA, n_aux=3, params={}, caps={},
                         abi_key="", model_hash="h", cxx="c++", std="c++20")
    # WENO5 est accepte par le package natif : il passe la garde Python et echoue seulement au dlopen
    # du package factice.
    expect_raises(RuntimeError, lambda: sys.add_equation("g", fake,
                  spatial=engine.Spatial(limiter=WENO5())),
                  "weno5 production : accepte (echec au dlopen)")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake,
                  spatial=engine.Spatial(flux=HLLC())), "hllc sans pression")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake, names=["x"]),
                  "names= sur production natif")
    print("OK  add_equation production : weno5 accepte, hllc sans p et names= rejetes")


def end_to_end_checks():
    n = 32
    model = build_euler("euler_production")
    artifact = compile_euler_artifact(model, cells=n)
    component = artifact.blocks[0].model
    assert isinstance(component, CompiledModel), "compile -> CompiledModel component"
    assert component.backend == "production" and component.target == "system"
    assert component.n_vars == 4 and abs((component.gamma or 0) - GAMMA) < 1e-12
    assert component.abi_key and component.model_hash, "abi_key + model_hash presents"
    assert "gamma" in component.params, "params porte le Param gamma"

    initial = np.asarray(initial_state(n), dtype=np.float64).reshape(4, n, n)
    simulation = _bind_final_artifact(
        artifact, initial_state={"gas": initial.copy()})
    report = pops.run(simulation, t_end=1.0e-4, max_steps=1)
    assert report.accepted_steps == 1
    final = np.asarray(simulation.get_state("gas"), dtype=np.float64)
    assert np.all(np.isfinite(final)), "production : etat fini"

    # The equivalent board spelling follows the same final transaction and numerical result.
    equivalent = compile_euler_artifact(build_euler_predef("euler_predef"), cells=n)
    equivalent_simulation = _bind_final_artifact(
        equivalent, initial_state={"gas": initial.copy()}
    )
    equivalent_report = pops.run(
        equivalent_simulation, t_end=1.0e-4, max_steps=1
    )
    assert equivalent_report.accepted_steps == 1
    equivalent_final = np.asarray(
        equivalent_simulation.get_state("gas"), dtype=np.float64
    )
    assert np.all(np.isfinite(equivalent_final))
    difference = float(np.max(np.abs(equivalent_final - final)))
    assert difference < 1e-10
    print("OK  final lifecycle + equivalent board model (dmax=%.3e)" % difference)


def modelspec_substeps_check():
    """substeps= doit etre forwarde pour un ModelSpec (pas seulement pour un CompiledModel) : la
    branche ModelSpec d'add_equation appelle _s.add_block DIRECTEMENT avec nsub (pas self.add_block,
    qui retomberait sur time.substeps et IGNORERAIT l'override). Verifie via un espion sur _s.add_block."""
    s = System(n=16, periodicity=(True, True))
    spec = engine.Model(state=engine.FluidState("isothermal", cs2=1.0), transport=engine.IsothermalFlux(),
                     source=engine.NoSource(), elliptic=engine.ChargeDensity(charge=-1.0))
    calls = []

    class _Spy:
        def add_block(self, *a):
            calls.append(a)

    s._s = _Spy()
    # _s.add_block positional : (name, model, limiter, flux, recon, time_kind, substeps, evolve)
    s.add_equation("ions", spec, time=engine.Explicit(), substeps=10)
    assert calls, "add_equation(ModelSpec) doit appeler _s.add_block"
    assert calls[0][6] == 10, "substeps= ignore pour ModelSpec : recu %r" % (calls[0][6],)
    calls.clear()
    s.add_equation("ions2", spec, time=engine.Explicit(substeps=3))   # defaut = time.substeps
    assert calls[0][6] == 3, "defaut substeps != time.substeps : recu %r" % (calls[0][6],)
    print("OK  substeps= override forwarde pour ModelSpec (10) ; defaut = time.substeps (3)")


def predef_primitive_selfref_check():
    """The final board route has one unambiguous primitive declaration per name."""
    m = build_euler_predef("euler_predef_pp")
    assert set(m.fluxes) == {"transport"}
    assert "transport" in m.module.operator_registry().names()
    print("OK  declarations primitives finalisees sans chemin self-reference legacy")


def main():
    pure_python_checks()
    predef_primitive_selfref_check()
    modelspec_substeps_check()
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        require_native_or_skip('skip  bout-en-bout (compilateur ou en-tetes pops absents)')
    else:
        end_to_end_checks()
    print("test_dsl_phase_a : tout est vert")


if __name__ == "__main__":
    main()
