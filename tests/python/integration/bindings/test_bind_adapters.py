#!/usr/bin/env python3
"""ADC-583 : System / AmrSystem cachés derrière les adaptateurs de runtime de pops.bind.

pops.bind ne rend plus le moteur C++ brut : il construit un adaptateur interne (Uniform ou AMR,
pops.runtime._bind_adapters), qui lowerise les objets validés de la Problem sur le seam interne
_install_compiled, puis emballe le moteur dans une VUE BoundSimulation (pops.runtime._bound_sim).
La vue expose la surface run / data / diagnostics / io et CACHE le vocabulaire d'assemblage
(add_block / add_equation / set_poisson / set_refinement / install_program / ...).

Ce que l'on prouve LOCALEMENT (sous un _pops déjà construit, sans codegen / Kokkos) :

  1. Sélection d'adaptateur : adapter_for('system', None) -> Uniform, adapter_for('amr_system',
     layout) -> AMR, et 'amr_system' sans layout lève le TypeError existant (message inchangé).
  2. Blocage de la facade : sur un moteur System REEL (route bas-niveau interne), chaque nom
     d'assemblage bloqué lève AttributeError, le message parle pops.Problem / pops.compile / pops.bind
     et ne recommande JAMAIS System / AmrSystem / set_poisson / install_program / set_refinement ;
     un attribut inconnu lève aussi.
  3. Délégation de la facade : on pilote un System (un bloc natif installé par la route interne)
     UNIQUEMENT via la vue (set_density / step_cfl / density / mass / inspect / str), les valeurs
     restent finies + masse conservée, et sim._engine EST le System.
  4. Facade AMR : idem sur un petit AmrSystem natif (step / mass / n_patches / patch_rectangles /
     vue amr atteignable ; set_refinement bloqué).
  5. Mapping _amr_config_from_layout (fonction déplacée dans le nouveau module).

Gated compilateur (skip propre comme les autres tests DSL) : le flux complet
pops.Problem -> pops.compile -> pops.bind sur Uniform (a besoin de cxx / include / Kokkos).

Ne FALSIFIE jamais le moteur pops : on construit un vrai System / AmrSystem par la route interne
(légitime dans un test bas-niveau) ou on appelle des helpers purs ; on skippe si l'environnement
manque une dépendance. Tourne sous pytest ET comme script (le garde __main__ ci-dessous)."""
import sys

try:
    import numpy as np
    import pops
    from tests.python.support.typed_program import program_states, synthetic_module
    from pops.runtime._bind_adapters import (
        adapter_for, _UniformRuntimeAdapter, _AmrRuntimeAdapter, _amr_config_from_layout)
    from pops.runtime._bound_sim import BoundSimulation
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    from pops.mesh.amr import PatchLayout, ProperNesting, RegridEvery
except Exception as exc:  # noqa: BLE001
    print("skip test_bind_adapters (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check
from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam


# Noms d'assemblage que la vue DOIT cacher (ceux qui existent sur au moins un moteur + le seam).
_BLOCKED_SAMPLE = ("add_block", "add_equation", "add_background", "add_coupling",
                   "add_elliptic_model", "set_poisson", "set_source_stage", "install_program",
                   "set_refinement", "set_phi_refinement", "_install_compiled")
# Vocabulaire hérité que le message d'un nom bloqué ne doit JAMAIS recommander comme le REMEDE.
_FORBIDDEN_IN_MESSAGE = ("System.", "AmrSystem", "set_poisson", "install_program", "set_refinement")


def _isothermal_model():
    """Un pops.Model(...) natif (briques composées, PAS de compile DSL) pour un System uniforme."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.PotentialForce(charge=1.0),
                      elliptic=pops.ChargeDensity(charge=1.0))


def _compressible_model():
    """Un pops.Model(...) natif compressible pour le transport pur sur une hierarchie AMR."""
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                      transport=pops.CompressibleFlux(), source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))


from tests.python.support.initial_states import bubble_offset as _bubble  # noqa: E402


# --- 1. Sélection d'adaptateur ---------------------------------------------------------------
def test_adapter_selection():
    """adapter_for choisit l'adaptateur d'apres le TARGET produit par le layout de la Problem."""
    _check(isinstance(adapter_for("system", None), _UniformRuntimeAdapter),
           "target='system' -> l'adaptateur Uniform")
    layout = AMR(CartesianMesh(n=16))
    _check(isinstance(adapter_for("amr_system", layout), _AmrRuntimeAdapter),
           "target='amr_system' avec layout -> l'adaptateur AMR")
    try:
        adapter_for("amr_system", None)
        raise AssertionError("un target AMR sans layout doit lever")
    except TypeError as exc:
        _check("no layout descriptor" in str(exc),
               "le message TypeError existant (layout manquant) est preserve")
    print("ok test_adapter_selection")


# --- 2. Blocage de la facade -----------------------------------------------------------------
def test_bound_simulation_blocks_assembly_vocabulary():
    """La vue cache tout le vocabulaire d'assemblage ; un inconnu leve aussi ; message propre."""
    engine = System(n=8, L=1.0, periodic=True)  # moteur REEL (route interne bas-niveau)
    sim = BoundSimulation(engine)
    _check(sim._engine is engine, "sim._engine est le moteur interne (echappatoire documentee)")
    for name in _BLOCKED_SAMPLE:
        try:
            getattr(sim, name)
            raise AssertionError("l'attribut d'assemblage %r doit etre cache" % name)
        except AttributeError as exc:
            msg = str(exc)
            _check("pops.Problem" in msg, "le message de %r mentionne pops.Problem" % name)
            _check("pops.compile" in msg and "pops.bind" in msg,
                   "le message de %r mentionne pops.compile / pops.bind" % name)
            # Le message nomme HONNETEMENT l'attribut bloque (le contexte), mais ne doit recommander
            # AUCUN AUTRE appel herite comme remede : on exclut donc le nom teste lui-meme.
            for bad in _FORBIDDEN_IN_MESSAGE:
                if bad.rstrip("(") == name or bad.rstrip(".") == name:
                    continue
                _check(bad not in msg,
                       "le message de %r ne recommande pas %r" % (name, bad))
    # Un attribut arbitraire inconnu leve aussi (la vue ne passe rien en silence).
    try:
        _ = sim.totally_unknown_attribute
        raise AssertionError("un attribut inconnu doit lever AttributeError")
    except AttributeError as exc:
        _check("bound simulation" in str(exc), "le message nomme la surface de la bound simulation")
    print("ok test_bound_simulation_blocks_assembly_vocabulary")


# --- 3. Délégation de la facade (Uniform) ----------------------------------------------------
def test_bound_simulation_delegates_uniform():
    """On pilote un System UNIQUEMENT via la vue : set_density / step_cfl / density / mass / str."""
    n = 16
    engine = System(n=n, L=1.0, periodic=True)  # route interne bas-niveau (legitime en test)
    engine.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    engine.add_block("ions", _isothermal_model(),
                     spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())

    sim = BoundSimulation(engine)
    # Toutes les mutations / lectures / le pas passent par la VUE seule (System multi-bloc : la
    # densite / la masse sont indexees par le nom du bloc).
    sim.set_density("ions", _bubble(n))
    m0 = sim.mass("ions")
    sim.solve_fields()
    for _ in range(4):
        sim.step_cfl(0.4)
    rho = np.array(sim.density("ions"))
    _check(np.isfinite(rho).all(), "la densite reste finie apres des pas via la vue")
    _check(sim.mass("ions") > 1e-6, "la masse reste positive")
    _check(abs(sim.mass("ions") - m0) < 1e-9 * (abs(m0) + 1.0),
           "la masse est conservee (transport periodique)")
    _check(sim.block_names() == ["ions"], "block_names() delegue au moteur")
    _check(isinstance(sim.inspect(), object), "inspect() delegue au moteur")
    _check(str(sim).startswith("BoundSimulation(System("),
           "str() nomme la bound simulation et resume le moteur")
    _check(sim._engine is engine, "sim._engine est bien le System")
    print("ok test_bound_simulation_delegates_uniform")


def test_set_potential_delegated():
    """set_potential -- the one allowlist mutation not yet driven through the view -- roundtrips.

    Write phi through the BoundSimulation view, read it back with potential() (also through the
    view): both forward to the engine, the value is finite and returned bit-for-bit.
    """
    n = 16
    engine = System(n=n, L=1.0, periodic=True)  # route interne bas-niveau (legitime en test)
    sim = BoundSimulation(engine)
    phi = (np.arange(n * n, dtype=np.float64) / (n * n)).reshape(n, n)
    sim.set_potential(phi.ravel())
    got = np.array(sim.potential())
    _check(np.isfinite(got).all(), "le potentiel reste fini apres set_potential via la vue")
    _check(got.shape == (n, n), "potential() rend une grille n x n via la vue")
    _check(np.array_equal(got.ravel(), phi.ravel()),
           "set_potential -> potential() roundtrip bit-a-bit a travers la vue")
    _check(sim._engine is engine, "les deux appels delegent au moteur (sim._engine)")
    print("ok test_set_potential_delegated")


# --- 4. Facade AMR ---------------------------------------------------------------------------
def test_bound_simulation_delegates_amr():
    """Idem sur un petit AmrSystem natif : step / mass / n_patches / vue amr ; set_refinement cache."""
    n = 16
    engine = AmrSystem(n=n, L=1.0)  # route interne bas-niveau
    engine.set_poisson("charge_density", "geometric_mg")
    engine.add_block("gas", _compressible_model(),
                     spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    engine.set_density("gas", _bubble(n))

    sim = BoundSimulation(engine)
    m0 = sim.mass()
    for _ in range(4):
        sim.step(2e-4)
    _check(sim.mass() > 1e-6, "la masse AMR reste positive")
    _check(abs(sim.mass() - m0) < 1e-9 * (abs(m0) + 1.0), "masse conservee sur la hierarchie AMR")
    _check(sim.n_patches() >= 0, "n_patches() delegue (la hierarchie est interrogeable)")
    _check(isinstance(sim.patch_rectangles(), list), "patch_rectangles() delegue au moteur AMR")
    # La vue amr est atteignable (propriete AmrSystem -> AmrRuntimeView).
    view = sim.amr
    _check(view is not None and hasattr(view, "patch_table"), "sim.amr rend la vue AMR")
    # set_refinement (assemblage / raffinement) est cache : il se declare sur le layout AMR.
    try:
        _ = sim.set_refinement
        raise AssertionError("set_refinement doit etre cache sur la bound simulation AMR")
    except AttributeError as exc:
        _check("pops.Problem" in str(exc), "le rejet AMR parle pops.Problem")
    _check(sim._engine is engine, "sim._engine est bien l'AmrSystem")
    print("ok test_bound_simulation_delegates_amr")


# --- 5. Mapping _amr_config_from_layout (fonction deplacee) ----------------------------------
def test_amr_config_from_layout_mapping():
    """La fonction deplacee dans _bind_adapters produit toujours le bon AmrSystemConfig."""
    layout = AMR(CartesianMesh(n=64, L=1.5, periodic=False), max_levels=2, ratio=2,
                 regrid=RegridEvery(8),
                 patches=PatchLayout(distribute_coarse=True, coarse_max_grid=16))
    cfg = _amr_config_from_layout(layout)
    _check(cfg.n == 64, "n depuis le CartesianMesh de base")
    _check(cfg.L == 1.5, "L depuis le CartesianMesh de base")
    _check(cfg.periodic is False, "periodic depuis le CartesianMesh de base")
    _check(cfg.regrid_every == 8, "regrid_every depuis RegridEvery(8)")
    _check(cfg.distribute_coarse is True, "distribute_coarse depuis PatchLayout")
    _check(cfg.coarse_max_grid == 16, "coarse_max_grid depuis PatchLayout")
    print("ok test_amr_config_from_layout_mapping")


def test_amr_config_refuses_untransported_hierarchy_semantics():
    """Level-count and nesting intent must never disappear in the legacy config adapter."""
    for layout, expected in (
        (AMR(CartesianMesh(n=32), max_levels=1), "no silent level-count substitution"),
        (AMR(CartesianMesh(n=32), nesting=ProperNesting(buffer=3)), "buffer/lookahead"),
    ):
        try:
            _amr_config_from_layout(layout)
            raise AssertionError("untransported AMR hierarchy semantics must be refused")
        except NotImplementedError as exc:
            _check(expected in str(exc), "structured pre-runtime hierarchy refusal")


# --- Gated compilateur : le flux complet Problem -> compile -> bind sur Uniform ------------------
def _dsl_isothermal_model(name="adc583_bind_iso"):
    """Un modele DSL isotherme MINIMAL et VALIDE (facade pops.physics), compilable en Program .so.

    Miroir simplifie de test_unified_install._lorentz_model (sans la source de Lorentz, donc sans
    aux B_z requis) : flux + valeurs propres isothermes cs2=0.5, primitives identite, rhs elliptique
    sur rho, rate_operator explicite."""
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model as FacadeModel

    m = FacadeModel(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs = sqrt(0.5)
    m.flux(x=[mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + 0.5 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)
    m.conservative_from([rho, mx, my])
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def _lie_program(block="ne", name="adc583_bind_prog"):
    """Un time Program Lie VALIDE (miroir de test_unified_install._lie_program) : lit l'etat du
    bloc, resout les champs, avance d'un pas d'Euler explicite et COMMIT le bloc (l'exigence du
    compile 'chaque bloc avance est committe exactement une fois' est donc satisfaite)."""
    P = pops.time.Program(name)
    module = synthetic_module("%s_state" % name, components=("rho", "mx", "my"))
    _case, states = program_states(P, module, (block,))
    temporal = states[block]
    u = temporal.n
    fields = P.solve_fields(u)
    r = P._rhs_legacy(state=u, fields=fields)
    P.commit(temporal.next, P.linear_combine(
        "u1", u + P.dt * r, at=temporal.next.point))
    return P


def test_full_bind_flow_uniform_gated():
    """pops.Problem -> pops.compile -> pops.bind (Uniform) rend une BoundSimulation ; add_equation cache.

    L'AUTHORING (modele DSL + Program Lie) est VALIDE et HORS du try : une erreur d'ecriture fait
    ECHOUER le test, jamais le skipper (pas de test fantome). La SEULE barriere locale est le
    compile .so (headers / cxx / Kokkos, ex. 'pops headers not found ... set POPS_INCLUDE'), qui
    skippe en nommant le TYPE d'exception (diagnosable en CI). Sur ROMEO / CI-Kokkos le compile
    aboutit et l'assertion complete tourne (BoundSimulation + run + rejet du setter)."""
    from pops.mesh.layouts import Uniform

    # Authoring valide, hors du try : une regression d'ecriture FAIT ECHOUER le test.
    # NB : la route Uniform de bind derive desormais la SystemConfig (n / L / periodic) du maillage
    # de la Problem (compile pose _layout=problem.layout sur Uniform), donc un n NON par defaut circule
    # jusqu'au moteur. On declare n=16 pour VERROUILLER le fix : avant, bind construisait un System a
    # n=64 par defaut et un etat 16x16 echouait a l'install ('taille != ncomp*n*n').
    n = 16
    m = _dsl_isothermal_model()
    prog = _lie_program(block="ne")
    case = (pops.Problem(layout=Uniform(CartesianMesh(n=n, L=1.0, periodic=True)))
            .block("ne", physics=m))

    # La SEULE barriere locale : l'emit + compile du Program .so (headers / cxx / Kokkos).
    try:
        compiled = pops.compile(case, time=prog)
    except Exception as exc:  # noqa: BLE001 - barriere toolchain -> skip diagnosable
        print("skip test_full_bind_flow_uniform_gated (toolchain %s: %s)"
              % (type(exc).__name__, str(exc)[:140]))
        return

    xs = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho0, 0.4 * rho0, -0.2 * rho0])
    # solvers= comme dans test_unified_install (le Program fait solve_fields -> Poisson 'phi').
    sim = pops.bind(compiled, state={"ne": u0},
                    solvers={"phi": pops.fields.catalog.GeometricMG()})
    _check(type(sim).__name__ == "BoundSimulation", "pops.bind rend une BoundSimulation")
    sim.run(t_end=0.01, cfl=0.4, max_steps=4)
    try:
        _ = sim.add_equation
        raise AssertionError("add_equation doit etre cache sur la bound simulation")
    except AttributeError:
        pass
    _check(hasattr(sim._engine, "_install_compiled"),
           "sim._engine expose le seam interne _install_compiled")
    print("ok test_full_bind_flow_uniform_gated")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
