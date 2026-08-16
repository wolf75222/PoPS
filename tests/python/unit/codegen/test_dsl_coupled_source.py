"""Source COUPLEE GENERIQUE inter-especes par le DSL (pops.dsl.CoupledSource, P5 phase 1, splitting
EXPLICITE). On decrit un couplage d'IONISATION en FORMULES -- pas la brique nommee add_ionization --
et on verifie qu'il s'applique comme un etage operator-split APRES le transport, AUCUN callback Python
par cellule (le bytecode est interprete cote C++ dans le for_each_cell device) :

    d_t n_e = +k n_e n_g     (un electron apparait)
    d_t n_i = +k n_e n_g     (un ion apparait)
    d_t n_g = -k n_e n_g     (un neutre disparait)

Invariants verifies :
(A) API : pops.dsl.CoupledSource(...).block(...).role(...) / .param(...) / .add(...) / .compile(...)
    construit l'ABI plate (bytecode) et sim.add_coupling(...) la branche sur System.add_coupled_source.
(B) Numerique : densites SPATIALEMENT UNIFORMES -> le transport (flux + derive E x B) est exactement
    nul a chaque pas (champ uniforme), donc seules les sources couplees evoluent l'etat. On compare la
    trajectoire a une REFERENCE NumPy de l'ODE forward-Euler (les MEMES Expr, evaluees par
    CompiledCoupledSource.reference_terms) -- bit-pour-bit la meme recurrence.
(C) Conservation : n_e et n_i CROISSENT, n_g DECROIT ; n_i + n_g (masse lourde) est conserve, et
    n_e - n_i reste constant (chaque ionisation cree une paire e/i). La masse totale n_e+n_i+n_g n'est
    PAS conservee (une paire e/i est creee par neutre consomme) -- c'est l'invariant ATTENDU.
(D) Defaut : un System SANS add_coupling reste BIT-IDENTIQUE (aucune evolution des densites uniformes).
(E) Pas de callback Python par cellule : la source est compilee en bytecode (verifie sur l'objet) ;
    l'evolution se produit dans System.step sans rappel Python.
"""
from pathlib import Path

import numpy as np

import pops
from pops.codegen import Production
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Density, Model
from pops.physics.multispecies import CoupledSource
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


ROOT = Path(__file__).resolve().parents[4]
_DENSITY_COMPONENTS = {}


def chk(cond, msg, fails):
    if not cond:
        print("FAIL", msg)
        fails[0] += 1
    return cond


def build_source(k):
    """Construit la source d'ionisation generique (3 especes) par le DSL et la compile."""
    src = CoupledSource("ionization")
    ne = src.block("electrons").role("density")
    src.block("ions").role("density")
    ng = src.block("neutrals").role("density")
    kp = src.param("Kiz", k)
    src.add("electrons", role="density", expr=+kp * ne * ng)
    src.add("ions", role="density", expr=+kp * ne * ng)
    src.add("neutrals", role="density", expr=-kp * ne * ng)
    return src.compile(backend="production")


def system_config_2d(n):
    """Return the complete exact-rank Cartesian authority for this uniform witness."""
    from pops.runtime._system import SystemConfig

    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    config.coordinate_system = "pops://coordinates/cartesian-2d@1"
    return config


def density_component(n, block_name):
    """Compile one fixture-unique passive density Module through the public Case lifecycle."""
    cache_key = (n, block_name)
    cached = _DENSITY_COMPONENTS.get(cache_key)
    if cached is not None:
        return cached

    model = Model("coupled-source-density-%s" % block_name, frame=FRAME)
    state = model.state("U", components=("density",), roles={"density": Density()})
    (rho,) = state
    flux = model.flux(
        "inert-transport",
        frame=FRAME,
        state=state,
        components={X_AXIS: (0.0 * rho,), Y_AXIS: (0.0 * rho,)},
        waves={X_AXIS: (0.0 * rho,), Y_AXIS: (0.0 * rho,)},
    )
    rate = model.rate("inert-rate", equation=ddt(state) == -div(flux))

    case = pops.Case("coupled-source-density-case-%s" % block_name)
    block = case.block("density", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    layout = Uniform(
        CartesianGrid(frame=FRAME, cells=(n, n), periodic=PeriodicAxes(FRAME.axes))
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    component = artifact.blocks[0].model
    _DENSITY_COMPONENTS[cache_key] = (component, artifact)
    return component, artifact


def install_runtime_lane(sim, components):
    """Install this fixture's exact runtime authority before package materialization."""
    context = artifact_execution_context(components[0][2])
    sim._execution_context = context
    sim._s._prepare_boundary_execution_lane(
        context.communicator.handle,
        context.identity.token,
    )
    for block_name, _component, artifact in components:
        (state_identity,) = artifact.plan.blocks[0].state_identities
        sim._s._install_block_state_route(block_name, state_identity)


def finalize_provider_pack(sim):
    """Materialize all staged PreparedSystemBlocks through their one sealed ProviderPack."""
    if sim._pending_native_packages:
        sim._s._finalize_native_packages()
        sim._pending_native_packages = 0


def make_system(n, ne0, ni0, ng0):
    # This two-axis fixture owns the exact Dim=2 variant before its private runtime seam imports.
    # Do not construct a ModelSpec or call the retired native add_block path here.
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from pops.runtime import _engine_descriptors as engine
    from pops.runtime._system import System

    components = [(name, *density_component(n, name))
                  for name in ("electrons", "ions", "neutrals")]
    sim = System(system_config_2d(n))
    install_runtime_lane(sim, components)
    for name, component, _artifact in components:
        sim.add_equation(name, model=component, spatial=engine.Spatial(none=True))
    finalize_provider_pack(sim)
    # The generated model has a zero elliptic contribution, so Poisson stays on the production path
    # without changing this uniform-source witness.
    sim.set_poisson(rhs="charge_density", solver="cartesian_cg")
    sim.set_density("electrons", np.full((n, n), ne0))
    sim.set_density("ions", np.full((n, n), ni0))
    sim.set_density("neutrals", np.full((n, n), ng0))
    return sim


def main():
    fails = [0]
    n = 16
    k = 0.7
    dt = 0.01
    nsteps = 25
    ne0, ni0, ng0 = 0.30, 0.10, 1.00

    compiled = build_source(k)

    # (A) ABI plate construite par le DSL : 3 entrees (densites), 1 constante (k), 3 termes de sortie.
    chk(compiled.in_blocks == ["electrons", "ions", "neutrals"], "in_blocks ordre/contenu", fails)
    chk(compiled.in_roles == ["density", "density", "density"], "in_roles canoniques", fails)
    chk(compiled.consts == [k], "constante k inlinee", fails)
    chk(compiled.out_blocks == ["electrons", "ions", "neutrals"], "out_blocks", fails)
    chk(len(compiled.prog_lens) == 3 and sum(compiled.prog_lens) == len(compiled.prog_ops),
        "programmes bytecode segmentes", fails)
    # (E) bytecode (pas de callback Python) : programmes non vides, pas de fonction Python embarquee.
    chk(all(L > 0 for L in compiled.prog_lens) and not hasattr(compiled, "callback"),
        "source compilee en bytecode (aucun callback par cellule)", fails)

    # --- (D) defaut : System SANS couplage reste a l'identique (densites uniformes inchangees) ---
    base = make_system(n, ne0, ni0, ng0)
    from tests.python.support.explicit_program import install_forward_euler_program

    install_forward_euler_program(base)
    for _ in range(nsteps):
        base.step(dt)
    chk(np.allclose(base.density("electrons"), ne0, atol=1e-12), "defaut: n_e inchange (pas de couplage)", fails)
    chk(np.allclose(base.density("ions"), ni0, atol=1e-12), "defaut: n_i inchange", fails)
    chk(np.allclose(base.density("neutrals"), ng0, atol=1e-12), "defaut: n_g inchange", fails)

    # --- couplage GENERIQUE branche via sim.add_coupling(compiled) ---
    sim = make_system(n, ne0, ni0, ng0)
    sim.add_coupling(compiled)
    install_forward_euler_program(sim, coupled_sources=True)

    # --- REFERENCE NumPy : MEME recurrence forward-Euler que l'etage C++ (sources evaluees par les
    #     memes Expr ; transport nul car etat uniforme). Etat scalaire (densites uniformes). ---
    ne, ni, ng = ne0, ni0, ng0
    traj = []
    for _ in range(nsteps):
        fields = {("electrons", "density"): np.array([ne]),
                  ("ions", "density"): np.array([ni]),
                  ("neutrals", "density"): np.array([ng])}
        terms = {(b, r): float(dS[0]) for (b, r, dS) in compiled.reference_terms(fields)}
        ne = ne + dt * terms[("electrons", "density")]
        ni = ni + dt * terms[("ions", "density")]
        ng = ng + dt * terms[("neutrals", "density")]
        traj.append((ne, ni, ng))

    for s, (rne, rni, rng) in enumerate(traj):
        sim.step(dt)
        ge = sim.density("electrons")
        gi = sim.density("ions")
        gg = sim.density("neutrals")
        # densite reste SPATIALEMENT UNIFORME (la source l'est) -> transport nul, etat = scalaire
        chk(np.ptp(ge) < 1e-12 and np.ptp(gi) < 1e-12 and np.ptp(gg) < 1e-12,
            "etat reste uniforme (transport nul) au pas %d" % s, fails)
        # (B) trajectoire == reference NumPy de l'ODE (tolerance serree : meme recurrence)
        chk(abs(ge.mean() - rne) < 1e-10, "n_e == ref ODE au pas %d (%.12g vs %.12g)" % (s, ge.mean(), rne), fails)
        chk(abs(gi.mean() - rni) < 1e-10, "n_i == ref ODE au pas %d" % s, fails)
        chk(abs(gg.mean() - rng) < 1e-10, "n_g == ref ODE au pas %d" % s, fails)

    ge = sim.density("electrons").mean()
    gi = sim.density("ions").mean()
    gg = sim.density("neutrals").mean()

    # (C) sens physique : electrons et ions ont CRU, neutres ont DECRU.
    chk(ge > ne0 + 1e-6, "n_e a cru", fails)
    chk(gi > ni0 + 1e-6, "n_i a cru", fails)
    chk(gg < ng0 - 1e-6, "n_g a decru", fails)
    # invariants de conservation : n_i + n_g (masse lourde) constant ; n_e - n_i constant (paires e/i).
    chk(abs((gi + gg) - (ni0 + ng0)) < 1e-9, "n_i + n_g conserve (masse lourde)", fails)
    chk(abs((ge - gi) - (ne0 - ni0)) < 1e-9, "n_e - n_i conserve (creation de paires)", fails)

    if fails[0] == 0:
        print("test_dsl_coupled_source : OK")
    else:
        print("test_dsl_coupled_source : %d FAIL" % fails[0])
    return fails[0]


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
