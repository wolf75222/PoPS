"""Echange CONSERVATIF par paire (pops.dsl.CoupledSource.add_pair) -- finding A2.

L'API manuelle .add(+expr) / .add(-expr) sur deux blocs NE GARANTIT PAS la conservation : rien
n'empeche d'ecrire par megarde deux formules legerement differentes -> la quantite totale derive
silencieusement. add_pair(block_a, block_b, role, expr) decrit l'echange par UNE seule expression
et emet +expr sur block_a, -expr sur block_b (MEME sous-arbre, signe oppose) : conservatif PAR
CONSTRUCTION, exactement comme les couplages NOMMES du C++ (add_collision / add_thermal_exchange).

Invariants verifies :
(A) add_pair construit DEUX termes opposes : out_blocks = [a, b], meme role, programmes de meme
    longueur, et la reference numpy donne dS_a == -dS_b a chaque point.
(B) Numerique de bout en bout : echange de DENSITE entre deux especes (gain pour A, perte pour B) sur
    un etat SPATIALEMENT UNIFORME (transport nul). On verifie a chaque pas que (i) l'etat est FINI
    (pas de nan/inf) AVANT toute autre assertion, (ii) la masse totale n_a + n_b est invariante a
    ~1e-12, (iii) la trajectoire colle a la reference forward-Euler (memes Expr).
(C) verify_conservation=True : un couplage NON conservatif ecrit a la main (deux .add avec des
    coefficients differents) leve une ValueError EXPLICITE ; un couplage par add_pair passe.
(D) add_pair refuse block_a == block_b.
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


def build_exchange(k):
    """Echange conservatif de densite de 'beta' vers 'alpha' : alpha GAGNE +k*na*nb, beta PERD
    -k*na*nb (transfert proportionnel au produit des densites). Conservatif par add_pair."""
    src = CoupledSource("exchange")
    na = src.block("alpha").role("density")
    nb = src.block("beta").role("density")
    kp = src.param("Kx", k)
    src.add_pair("alpha", "beta", role="density", expr=kp * na * nb)
    return src.compile(backend="production", verify_conservation=True)


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

    model = Model("coupled-source-conservation-density-%s" % block_name, frame=FRAME)
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

    case = pops.Case("coupled-source-conservation-density-case-%s" % block_name)
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


def make_system(n, na0, nb0):
    from pops._native_selector import select_native_dimension

    select_native_dimension(2)
    from pops.runtime import _engine_descriptors as engine
    from pops.runtime._system import System

    components = [(name, *density_component(n, name)) for name in ("alpha", "beta")]
    sim = System(system_config_2d(n))
    install_runtime_lane(sim, components)
    for name, component, _artifact in components:
        sim.add_equation(
            name,
            component,
            spatial=engine.Spatial(none=True),
            time=engine.Explicit(),
        )
    finalize_provider_pack(sim)
    sim.set_poisson(rhs="charge_density", solver="cartesian_cg")
    sim.set_density("alpha", np.full((n, n), na0))
    sim.set_density("beta", np.full((n, n), nb0))
    return sim


def all_finite(*arrs):
    return all(np.all(np.isfinite(a)) for a in arrs)


def main():
    fails = [0]
    n = 16
    k = 0.5
    dt = 0.01
    nsteps = 30
    na0, nb0 = 0.20, 0.90

    compiled = build_exchange(k)

    # --- (A) add_pair = deux termes opposes (meme role, meme corps, signe inverse) ---
    chk(compiled.out_blocks == ["alpha", "beta"], "add_pair : out_blocks = [a, b]", fails)
    chk(compiled.out_roles == ["density", "density"], "add_pair : meme role sur les deux legs", fails)
    # Le leg PERTE est EXACTEMENT le leg GAIN suivi d'un seul opcode NEG (5) : meme programme evalue,
    # signe oppose -> echange conservatif par construction (et non deux formules ecrites separement).
    chk(len(compiled.prog_lens) == 2, "add_pair : deux programmes", fails)
    if len(compiled.prog_lens) == 2:
        lg, ll = compiled.prog_lens
        gain_ops = compiled.prog_ops[:lg]
        gain_args = compiled.prog_args[:lg]
        loss_ops = compiled.prog_ops[lg:lg + ll]
        loss_args = compiled.prog_args[lg:lg + ll]
        chk(ll == lg + 1 and loss_ops == gain_ops + [5] and loss_args == gain_args + [0],
            "add_pair : leg perte = leg gain + NEG (meme Expr, signe oppose)", fails)
    # reference numpy : sur un etat test, dS_alpha == -dS_beta point par point (echange exact).
    probe = {("alpha", "density"): np.array([0.2, 0.5, 0.7]),
             ("beta", "density"): np.array([0.9, 0.4, 0.3])}
    ref = {b: dS for (b, _r, dS) in compiled.reference_terms(probe)}
    chk(all_finite(ref["alpha"], ref["beta"]), "reference termes finis", fails)
    chk(np.allclose(ref["alpha"], -ref["beta"], atol=1e-14),
        "add_pair : dS_alpha == -dS_beta (echange exact)", fails)

    # --- (B) bout en bout : masse n_a + n_b conservee, trajectoire == reference forward-Euler ---
    sim = make_system(n, na0, nb0)
    from tests.python.support.explicit_program import install_forward_euler_program

    sim.add_coupling(compiled)
    install_forward_euler_program(sim, coupled_sources=True)

    na, nb = na0, nb0
    traj = []
    for _ in range(nsteps):
        fields = {("alpha", "density"): np.array([na]),
                  ("beta", "density"): np.array([nb])}
        terms = {b: float(dS[0]) for (b, _r, dS) in compiled.reference_terms(fields)}
        na = na + dt * terms["alpha"]
        nb = nb + dt * terms["beta"]
        traj.append((na, nb))

    total0 = na0 + nb0
    for s, (rna, rnb) in enumerate(traj):
        sim.step(dt)
        ga = sim.density("alpha")
        gb = sim.density("beta")
        # (i) FINITUDE d'abord : on rejette explicitement nan/inf avant toute tolerance.
        if not chk(all_finite(ga, gb), "etat fini (pas de nan/inf) au pas %d" % s, fails):
            break
        # etat reste spatialement uniforme (source uniforme) -> transport nul
        chk(np.ptp(ga) < 1e-12 and np.ptp(gb) < 1e-12, "etat reste uniforme au pas %d" % s, fails)
        # (ii) CONSERVATION : masse totale invariante a ~1e-12
        total = ga.mean() + gb.mean()
        chk(abs(total - total0) < 1e-12, "n_a + n_b conserve a 1e-12 au pas %d (%.16g vs %.16g)"
            % (s, total, total0), fails)
        # (iii) trajectoire == reference forward-Euler (memes Expr)
        chk(abs(ga.mean() - rna) < 1e-10, "n_a == ref ODE au pas %d" % s, fails)
        chk(abs(gb.mean() - rnb) < 1e-10, "n_b == ref ODE au pas %d" % s, fails)

    # sens physique : alpha a GAGNE, beta a PERDU (na0 < nb0 -> transfert positif net de beta vers alpha)
    chk(sim.density("alpha").mean() > na0 + 1e-6, "alpha a gagne (+expr)", fails)
    chk(sim.density("beta").mean() < nb0 - 1e-6, "beta a perdu (-expr)", fails)

    # --- (C) verify_conservation=True attrape un couplage manuel NON conservatif ---
    bad = CoupledSource("bad")
    na2 = bad.block("alpha").role("density")
    nb2 = bad.block("beta").role("density")
    k1 = bad.param("k1", 0.5)
    k2 = bad.param("k2", 0.7)  # coefficient DIFFERENT -> les deux legs ne se compensent pas
    bad.add("alpha", role="density", expr=+k1 * na2 * nb2)
    bad.add("beta", role="density", expr=-k2 * na2 * nb2)
    raised = False
    try:
        bad.compile(backend="production", verify_conservation=True)
    except ValueError as e:
        raised = "verify_conservation" in str(e).lower()
    chk(raised, "verify_conservation attrape le couplage manuel non conservatif", fails)
    # le MEME couplage SANS le flag reste licite (retro-compat : opt-in)
    ok_without_flag = True
    try:
        bad.compile(backend="production")  # defaut verify_conservation=False
    except Exception:
        ok_without_flag = False
    chk(ok_without_flag, "sans le flag, le couplage non conservatif compile (opt-in)", fails)

    # couplage par add_pair PASSE verify_conservation (deja exerce dans build_exchange, redondance ciblee)
    ok_pair = True
    try:
        build_exchange(0.3)
    except Exception as e:
        print("FAIL add_pair rejete a tort par verify_conservation :", e)
        ok_pair = False
    chk(ok_pair, "add_pair passe verify_conservation", fails)

    # --- (D) add_pair refuse block_a == block_b ---
    raised_same = False
    try:
        s2 = CoupledSource("degenere")
        f = s2.block("alpha").role("density")
        s2.add_pair("alpha", "alpha", role="density", expr=f)
    except ValueError:
        raised_same = True
    chk(raised_same, "add_pair refuse block_a == block_b", fails)

    # --- (E) NAMED PRESETS lower to a declared coupling contract (ADC-595) ---
    from pops.physics.coupling_presets import (collision_preset, ionization_preset,
                                               thermal_exchange_preset)
    from pops.physics.roles import StateSchema
    fluid = StateSchema.resolve(
        ("density", "momentum:0", "momentum:1", "energy"), dimension=2,
        where="coupled-source conservation fluid")
    density = StateSchema.resolve(("density",), dimension=2,
                                  where="coupled-source conservation density")
    # Collision conserves momentum: its declared contract passes verify_declared_contract, and the
    # add_pair legs make every resolved momentum-axis term cancel structurally.
    col = collision_preset("a", "b", 0.7, a_schema=fluid, b_schema=fluid)
    chk(col.conserved == ["momentum:0", "momentum:1"], "collision preset declares momentum conserved",
        fails)
    col_ok = True
    try:
        col.source.verify_declared_contract(conserved=col.conserved, created=col.created)
    except Exception as e:
        print("FAIL collision preset contract:", e)
        col_ok = False
    chk(col_ok, "collision preset satisfies its declared conserved-momentum contract", fails)
    # A collision that FALSELY declared it also conserves density must be rejected (density is only read,
    # never a source term), proving the contract is checked, not trusted.
    raised_bad_col = False
    try:
        col.source.verify_declared_contract(conserved=["momentum:0", "momentum:1", "density"])
    except ValueError:
        raised_bad_col = True
    chk(raised_bad_col, "collision preset rejects a bogus extra conserved role (density)", fails)

    # Ionization legally NET-SOURCES density (an electron/ion pair is created): it is declared CREATED,
    # so the contract validator accepts the net source; declaring density CONSERVED must instead raise.
    ion = ionization_preset(
        "e", "i", "g", 1.7,
        electron_schema=density, ion_schema=density, neutral_schema=density)
    chk(ion.created == ["density"] and ion.conserved == [],
        "ionization preset declares density created (net source)", fails)
    ion_ok = True
    try:
        ion.source.verify_declared_contract(conserved=ion.conserved, created=ion.created)
    except Exception as e:
        print("FAIL ionization preset created-contract:", e)
        ion_ok = False
    chk(ion_ok, "ionization preset legally net-sources under declared created", fails)
    raised_ion = False
    try:
        ion.source.verify_declared_contract(conserved=["density"])
    except ValueError:
        raised_ion = True
    chk(raised_ion, "ionization density-declared-conserved raises (it net-sources)", fails)

    # ThermalExchange conserves energy (add_pair on energy); its contract passes.
    th = thermal_exchange_preset(
        "a", "b", 0.3, 1.4, 1.6667, a_schema=fluid, b_schema=fluid)
    th_ok = True
    try:
        th.source.verify_declared_contract(conserved=th.conserved, created=th.created)
    except Exception as e:
        print("FAIL thermal preset contract:", e)
        th_ok = False
    chk(th.conserved == ["energy"] and th_ok, "thermal_exchange preset conserves energy", fails)

    if fails[0] == 0:
        print("test_dsl_coupled_source_conservation : OK")
    else:
        print("test_dsl_coupled_source_conservation : %d FAIL" % fails[0])
    return fails[0]


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
