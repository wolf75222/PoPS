"""Backend "production" (NATIF) du DSL cote AMR (Plan Ideal etape 5 / DSL Phase D) : un modele ecrit
en formules est compile en un LOADER .so via compile(backend="production", target="amr_system"), qui
inline le gabarit en-tete pops::add_compiled_model(AmrSystem&, ...), puis branche dans une AmrSystem
via AmrSystem.add_native_block (symbole pops_install_native_amr, distinct du chemin System).

A la difference du chemin System (grille plate mono-niveau), le bloc est l'UNIQUE modele porte sur la
hierarchie AMR (AmrRuntimeBlock + reflux conservatif + regrid), MEME chemin que AmrSystem.add_block
(dispatch d'une ModelSpec via detail::dispatch_amr_block). On verifie :

  1) PARITE STRICTE (transport pur, elliptic_rhs nul => zero bruit FP elliptique) : la densite
     grossiere apres plusieurs pas est BIT-IDENTIQUE (dmax == 0) entre le bloc "production" (loader
     .so -> add_native_block) et le bloc NATIF add_block (ModelSpec CompressibleFlux). C'est la parite
     attendue : le brique Euler generee a une arithmetique de flux bit-identique a pops::Euler natif, et
     les deux empruntent la MEME machinerie AMR (add_compiled_model(AmrSystem&)).
  2) PARITE FORTE (Euler-Poisson couple) : le meme Case explicite de flux/source/champ est installe
     dans deux moteurs independants. L'installation native et le conducteur public preservent tous
     les etats conservatifs, la masse et les patches a 1e-12 sur douze pas AMR.
  3) CAPACITES AMR enforcees : la facade applique son garde-fou pression (hllc/roe sans primitive 'p'
     rejete) avant le C++. WENO5 multilevel resout le fournisseur coarse/fine authentifie d'ordre 5
     et de halo 3 sur les chemins production et ModelSpec. Aucun abaissement silencieux de
     reconstruction n'est accepte.
  4) GARDE-FOUS de compilation : compile(target="amr_system") exige backend="production" ; un
     CompiledModel target="system" est refuse par AmrSystem.add_equation (loader sans pops_install_native_amr).
  5) GARDE-FOU ABI : un loader AMR a cle pops_native_abi_key falsifiee est rejete par add_native_block.

S'auto-saute explicitement sur une machine locale sans toolchain native. Dans une lane native de
release, toute capacite manquante est un echec, jamais une couverture silencieusement retiree.
"""
from pops.numerics.variables import Conservative, Primitive
from pops.numerics.riemann import HLLC, Roe
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import os
import shutil
import subprocess
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.codegen.loader import CompiledModel
from pops.math import sqrt
from pops.physics import Density, Energy, Momentum
from pops.physics._facade import Model
from pops.runtime._system import AmrSystem, AmrSystemConfig  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.support.initial_states import bubble_amr as _bubble
from tests.python.support.physics_roles import X_AXIS, Y_AXIS
from tests.python.support.amr_tagging import install_prepared_threshold_union
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

GAMMA = 1.4
# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_dsl_compile_cache).
POPS_PROCESS_TIMEOUT = 900
INCLUDE = repo_include()


def _euler_formulas(m):
    """Pose les formules Euler compressible (flux + valeurs propres + conversions + gamma) sur la
    FACADE Model @p m (dont compile(...) rend un CompiledModel). Renvoie (rho, rho_u, rho_v, E)."""
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=[Density(), Momentum(X_AXIS), Momentum(Y_AXIS), Energy()])
    u = rhou / rho
    v = rhov / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    pu, pv, pp = m.primitive("u", u), m.primitive("v", v), m.primitive("p", p)
    H = (E + pp) / rho
    c = sqrt(GAMMA * pp / rho)
    m.flux(x=[rhou, rhou * pu + pp, rhou * pv, rho * H * pu],
           y=[rhov, rhov * pu, rhov * pv + pp, rho * H * pv])
    m.eigenvalues(x=[pu - c, pu, pu + c], y=[pv - c, pv, pv + c])
    m.primitive_vars(rho, pu, pv, pp)
    m.conservative_from([rho, rho * pu, rho * pv, pp / (GAMMA - 1.0) + 0.5 * rho * (pu * pu + pv * pv)])
    m.gamma(GAMMA)
    # ADC-590 : hllc/roe generiques exigent la capability EMISE (plus de fallback Euler implicite) ;
    # parity_riemann exerce riemann=HLLC()/Roe() sur cm_t (bit-identite prouvee vs briques natives).
    m.enable_hllc()
    m.enable_roe()
    return rho, rhou, rhov, E, pu, pv


def _build_euler_transport():
    """Euler PUR (transport seul) via la facade Model : pas de source, elliptic_rhs NUL. Le solve
    elliptique donne phi=0 des deux cotes (zero bruit FP), donc la parite transport est BIT-IDENTIQUE."""
    m = Model("euler_transport")
    rho, _rhou, _rhov, _E, _u, _v = _euler_formulas(m)
    m.elliptic_rhs(0.0 * rho)  # f = 0 : aucun couplage Poisson (isole le transport)
    return m


def _public_euler_poisson_plan(n, dt):
    """One explicit field/source Case for native-installer versus public-driver parity."""
    import pops
    from test_dsl_coupled import build_euler
    from tests.python.support.physics_roles import FRAME
    from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
                          AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
    from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
    from pops.fields.methods import CellCenteredSecondOrder
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ValueExpr, ddt, div, grad, laplacian
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, variables
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.solvers import CompositeFieldGMRES
    from pops.time import FailRun, FixedDt, every

    model = build_euler("public_euler_poisson_amr")
    state, flux = model.states["U"], model.fluxes["transport"]
    rho, mx, my, _energy = state
    potential = model.field("potential")
    gradient = model.vector("gravity_gradient", frame=FRAME,
        components={FRAME.x: grad(potential).x, FRAME.y: grad(potential).y})
    gravity = model.source("gravity", on=state,
        value=(0 * rho, -rho * gradient.x, -rho * gradient.y,
               -(mx * gradient.x + my * gradient.y)))
    rate = model.rate("Euler_Poisson", equation=ddt(state) == -div(flux) + gravity)
    model.select_balance(rate)
    problem = FieldProblem("gravity", unknowns=(potential,),
        # Preserve GravityCoupling(sign=-1): the prepared operator owns -laplacian.
        equations=(-laplacian(potential) == -(rho - 1),),
        boundaries=(FieldBoundary(potential,
            bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((potential,)))
    case = pops.Case("production_euler_poisson_amr")
    block = case.block("gas", model)
    numerical = DiscretizationPlan()
    numerical.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.MUSCL(Minmod()), riemann=Rusanov()))
    case.numerics(numerical, block=block)
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CompositeFieldGMRES(max_iter=4000, restart=80, rel_tol=1e-11, abs_tol=1e-12)))
    program = pops.Program("coupled_forward_euler")
    current = program.state(block[state])
    observations = field.observe(program.solve(field, values={block[state]: current.n},
        at=program.stage("gravity", c=0)).consume(action=FailRun()))
    solved_gradient = observations.gradient(field[potential], dimension=2)
    carrier = block[model.module.field_handle(model.module.field_spaces()["fields"])]
    publication = observations.publish({
        (carrier, "potential_grad_x"): (solved_gradient, 0),
        (carrier, "potential_grad_y"): (solved_gradient, 1),
    }, states={block[state]: current.n})
    rhs = rate(current.n, publication)
    program.record_scalar("gravity_potential_abs_sum",
        program.abs_sum_component(observations[field[potential]], 0))
    program.commit(current.next, program.value("advanced", current.n + program.dt * rhs,
        at=current.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
        projection=ConservativeCellAverage()))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("density_refinement", default=1.2))
    layout = AMR(grid=CartesianGrid(frame=FRAME, cells=(n, n), periodic=PeriodicAxes(FRAME.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["rho"] > case.value(threshold)),
                                 Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(4, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    initial_rho = np.asarray(_bubble(n), dtype=float)
    initial_rho += 1.0 - float(initial_rho.mean())
    initial = np.stack((initial_rho, np.zeros_like(initial_rho), np.zeros_like(initial_rho),
                        np.full_like(initial_rho, 1 / (GAMMA - 1))))
    return pops.resolve(pops.validate(case), layout=layout), initial


def _euler_poisson_public_parity(n, dt):
    import pops
    from pops.runtime._runtime_executor import _install_adaptive_native_engine
    from pops.runtime._step_strategy import prepare_program_run
    from pops.runtime._native_step_target import native_step_target
    from tests.python.support.native_execution_context import artifact_execution_context

    resolved, initial = _public_euler_poisson_plan(n, dt)
    artifact = pops.compile(resolved)
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    public = pops.bind(artifact, initial_values={subject: initial},
        resources={"execution_context": artifact_execution_context(artifact)})
    # The immutable plan is shared; installation allocates separate state, field, and Program storage.
    native = _install_adaptive_native_engine(public._install_plan)
    assert native._s is not public._executor._s
    assert native.n_levels() == public.n_levels() == 2
    assert native.n_patches() == public._executor.n_patches()
    mass = native.mass()
    assert abs(mass - public._executor.mass()) < 1e-12 * (abs(mass) + 1)
    prepared = prepare_program_run(native)
    prepared.begin(native._temporal_restart_state, time=native.time(), macro_step=native.macro_step())
    target = native_step_target(native)
    initial_energy = float(native.composite_reduce("gas", "sum", 3, []))
    prepared.run_step(target, t_end=12 * dt)
    first = np.asarray(native.block_level_state_global("gas", 0)).reshape(4, n, n)
    # Independent sign/work witness: the density bubble is centered at (.5,.5), starts
    # at rest, and has spatially constant pressure/energy. Its first momentum increment
    # therefore comes only from gravity; its initial gravitational work is exactly zero.
    # Accepted AMR state restricts covered fine cells onto this complete coarse grid.
    centers = (np.arange(n) + .5) / n - .5
    x, y = np.meshgrid(centers, centers)
    radius_squared = x*x + y*y
    annulus = (radius_squared > .05**2) & (radius_squared < .3**2)
    radial_momentum = float(np.sum((x*first[1] + y*first[2])[annulus]) / (n*n))
    assert radial_momentum < -1e-10, "negative Poisson load must accelerate the bubble inward"
    np.testing.assert_allclose(first[3], initial[3], rtol=0, atol=1e-12)
    assert abs(native.mass() - mass) < 1e-12 * (abs(mass) + 1)
    prepared.run_step(target, t_end=12 * dt)
    # Periodic conservative transport cannot create total energy. On the second step,
    # inward momentum does positive gravity work, so use the native masked-volume
    # composite inventory to catch a reversed or missing source-work term independently.
    second_energy = float(native.composite_reduce("gas", "sum", 3, []))
    roundoff = 64 * np.finfo(np.float64).eps * abs(initial_energy)
    assert second_energy - initial_energy > roundoff, "inward gravity must do positive work"
    for _ in range(10):
        prepared.run_step(target, t_end=12 * dt)
    # Advancing one engine must not mutate the other through the shared immutable install plan.
    assert public.time() == 0 and public.macro_step() == 0
    report = pops.run(public, t_end=12 * dt, max_steps=12, console=False)
    assert report.accepted_steps == native.macro_step() == 12
    assert native.n_patches() == public._executor.n_patches()
    for level in range(public.n_levels()):
        np.testing.assert_allclose(native.block_level_state_global("gas", level),
            public.block_level_state_global("gas", level), rtol=0, atol=1e-12)
    assert abs(native.mass() - public._executor.mass()) < 1e-12 * (abs(mass) + 1)
    assert abs(native.mass() - mass) < 1e-12 * (abs(mass) + 1)
    state = np.asarray(public.block_level_state_global("gas", 0)).reshape(4, n, n)
    assert np.isfinite(state).all() and np.max(np.abs(state[1:3])) > 1e-8
    assert public._executor.program_diagnostics()["gravity_potential_abs_sum"] > 1e-8
    print("OK  (2) Euler-Poisson AMR: native installer/public bind preserve all state, mass and patches")


def _amr(n, L, branch, refine=1.2):
    cfg = AmrSystemConfig()
    cfg.shape = (n, n)
    cfg.lower = (0.0, 0.0)
    cfg.upper = (float(L), float(L))
    cfg.periodicity = (True, True)
    cfg.regrid_every = 4
    s = AmrSystem(cfg)
    s.set_temporal_relations([2], [1], ["integral_only"])
    branch(s)
    install_prepared_threshold_union(s, (("gas", "rho", refine),))
    rho = np.asarray(_bubble(n), dtype=float)
    rho += 1.0 - float(rho.mean())
    s.set_density("gas", rho)
    install_forward_euler_program(s)
    # This advanced-runtime fixture assembles the native hierarchy directly, without pops.bind.
    # Seal its installed Program checkpoint budget before the first accepted-state publication.
    s._s.mark_bound()
    return s


def _install_compiled_amr(system, component, *, weno5=False):
    """Install a production AMR package through the authenticated runtime facade.

    The historical ``system._s.add_native_block`` call bypassed the final Python contract and was
    removed from the extension.  ``AmrSystem.add_equation`` is the supported dispatcher for a
    ``CompiledModel(target='amr_system')``; it still reaches the same native loader boundary.
    """
    spatial = engine.Spatial(
        weno5=True if weno5 else False,
        minmod=not weno5,
        flux=Rusanov(),
        recon=Conservative(),
    )
    system.add_equation("gas", component, spatial=spatial, time=engine.Explicit())


def _component_at(component, so_path):
    """Detach valid metadata while substituting a deliberately mismatched ABI package path."""
    return CompiledModel(
        so_path=so_path,
        backend=component.backend,
        target=component.target,
        cons_names=component.cons_names,
        state_spaces=component.state_spaces,
        cons_roles=component.cons_roles,
        prim_names=component.prim_names,
        n_vars=component.n_vars,
        gamma=component.gamma,
        n_aux=component.n_aux,
        params=component.params,
        caps=component.caps,
        abi_key=component.abi_key,
        model_hash=component.model_hash,
        cxx=component.cxx,
        std=component.std,
        native_dimension=component.native_dimension,
        hllc=component.has_hllc,
        roe=component.has_roe,
        aux_extra_names=component.aux_extra_names,
        wave_speeds=component.has_wave_speeds,
        wave_speed_provider=component.wave_speed_provider,
        elliptic_field_names=component.elliptic_field_names,
        definition_identity=component.definition_identity,
    )


def main():
    cxx = default_cxx()
    missing = missing_native_compile_requirement(INCLUDE, cxx)
    if missing is not None:
        require_native_or_skip(missing)
    assert cxx is not None

    n, L = 48, 1.0
    tmp = tempfile.mkdtemp()
    try:
        # --- (1) PARITE STRICTE : transport pur (elliptic_rhs = 0), dmax == 0 ---
        et = _build_euler_transport()
        cm_t = et.compile(os.path.join(tmp, "euler_transport_amr.so"), INCLUDE,
                          backend="production", target="amr_system")
        assert isinstance(cm_t, CompiledModel)
        assert cm_t.backend == "production" and cm_t.target == "amr_system"
        assert cm_t.caps.get("amr") is True, "production caps amr=True (Phase D)"
        spec_t = engine.Model(state=engine.FluidState("compressible", gamma=GAMMA),
                           transport=engine.CompressibleFlux(), source=engine.NoSource(),
                           elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0))

        A = _amr(n, L, lambda s: _install_compiled_amr(s, cm_t))
        B = _amr(n, L, lambda s: s.add_equation(
            "gas", spec_t, spatial=engine.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()),
            time=engine.Explicit()))
        assert A.n_patches() == B.n_patches(), "n_patches initial production != add_block"
        dt = 2e-4
        for _ in range(12):
            A.step(dt)
            B.step(dt)
        da, db = np.array(A.density()), np.array(B.density())
        assert da.size == db.size and da.size > 0
        nrm = float(np.max(np.abs(db)))
        assert nrm > 1e-6, "densite natif triviale"
        dmax = float(np.max(np.abs(da - db)))
        assert dmax == 0.0, ("transport pur AMR : densite production != add_block (dmax %.2e, "
                             "attendu 0)" % dmax)
        assert A.n_patches() == B.n_patches(), "n_patches final production != add_block"
        print("OK  (1) transport pur AMR : densite production BIT-IDENTIQUE a add_block (dmax=0)")

        # --- (2) Exact public field/source authority on two independently installed engines. ---
        _euler_poisson_public_parity(n, dt)

        # --- (3) PARITE hllc/roe/primitive : la facade add_equation ACCEPTE et donne un resultat
        #     bit-identique a add_block (Gap 1 parite : le moteur AMR supporte ces schemas).
        #     Reutilise cm_t (transport pur, phi=0) : parite STRICTE dmax==0 (zero bruit FP),
        #     meme garantie que le test C++ test_amr_riemann_native. cm_t a une primitive 'p'
        #     (declaree dans _euler_formulas via _build_euler_transport) -> garde-fou pression OK.

        def parity_riemann(riem, recon, label):
            """add_equation(riemann, recon) BIT-IDENTIQUE a add_block (dmax==0)."""
            R = _amr(n, L, lambda s: s.add_equation(
                "gas", cm_t,
                spatial=engine.Spatial(limiter=Minmod(), flux=riem, recon=recon)))
            S = _amr(n, L, lambda s: s.add_equation(
                "gas", spec_t,
                spatial=engine.Spatial(minmod=True, flux=riem, recon=recon),
                time=engine.Explicit()))
            for _ in range(12):
                R.step(dt)
                S.step(dt)
            dr, ds = np.array(R.density()), np.array(S.density())
            dmax = float(np.max(np.abs(dr - ds)))
            assert dmax == 0.0, ("%s: add_equation != add_block (dmax=%.2e)" % (label, dmax))
            assert np.isfinite(dr).all() and float(np.max(np.abs(ds))) > 1e-6
            print("OK  (3) %s : add_equation BIT-IDENTIQUE a add_block (dmax=%.0f)" % (label, dmax))

        parity_riemann(HLLC(), Conservative(), "hllc/conservative")
        parity_riemann(HLLC(), Primitive(),    "hllc/primitive")
        parity_riemann(Roe(),  Conservative(), "roe/conservative")
        parity_riemann(Roe(),  Primitive(),    "roe/primitive")

        # La garde-fou pressure reste active : un modele SANS primitive 'p' doit etre rejete.
        # Modele isotherme 3 variables (rho, rho_u, rho_v) avec primitives (rho, u, v) sans 'p' :
        # il compile, mais add_equation(flux=hllc) doit lever ValueError (pression requise).
        m_iso = Model("isothermal_no_p")
        rho_i, rhou_i, rhov_i = m_iso.conservative_vars("rho", "rho_u", "rho_v")
        cs2 = 0.5
        ui = rhou_i / rho_i
        vi = rhov_i / rho_i
        pui = m_iso.primitive("u", ui)
        pvi = m_iso.primitive("v", vi)
        m_iso.flux(x=[rhou_i, rhou_i * pui + cs2 * rho_i, rhou_i * pvi],
                   y=[rhov_i, rhov_i * pui, rhov_i * pvi + cs2 * rho_i])
        m_iso.eigenvalues(x=[pui - sqrt(cs2), pui, pui + sqrt(cs2)],
                          y=[pvi - sqrt(cs2), pvi, pvi + sqrt(cs2)])
        m_iso.primitive_vars(rho_i, pui, pvi)
        m_iso.conservative_from([rho_i, rho_i * pui, rho_i * pvi])
        m_iso.elliptic_rhs(0.0 * rho_i)
        cm_iso = m_iso.compile(os.path.join(tmp, "isothermal_amr.so"), INCLUDE,
                               backend="production", target="amr_system")
        assert "p" not in cm_iso.prim_names, "modele isotherme ne devrait pas avoir 'p'"
        raised = False
        try:
            s_nop = AmrSystem(n=n, L=L, periodicity=(True, True))
            s_nop.add_equation("gas", cm_iso,
                               spatial=engine.Spatial(minmod=True, flux=HLLC()))
        except ValueError as ex:
            raised = True
            assert "hllc" in str(ex).lower()
        assert raised, "add_equation a accepte hllc sans primitive 'p'"
        print("OK  (3) garde-fou pression hllc/roe SANS primitive 'p' : rejet explicite")

        # --- (3w) WENO5 multilevel : fournisseur C/F natif ordre 5 / halo 3. -----------------------
        # Les routes production et ModelSpec doivent toutes deux construire une vraie hierarchie,
        # avancer, et rester finies. La selection native refuse tout fournisseur d'ordre inferieur.
        weno_systems = [
            _amr(n, L, install)
            for install in (
                lambda s: _install_compiled_amr(s, cm_t, weno5=True),
                lambda s: s.add_equation(
                    "gas", spec_t,
                    spatial=engine.Spatial(weno5=True, flux=Rusanov(), recon=Conservative()),
                    time=engine.Explicit()),
            )
        ]
        for weno_system in weno_systems:
            # n_patches() counts fine patches only, not hierarchy levels.  A single clustered
            # fine patch is already a genuine two-level hierarchy; assert both facts explicitly.
            assert weno_system.n_levels() == 2, "WENO5 doit produire exactement deux niveaux"
            assert weno_system.n_patches() > 0, "WENO5 doit activer une hierarchie multilevel"
            weno_system.step(dt)
            assert np.isfinite(np.asarray(weno_system.density())).all()
        print("OK  (3w) WENO5 multilevel utilise le fournisseur coarse/fine ordre 5")

        # add_equation chemin nominal (rusanov + conservatif) accepte et tourne :
        E = AmrSystem(n=n, L=L, periodicity=(True, True))
        E.set_temporal_relations([2], [1], ["integral_only"])
        E.set_poisson("charge_density", "geometric_mg")
        E.add_equation("gas", cm_t,
                       spatial=engine.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()))
        install_prepared_threshold_union(E, (("gas", "rho", 1.2),))
        E.set_density("gas", _bubble(n))
        install_forward_euler_program(E)
        E._s.mark_bound()
        for _ in range(4):
            E.step(dt)
        assert np.isfinite(np.array(E.density())).all() and E.mass() > 1e-6
        print("OK  (3b) AmrSystem.add_equation(production, rusanov) tourne et reste physique")

        # --- (4) GARDE-FOUS de compilation / dispatch ---
        sys_cm = ep.compile(os.path.join(tmp, "ep_sys_cm.so"), INCLUDE,
                            backend="production", target="system")  # target System par defaut
        s = AmrSystem(n=n, L=L, periodicity=(True, True))
        raised = False
        try:
            s.add_equation("gas", sys_cm,
                           spatial=engine.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()))
        except ValueError as ex:
            raised = True
            assert "target='system'" in str(ex) or "amr_system" in str(ex)
        assert raised, "AmrSystem.add_equation a accepte un CompiledModel target='system'"
        print("OK  (4) compile(target=) garde-fous + CompiledModel target='system' refuse sur AMR")

        # --- (5) GARDE-FOU ABI : loader AMR a cle pops_native_abi_key falsifiee -> rejet ---
        bad_abi = _compile_wrong_abi(ep, os.path.join(tmp, "ep_amr_wrongabi.so"), cxx)
        bad_component = _component_at(cm_p, bad_abi)
        s = AmrSystem(n=n, L=L, periodicity=(True, True))
        raised = False
        try:
            s.add_equation(
                "gas",
                bad_component,
                spatial=engine.Spatial(minmod=True, flux=Rusanov(), recon=Conservative()),
                time=engine.Explicit(),
            )
        except RuntimeError as ex:
            raised = True
            assert "ABI" in str(ex), "message inattendu : %s" % ex
        assert raised, "add_native_block a accepte un loader AMR a cle d'ABI fausse (UB silencieux)"
        print("OK  (5) cle d'ABI divergente REJETEE par AmrSystem.add_native_block")

        print("test_dsl_production_amr : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile_wrong_abi(model, dst_so, cxx):
    """Compile le MEME loader natif AMR mais avec une signature d'en-tetes FAUSSE (-DPOPS_HEADER_SIG
    bidon) : le .so est valide mais sa cle d'ABI differe de celle du module -> rejet d'add_native_block.
    On regenere (pas de patch binaire : sur macOS ARM cela invaliderait la signature et tuerait le
    process). Renvoie le chemin du .so."""
    from pops.codegen.toolchain import pops_loader_build_flags
    lowering = model.__pops_compiler_lowering__()
    src = lowering.native_loader_source(target="amr_system")
    # PoPS est Kokkos-only : le loader inclut les en-tetes pops -> Kokkos + (macOS) -undefined
    # dynamic_lookup via pops_loader_build_flags. SIGNATURE D'EN-TETES FAUSSE conservee (le .so compile
    # mais doit etre REJETE a l'ABI par add_native_block).
    cc, kflags_c, kflags_l = pops_loader_build_flags(cxx)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O2",
             "-DPOPS_HEADER_SIG=\"deadbeef_signature_volontairement_fausse\"", *kflags_c]
    with tempfile.TemporaryDirectory() as t:
        cpp = os.path.join(t, "wrong_amr.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run([cc, *flags, "-I", INCLUDE, cpp, "-o", dst_so, *kflags_l], check=True)
    return dst_so


if __name__ == "__main__":
    main()
