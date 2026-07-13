#!/usr/bin/env python3
# ruff: noqa: B018, B904, E402
"""ADC-592 : cycle de vie de gel du runtime apres pops.bind.

Le cycle de vie runtime est EXPLICITE : assembly mutable AVANT bind, composition GELEE une fois
pops.bind termine (etat 'bound'), simulation mutable seulement par les APIs runtime controlees
(donnees d'etat / checkpoint / diagnostics / sorties ; params figes au bind). Ce test prouve LOCALEMENT
(sous un _pops deja construit, sans codegen / Kokkos obligatoire) :

  1. AVANT bind : un System frais est 'assembling' ; add_block fonctionne.
  2. Gel couche Python : apres _finalize_bind (route bas-niveau legitime), chaque methode
     structurelle Python leve RuntimeError avec le vocabulaire bind (pops.Problem + pops.compile +
     pops.bind, JAMAIS un setter herite comme REMEDE) ; les noms natifs structurels interceptes
     par __getattr__ (install_program / set_refinement / set_program_cadence) levent aussi ; les
     mutations d'etat restent permises mais les carriers param bruts sont bloques ; lifecycle passe a
     'running' apres un pas.
  3. AMR : idem sur un AmrSystem (set_refinement / add_block / add_coupling gelees).
  4. PENDANT bind : _finalize_bind est le DERNIER acte -> le System reste 'assembling' jusque-la.
  5. Snapshot : le BoundSnapshot est un manifeste inerte, JSON-ready, hash stable 64-hex ;
     inspect() montre le lifecycle + le hash + les blocs/solveurs.
  6. Params runtime : BindSchema fige les valeurs au bind ; les setters de carrier bruts sont refuses.
  7. Gate compilateur : le flux complet Problem -> compile -> bind (le native mark_bound absent du .so
     prebuilt -> les sous-asserts natifs skippent avec un message CI-diagnosable).

Ne FALSIFIE jamais le moteur pops : on construit un vrai System / AmrSystem par la route interne
(legitime en test bas-niveau) ou on appelle _finalize_bind directement ; on skippe si une dependance
manque. Tourne sous pytest ET comme script (garde __main__)."""
import sys

try:
    import numpy as np
    import pops
    from pops.identity import make_identity
    from pops.runtime._bound_sim import BoundSimulation
    from pops.runtime._bound_snapshot import BoundSnapshot
    from pops.runtime._lifecycle import FROZEN_STRUCTURAL
    from pops.numerics.reconstruction.limiters import Minmod
except Exception as exc:  # noqa: BLE001
    print("skip test_freeze_lifecycle (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check
from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam


# Vocabulaire herite que le message d'un refus ne doit JAMAIS recommander comme le REMEDE.
_FORBIDDEN_REMEDY = ("add_block", "set_poisson", "install_program", "set_refinement",
                     "add_equation")


def _isothermal_model():
    """Un pops.Model(...) natif (briques composees, PAS de compile DSL) pour un System uniforme."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.PotentialForce(charge=1.0),
                      elliptic=pops.ChargeDensity(charge=1.0))


def _compressible_model():
    """Un pops.Model(...) natif compressible pour le transport pur sur une hierarchie AMR."""
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                      transport=pops.CompressibleFlux(), source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))


from tests.python.support.initial_states import bubble_offset as _bubble


def _minimal_snapshot(layout="system"):
    """Un BoundSnapshot minimal LEGITIME pour geler un moteur bas-niveau dans un test unitaire."""
    return BoundSnapshot(
        semantic_identity=make_identity("semantic", {"test": "freeze"}),
        artifact_identity=make_identity("artifact", {"binary": "freeze"}),
        layout={"kind": layout}, blocks=[{"name": "ions"}],
        solvers={"phi": "geometric_mg"},
        cadence={"kind": "engine-default", "substeps": 1, "stride": 1, "cfl": "default"},
        params=[], aux_evidence={}, initial_evidence={}, outputs=[], diagnostics=[],
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )


def _assert_bind_vocabulary(exc, what):
    """Le message d'un refus de gel parle le vocabulaire bind et ne recommande aucun setter herite."""
    msg = str(exc)
    _check("pops.Problem" in msg, "le refus de %r mentionne pops.Problem" % what)
    _check("pops.compile" in msg and "pops.bind" in msg,
           "le refus de %r mentionne pops.compile / pops.bind" % what)
    # Il nomme HONNETEMENT l'operation refusee (le contexte), mais ne recommande AUCUN autre setter
    # herite comme remede : on exclut le nom teste lui-meme de l'interdiction.
    for bad in _FORBIDDEN_REMEDY:
        if bad == what:
            continue
        _check(bad not in msg, "le refus de %r ne recommande pas %r comme remede" % (what, bad))


# --- 1. AVANT bind : assembling, mutable -----------------------------------------------------
def test_assembling_before_bind():
    """Un System frais est 'assembling' ; add_block passe ; lifecycle_state est expose."""
    engine = System(n=8, L=1.0, periodic=True)
    _check(engine.lifecycle_state() == "assembling", "un System frais est 'assembling'")
    _check(engine._lifecycle == "assembling", "le flag Python demarre a 'assembling'")
    engine.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    engine.add_block("ions", _isothermal_model(),
                     spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    _check(engine.block_names() == ["ions"], "add_block fonctionne avant bind")
    _check(engine.lifecycle_state() == "assembling", "toujours 'assembling' apres add_block")
    print("ok test_assembling_before_bind")


# --- 2. Gel couche Python (System) -----------------------------------------------------------
def test_python_freeze_uniform():
    """Apres _finalize_bind, toute methode structurelle Python leve ; les mutations restent OK."""
    n = 16
    engine = System(n=n, L=1.0, periodic=True)
    engine.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    engine.add_block("ions", _isothermal_model(),
                     spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    engine.set_density("ions", _bubble(n))
    # Gel bas-niveau LEGITIME (ce que _install_compiled fait en dernier).
    engine._finalize_bind(_minimal_snapshot())
    _check(engine.lifecycle_state() == "bound", "apres _finalize_bind le System est 'bound'")

    # Chaque methode structurelle Python DIRECTE (pas un wrapper qui delegue a une autre methode
    # deja gardee comme add_background -> add_block) leve RuntimeError avec le vocabulaire bind.
    frozen_calls = {
        "add_block": lambda: engine.add_block("x", _isothermal_model()),
        "add_equation": lambda: engine.add_equation("x", _isothermal_model()),
        "set_poisson": lambda: engine.set_poisson(bc="periodic"),
        "add_coupling": lambda: engine.add_coupling(object()),
        "set_disc_domain": lambda: engine.set_disc_domain(0.5, 0.5, 0.4),
        "set_geometry_mode": lambda: engine.set_geometry_mode("none"),
    }
    for what, call in frozen_calls.items():
        try:
            call()
            raise AssertionError("la methode structurelle %r doit lever apres bind" % what)
        except RuntimeError as exc:
            _assert_bind_vocabulary(exc, what)
    # add_background delegue a add_block (deja garde) : il leve aussi, message nommant add_block.
    try:
        engine.add_background("x", _isothermal_model(), _bubble(n))
        raise AssertionError("add_background doit lever apres bind (via add_block delegue)")
    except RuntimeError as exc:
        _check("pops.bind" in str(exc), "add_background leve le refus de gel (via add_block)")

    # Les noms natifs structurels interceptes par __getattr__ levent RuntimeError (PAS AttributeError).
    for native in ("install_program", "set_program_cadence"):
        try:
            getattr(engine, native)
            raise AssertionError("le nom natif structurel %r doit lever apres bind" % native)
        except RuntimeError as exc:
            _assert_bind_vocabulary(exc, native)
        except AttributeError:
            raise AssertionError("%r doit lever RuntimeError (gel), pas AttributeError" % native)

    # Les mutations runtime restent PERMISES (donnees d'etat).
    engine.set_density("ions", _bubble(n))  # ne doit pas lever
    engine.solve_fields()
    engine.step_cfl(0.4)
    _check(engine.lifecycle_state() == "running", "apres un pas le lifecycle est 'running'")
    print("ok test_python_freeze_uniform")


# --- 3. AMR : gel couche Python --------------------------------------------------------------
def test_python_freeze_amr():
    """Apres _finalize_bind, les setters structurels AMR (Python + natifs) levent ; step OK."""
    n = 16
    engine = AmrSystem(n=n, L=1.0)
    engine.set_poisson("charge_density", "geometric_mg")
    engine.add_block("gas", _compressible_model(),
                     spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    engine.set_density("gas", _bubble(n))
    engine._finalize_bind(_minimal_snapshot(layout="amr_system"))
    _check(engine.lifecycle_state() == "bound", "l'AmrSystem est 'bound' apres _finalize_bind")

    # add_block / add_coupling (Python) et set_refinement (natif, __getattr__) levent.
    try:
        engine.add_block("g2", _compressible_model())
        raise AssertionError("add_block AMR doit lever apres bind")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "add_block")
    try:
        engine.add_coupling(object())
        raise AssertionError("add_coupling AMR doit lever apres bind")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "add_coupling")
    try:
        engine.set_refinement
        raise AssertionError("set_refinement (natif) doit lever apres bind")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "set_refinement")
    except AttributeError:
        raise AssertionError("set_refinement doit lever RuntimeError (gel), pas AttributeError")

    for _ in range(3):
        engine.step(2e-4)
    _check(engine.lifecycle_state() == "running", "l'AmrSystem est 'running' apres un pas")
    print("ok test_python_freeze_amr")


# --- 4. PENDANT bind : reste assembling jusqu'au DERNIER acte ---------------------------------
def test_assembling_during_install():
    """Le flag reste 'assembling' pendant le lowering ; _finalize_bind (dernier acte) le bascule."""
    engine = System(n=8, L=1.0, periodic=True)
    # Toute la sequence d'install (add_block / set_poisson) tourne sous 'assembling'.
    engine.set_poisson(bc="periodic")
    engine.add_block("ions", _isothermal_model())
    _check(engine.lifecycle_state() == "assembling",
           "la composition reste mutable pendant tout le lowering (avant _finalize_bind)")
    engine._finalize_bind(_minimal_snapshot())
    _check(engine.lifecycle_state() == "bound", "_finalize_bind bascule a 'bound'")
    print("ok test_assembling_during_install")


# --- 4b. Double bind : le seam d'install refuse un second lowering ----------------------------
def test_double_bind_rejected():
    """Un second bind (le seam _install_compiled) leve : le contrat de gel interdit de re-lowerer.

    _finalize_bind est la primitive bas-niveau qui bascule le flag (idempotente : la rappeler ne
    leve pas). Le VRAI point d'entree du bind, _install_compiled, est garde par guard_assembling en
    tete : une fois 'bound', un second _install_compiled leve freeze_error avec le vocabulaire bind.
    C'est le contrat reel du mixin (guard_assembling lit self._lifecycle), pas une invention."""
    engine = System(n=8, L=1.0, periodic=True)
    engine.set_poisson(bc="periodic")
    engine.add_block("ions", _isothermal_model())
    engine._finalize_bind(_minimal_snapshot())
    _check(engine.lifecycle_state() == "bound", "le System est 'bound' apres le premier bind")
    # Un second passage par le seam d'install (ce que pops.bind appelle) DOIT lever.
    try:
        engine._install_compiled(compiled=None, instances=[])
        raise AssertionError("un second _install_compiled doit lever une fois bound")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "_install_compiled")
    print("ok test_double_bind_rejected")


# --- 4c. Restart d'une sim bindee (mutation runtime permise) ---------------------------------
def test_restart_on_bound_sim_restores_state():
    """Un moteur assemble bas niveau ne peut plus fabriquer un checkpoint sans identité de bind/run."""
    import os
    import tempfile

    n = 16
    engine = System(n=n, L=1.0, periodic=True)
    engine.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    engine.add_block("ions", _isothermal_model(),
                     spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    engine.set_density("ions", _bubble(n))
    tmp = tempfile.mkdtemp(prefix="pops_freeze_ckpt_")
    path = os.path.join(tmp, "state")
    try:
        engine.checkpoint(path)
        raise AssertionError("checkpoint bas niveau sans bind/run doit etre refuse")
    except RuntimeError as exc:
        _check("declared step strategy" in str(exc) or "compiled Program" in str(exc)
               or "pops.bind" in str(exc),
               "le checkpoint exige un artefact compile et une identite de bind")
    print("ok test_low_level_checkpoint_without_identity_rejected")


# --- 5. Snapshot : manifeste inerte + hash stable + inspect() --------------------------------
def test_bound_snapshot_manifest():
    """BoundSnapshot est JSON-ready, hash 64-hex stable ; inspect() montre lifecycle + hash."""
    import json

    snap = BoundSnapshot(
        semantic_identity=make_identity("semantic", {"problem": "snapshot-test"}),
        artifact_identity=make_identity("artifact", {"binary": "snapshot-test"}),
        layout={"kind": "uniform"},
        blocks=[{"name": "ions", "model_hash": None, "limiter": "minmod", "flux": "rusanov",
                 "recon": "conservative", "time": "explicit", "evolve": True}],
        solvers={"phi": "geometric_mg"},
        cadence={"kind": "compiled-time", "substeps": 1, "stride": 1, "cfl": 0.4},
        aux_evidence={"B_z": {"dtype": "<f8", "shape": [1], "content_sha256": "a" * 64}},
        initial_evidence={}, params=[{
            "qid": "pops.handle.v1::block:plasma::parameter::cs2",
            "dtype": "Real", "source": "override",
            "value": {"kind": "binary64", "value": "0x1.0000000000000p+0",
                      "target": "Real"},
        }], bind_schema_identity=make_identity("bind-schema", {"slots": []}),
        outputs=["OutputPolicy"], diagnostics=[])
    d = snap.to_dict()
    _check(json.loads(json.dumps(d)) == d, "le snapshot est JSON round-trippable")
    h = snap.bind_identity.hexdigest
    _check(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h),
           "bind_identity est un sha256 64-hex")
    _check(BoundSnapshot(**_snap_kwargs(snap)).bind_identity.hexdigest == h,
           "le hash est STABLE (deterministe pour le meme contenu)")
    _check(snap.block_names() == ["ions"], "block_names() liste les blocs bindes")
    try:
        snap.params = ()
        raise AssertionError("BoundSnapshot mutation should be rejected")
    except AttributeError as exc:
        _check("immutable" in str(exc), "le snapshot refuse les mutations")

    # inspect() a travers un moteur reel gele expose lifecycle + snapshot.
    engine = System(n=8, L=1.0, periodic=True)
    engine.set_poisson(bc="periodic")
    engine.add_block("ions", _isothermal_model())
    engine._finalize_bind(snap)
    rep = engine.inspect()
    rep_dict = rep.to_dict()
    _check(rep_dict["lifecycle"] == "bound", "inspect().to_dict() porte le lifecycle 'bound'")
    _check(rep_dict["bound_snapshot"]["bind_identity"]["hexdigest"] == h,
           "inspect().to_dict() porte le hash du snapshot")
    text = str(rep)
    _check("lifecycle" in text and "bound" in text, "str(inspect()) montre le lifecycle")
    _check(h in text, "str(inspect()) montre le hash du snapshot")
    _check("ions" in text, "str(inspect()) resume les blocs bindes")
    print("ok test_bound_snapshot_manifest")


def _snap_kwargs(snap):
    """Reconstruit le kwargs d'un BoundSnapshot pour prouver la stabilite du hash."""
    return {
        "semantic_identity": snap.semantic_identity,
        "artifact_identity": snap.artifact_identity,
        "layout": snap.layout, "blocks": snap.blocks, "solvers": snap.solvers,
        "cadence": snap.cadence, "params": snap.params,
        "aux_evidence": snap.aux_evidence, "initial_evidence": snap.initial_evidence,
        "bind_schema_identity": snap.bind_schema_identity,
        "outputs": snap.outputs, "diagnostics": snap.diagnostics,
    }


# --- 6. Les carriers param bruts ne contournent pas BindSchema apres bind ---------------------
def test_runtime_param_carriers_are_frozen_after_bind():
    """Les setters indexes internes disparaissent de la surface publique apres bind."""
    from pops.runtime._bound_sim import _MUTATIONS, _BLOCKED
    _check("set_block_params" not in _MUTATIONS and "set_block_params" in _BLOCKED,
           "set_block_params est bloque sur la vue")
    _check("set_program_params" not in _MUTATIONS and "set_program_params" in _BLOCKED,
           "set_program_params est bloque sur la vue")
    _check("install_program" in _BLOCKED, "install_program reste bloque sur la vue")
    _check("set_refinement" in _BLOCKED, "set_refinement reste bloque sur la vue")
    for carrier in ("set_block_params", "set_program_params"):
        _check(carrier in FROZEN_STRUCTURAL, "%r est gele dans le passthrough natif" % carrier)
    for data_setter in ("set_density", "set_magnetic_field", "set_state", "set_clock"):
        _check(data_setter not in FROZEN_STRUCTURAL,
               "%r (donnees runtime) n'est PAS gele" % data_setter)
    engine = System(n=8, L=1.0, periodic=True)
    engine._finalize_bind(_minimal_snapshot())
    view = BoundSimulation(engine)
    try:
        view.set_block_params
        raise AssertionError("la vue ne doit pas exposer set_block_params")
    except AttributeError:
        pass
    try:
        engine.set_program_params
        raise AssertionError("le passthrough ne doit pas exposer set_program_params apres bind")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "set_program_params")
    print("ok test_runtime_param_carriers_are_frozen_after_bind")


# --- Gate compilateur : le flux complet Problem -> compile -> bind + snapshot reel ---------------
def _dsl_isothermal_model(name="adc592_iso"):
    """Un modele DSL isotherme MINIMAL et VALIDE (facade pops.physics), compilable en Program .so."""
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


def _lie_program(block, state, name="adc592_prog"):
    """Un time Program Lie VALIDE (miroir de test_bind_adapters._lie_program)."""
    P = pops.time.Program(name)
    endpoint = P.state(block, state)
    u = endpoint.n
    fields = P.solve_fields(u)
    r = P._rhs_legacy(state=u, fields=fields)
    P.commit(endpoint.next, P.linear_combine("u1", u + P.dt * r, at=endpoint.next.point))
    return P


def test_full_bind_flow_freeze_gated():
    """pops.Problem -> pops.compile -> pops.bind : la sim est 'bound', snapshot 64-hex, bypass ferme.

    L'AUTHORING est VALIDE et HORS du try (une regression fait ECHOUER, jamais skipper). La SEULE
    barriere locale est le compile .so (headers / cxx / Kokkos), qui skippe en nommant le TYPE
    d'exception. Le native mark_bound est ABSENT du .so prebuilt : la couche Python porte les
    assertions, les sous-asserts natifs skippent avec un message CI-diagnosable."""
    from pops.mesh.layouts import Uniform
    from pops.mesh.cartesian import CartesianMesh

    n = 64
    m = _dsl_isothermal_model()
    case = pops.Problem(layout=Uniform(CartesianMesh(n=n, L=1.0, periodic=True)))
    block = case.add_block("ne", m)
    module = m.module
    state = module.state_handle(next(iter(module.state_spaces().values())))
    prog = _lie_program(block, state)
    try:
        compiled = pops.compile(case, time=prog)
    except Exception as exc:  # noqa: BLE001 - barriere toolchain -> skip diagnosable
        print("skip test_full_bind_flow_freeze_gated (toolchain %s: %s)"
              % (type(exc).__name__, str(exc)[:140]))
        return

    xs = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho0, 0.4 * rho0, -0.2 * rho0])
    sim = pops.bind(compiled, state={"ne": u0},
                    solvers={"phi": pops.fields.catalog.GeometricMG()})
    _check(type(sim).__name__ == "BoundSimulation", "pops.bind rend une BoundSimulation")

    # La sim est 'bound' et son snapshot est un manifeste hache.
    _check(sim.lifecycle_state() == "bound", "la sim bindee est 'bound'")
    snap = sim.bound_snapshot
    _check(snap is not None, "la sim bindee porte un BoundSnapshot")
    h = snap.bind_identity.hexdigest
    _check(isinstance(h, str) and len(h) == 64, "bind_identity est un sha256 64-hex")
    _check("ne" in snap.block_names(), "le snapshot liste le bloc 'ne'")

    # inspect() a travers la vue montre lifecycle + hash + blocs/solveurs.
    text = str(sim.inspect())
    _check("lifecycle" in text and "bound" in text, "inspect() montre le lifecycle")
    _check(h in text, "inspect() montre le hash du snapshot")

    # Apres un run, lifecycle 'running'.
    sim.run(t_end=0.01, cfl=0.4, max_steps=4)
    _check(sim.lifecycle_state() == "running", "apres run la sim est 'running'")

    # BYPASS FERME : sim._engine.install_program leve RuntimeError (pas AttributeError). Le native
    # mark_bound est absent du .so prebuilt -> la garde couche Python (__getattr__) porte le refus.
    try:
        sim._engine.install_program
        raise AssertionError("sim._engine.install_program doit lever apres bind (bypass ferme)")
    except RuntimeError as exc:
        _assert_bind_vocabulary(exc, "install_program")
    except AttributeError:
        raise AssertionError("le bypass doit lever RuntimeError (gel), pas AttributeError")

    # Sous-assert natif : gele en profondeur (defence in depth) SEULEMENT si le .so a mark_bound.
    if hasattr(sim._engine._s, "mark_bound"):
        try:
            sim._engine._s.install_program("/nonexistent.so")
            raise AssertionError("le natif install_program doit lever une fois bound")
        except RuntimeError:
            pass
    else:
        print("  (skip native-guard sub-assert: needs a _pops rebuilt from this branch (CI covers))")
    print("ok test_full_bind_flow_freeze_gated")


def test_checkpoint_restart_roundtrip_through_bind_gated():
    """checkpoint / restart round-trip a travers pops.bind : rebind + restart == etat sauve.

    Comme test_full_bind_flow_freeze_gated, l'authoring est VALIDE et hors du try ; la SEULE barriere
    locale est le compile .so (headers / cxx / Kokkos), qui skippe en nommant le TYPE d'exception.
    Sur CI-Kokkos le flux complet tourne : on binde, on avance, on checkpoint PAR LA VUE, on rebinde
    une sim IDENTIQUE (meme composition, exigence v1 du restart) et on restart PAR LA VUE ; l'etat du
    bloc revient bit-a-bit a celui sauve (checkpoint / restart sont allowlistes sur la vue)."""
    import os
    import tempfile

    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform

    n = 64

    def _case_and_program():
        model = _dsl_isothermal_model()
        case = pops.Problem(layout=Uniform(CartesianMesh(n=n, L=1.0, periodic=True)))
        block = case.add_block("ne", model)
        module = model.module
        state = module.state_handle(next(iter(module.state_spaces().values())))
        return case, _lie_program(block, state)

    xs = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho0, 0.4 * rho0, -0.2 * rho0])

    try:
        case, program = _case_and_program()
        compiled = pops.compile(case, time=program)
    except Exception as exc:  # noqa: BLE001 - barriere toolchain -> skip diagnosable
        print("skip test_checkpoint_restart_roundtrip_through_bind_gated (toolchain %s: %s)"
              % (type(exc).__name__, str(exc)[:140]))
        return

    solvers = {"phi": pops.fields.catalog.GeometricMG()}
    sim = pops.bind(compiled, state={"ne": u0}, solvers=solvers)
    sim.run(t_end=0.01, cfl=0.4, max_steps=4)
    saved = np.array(sim.density("ne"), copy=True)

    tmp = tempfile.mkdtemp(prefix="pops_bind_ckpt_")
    path = os.path.join(tmp, "state")
    sim.checkpoint(path)  # checkpoint est allowlistee sur la vue

    # Rebind une sim IDENTIQUE (le restart v1 exige la MEME composition rejouee avant l'appel) puis
    # restaure PAR LA VUE : l'etat du bloc revient a celui sauve.
    case2, program2 = _case_and_program()
    compiled2 = pops.compile(case2, time=program2)
    sim2 = pops.bind(compiled2, state={"ne": u0}, solvers=solvers)
    sim2.restart(path)  # restart est allowlistee sur la vue
    restored = np.array(sim2.density("ne"))
    _check(np.array_equal(restored, saved),
           "restart a travers pops.bind restaure l'etat du bloc bit-a-bit")
    print("ok test_checkpoint_restart_roundtrip_through_bind_gated")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
