#!/usr/bin/env python3
"""Checkpoint / restart AMR BIT-IDENTIQUE (ADC-65 mono-bloc mono-rang ; ADC-509 multi-blocs + np>1).

Comble les manques d'ABI signales par l'ancienne docstring de AmrSystem.checkpoint (etats fins par
patch + etat conservatif COMPLET illisibles/inecrivables, hierarchie non imposable). On expose
desormais :

  - level_state(k) / set_level_state(k, .) : etat conservatif COMPLET du niveau k (toutes
    composantes ; grossier ET patchs fins), plat composante-majeur c*nf*nf + j*nf + i (nf = n << k) ;
    MONO-BLOC. level_state_global(k) = gather np>1 (collectif) ;
  - block_level_state(name, k) / set_block_level_state(name, k, .) : idem PAR BLOC, MULTI-BLOCS
    (moteur AmrRuntime, layout + aux PARTAGES) ; block_level_state_global = gather np>1 ;
  - level_potential(k) / set_level_potential(k, .) : phi du niveau k PARTAGE (niveau 0 = warm-start du
    multigrille, load-bearing pour la reprise bit-identique) ; level_potential_global = gather np>1 ;
  - set_hierarchy(boxes) : IMPOSE la hierarchie fine sauvee (au lieu du clustering Berger-Rigoutsos),
    MONO-BLOC. MULTI-BLOCS : la hierarchie partagee est le patch central FIGE deterministe, reproduit
    par le rejeu de la composition (regrid_every=0) -> pas d'imposition.

VERROUILLE (mono-rang, executables sur Mac sans Kokkos) :
  T1 (BIT-IDENTIQUE) : run A de 10 pas AVEC patchs fins actifs (refinement bas, regrid_every=0) ;
       checkpoint a 5 pas ; restart dans un systeme NEUF ; 5 pas ; etat FINAL identique a run A,
       grossier (niveau 0) ET patchs (niveau 1) -- dmax == 0.0 EXACT.
  T2 (HIERARCHIE) : patch_boxes() identiques au checkpoint et apres restart.
  T3 (HORLOGE) : time() / macro_step() restaures (la cadence reprend exactement).
  T4 (REJETS) : composition differente (bloc), n different, regrid_every > 0 au restart -- chacun leve.
  T5 (MULTI-BLOCS BIT-IDENTIQUE) : 2 blocs (ions/elec) sur la hierarchie partagee ; checkpoint a
       mi-course ; restart NEUF ; etat FINAL identique BIT A BIT par bloc et par niveau.

np>1 (gather par niveau / par bloc) : code en place (accesseurs *_global collectifs + ecriture
rang 0), NON declenchable en serie sur Mac -> valide par CI-MPI / ROMEO. regrid_every > 0 est refuse
(la cadence regrid post-restart re-divergerait la hierarchie).

Lancement : PYTHONPATH=<build>/python python3 tests/python/integration/io/test_amr_checkpoint.py
"""
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import json
import os
import tempfile

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam


def _advance(sim, nsteps, dt):
    strategy = pops.time.FixedDt(dt)
    if sim._step_strategy is None:
        sim._step_strategy = strategy
        sim._step_transaction_plan = pops.time.StepTransactionPlan(strategy)
    elif sim._step_strategy != strategy:
        raise ValueError("test runtime already owns a different installed StepStrategy")
    return sim.run(
        t_end=float(sim.time()) + nsteps * dt,
        max_steps=nsteps,
    )


def _bump(n, L=1.0, amp=1.0, w=0.10):
    """Bosse gaussienne centrale (floor 1.0, pic ~2.0) : compacte -> patchs fins centraux quand
    refine_threshold est entre le plancher et le pic. field[j, i] (X selon i, Y selon j)."""
    xs = (np.arange(n) + 0.5) / n * L
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    r2 = (X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2
    return np.ascontiguousarray(1.0 + amp * np.exp(-r2 / w ** 2))


def _build(n=32, regrid_every=0, block="ne"):
    """AMR mono-bloc scalaire (ExB, fond neutralisant pour le Poisson periodique), refinement bas
    (patchs fins actifs des le seed), hierarchie FIGEE (regrid_every=0 -> reprise bit-identique)."""
    rho0 = _bump(n)
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every)
    sim.block(block,
                  engine.Model(engine.Scalar(), engine.ExB(B0=1.0), engine.NoSource(),
                            engine.BackgroundDensity(alpha=1.0, n0=float(rho0.mean()))),
                  spatial=engine.Spatial(limiter=Minmod(), flux=Rusanov()),
                  time=engine.Explicit())
    sim.set_refinement(threshold=1.5)  # rho > 1.5 -> patchs centraux (pic 2.0, plancher 1.0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density(block, rho0)
    return sim


def test_amr_checkpoint_bit_identical():
    """T1 + T2 : reprise BIT-IDENTIQUE (etat final niveau 0 ET 1) + hierarchie restauree identique."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")

        # --- RUN A : 10 pas, checkpoint a mi-course (5 pas) ---
        simA = _build()
        _advance(simA, 5, dt)
        simA.checkpoint(path)
        boxes_chk = simA.patch_boxes()
        assert any(b[0] == 1 for b in boxes_chk), "patchs fins inactifs : test sans interet"
        _advance(simA, 5, dt)
        finalA = [np.asarray(simA.level_state(k), dtype=np.float64)
                  for k in range(simA.n_levels())]

        # --- RESTART : systeme NEUF, MEME composition, reprise puis 5 pas ---
        simB = _build()
        simB.restart(path)
        boxes_rst = simB.patch_boxes()
        # T2 : la hierarchie imposee == celle du checkpoint
        assert boxes_rst == boxes_chk, "patch_boxes() apres restart != checkpoint"
        assert simB.n_levels() == simA.n_levels(), "nombre de niveaux divergent"
        _advance(simB, 5, dt)
        finalB = [np.asarray(simB.level_state(k), dtype=np.float64)
                  for k in range(simB.n_levels())]

        # T1 : etat FINAL identique BIT A BIT, niveau par niveau (grossier ET patchs fins)
        for k in range(len(finalA)):
            assert finalA[k].shape == finalB[k].shape, "forme du niveau %d divergente" % k
            dmax = float(np.max(np.abs(finalA[k] - finalB[k]))) if finalA[k].size else 0.0
            assert dmax == 0.0, "niveau %d : dmax = %r != 0 (reprise NON bit-identique)" % (k, dmax)


def test_amr_checkpoint_restores_clock():
    """T3 : t et macro_step restaures par le restart (la cadence reprend exactement)."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")
        simA = _build()
        _advance(simA, 5, dt)
        simA.checkpoint(path)
        t_chk, ms_chk = simA.time(), simA.macro_step()
        assert ms_chk == 5, "macro_step() = %d apres 5 pas (attendu 5)" % ms_chk
        with np.load(path + ".npz", allow_pickle=False) as payload:
            temporal = json.loads(str(payload["temporal_restart_state"]))
        assert temporal["strategy"]["strategy"]["kind"] == "fixed_dt"
        assert temporal["transaction_stats"]["accepted"] == 5

        simB = _build()
        simB.restart(path)
        assert simB.macro_step() == ms_chk, "macro_step() = %d apres restart (attendu %d)" % (
            simB.macro_step(), ms_chk)
        assert abs(simB.time() - t_chk) < 1e-15, "time() = %r apres restart (attendu %r)" % (
            simB.time(), t_chk)
        replacement = pops.time.FixedDt(2 * dt)
        simB._step_strategy = replacement
        simB._step_transaction_plan = pops.time.StepTransactionPlan(replacement)
        _expect(
            RuntimeError,
            lambda: simB.run(t_end=t_chk + 2 * dt, max_steps=1),
            "le premier pas AMR post-restart doit conserver la strategie authentifiee",
        )
        assert simB.macro_step() == ms_chk and simB.time() == t_chk


def _expect(exc_types, fn, label):
    raised = False
    try:
        fn()
    except exc_types:
        raised = True
    assert raised, label


def test_amr_checkpoint_rejects():
    """T4 : rejets explicites (composition differente, n different, regrid_every>0, multi-blocs)."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")
        simA = _build()
        _advance(simA, 3, dt)
        simA.checkpoint(path)

        # composition differente : nom de bloc != celui du checkpoint
        sim_diffblock = _build(block="autre")
        _expect(ValueError, lambda: sim_diffblock.restart(path),
                "restart aurait du lever sur une composition (bloc) differente")

        # grille differente : n != celui du checkpoint
        sim_diffn = _build(n=48)
        _expect(ValueError, lambda: sim_diffn.restart(path),
                "restart aurait du lever sur une grille (n) differente")

        # regrid_every > 0 au restart : la cadence regrid re-divergerait
        sim_regrid = _build(regrid_every=10)
        _expect(ValueError, lambda: sim_regrid.restart(path),
                "restart aurait du lever sur regrid_every > 0")


def _build_multiblock(n=32, regrid_every=0):
    """AMR MULTI-BLOCS (2 especes ions/elec sur la hierarchie partagee, moteur AmrRuntime). Densites
    decalees (les deux blocs n'evoluent pas identiquement -> le checkpoint par bloc est verifiable).
    refinement bas (patchs fins actifs des le seed), hierarchie FIGEE (regrid_every=0)."""
    rho_i = _bump(n, amp=1.0, w=0.10)
    rho_e = _bump(n, amp=0.6, w=0.14)  # profil DIFFERENT -> trajectoires par bloc distinctes
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every)
    for nm, q in (("ions", +1.0), ("elec", -1.0)):
        sim.block(nm,
                      engine.Model(engine.Scalar(), engine.ExB(B0=1.0), engine.NoSource(),
                                engine.ChargeDensity(charge=q)),
                      spatial=engine.Spatial(limiter=Minmod(), flux=Rusanov()),
                      time=engine.Explicit())
    sim.set_refinement(threshold=1.5)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.set_density("ions", rho_i)
    sim.set_density("elec", rho_e)
    return sim


def test_amr_checkpoint_multiblock_bit_identical():
    """T5 : reprise MULTI-BLOCS BIT-IDENTIQUE (etat final par BLOC et par niveau, hierarchie partagee).

    Le checkpoint serialise l'etat conservatif PAR BLOC et le phi PARTAGE ; le restart rejoue la meme
    composition (la hierarchie centrale figee est reproduite deterministiquement) puis restaure chaque
    bloc. Mono-rang -> executable sur Mac sans compilation Kokkos."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt_mb")

        simA = _build_multiblock()
        _advance(simA, 5, dt)
        simA.checkpoint(path)
        boxes_chk = simA.patch_boxes()
        assert any(b[0] == 1 for b in boxes_chk), "patchs fins inactifs : test sans interet"
        _advance(simA, 5, dt)
        names = list(simA.block_names())
        finalA = {b: [np.asarray(simA.block_level_state(b, k), dtype=np.float64)
                      for k in range(simA.n_levels())] for b in names}

        simB = _build_multiblock()
        simB.restart(path)
        assert simB.patch_boxes() == boxes_chk, "patch_boxes() multi-blocs apres restart != checkpoint"
        assert list(simB.block_names()) == names, "blocs divergents apres restart"
        _advance(simB, 5, dt)
        finalB = {b: [np.asarray(simB.block_level_state(b, k), dtype=np.float64)
                      for k in range(simB.n_levels())] for b in names}

        for b in names:
            for k in range(simA.n_levels()):
                a, c = finalA[b][k], finalB[b][k]
                assert a.shape == c.shape, "bloc %s niveau %d : forme divergente" % (b, k)
                dmax = float(np.max(np.abs(a - c))) if a.size else 0.0
                assert dmax == 0.0, ("bloc %s niveau %d : dmax = %r != 0 (reprise multi-blocs NON "
                                     "bit-identique)" % (b, k, dmax))


if __name__ == "__main__":
    test_amr_checkpoint_bit_identical()
    print("OK T1/T2 : reprise bit-identique (niveau 0 + patchs fins) + hierarchie restauree")
    test_amr_checkpoint_restores_clock()
    print("OK T3 : horloge (t, macro_step) restauree")
    test_amr_checkpoint_rejects()
    print("OK T4 : rejets composition / grille / regrid_every")
    test_amr_checkpoint_multiblock_bit_identical()
    print("OK T5 : reprise multi-blocs bit-identique (par bloc + par niveau)")
    print("test_amr_checkpoint : OK")
