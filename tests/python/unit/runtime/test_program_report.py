#!/usr/bin/env python3
"""ADC-594 : report structure du sous-systeme Program compile (ProgramRuntimeReport).

L'etat runtime du Program compile (step installe / hash, cadence, block map, params runtime,
diagnostics, histories, cache, profiler) est EXTRAIT de System::Impl / AmrSystem::Impl dans une
struct partagee (pops::runtime::program::ProgramRuntimeState). Cote Python, System.program_report()
et AmrSystem.program_report() AGREGENT les accessors deja exposes en UN report inspectable, JSON-ready
(to_dict / to_json sans tableau), source UNIQUE de la section program de l'inspection ADC-591.

Ce test prouve LOCALEMENT (sous un _pops prebuilt, sans codegen / Kokkos obligatoire) :

  1. System frais : report installed=False, sections vides, cadence 1/1 (ou None si le .so prebuilt
     precede les getters ADC-594 program_substeps/program_stride -> skip gracieux, message CI).
  2. Apres avoir peuple une ring d'history (via restore_history, la route bas-niveau bindee), le
     report liste la ring (name / depth / ncomp / initialized).
  3. Le report passe par la vue BoundSimulation (surface DIAGNOSTIC allowlistee, ADC-583).
  4. La section program de l'inspection ADC-591 est bien construite DEPUIS le report (source unique).
  5. to_dict / to_json restent inertes et JSON-serialisables (aucun tableau de champ).

Ne FALSIFIE jamais le moteur pops : on construit un vrai System / AmrSystem, on skippe si une
dependance manque. Tourne sous pytest ET comme script (garde __main__)."""
import json
import sys

try:
    import numpy as np
    import pops
    from pops.runtime._bound_sim import BoundSimulation
    from pops.numerics.reconstruction.limiters import Minmod
except Exception as exc:  # noqa: BLE001
    print("skip test_program_report (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


def _isothermal_model():
    """Un pops.Model(...) natif (briques composees, PAS de compile DSL) -- aucun .so requis."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.PotentialForce(charge=1.0),
                      elliptic=pops.ChargeDensity(charge=1.0))


def _fresh_system(n=8):
    """Un System n x n periodique avec UN bloc natif (aucun program installe)."""
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("ions", _isothermal_model(),
                  spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    return sim


def test_fresh_report_is_empty():
    """Un System frais : rien d'installe, sections vides, report JSON-ready."""
    sim = _fresh_system()
    if not hasattr(sim, "program_report"):
        print("skip test_fresh_report_is_empty (pops build lacks program_report; rebuild pops)")
        return
    rep = sim.program_report()
    _check(rep.installed is False, "un System sans program : installed=False")
    _check(rep.program_hash == "", "pas de hash sans program installe")
    _check(rep.histories == [], "pas d'history sur un System frais")
    _check(rep.cache == [], "pas de slot de cache sur un System frais")
    _check(rep.diagnostics == {}, "pas de diagnostic recorded sur un System frais")
    _check(rep.block_map == [], "block_map vide (identite) sans program")
    # params : liste de lignes {program_block, count, limit}. Sans program installe AUCUNE ligne ne
    # declare de param runtime (count None ou 0) -- le contrat reel du builder (_params), pas une
    # invention ; limit = kMaxRuntimeParams (ADC-610 #432) expose la capacite du tableau fixe.
    _check(isinstance(rep.params, list), "params est une liste de lignes par bloc program")
    for row in rep.params:
        _check(set(row) == {"program_block", "count", "limit"},
               "chaque ligne params porte program_block+count+limit")
        _check(row["count"] in (None, 0), "aucun param runtime declare sur un System frais")
        _check(isinstance(row["limit"], int) and row["limit"] > 0,
               "limit surface kMaxRuntimeParams (entier > 0)")
    # profiler : dict portant la cle 'enabled' ; le profiling est ETEINT sur un System frais.
    _check(isinstance(rep.profiler, dict) and "enabled" in rep.profiler,
           "profiler est un dict portant la cle 'enabled'")
    _check(rep.profiler["enabled"] in (False, None), "profiler eteint sur un System frais")
    # Cadence : 1/1 si le .so expose les getters ADC-594, sinon None (skip gracieux, CI couvre).
    cad = rep.cadence
    if cad.get("substeps") is None:
        print("  (skip cadence sub-assert: _pops lacks program_substeps/stride (rebuild pops; CI covers))")
    else:
        _check(cad["substeps"] == 1 and cad["stride"] == 1, "cadence par defaut 1/1")
    # JSON-ready : to_dict serialisable, to_json round-trip.
    d = rep.to_dict()
    _check(d["report_type"] == "program_runtime", "report_type nomme le sous-systeme")
    _check(json.loads(rep.to_json()) == d, "to_json round-trip == to_dict")
    print("ok test_fresh_report_is_empty")


def test_report_lists_history_after_restore():
    """Apres avoir peuple une ring d'history (restore_history, route bas-niveau bindee), le report la
    liste. restore_history enregistre la ring co-distribuee avec le bloc 0 puis y scatter les valeurs."""
    sim = _fresh_system(n=8)
    if not hasattr(sim, "program_report") or not hasattr(sim._s, "restore_history"):
        print("skip test_report_lists_history_after_restore (_pops lacks the history seam; rebuild pops)")
        return
    ncell = 8 * 8
    ncomp = sim._s.n_vars("ions")
    values = np.zeros(ncomp * ncell, dtype=np.float64)  # slot 0, buffer global component-major
    sim._s.restore_history("u_prev", 0, values)  # enregistre la ring + scatter (register interne)
    sim._s.set_history_initialized("u_prev", True)
    rep = sim.program_report()
    names = [row["name"] for row in rep.histories]
    _check("u_prev" in names, "le report liste la ring d'history restauree")
    row = next(r for r in rep.histories if r["name"] == "u_prev")
    # restore_history(name, 0, .) enregistre la ring avec register_history(name, lag>=1 -> 1),
    # donc depth = lag + 1 = 2 (slot courant + un slot plus profond, zero-rempli).
    _check(row["depth"] == 2, "depth de la ring = 2 (register_history lag>=1)")
    _check(row["ncomp"] == ncomp, "ncomp de la ring = ncomp du bloc")
    _check(row["initialized"] is True, "la ring est initialisee")
    print("ok test_report_lists_history_after_restore")


def test_report_through_bound_simulation_view():
    """program_report est expose sur la vue BoundSimulation (surface DIAGNOSTIC allowlistee)."""
    sim = _fresh_system()
    if not hasattr(sim, "program_report"):
        print("skip test_report_through_bound_simulation_view (pops lacks program_report; rebuild pops)")
        return
    view = BoundSimulation(sim)
    rep = view.program_report()  # doit passer par l'allowlist _DIAGNOSTICS, pas lever
    _check(rep.report_type == "program_runtime", "la vue relaie program_report")
    _check(rep.installed is False, "meme report inerte via la vue")
    print("ok test_report_through_bound_simulation_view")


def test_inspection_program_section_from_report():
    """La section program de l'inspection ADC-591 est construite DEPUIS le report (source unique)."""
    sim = _fresh_system()
    if not hasattr(sim, "program_report"):
        print("skip test_inspection_program_section_from_report (pops lacks program_report; rebuild pops)")
        return
    from pops.runtime.program_report import build_program_report
    from pops.runtime.inspection import _program
    rep = build_program_report(sim)
    section = _program(sim)
    _check(section["installed"] == rep.installed, "section installed == report installed")
    _check(section["hash"] == rep.program_hash, "section hash == report hash (source unique)")
    _check(section["histories"] == [dict(r) for r in rep.histories],
           "section histories == report histories")
    _check("cadence" in section and "block_map" in section,
           "la section porte la cadence + block_map (plus seulement les maps string-only)")
    print("ok test_inspection_program_section_from_report")


def test_amr_report_shares_the_contract():
    """AmrSystem.program_report renvoie le MEME value object (sous-systeme partage, ADC-594)."""
    sim = pops.AmrSystem(n=8, L=1.0)
    if not hasattr(sim, "program_report"):
        print("skip test_amr_report_shares_the_contract (pops lacks program_report; rebuild pops)")
        return
    rep = sim.program_report()
    _check(rep.report_type == "program_runtime", "AMR : meme type de report")
    _check(rep.installed is False, "AMR frais : installed=False")
    _check(rep.cache == [], "AMR ne cable pas le cache scheduler (contrat commun documente)")
    print("ok test_amr_report_shares_the_contract")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
