#!/usr/bin/env python3
"""AmrSystem : alignement de table System/AMR pour limiter='weno5' x riemann={'hllc','roe'}.

DIVERGENCE CORRIGEE (audit GENERICITY_2026-06 §8 "registry des tags") : les branches hllc et roe du
dispatch AMR (detail::dispatch_amr_block, amr_dsl_block.hpp) n'avaient PAS de
cas 'weno5' (seulement les routes de halo <= 2) alors que System::make_block (block_builder.hpp) le route.
Resultat : un utilisateur AmrSystem demandant un schema compressible weno5+hllc (ou weno5+roe) recevait
"limiter inconnu 'weno5'" la ou le MEME modele buildait sous System. Les deux branches AMR portent
desormais le cas weno5 (build_amr_block supporte deja Weno5, cable sur
rusanov/hll) -> parite STRICTE de surface System/AMR.

Verifie (engine.Model compile en package NATIF ; le dispatch est exerce au BUILD) :
  - UN BLOC (un seul add_block -> dispatch_amr_block) : Euler compressible + weno5 + hllc TOURNE
    fini ; idem weno5 + roe. (Avant le fix : RuntimeError "limiter inconnu 'weno5'".)
  - MULTI-NIVEAUX : WENO5 selectionne le fournisseur coarse/fine conservatif d'ordre 5 / halo 3
    et avance sans abaissement silencieux vers l'interpolation MUSCL.
  - ISOTHERME : sa structure de contact physique autorise maintenant WENO5 + HLLC.
  - GARDE INTACTE : le transport scalaire sans structure de contact reste REJETE par la
    CAPABILITE HLLC, PAS par le limiteur ; WENO5 ne court-circuite pas la garde de flux.

Invariants par assert ; imprime "OK test_amr_weno5_hllc_roe" en cas de succes.
"""
from pops.numerics.riemann import HLLC, Roe
from pops.numerics.reconstruction import WENO5
import sys

import numpy as np
import pops

import pops.runtime._engine_descriptors as engine
from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam
from tests.python.integration._final_field_program import compile_block_model, scalar_advection_model
from tests.python.support.explicit_program import install_forward_euler_program

GAMMA = 1.4
fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def euler_spec():
    """Bloc natif compressible (4 var, pression -> capability HLLC/Roe canonique)."""
    return engine.Model(state=engine.FluidState("compressible", gamma=GAMMA),
                     transport=engine.CompressibleFlux(), source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0))


def iso_spec():
    """Bloc natif isotherme (3 var, pression barotrope et structure de contact HLLC)."""
    return engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                     transport=engine.IsothermalFlux(), source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0))


def _install_state_routes(system, names, components):
    """Declare the complete native composition before a package seals its state routes."""
    model = pops.Model("amr-weno5-state")
    state = model.state("U", components=components)
    case = pops.Case("amr-weno5")
    blocks = {name: case.block(name, model, states=(state,)) for name in names}
    validated = pops.validate(case)
    for name, block in blocks.items():
        system._s._install_block_state_route(name, validated.resolve(block[state]).qualified_id)


def bump(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)


def euler_state(density):
    """A density bump at rest with unit pressure, including its required thermal energy."""
    state = np.zeros((4, *density.shape), dtype=np.float64)
    state[0] = density
    state[3] = 1.0 / (GAMMA - 1.0)
    return state


n = 32
rho = bump(n)

# --- 1. UN BLOC (dispatch_amr_block) : weno5 + hllc, puis weno5 + roe --------------------------------
for riem in (HLLC(), Roe()):
    print(f"== un bloc Euler : weno5 + {riem.scheme} (dispatch_amr_block) ==")
    s = AmrSystem(n=n, L=1.0, periodicity=(True, True))
    _install_state_routes(s, ("gas",), ("rho", "mx", "my", "E"))
    s.set_temporal_relations([2], [1], ["integral_only"])
    s.add_equation("gas", euler_spec(),
                spatial=engine.Spatial(limiter=WENO5(), flux=riem), time=engine.Explicit())
    s.set_conservative_state("gas", euler_state(rho))
    install_forward_euler_program(s)
    s.mark_bound()  # seals the real installed Program accepted-state capacity
    for _ in range(3):
        s.step(1e-4)
    d = np.asarray(s.density("gas"))
    chk(np.all(np.isfinite(d)), f"weno5 + {riem.scheme} : densite finie sur 3 pas (build OK, plus de 'limiter inconnu')")

# --- 2. MULTI-NIVEAUX : le registre choisit le fournisseur C/F ordre 5 -----------------------------
print("== multi-niveaux Euler : weno5 + fournisseur coarse/fine ordre 5 ==")
s = AmrSystem(n=n, L=1.0, periodicity=(True, True))
_install_state_routes(s, ("a", "b"), ("rho", "mx", "my", "E"))
s.set_temporal_relations([2], [1], ["integral_only"])
s.add_equation("a", euler_spec(),
               spatial=engine.Spatial(limiter=WENO5(), flux=HLLC()), time=engine.Explicit())
s.add_equation("b", euler_spec(),
               spatial=engine.Spatial(limiter=WENO5(), flux=HLLC()), time=engine.Explicit())
s.set_conservative_state("a", euler_state(rho))
s.set_conservative_state("b", euler_state(rho))
install_forward_euler_program(s)
s.mark_bound()
s.step(1e-4)
chk(np.all(np.isfinite(np.asarray(s.density("a")))),
    "weno5 multi-niveaux avance avec le fournisseur coarse/fine ordre 5")

# --- 3. ISOTHERME : la structure de contact physique est maintenant supportee --------------------
print("== isotherme 3-var : weno5 + hllc avec la structure de contact physique ==")
s = AmrSystem(n=n, L=1.0, periodicity=(True, True))
_install_state_routes(s, ("iso",), ("rho", "mx", "my"))
s.set_temporal_relations([2], [1], ["integral_only"])
s.add_equation("iso", iso_spec(),
               spatial=engine.Spatial(limiter=WENO5(), flux=HLLC()), time=engine.Explicit())
s.set_density("iso", rho.copy())
install_forward_euler_program(s)
s.mark_bound()
s.step(1e-4)
chk(np.isfinite(np.asarray(s.density("iso"))).all(), "WENO5 + HLLC isotherme reste fini")

# --- 4. GARDE INTACTE : un scalaire sans structure de contact est refuse --------------------------
print("== transport scalaire : weno5 + hllc rejete par la CAPABILITE ==")
try:
    s = AmrSystem(n=n, L=1.0, periodicity=(True, True))
    _install_state_routes(s, ("scalar",), ("rho",))
    s.set_temporal_relations([2], [1], ["integral_only"])
    scalar = compile_block_model(
        scalar_advection_model("weno5-scalar-capability"), target="amr_system")
    s.add_equation("scalar", scalar,
                   spatial=engine.Spatial(limiter=WENO5(), flux=HLLC()), time=engine.Explicit())
    chk(False, "WENO5 + HLLC sans structure de contact aurait du lever")
except ValueError as error:
    message = str(error).lower()
    chk("hllc_star_state" in message and "requirement:" in message,
        f"rejet par la capability HLLC : {message[:100]}")
    chk("limiter inconnu" not in message, "WENO5 ne court-circuite pas la garde de flux")

if fails:
    print(f"FAIL test_amr_weno5_hllc_roe : {fails} echec(s)")
    sys.exit(1)
print("OK test_amr_weno5_hllc_roe")
