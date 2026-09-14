"""Facade Python AmrSystem MULTI-BLOCS (capstone PR1, docs/AMR_MULTIBLOCK_DESIGN.md).

Deux transports scalaires DECLARES a SCHEMAS SPATIAUX DIFFERENTS (none/rusanov vs minmod/rusanov),
co-localises sur UNE hierarchie AMR PARTAGEE, avec un Poisson de SYSTEME a second membre SOMME
co-localise (Sum_b q_b n_b). Le transport est une advection constante explicite ; le potentiel sert
ici d'oracle du carrier elliptique, sans revendiquer un couplage ExB. On verifie cote Python :
  (a) les DEUX blocs sont enregistres (n_blocks == 2) et evoluent (densite changee par le transport) ;
  (b) la MASSE de CHAQUE bloc est conservee a ~machine (reflux + average_down, PAR BLOC) ;
  (c) le potentiel de systeme est non trivial (le Poisson somme co-localise produit un phi) ;
  (d) le chemin MONO-BLOC reste deterministe / bit-identique (run x2 -> dmax == 0, chemin AmrCouplerMP) ;
  (e) multi-blocs + regrid_every > 0 est ACCEPTE (deverrouillage Phase 2, C.6 : regrid d'union des tags).

L'ancien ModelSpec ExB est conserve comme controle negatif : son adaptation acceptait des gradients
InputAux non relies au Poisson et produisait un transport nul. Il doit etre refuse avant mutation.
Le couplage general champ/transport AMR reste hors de ce test de carrier.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import numpy as np
import pops
from pops.codegen.loader import CompiledModel
from pops.physics import Density
from pops.physics._facade import Model

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import AmrSystem  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.support.amr_tagging import install_prepared_threshold_union
from tests.python.support.native_execution_context import (
    install_compiled_model_amr_test_lane,
    runtime_aligned_abi,
)


def _bump(n, amp):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())  # offset moyen nul -> Sum q n solvable en periodique


def _amr_system(n, *, regrid_every):
    return AmrSystem(
        shape=(n, n),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        periodicity=(True, True),
        regrid_every=regrid_every,
    )


def _amr_lane_model():
    """Detached exact-rank package metadata used solely to authenticate this test lane."""
    abi_key, std = runtime_aligned_abi()
    return CompiledModel(
        so_path="<amr-multiblock-lane>",
        backend="production",
        cons_names=["n"],
        cons_roles=["density"],
        prim_names=["n"],
        n_vars=1,
        gamma=None,
        n_aux=0,
        params={},
        caps={},
        abi_key=abi_key,
        model_hash="amr-multiblock-lane",
        cxx="c++",
        std=std,
        native_dimension=2,
        target="amr_system",
    )


def _install_state_routes(sim, names):
    """Freeze every actual Case state identity before the first native package install."""
    model = pops.Model("amr-multiblock-state")
    state = model.state("U", components=("n",))
    case = pops.Case("amr-multiblock")
    blocks = {name: case.block(name, model, states=(state,)) for name in names}
    validated = pops.validate(case)
    for name, block in blocks.items():
        sim._s._install_block_state_route(name, validated.resolve(block[state]).qualified_id)


def _scalar_charge(name, q, *, background=0.0):
    """Explicit nonzero advection plus an independent scalar contribution to shared Poisson."""
    model = Model("%s-scalar-advection" % name)
    (rho,) = model.conservative_vars("n", roles=(Density(),))
    model.flux(x=[0.3 * rho], y=[0.2 * rho])
    model.eigenvalues(x=[0.3 + 0.0 * rho], y=[0.2 + 0.0 * rho])
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.elliptic_rhs(q * (rho - background))
    return model.compile(
        backend="production", target="amr_system", name=name,
        consumer_owner_qid="tests.amr-multiblock.%s" % name,
    )


def _reject_legacy_exb():
    """An unsupported legacy request must not install a block or default field slot."""
    sim = _amr_system(32, regrid_every=0)
    before = (sim.n_blocks(), tuple(sim.field_provider_slots()))
    request = engine.Model(
        engine.Scalar(), engine.ExB(), engine.NoSource(), engine.ChargeDensity(charge=1.0))
    try:
        sim.add_equation(
            "legacy-exb", request,
            spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()))
    except ValueError as error:
        assert "ModelSpec ExB" in str(error) and "field-output provider plan" in str(error)
    else:
        raise AssertionError("legacy ExB without an exact gradient provider was accepted")
    assert (sim.n_blocks(), tuple(sim.field_provider_slots())) == before


def _build(n=32, regrid_every=0, *, tagging=()):
    sim = _amr_system(n, regrid_every=regrid_every)
    install_compiled_model_amr_test_lane(sim, _amr_lane_model())
    _install_state_routes(sim, ("ions", "electrons"))
    sim.set_temporal_relations([2], [1], ["integral_only"])
    sim.set_poisson(bc=Periodic())
    sim.add_equation("ions", _scalar_charge("ions", +1.0),
                  spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()))
    sim.add_equation("electrons", _scalar_charge("electrons", -1.0),
                  spatial=engine.Spatial(limiter=Minmod(), flux=Rusanov()))  # SCHEMA DIFFERENT
    sim.set_density("ions", _bump(n, 0.40))
    sim.set_density("electrons", np.roll(_bump(n, 0.20), (3, 5), axis=(0, 1)))
    if tagging:
        install_prepared_threshold_union(sim, tagging)
    install_forward_euler_program(sim)
    sim.mark_bound()  # seals exact Program accepted-state capacity before stepping
    return sim


def main():
    n = 32
    _reject_legacy_exb()

    # (a)(b)(c) deux blocs, schemas differents, hierarchie partagee, Poisson somme co-localise.
    sim = _build(n=n, regrid_every=0)
    assert sim.n_blocks() == 2, "n_blocks != 2"

    d0i = np.array(sim.density("ions"), copy=True)
    d0e = np.array(sim.density("electrons"), copy=True)
    m0i, m0e = sim.mass("ions"), sim.mass("electrons")

    sim.advance(0.001, 10)

    d1i = np.asarray(sim.density("ions"))
    d1e = np.asarray(sim.density("electrons"))
    m1i, m1e = sim.mass("ions"), sim.mass("electrons")
    phi = np.asarray(sim.potential())

    # (a) les DEUX blocs ont evolue sous leur advection constante explicite.
    assert float(np.abs(d1i - d0i).max()) > 1e-6, "bloc ions non avance"
    assert float(np.abs(d1e - d0e).max()) > 1e-6, "bloc electrons non avance"
    # (b) masse de CHAQUE bloc conservee a ~machine (par bloc, reflux + average_down).
    assert abs(m1i - m0i) < 1e-9, "masse ions non conservee (dm=%.2e)" % abs(m1i - m0i)
    assert abs(m1e - m0e) < 1e-9, "masse electrons non conservee (dm=%.2e)" % abs(m1e - m0e)
    # (c) potentiel de systeme non trivial (Poisson somme co-localise q0 n0 + q1 n1).
    assert float(np.abs(phi).max()) > 1e-8, "potentiel trivial (Poisson somme inactif)"
    print("OK  AmrSystem multi-blocs : 2 blocs schemas differents, masse par bloc conservee, "
          "Poisson somme co-localise")

    # (d) MONO-BLOC deterministe (chemin AmrCouplerMP intouche) : run x2 -> dmax == 0.
    def run_mono():
        s = _amr_system(n, regrid_every=0)
        install_compiled_model_amr_test_lane(s, _amr_lane_model())
        _install_state_routes(s, ("ne",))
        s.set_temporal_relations([2], [1], ["integral_only"])
        s.set_poisson(bc=Periodic())
        s.add_equation(
            "ne",
            _scalar_charge("ne", 1.0, background=1.0),
            spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        )
        s.set_density("ne", _bump(n, 0.40))
        install_forward_euler_program(s)
        s.mark_bound()
        s.advance(0.001, 10)
        return np.asarray(s.density())

    a = run_mono()
    b = run_mono()
    assert float(np.abs(a - b).max()) == 0.0, "mono-bloc non bit-identique (dmax != 0)"
    print("OK  mono-bloc bit-identique (dmax == 0, chemin AmrCouplerMP intouche)")

    # (e) multi-blocs + regrid_every > 0 ACCEPTE (deverrouillage Phase 2, C.6 : regrid d'union des tags).
    # L'ancien refus (hierarchie figee) est leve : la grille re-grille a partir de l'union des tags de
    # tous les blocs. On verifie que le build paresseux + l'avance NE LEVENT PLUS et que la hierarchie
    # reste valide (au moins un patch fin). Le mouvement effectif de la grille est verrouille en C++
    # (test_amr_multiblock_regrid_union) ; ici on assure la non-regression de la facade Python.
    s = _build(
        n=n,
        regrid_every=2,
        tagging=(("ions", "n", 1.05), ("electrons", "n", 1.05)),
    )  # regrid_every > 0 en multi-blocs : DESORMAIS supporte
    s.advance(0.001, 6)     # declenche le build paresseux + plusieurs regrids
    assert s.n_patches() >= 1, "hierarchie sans patch fin apres regrid d'union"
    assert np.isfinite(np.asarray(s.density("ions"))).all(), "etat ions non fini apres regrid"
    print("OK  multi-blocs + regrid_every > 0 accepte (regrid d'union des tags, deverrouillage Phase 2)")

    print("OK test_amr_multiblock")


if __name__ == "__main__":
    main()
