"""Champs aux NOMMES declares par le modele -- ADC-70 phase 1 (System cartesien).

Un modele DSL peut declarer un champ auxiliaire ARBITRAIRE via m.aux_field("nom") (au-dela des champs
canoniques phi/grad/B_z/T_e). Le k-ieme nom reserve la composante AUX_NAMED_BASE + k (= 5 + k) du canal
aux, lue en C++ via aux.extra_field(k). La FACADE resout nom -> composante par bloc
(System.set_aux_field / aux_field) ; le C++ ne manipule que des indices.

Couvre :
  (forme, sans compilateur) helpers de largeur, emission n_aux + extra_field, retro-compat (modele sans
    champ nomme -> pas de n_aux), rejets DSL (nom canonique, doublon, depassement), rejets FACADE
    (B_z/T_e/canonique rediriges, bloc inconnu) ;
  (bout en bout, saute sans compilateur C++) source S = -kappa*n lisant aux_field("kappa") :
    (a) kappa CONSTANT -> residu == -kappa*n exact ; kappa SPATIAL (gaussien) -> residu suit le champ ;
    (b) PERSISTANCE : le champ nomme survit a plusieurs step() (relecture aux_field) ;
    (c) defaut : un modele simple SANS aux_field garde n_aux=3 ;
    (d) lecture AVANT ecriture -> zeros (documente) ; champ inconnu d'un bloc enregistre -> rejet.
"""
import os
import shutil
import tempfile

import numpy as np

import adc
from adc import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_decay_model():
    """Scalaire 'n' sans flux, source S = -kappa * n ou kappa est un champ aux NOMME (aux_field).
    flux nul -> eval_rhs = source = -kappa * n (verifiable exactement)."""
    m = dsl.Model("kappadecay")
    (nn,) = m.conservative_vars("n")
    zero = 0.0 * nn                      # expression nulle (flux/eig n'enrobent pas un float brut)
    m.flux(x=[zero], y=[zero])
    m.eigenvalues(x=[zero], y=[zero])
    m.primitive_vars(n=nn)               # layout Prim = [n]
    m.conservative_from([nn])
    kappa = m.aux_field("kappa")         # champ aux NOMME -> composante 5
    m.source([-(kappa * nn)])            # S = -kappa n
    return m


def test_form():
    """Garde-fous PUR-PYTHON (aucun compilateur) : largeurs, emission, retro-compat, rejets DSL."""
    # (1) largeur totale du canal : base seule = 3 ; un champ nomme -> AUX_NAMED_BASE + 1 = 6.
    assert dsl.aux_total_n_aux([], []) == 3
    assert dsl.aux_total_n_aux([], ["kappa"]) == 6
    assert dsl.aux_total_n_aux([], ["kappa", "sigma"]) == 7
    assert dsl.aux_total_n_aux(["B_z"], ["kappa"]) == 6  # max(4, 6)
    assert dsl.AUX_NAMED_BASE == 5
    print("OK  aux_total_n_aux : base=3, 1 nomme=6 (AUX_NAMED_BASE=5), 2 nommes=7")

    # (2) emission : un modele lisant aux_field('kappa') declare n_aux=6 et lit a.extra_field(0).
    m = dsl.HyperbolicModel("decay")
    (nn,) = m.conservative_vars("n")
    kappa = m.aux_field("kappa")
    m.set_source([-(kappa * nn)])
    src = m.emit_cpp_source(name="GenDecaySrc")
    assert "static constexpr int n_aux = 6;" in src, "n_aux=6 absent : %s" % src
    assert "const adc::Real kappa = a.extra_field(0);" in src, "lecture extra_field(0) absente : %s" % src
    print("OK  emit_cpp_source(aux_field) : n_aux=6 + a.extra_field(0)")

    # (3) retro-compat : un modele SANS aux_field n'emet PAS de n_aux (bit-identique a l'historique).
    m2 = dsl.HyperbolicModel("plain")
    (n2,) = m2.conservative_vars("n")
    m2.set_source([0.0 * n2])
    src2 = m2.emit_cpp_source(name="GenPlainSrc")
    assert "n_aux" not in src2, "n_aux ne doit pas etre emis pour un modele sans champ aux : %s" % src2
    assert m2._total_n_aux() == 3, "modele simple : n_aux total doit rester 3"
    print("OK  modele sans aux_field : pas de n_aux emis, largeur 3 (defaut)")

    # (4) rejets DSL : nom canonique, doublon, depassement de la borne kAuxMaxExtra.
    m3 = dsl.HyperbolicModel("rej")
    m3.conservative_vars("n")
    for bad in ("B_z", "T_e", "phi", "grad_x"):
        try:
            m3.aux_field(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("aux_field(%r) aurait du lever (nom canonique)" % bad)
    m3.aux_field("kappa")
    try:
        m3.aux_field("kappa")  # doublon
    except ValueError:
        pass
    else:
        raise AssertionError("aux_field doublon aurait du lever")
    # remplir jusqu'a la borne (kappa deja pose -> 3 de plus = 4 max), le 5e leve.
    m3.aux_field("a")
    m3.aux_field("b")
    m3.aux_field("c")
    try:
        m3.aux_field("d")  # 5e champ : depasse AUX_NAMED_MAX
    except ValueError:
        pass
    else:
        raise AssertionError("aux_field au-dela de AUX_NAMED_MAX aurait du lever")
    print("OK  aux_field rejette : nom canonique, doublon, > %d champs" % dsl.AUX_NAMED_MAX)


def test_facade_rejects():
    """Rejets de la FACADE qui ne demandent aucun bloc compile (resolution avant la table) : B_z / T_e
    rediriges vers leur chemin dedie, nom canonique non fixable, bloc inconnu."""
    sim = adc.System(n=8, L=1.0, periodic=True)
    field = np.ones((8, 8))
    # B_z -> set_magnetic_field (message redirigeant)
    try:
        sim.set_aux_field("blk", "B_z", field)
    except ValueError as ex:
        assert "set_magnetic_field" in str(ex), "le message B_z devrait rediriger : %r" % str(ex)
    else:
        raise AssertionError("set_aux_field('B_z') aurait du lever")
    # T_e -> set_electron_temperature_from
    try:
        sim.set_aux_field("blk", "T_e", field)
    except ValueError as ex:
        assert "set_electron_temperature_from" in str(ex), "le message T_e devrait rediriger : %r" % str(ex)
    else:
        raise AssertionError("set_aux_field('T_e') aurait du lever")
    # autre nom canonique (phi) non fixable
    try:
        sim.set_aux_field("blk", "phi", field)
    except ValueError:
        pass
    else:
        raise AssertionError("set_aux_field('phi') aurait du lever")
    # bloc inconnu (aucun champ nomme enregistre)
    try:
        sim.set_aux_field("inexistant", "kappa", field)
    except ValueError as ex:
        assert "inexistant" in str(ex)
    else:
        raise AssertionError("set_aux_field sur bloc inconnu aurait du lever")
    print("OK  facade : B_z/T_e/phi rediriges, bloc inconnu rejete")


def test_end_to_end():
    """Bout en bout : source lisant aux_field('kappa'), branchee via add_equation (backend AOT)."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> bout-en-bout saute (%s)" % INCLUDE)
        print("test_aux_named : OK (forme seulement)")
        return

    n, L = 16, 1.0
    tmp = tempfile.mkdtemp()
    try:
        m = build_decay_model()
        compiled = m.compile(os.path.join(tmp, "kappadecay.so"), include=INCLUDE, backend="aot")
        assert compiled.aux_extra_names == ["kappa"], "aux_extra_names attendu ['kappa']"
        assert compiled.n_aux == 6, "n_aux=6 attendu (5 + 1 champ nomme)"

        sim = adc.System(n=n, L=L, periodic=True)
        sim.set_poisson(rhs="charge_density", solver="geometric_mg")
        sim.add_equation("decay", model=compiled,
                         spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                         time=adc.Explicit())
        sim.set_density("decay", np.ones((n, n)))

        # (d) lecture AVANT ecriture : le champ nomme vaut 0 partout (canal initialise a zero).
        before = sim.aux_field("decay", "kappa")
        assert before.shape == (n, n) and float(np.max(np.abs(before))) == 0.0, \
            "kappa avant ecriture devrait etre 0 partout"
        print("OK  lecture avant ecriture : kappa == 0 (documente)")

        # (a1) kappa CONSTANT : eval_rhs = S = -kappa*n = -2 (n=1 partout).
        kc = 2.0
        sim.set_aux_field("decay", "kappa", kc * np.ones((n, n)))
        sim.solve_fields()
        R = np.array(sim.eval_rhs("decay"))
        err = float(np.max(np.abs(R + kc)))  # R = -kappa*n = -2
        assert err < 1e-12, "kappa constant non lu (max|R+kappa| = %.2e)" % err
        # relecture : aux_field rend bien le champ pose.
        rk = sim.aux_field("decay", "kappa")
        assert float(np.max(np.abs(rk - kc))) < 1e-12, "aux_field ne relit pas kappa constant"
        print("OK  kappa constant : eval_rhs == -kappa*n (max ecart %.2e)" % err)

        # (a2) kappa SPATIAL (gaussien) : eval_rhs = -kappa(x)*n suit le champ exactement.
        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        ks = 1.0 + 3.0 * np.exp(-30.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        sim.set_aux_field("decay", "kappa", ks)
        sim.solve_fields()
        R2 = np.array(sim.eval_rhs("decay"))
        err2 = float(np.max(np.abs(R2 + ks)))  # n=1 -> R = -kappa(x)
        assert err2 < 1e-12, "kappa spatial non suivi (max|R+kappa| = %.2e)" % err2
        print("OK  kappa spatial (gaussien) : eval_rhs suit -kappa(x) (max ecart %.2e)" % err2)

        # (b) PERSISTANCE : plusieurs step() ; kappa (champ statique) reste inchange.
        for _ in range(5):
            sim.step_cfl(0.4)
        rk2 = sim.aux_field("decay", "kappa")
        errp = float(np.max(np.abs(rk2 - ks)))
        assert errp < 1e-12, "kappa n'a pas persiste apres 5 step (max ecart %.2e)" % errp
        print("OK  persistance : kappa intact apres 5 step (max ecart %.2e)" % errp)

        # (d) champ inconnu d'un bloc ENREGISTRE -> rejet listant les champs connus.
        try:
            sim.set_aux_field("decay", "sigma", np.ones((n, n)))
        except ValueError as ex:
            assert "sigma" in str(ex) and "kappa" in str(ex), "le rejet devrait lister les champs : %r" % str(ex)
        else:
            raise AssertionError("set_aux_field('decay','sigma') aurait du lever (non declare)")
        print("OK  champ aux nomme inconnu d'un bloc enregistre rejete")

        print("test_aux_named : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_form()
    test_facade_rejects()
    test_end_to_end()


if __name__ == "__main__":
    main()
