"""Test des ROLES physiques portes par une brique generee (pops.dsl.emit_cpp_brick).

Une brique generee DECLARE desormais le SENS de ses composantes (densite, qte de mvt, energie...)
via pops::VariableSet::roles, et non plus seulement leurs noms. Les couplages inter-especes du System
resolvent ainsi une composante par index_of(role) au lieu d'un indice litteral.

Ce test verifie :
(1) FORME (sans compilateur) : Euler (noms standards) emet les roles CANONIQUES (Density, MomentumX,
    MomentumY, Energy / Pressure) ; un layout NON STANDARD (qte de mvt avant densite) avec roles=
    explicites emet ces roles dans l'ordre demande. Les noms sans role canonique exportent
    explicitement ``Custom`` : le contrat ABI courant exige
    un descripteur total et n'autorise plus l'absence ambigue de roles.
(2) RESOLUTION (si compilateur + en-tetes pops) : la brique au layout non standard compile, satisfait
    pops::HyperbolicModel, et index_of(MomentumX/MomentumY/Density/Energy) retrouve la BONNE composante
    QUELLE QUE SOIT sa position -- c'est exactement ce dont depend la resolution par role des couplages.
Lance avec python3.
"""
from tests.python.support.requirements import require_native_or_skip
import os
import shutil
import subprocess
import tempfile

from pops.math import sqrt
from pops.physics._model import HyperbolicModel
from tests.python.support.models import build_euler_brick
from tests.python.support.requirements import repo_include

GAMMA = 1.4
INCLUDE = repo_include()


def build_shuffled_brick():
    """Euler au layout NON STANDARD : composantes rangees (mom_y, E, mom_x, rho). Les noms ne suivent
    pas la convention, donc on impose les roles explicitement via roles=. La physique reste Euler ;
    seule la POSITION des composantes change. Sert a prouver que index_of(role) resout par le SENS."""
    e = HyperbolicModel("euler_shuf")
    # ordre des conservatives : [rho_v(my), E, rho_u(mx), rho]
    my, E, mx, rho = e.conservative_vars(
        "my", "ee", "mx", "rho",
        roles=["MomentumY", "Energy", "MomentumX", "Density"])
    u = e.primitive("u", mx / rho)
    v = e.primitive("v", my / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    # flux Euler reordonne pour suivre le layout [my, E, mx, rho]
    e.set_flux(x=[mx * v, rho * H * u, mx * u + p, mx],
               y=[my * v + p, rho * H * v, my * u, my])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # Prim au layout primitif STANDARD (rho, u, v, p) avec ses roles ; to_conservative produit
    # ensuite le layout conservatif SHUFFLE [my, E, mx, rho] a partir de ces primitives.
    e.set_primitive_state(rho, u, v, p,
                          roles=["Density", "VelocityX", "VelocityY", "Pressure"])
    e.set_conservative_from([rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
                             rho * u, rho])
    return e


def build_scalar_brick():
    """Modele a NOM inconnu (q) : aucun role canonique -> role Custom explicite."""
    e = HyperbolicModel("scal")
    (q,) = e.conservative_vars("q")
    e.set_flux(x=[q], y=[q])
    e.set_eigenvalues(x=[q], y=[q])
    e.set_primitive_state(q)
    e.set_conservative_from([q])
    return e


HARNESS = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cstdio>

using R = pops::VariableRole;

static_assert(pops::HyperbolicModel<pops_generated::ShufGen>, "brique non standard non conforme au concept");

int main() {
  const pops::VariableSet c = pops_generated::ShufGen::conservative_vars();
  // layout = [my, E, mx, rho] : index_of(role) doit retrouver la composante par son SENS.
  if (c.index_of(R::MomentumY) != 0) { printf("FAIL MomentumY=%%d\n", c.index_of(R::MomentumY)); return 1; }
  if (c.index_of(R::Energy)    != 1) { printf("FAIL Energy=%%d\n",    c.index_of(R::Energy));    return 1; }
  if (c.index_of(R::MomentumX) != 2) { printf("FAIL MomentumX=%%d\n", c.index_of(R::MomentumX)); return 1; }
  if (c.index_of(R::Density)   != 3) { printf("FAIL Density=%%d\n",   c.index_of(R::Density));   return 1; }
  if (c.index_of(R::Pressure)  != -1){ printf("FAIL Pressure devrait etre absente\n");          return 1; }
  printf("OK\n");
  return 0;
}
"""


def main():
    # (1) FORME : roles emis pour Euler standard ----------------------------------------------
    euler = build_euler_brick().emit_cpp_brick(name="EulerGen")
    assert ("conservative_vars() { return {pops::VariableKind::Conservative, "
            '{"rho", "rho_u", "rho_v", "E"}, 4, {pops::VariableRole::Density, '
            "pops::VariableRole::MomentumX, pops::VariableRole::MomentumY, "
            "pops::VariableRole::Energy}}; }") in euler, "roles conservatifs Euler absents/incorrects"
    assert ("primitive_vars() { return {pops::VariableKind::Primitive, "
            '{"rho", "u", "v", "p"}, 4, {pops::VariableRole::Density, '
            "pops::VariableRole::VelocityX, pops::VariableRole::VelocityY, "
            "pops::VariableRole::Pressure}}; }") in euler, "roles primitifs Euler absents/incorrects"
    print("OK  Euler (noms standards) : roles canoniques emis (Density/Momentum/Energy/Pressure)")

    # layout non standard : roles dans l'ordre demande
    shuf = build_shuffled_brick().emit_cpp_brick(name="ShufGen")
    assert ("{pops::VariableRole::MomentumY, pops::VariableRole::Energy, "
            "pops::VariableRole::MomentumX, pops::VariableRole::Density}") in shuf, \
        "roles du layout non standard incorrects"
    print("OK  layout non standard : roles explicites emis dans l'ordre du layout")

    # Contrat strict : un nom inconnu conserve son identite avec un role Custom explicite.
    scal = build_scalar_brick().emit_cpp_brick(name="ScalGen")
    assert ('conservative_vars() { return {pops::VariableKind::Conservative, {"q"}, 1, '
            '{pops::VariableRole::Custom}}; }') in scal, \
        "modele a nom inconnu doit emettre un role Custom explicite"
    assert ('primitive_vars() { return {pops::VariableKind::Primitive, {"q"}, 1, '
            '{pops::VariableRole::Custom}}; }') in scal, \
        "etat primitif identite doit emettre un role Custom explicite"
    print("OK  noms inconnus : role Custom explicite (metadata ABI totale)")

    # (2) RESOLUTION par role a travers le C++ (si compilateur dispo) --------------------------
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        require_native_or_skip('skip  compilateur ou en-tetes pops absents -> resolution sautee (%s)' % INCLUDE)
        print("test_dsl_roles : OK (forme des roles seulement)")
        return

    prog = HARNESS % shuf
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "roles.cpp")
        exe = os.path.join(tmp, "roles")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "OK", "index_of(role) n'a pas retrouve la bonne composante : %s" % out.strip()
    print("OK  index_of(role) retrouve la composante par son SENS dans un layout non standard")
    print("test_dsl_roles : tout est vert")


if __name__ == "__main__":
    main()
