"""Test des ROLES physiques portes par une brique generee (pops.dsl.emit_cpp_brick).

Une brique generee DECLARE desormais le SENS de ses composantes (densite, qte de mvt, energie...)
via pops::VariableSet::roles, et non plus seulement leurs noms. Les couplages inter-especes du System
resolvent ainsi une composante par index_of(role) au lieu d'un indice litteral.

Ce test verifie :
(1) FORME (sans compilateur) : Euler (noms standards) emet les semantiques structurees
    (density, momentum:<axis>, energy / pressure) ; un layout NON STANDARD avec roles=
    explicites emet ces roles dans l'ordre demande. Les noms sans role canonique exportent
    explicitement ``Custom`` avec un label utilisateur parallèle : le contrat ABI courant exige
    un descripteur total et n'autorise ni l'absence ni plusieurs ``Custom`` anonymes ambigus.
(2) RESOLUTION (si compilateur + en-tetes pops) : la brique au layout non standard compile, satisfait
    pops::HyperbolicModel, et index_of(momentum(axis)/density/energy) retrouve la BONNE composante
    QUELLE QUE SOIT sa position -- c'est exactement ce dont depend la resolution par role des couplages.
Lance avec python3.
"""
from tests.python.support.requirements import require_native_or_skip
import os
import subprocess
import tempfile

from pops.codegen.toolchain import native_compile_environment, pops_loader_build_flags
from pops.math import sqrt
from pops.frames import X_AXIS, Y_AXIS
from pops.codegen.module_lowering import lower_and_validate
from pops.physics import Density, Energy, Momentum, Pressure, Velocity
from pops.physics._facade import Model
from tests.python.support.requirements import repo_include

GAMMA = 1.4
INCLUDE = repo_include()


def emit_brick(model, *, name):
    # Resolve the exact empty ProviderPack and consumer plan through the same
    # canonical Module lowering used by production.
    emit_model, _ = lower_and_validate(model, facade=model)
    return emit_model._m.emit_cpp_brick(name=name)


def build_euler_brick():
    """Euler facade used by this role test through the canonical Module route."""
    e = Model("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    e.flux(
        x=[rhou, rhou * u + p, rhou * v, rho * H * u],
        y=[rhov, rhov * u, rhov * v + p, rho * H * v],
    )
    e.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.primitive_vars(rho, u, v, p)
    e.conservative_from(
        [rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)]
    )
    return e


def build_shuffled_brick():
    """Euler au layout NON STANDARD : composantes rangees (mom_y, E, mom_x, rho). Les noms ne suivent
    pas la convention, donc on impose les roles explicitement via roles=. La physique reste Euler ;
    seule la POSITION des composantes change. Sert a prouver que index_of(role) resout par le SENS."""
    e = Model("euler_shuf")
    # ordre des conservatives : [rho_v(my), E, rho_u(mx), rho]
    my, E, mx, rho = e.conservative_vars(
        "my", "ee", "mx", "rho",
        roles=[Momentum(Y_AXIS), Energy(), Momentum(X_AXIS), Density()])
    u = e.primitive("u", mx / rho)
    v = e.primitive("v", my / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    # flux Euler reordonne pour suivre le layout [my, E, mx, rho]
    e.flux(x=[mx * v, rho * H * u, mx * u + p, mx],
           y=[my * v + p, rho * H * v, my * u, my])
    e.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # Prim au layout primitif STANDARD (rho, u, v, p) avec ses roles ; to_conservative produit
    # ensuite le layout conservatif SHUFFLE [my, E, mx, rho] a partir de ces primitives.
    e.primitive_vars(rho, u, v, p,
                     roles=[Density(), Velocity(X_AXIS), Velocity(Y_AXIS), Pressure()])
    e.conservative_from([rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
                         rho * u, rho])
    return e


def build_scalar_brick():
    """Deux noms inconnus gardent deux labels Custom distincts dans l'ABI."""
    e = Model("scal")
    q1, q2 = e.conservative_vars("q1", "q2")
    e.flux(x=[q1, q2], y=[q2, q1])
    e.eigenvalues(x=[q1, q2], y=[q2, q1])
    e.primitive_vars(q1, q2)
    e.conservative_from([q1, q2])
    return e


HARNESS = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cmath>
#include <cstdio>

using R = pops::VariableSemantic;
using Providers = pops::ProviderValues<0>;
using NonzeroProviders = pops::ProviderValues<1>;

static_assert(pops::HyperbolicModel<pops_generated::ShufGen>, "brique non standard non conforme au concept");
static_assert(requires(const pops_generated::ShufGen m,
                       const pops_generated::ShufGen::State u,
                       const NonzeroProviders providers) {
  { m.flux(u, providers, 0) } -> std::same_as<pops_generated::ShufGen::State>;
  { m.max_wave_speed(u, providers, 0) } -> std::convertible_to<pops::Real>;
}, "le pont runtime doit accepter les ProviderValues non vides");

int main() {
  const pops::VariableSet c = pops_generated::ShufGen::conservative_vars();
  // layout = [my, E, mx, rho] : index_of(role) doit retrouver la composante par son SENS.
  if (c.index_of(R::momentum(1)) != 0) { printf("FAIL momentum:1=%%d\n", c.index_of(R::momentum(1))); return 1; }
  if (c.index_of(R::Energy)    != 1) { printf("FAIL Energy=%%d\n",    c.index_of(R::Energy));    return 1; }
  if (c.index_of(R::momentum(0)) != 2) { printf("FAIL momentum:0=%%d\n", c.index_of(R::momentum(0))); return 1; }
  if (c.index_of(R::Density)   != 3) { printf("FAIL Density=%%d\n",   c.index_of(R::Density));   return 1; }
  if (c.index_of(R::Pressure)  != -1){ printf("FAIL Pressure devrait etre absente\n");          return 1; }
  // Module lowering resolved ShufGen's exact zero-width ProviderPack.  The runtime calls compile
  // through that real pack; no legacy auxiliary carrier or test-only attachment is involved.
  const Providers providers{};
  const pops_generated::ShufGen::State state{0.0, 2.5, 0.0, 1.0};
  const auto invalid_flux = pops_generated::ShufGen{}.flux(
      state, providers, pops_generated::ShufGen::dimension);
  for (int component = 0; component < pops_generated::ShufGen::State::size(); ++component)
    if (!std::isnan(invalid_flux[component])) { printf("FAIL invalid flux not NaN\n"); return 1; }
  if (!std::isnan(pops_generated::ShufGen{}.max_wave_speed(
          state, providers, pops_generated::ShufGen::dimension))) {
    printf("FAIL invalid speed not NaN\n");
    return 1;
  }
  printf("OK\n");
  return 0;
}
"""


def main():
    # (1) FORME : roles emis pour Euler standard ----------------------------------------------
    euler = emit_brick(build_euler_brick(), name="EulerGen")
    assert ("conservative_vars() { return {pops::VariableKind::Conservative, "
            '{"rho", "rho_u", "rho_v", "E"}, 4, {pops::VariableSemantic::Density, '
            "pops::VariableSemantic::momentum(0), pops::VariableSemantic::momentum(1), "
            "pops::VariableSemantic::Energy}}; }") in euler, "roles conservatifs Euler absents/incorrects"
    assert ("primitive_vars() { return {pops::VariableKind::Primitive, "
            '{"rho", "u", "v", "p"}, 4, {pops::VariableSemantic::Density, '
            "pops::VariableSemantic::velocity(0), pops::VariableSemantic::velocity(1), "
            "pops::VariableSemantic::Pressure}}; }") in euler, "roles primitifs Euler absents/incorrects"
    print("OK  Euler : semantiques structurees exactes emises")

    # layout non standard : roles dans l'ordre demande
    shuf = emit_brick(build_shuffled_brick(), name="ShufGen")
    assert ("{pops::VariableSemantic::momentum(1), pops::VariableSemantic::Energy, "
            "pops::VariableSemantic::momentum(0), pops::VariableSemantic::Density}") in shuf, \
        "roles du layout non standard incorrects"
    assert "return flux<Axis>(U, a);" in shuf
    assert "return max_wave_speed<Axis>(U, a);" in shuf
    assert shuf.count("if constexpr (Axis + 1 < dimension)") == 2
    print("OK  layout non standard : roles explicites emis dans l'ordre du layout")

    # Contrat strict : deux noms inconnus conservent deux identites Custom distinctes.
    scal = emit_brick(build_scalar_brick(), name="ScalGen")
    assert ('conservative_vars() { return {pops::VariableKind::Conservative, {"q1", "q2"}, 2, '
            '{pops::VariableSemantic::Custom, pops::VariableSemantic::Custom}, '
            '{"q1", "q2"}}; }') in scal, \
        "les composantes generiques doivent emettre deux labels Custom exacts"
    assert ('primitive_vars() { return {pops::VariableKind::Primitive, {"q1", "q2"}, 2, '
            '{pops::VariableSemantic::Custom, pops::VariableSemantic::Custom}, '
            '{"q1", "q2"}}; }') in scal, \
        "l'etat primitif generique doit garder deux labels Custom exacts"
    print("OK  noms inconnus : labels Custom distincts (metadata ABI totale)")

    # (2) RESOLUTION par role a travers le C++ (si compilateur dispo) --------------------------
    if not os.path.isdir(INCLUDE):
        require_native_or_skip('skip  en-tetes pops absents -> resolution sautee (%s)' % INCLUDE)
        print("test_dsl_roles : OK (forme des roles seulement)")
        return
    try:
        cxx, compile_flags, link_flags = pops_loader_build_flags()
    except RuntimeError as exc:
        require_native_or_skip(str(exc))
        return

    prog = HARNESS % shuf
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "roles.cpp")
        exe = os.path.join(tmp, "roles")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run(
            [
                cxx,
                "-std=c++20",
                "-O2",
                "-I",
                INCLUDE,
                *compile_flags,
                cpp,
                *link_flags,
                "-o",
                exe,
            ],
            check=True,
            env=native_compile_environment(),
        )
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "OK", "index_of(role) n'a pas retrouve la bonne composante : %s" % out.strip()
    print("OK  index_of(role) retrouve la composante par son SENS dans un layout non standard")
    print("test_dsl_roles : tout est vert")


if __name__ == "__main__":
    main()
