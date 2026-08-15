"""Test de COMPOSITION de la brique generee (etape 2bis du DSL, suite).

emit_cpp_brick() produit un struct hyperbolique ; ce test verifie qu'il se COMPOSE comme n'importe
quelle brique manuelle. On l'insere dans un pops::CompositeModel<EulerGen, NoSource, ChargeDensity> et
on exige : (1) static_assert(pops::PhysicalModel<Gen>) et static_assert(pops::HyperbolicModel<EulerGen>)
(la composition compile et satisfait le contrat du modele physique) ; (2) sur des etats deterministes
(rho>0, p>0) et dir 0/1, le compose genere egale le compose ECRIT A LA MAIN (Euler oracle) sur flux,
max_wave_speed et elliptic_rhs. La compilation echoue si un concept n'est pas satisfait ; le programme
imprime l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import subprocess
import tempfile
from functools import cache

from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.toolchain import pops_loader_build_flags
from pops.math import sqrt
from pops.physics._facade import Model

GAMMA = 1.4
INCLUDE = repo_include()


@cache
def _header_only_toolchain():
    cxx = default_cxx()
    reason = missing_compiler_requirement(INCLUDE)
    if reason or cxx is None:
        require_native_or_skip(reason or "compilateur C++ absent (CXX, c++, clang++)")
        return None
    try:
        selected_cxx, compile_flags, link_flags = pops_loader_build_flags(cxx)
    except RuntimeError as exc:
        require_native_or_skip(str(exc))
        return None
    return selected_cxx, tuple(compile_flags), tuple(link_flags)


def header_only_cxx():
    toolchain = _header_only_toolchain()
    return None if toolchain is None else toolchain[0]


def header_only_flags():
    toolchain = _header_only_toolchain()
    if toolchain is None:
        return []
    _cxx, compile_flags, link_flags = toolchain
    return [
        "-std=c++20",
        "-O2",
        "-I",
        INCLUDE,
        *compile_flags,
        *link_flags,
    ]


def build_euler():
    """Author Euler through the internal codegen facade and canonical Module/ProviderPack route."""
    model = Model("euler")
    rho, rhou, rhov, energy = model.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = model.primitive("u", rhou / rho)
    v = model.primitive("v", rhov / rho)
    pressure = model.primitive(
        "p", (GAMMA - 1.0) * (energy - 0.5 * rho * (u * u + v * v))
    )
    enthalpy = (energy + pressure) / rho
    sound_speed = sqrt(GAMMA * pressure / rho)
    model.flux(
        x=[rhou, rhou * u + pressure, rhov * u, rho * enthalpy * u],
        y=[rhov, rhou * v, rhov * v + pressure, rho * enthalpy * v],
    )
    model.eigenvalues(
        x=[u - sound_speed, u, u + sound_speed],
        y=[v - sound_speed, v, v + sound_speed],
    )
    model.primitive_vars(rho, u, v, pressure)
    model.conservative_from([
        rho, rho * u, rho * v,
        pressure / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
    ])
    return model


def emit_brick(model, *, name, cse=True):
    """Emit only after the canonical Module resolves its exact provider packs."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    assert type(emit_model._component_flux_provider_pack).__name__ == "ProviderPack"
    return emit_model._m.emit_cpp_brick(name=name, cse=cse)

HARNESS = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cstdio>
#include <cmath>

// La brique generee doit etre un modele hyperbolique conforme...
static_assert(pops::HyperbolicModel<pops_generated::EulerGen>, "brique generee non conforme au concept");

// ...et se composer en un PhysicalModel complet (hyperbolique + source + elliptique).
using Gen = pops::CompositeModel<pops_generated::EulerGen, pops::NoSource, pops::ChargeDensity>;
using Ref = pops::CompositeModel<pops::Euler,              pops::NoSource, pops::ChargeDensity>;
static_assert(pops::PhysicalModel<Gen>, "compose genere non conforme au concept PhysicalModel");
static_assert(pops::PhysicalModel<Ref>, "compose oracle non conforme (setup du test)");

int main() {
  Gen gen;                       // EulerGen inline gamma dans ses formules (pas de membre gamma).
  Ref ref;  ref.hyp.gamma = %r;   // on aligne l'oracle ; q par defaut = 1 (ChargeDensity) des deux cotes.
  const pops::ProviderValues<0> providers{};
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},{0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0}};
  const int n = sizeof(S)/sizeof(S[0]);
  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<n;++k){
    pops::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int dir=0; dir<2; ++dir){
      auto fr = ref.flux(u,providers,dir); auto fg = gen.flux(u,providers,dir);
      for(int i=0;i<4;++i) upd(fr[i], fg[i]);
      upd(ref.max_wave_speed(u,providers,dir), gen.max_wave_speed(u,providers,dir));
    }
    upd(ref.elliptic_rhs(u), gen.elliptic_rhs(u));
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    brick = emit_brick(build_euler(), name="EulerGen")

    # (1) forme de la brique (sans compilateur)
    assert "struct EulerGen {" in brick
    for m in ("State flux(", "max_wave_speed(", "to_primitive(", "to_conservative(",
              "conservative_vars()", "primitive_vars()", "using State", "using Prim"):
        assert m in brick, "membre attendu absent : %s" % m
    print("OK  emit_cpp_brick : struct genere (%d lignes)" % brick.count("\n"))

    cxx = header_only_cxx()
    if cxx is None:
        return

    prog = HARNESS % (brick, GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "compose.cpp")
        exe = os.path.join(tmp, "compose")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run(
            [cxx, *header_only_flags(), cpp, "-o", exe],
            check=True,
        )
        out = subprocess.run(
            [exe], capture_output=True, text=True, check=True,
        ).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "compose genere != compose oracle (ecart max %.2e)" % maxdiff
    print("OK  static_assert(PhysicalModel<Gen>) + CompositeModel(EulerGen) == CompositeModel(Euler)"
          " (ecart max %.1e)" % maxdiff)
    print("test_dsl_compose : tout est vert")


if __name__ == "__main__":
    main()
