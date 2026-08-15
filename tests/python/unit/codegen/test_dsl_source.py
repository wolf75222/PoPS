"""Test de la brique de SOURCE generee (etape 2ter du DSL).

emit_cpp_source() produit un struct C++ expose apply(U, a) cense reproduire une brique de source
ECRITE A LA MAIN. Ce test : (1) construit le modele a 4 variables avec la source (q/m) rho E (forme
electrostatique), aux = grad_x/grad_y ; (2) genere la brique GenForce via le Module canonique et
son ProviderPack exact ; (3) si un compilateur et les en-tetes pops sont presents, compile un
programme qui inclut les vrais en-tetes pops et compare, sur des etats ET des aux deterministes,
GenForce::apply a pops::PotentialForce{-1.0}::apply composante par composante. Le programme imprime
l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pops.codegen.module_lowering import lower_and_validate
from pops.model import ProviderPack
from pops.physics._facade import Model
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)

QOM = -1.0
INCLUDE = repo_include()


def _lowered(model):
    """Emit only after the canonical Module resolves its exact provider packs."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert emit_model is model
    assert source_module is model.module
    assert type(emit_model._m._auxiliary_provider_pack) is ProviderPack
    return emit_model


def build_force_model():
    """Modele a 4 var avec la source (q/m) rho E, E = -grad phi (aux grad_x/grad_y).

    Source par composante (layout (rho, rho u, rho v, E)) :
      S[0] = 0
      S[1] = qom * rho * Ex          avec Ex = -grad_x
      S[2] = qom * rho * Ey          avec Ey = -grad_y
      S[3] = qom * (rho_u Ex + rho_v Ey)   (travail sur l'energie)
    Identique a pops::PotentialForce{qom} sur 4 variables.
    """
    m = Model("force")
    rho, rho_u, rho_v, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    zeros = [0.0 * value for value in (rho, rho_u, rho_v, E)]
    m.flux(x=zeros, y=zeros)
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.source([
        0,
        QOM * rho * (-gx),
        QOM * rho * (-gy),
        QOM * (rho_u * (-gx) + rho_v * (-gy)),
    ])
    return m


def emit_source(model, *, name):
    return _lowered(model)._m.emit_cpp_source(name=name)


HARNESS = r"""
#include <pops/physics/bricks/bricks.hpp>
%s
#include <cstdio>
#include <cmath>

int main() {
  pops_generated::GenForce gen;
  pops::PotentialForce ref{%r};

  // etats deterministes (rho > 0) et aux deterministes (gradients varies, signes mixtes).
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},
                         {0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0},{3.0,-1.2,0.7,9.0}};
  const double G[][2] = {{-0.3,0.7},{0.4,-0.9},{1.1,0.2},{0.0,-0.5},{-0.6,-0.6}};
  const int ns = sizeof(S)/sizeof(S[0]);
  const int ng = sizeof(G)/sizeof(G[0]);

  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<ns;++k){
    pops::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int j=0;j<ng;++j){
      pops::ProviderValues<pops_generated::GenForce::n_aux> a{};
      a[0] = G[j][0]; a[1] = G[j][1];
      auto sg = gen.apply(u, a);
      auto sr = ref.apply(u, a);
      for(int i=0;i<4;++i) upd(sg[i], sr[i]);
    }
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def _header_only_cxx():
    reason = missing_compiler_requirement(INCLUDE)
    cxx = default_cxx()
    if reason or not cxx:
        require_native_or_skip(reason or "compilateur C++ absent (CXX, c++, clang++)")
        return None
    return cxx


def _header_only_flags():
    return [
        "-std=c++20",
        "-O2",
        "-DPOPS_NATIVE_DIM=" + os.environ.get("POPS_NATIVE_DIM", "2"),
        "-I",
        INCLUDE,
        "-I",
        str(Path(sys.prefix) / "include"),
    ]


def _compile_and_run(source, stem):
    cxx = _header_only_cxx()
    if cxx is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, stem + ".cpp")
        exe = os.path.join(tmp, stem)
        with open(cpp, "w") as f:
            f.write(source)
        subprocess.run([cxx, *_header_only_flags(), cpp, "-o", exe], check=True)
        return subprocess.run([exe], capture_output=True, text=True, check=True).stdout


def main():
    m = build_force_model()
    struct = emit_source(m, name="GenForce")

    # (1) forme de la brique (sans compilateur)
    assert "struct GenForce {" in struct
    for token in ("apply(const pops::StateVec<4>&", "const auto& a",
        "const pops::Real grad_x = pops::provider_value<0>(a);",
        "const pops::Real grad_y = pops::provider_value<1>(a);",
                  "pops::StateVec<4> S{};"):
        assert token in struct, "membre attendu absent : %s" % token
    assert "static constexpr int n_aux = 2;" in struct
    print("OK  emit_cpp_source : struct genere (%d lignes)" % struct.count("\n"))

    out = _compile_and_run(HARNESS % (struct, QOM), "source")
    if out is None:
        print("test_dsl_source : OK (forme du struct seulement)")
        return

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "source generee != pops::PotentialForce (ecart max %.2e)" % maxdiff
    print("OK  GenForce::apply == pops::PotentialForce{%.1f} (ecart max %.1e)" % (QOM, maxdiff))
    print("test_dsl_source : tout est vert")


if __name__ == "__main__":
    main()
