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
import shutil
import subprocess
import tempfile

GAMMA = 1.4
from tests.python.support.requirements import repo_include
INCLUDE = repo_include()


from tests.python.support.models import build_euler_brick


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
  pops::Aux aux{};
  const double S[][4] = {{1.0,0.2,-0.1,2.5},{2.0,0.5,0.3,6.0},{0.5,-0.2,0.1,1.8},{1.5,0.0,0.0,3.0}};
  const int n = sizeof(S)/sizeof(S[0]);
  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<n;++k){
    pops::StateVec<4> u{}; for(int i=0;i<4;++i) u[i]=S[k][i];
    for (int dir=0; dir<2; ++dir){
      auto fr = ref.flux(u,aux,dir); auto fg = gen.flux(u,aux,dir);
      for(int i=0;i<4;++i) upd(fr[i], fg[i]);
      upd(ref.max_wave_speed(u,aux,dir), gen.max_wave_speed(u,aux,dir));
    }
    upd(ref.elliptic_rhs(u), gen.elliptic_rhs(u));
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def main():
    e = build_euler_brick()
    brick = e.emit_cpp_brick(name="EulerGen")

    # (1) forme de la brique (sans compilateur)
    assert "struct EulerGen {" in brick
    for m in ("State flux(", "max_wave_speed(", "to_primitive(", "to_conservative(",
              "conservative_vars()", "primitive_vars()", "using State", "using Prim"):
        assert m in brick, "membre attendu absent : %s" % m
    print("OK  emit_cpp_brick : struct genere (%d lignes)" % brick.count("\n"))

    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> verification sautee (%s)" % INCLUDE)
        print("test_dsl_compose : OK (forme du struct seulement)")
        return

    prog = HARNESS % (brick, GAMMA)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "compose.cpp")
        exe = os.path.join(tmp, "compose")
        with open(cpp, "w") as f:
            f.write(prog)
        # le coeur pops est propre en C++20 (concepts) ; -I include suffit (header-only).
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "compose genere != compose oracle (ecart max %.2e)" % maxdiff
    print("OK  static_assert(PhysicalModel<Gen>) + CompositeModel(EulerGen) == CompositeModel(Euler)"
          " (ecart max %.1e)" % maxdiff)
    print("test_dsl_compose : tout est vert")


if __name__ == "__main__":
    main()
