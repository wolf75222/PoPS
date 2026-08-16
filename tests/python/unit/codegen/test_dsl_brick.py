"""Test de l'emballage en BRIQUE compilee (etape 2bis du DSL).

emit_cpp_brick() produit un struct C++ cense satisfaire le concept pops::HyperbolicModel. Ce test :
(1) genere la brique pour Euler via le Module canonique et son ProviderPack exact ; (2) si un
compilateur + les en-tetes pops sont presents, compile un programme qui inclut les vrais en-tetes
pops, AFFIRME static_assert(pops::HyperbolicModel<brique>), et compare chaque methode (flux,
max_wave_speed, to_primitive, to_conservative) a la brique ECRITE A LA MAIN pops::Euler sur des
etats deterministes. La compilation echoue si le concept n'est pas satisfait ; le programme imprime
l'ecart max, qu'on exige < 1e-12. Lance avec python3.
"""
import os
import subprocess
import tempfile
from functools import cache

from pops._native_selector import select_native_dimension, selected_native_dimension
from pops.codegen.toolchain import (
    loader_cxx_std,
    native_compile_environment,
    pops_loader_build_flags,
)
from pops.codegen.module_lowering import lower_and_validate
from pops.math import sqrt
from pops.physics._facade import Model
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)

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
        dimension = selected_native_dimension()
        if dimension is None:
            configured_dimension = os.environ.get("POPS_NATIVE_DIM")
            if configured_dimension is None:
                raise RuntimeError(
                    "header-only brick harness requires a selected native dimension or POPS_NATIVE_DIM"
                )
            if configured_dimension not in {"1", "2", "3"}:
                raise RuntimeError(
                    "header-only brick harness requires canonical POPS_NATIVE_DIM text 1, 2, or 3"
                )
            dimension = int(configured_dimension)
            select_native_dimension(dimension)
        selected_cxx, compile_flags, link_flags = pops_loader_build_flags(cxx)
    except (RuntimeError, ValueError) as exc:
        require_native_or_skip(str(exc))
        return None
    return selected_cxx, loader_cxx_std(), tuple(compile_flags), tuple(link_flags)


def _header_only_cxx():
    toolchain = _header_only_toolchain()
    return None if toolchain is None else toolchain[0]


def _header_only_flags():
    toolchain = _header_only_toolchain()
    if toolchain is None:
        return []
    _cxx, standard, compile_flags, link_flags = toolchain
    return [
        "-std=" + standard,
        "-O2",
        "-I",
        INCLUDE,
        *compile_flags,
        *link_flags,
    ]


def _header_only_compile_flags():
    toolchain = _header_only_toolchain()
    if toolchain is None:
        return []
    _cxx, standard, compile_flags, _link_flags = toolchain
    return ["-std=" + standard, "-O2", "-I", INCLUDE, *compile_flags]


def _header_only_link_flags():
    toolchain = _header_only_toolchain()
    return [] if toolchain is None else list(toolchain[3])


def emit_brick(model, *, name):
    """Emit only after the canonical Module resolves its exact provider packs."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    assert type(emit_model._component_flux_provider_pack).__name__ == "ProviderPack"
    return emit_model._m.emit_cpp_brick(name=name)


def build_euler_brick():
    """Internal codegen facade lowered through the canonical Module/ProviderPack route."""
    e = Model("euler")
    rho, rhou, rhov, energy = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    pressure = e.primitive(
        "p", (GAMMA - 1.0) * (energy - 0.5 * rho * (u * u + v * v))
    )
    enthalpy = (energy + pressure) / rho
    sound_speed = sqrt(GAMMA * pressure / rho)
    e.flux(
        x=[rhou, rhou * u + pressure, rhou * v, rho * enthalpy * u],
        y=[rhov, rhou * v, rhov * v + pressure, rho * enthalpy * v],
    )
    e.eigenvalues(
        x=[u - sound_speed, u, u + sound_speed],
        y=[v - sound_speed, v, v + sound_speed],
    )
    e.primitive_vars(rho, u, v, pressure)
    e.conservative_from([
        rho, rho * u, rho * v,
        pressure / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
    ])
    return e


HARNESS = r"""
#include <pops/physics/fluids/euler.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cstdio>
#include <cmath>

static_assert(pops::HyperbolicModel<pops::Euler>, "oracle Euler non conforme (setup du test)");
static_assert(pops::HyperbolicModel<pops_generated::EulerGen>, "brique generee non conforme au concept");

int main() {
  pops::Euler ref; ref.gamma = %r;
  pops_generated::EulerGen gen;
  const pops::ProviderValues<pops_generated::EulerGen::n_flux_providers> providers{};
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
    auto pr = ref.to_primitive(u); auto pg = gen.to_primitive(u);
    for(int i=0;i<4;++i) upd(pr[i], pg[i]);
    auto ur = ref.to_conservative(pr); auto ug = gen.to_conservative(pg);
    for(int i=0;i<4;++i) upd(ur[i], ug[i]);
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def build_exb_brick():
    """Transport scalaire par derive E x B : flux qui DEPEND des champs auxiliaires (grad phi).
    Sert a verifier que la brique generee lit le pack provider exact dans flux et max_wave_speed,
    et reproduit la brique manuelle pops::CartesianExBDrift avec B=(0,0,1)."""
    e = Model("exb")
    (n,) = e.conservative_vars("n")
    gx = e.aux("grad_x")
    gy = e.aux("grad_y")
    e.flux(x=[n * (-gy)], y=[n * gx])     # B=(0,0,1), v = (-d_y phi, d_x phi)
    e.eigenvalues(x=[-gy], y=[gx])        # |v_dir| comme borne
    e.primitive_vars(n)                  # scalaire : primitif = conservatif
    e.conservative_from([n])
    return e


EXB_HARNESS = r"""
#include <pops/physics/bricks/bricks.hpp>
#include <pops/core/model/physical_model.hpp>
%s
#include <cstdio>
#include <cmath>

static_assert(pops::HyperbolicModel<pops::CartesianExBDrift>, "oracle ExB non conforme (setup du test)");
static_assert(pops::HyperbolicModel<pops_generated::ExBGen>, "brique ExB generee non conforme au concept");
static_assert(pops_generated::ExBGen::n_flux_providers == 2, "ExB consumer pack is two compact slots");

int main() {
  pops::CartesianExBDrift ref;
  pops_generated::ExBGen gen;
  const double S[] = {0.5, 1.0, 2.0, -0.3};
  const double A[][2] = {{0.5,-0.3},{-0.2,0.7},{0.0,0.4},{1.1,-0.9}};
  double maxdiff = 0.0;
  auto upd = [&](double a, double b){ double d = std::fabs(a-b); if (d>maxdiff) maxdiff=d; };
  for (int k=0;k<4;++k){
    pops::StateVec<1> u{}; u[0]=S[k];
    for (int j=0;j<4;++j){
      pops::ProviderValues<pops::CartesianExBDrift::n_providers> ref_a{};
      ref_a[0] = A[j][0]; ref_a[1] = A[j][1]; ref_a[2] = 0.0; ref_a[3] = 0.0; ref_a[4] = 1.0;
      pops::ProviderValues<pops_generated::ExBGen::n_flux_providers> gen_a{};
      gen_a[0] = A[j][0]; gen_a[1] = A[j][1];
      for (int dir=0; dir<2; ++dir){
        upd(ref.flux(u,ref_a,dir)[0], gen.flux(u,gen_a,dir)[0]);
        upd(ref.max_wave_speed(u,ref_a,dir), gen.max_wave_speed(u,gen_a,dir));
      }
    }
  }
  printf("%%.17g\n", maxdiff);
  return 0;
}
"""


def _compile_and_run(source, stem):
    cxx = _header_only_cxx()
    if cxx is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, stem + ".cpp")
        exe = os.path.join(tmp, stem)
        with open(cpp, "w") as f:
            f.write(source)
        subprocess.run(
            [cxx, *_header_only_compile_flags(), cpp, "-o", exe, *_header_only_link_flags()],
            check=True,
            env=native_compile_environment(),
        )
        return subprocess.run(
            [exe], capture_output=True, text=True, check=True,
            env=native_compile_environment(),
        ).stdout


def main():
    brick = emit_brick(build_euler_brick(), name="EulerGen")

    # (1) forme de la brique (sans compilateur)
    assert "struct EulerGen {" in brick
    for m in ("State flux(", "max_wave_speed(", "to_primitive(", "to_conservative(",
              "conservative_vars()", "primitive_vars()", "using State", "using Prim"):
        assert m in brick, "membre attendu absent : %s" % m
    assert "static constexpr int n_flux_providers = 0;" in brick
    print("OK  emit_cpp_brick : struct genere (%d lignes)" % brick.count("\n"))

    out = _compile_and_run(HARNESS % (brick, GAMMA), "brick")
    if out is None:
        print("test_dsl_brick : OK (forme du struct seulement)")
        return

    maxdiff = float(out.strip())
    assert maxdiff < 1e-12, "brique generee != pops::Euler (ecart max %.2e)" % maxdiff
    print("OK  static_assert(HyperbolicModel<EulerGen>) + brique == pops::Euler (ecart max %.1e)"
          % maxdiff)

    # (2) brique a flux dependant des AUXILIAIRES (ExB) : les locals aux doivent etre emis dans
    # flux ET max_wave_speed, et la brique doit egaler pops::CartesianExBDrift ecrite a la main.
    exb = emit_brick(build_exb_brick(), name="ExBGen")
    assert exb.count("const pops::Real grad_x = pops::provider_value<0>(a);") >= 2, \
        "lectures provider absentes (flux/vitesse)"
    assert exb.count("const pops::Real grad_y = pops::provider_value<1>(a);") >= 2, \
        "lectures provider compactes absentes pour grad_y"
    assert "template <int Axis>\n  POPS_HD State flux(const State& U, const auto& a)" in exb, \
        "parametre provider exact non nomme dans le flux"
    assert "static constexpr int n_flux_providers = 2;" in exb
    out2 = _compile_and_run(EXB_HARNESS % exb, "exb")
    if out2 is None:
        print("test_dsl_brick : OK (forme ExB seulement)")
        return
    d2 = float(out2.strip())
    assert d2 < 1e-12, "brique ExB generee != pops::CartesianExBDrift (ecart max %.2e)" % d2
    print("OK  brique a flux auxiliaire (ExB) == pops::CartesianExBDrift (ecart max %.1e)" % d2)
    print("test_dsl_brick : tout est vert")


if __name__ == "__main__":
    main()
