"""Source/host arithmetic witnesses, never a PoPS Native reception."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import pytest

from pops.codegen.moment_path_kernel import emit_moment_path_kernel
from pops.moments import CartesianMonomialBasis
from pops.moments.fan_li import FAN_LI15_INDICES, fan_li15_native_plan
from pops.moments.polynomial_path import (
    EndpointPathInputs, NormalizedPathInputs, derivative, endpoint_polynomial_path,
    normalized_polynomial_path,
)


@pytest.fixture(scope="session")
def native_cxx():
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None, "the host formula witness requires a compiler"
    return compiler


def test_authored_degree_two_integral_and_permuted_support():
    basis = CartesianMonomialBasis(((1, 1), (0, 0), (0, 2), (1, 0), (2, 0), (0, 1)))
    endpoint, pair = NormalizedPathInputs(basis), EndpointPathInputs(basis)
    # A deliberately different law: B_g = (g_x-g_y/2) rho I.
    integral = tuple((pair.direction(0)-pair.direction(1)/2)
                     * (pair.left((0, 0))+pair.right((0, 0)))/2
                     * (pair.right(index)-pair.left(index)) for index in basis.indices)
    plan = endpoint_polynomial_path(basis, flux=(0,)*6, integral=integral, speed=1)
    source = "\n".join(emit_moment_path_kernel(plan, "DifferentLaw"))
    assert "slots[] = {1, 3, 4, 5, 0, 2}" in source
    assert "raw_left[1]" in source and "raw_right[1]" in source
    assert "hermite" not in source
    assert "analytic_endpoint" == plan["integral_mode"]


@pytest.mark.parametrize("mutation", ("cycle", "bool_ref", "degree", "op", "missing_flux", "forged_capacity"))
def test_graph_contract_rejects_malformed_operations(mutation):
    plan = deepcopy(fan_li15_native_plan())
    target = plan["flux"][0]
    node = plan["nodes"][target]
    if mutation == "cycle":
        node["args"] = (target, target)
    elif mutation == "bool_ref":
        node["args"] = (True, node["args"][1])
    elif mutation == "degree":
        node["degree"] = True
    elif mutation == "op":
        node["op"] = "hermite_closure"
    elif mutation == "missing_flux":
        plan["flux"] = plan["flux"][:-1]
    else:
        plan["polynomial_degree"] = 6
    with pytest.raises((ValueError, TypeError)):
        emit_moment_path_kernel(plan, "Invalid")


def test_endpoint_integral_has_no_implicit_endpoint_retargeting():
    basis = CartesianMonomialBasis(((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)))
    endpoint = NormalizedPathInputs(basis)
    plan = endpoint_polynomial_path(basis, flux=(0,)*6,
        integral=(endpoint.raw((0, 0)), 0, 0, 0, 0, 0), speed=1)
    with pytest.raises(ValueError, match="qualified left/right"):
        emit_moment_path_kernel(plan, "Invalid")


def test_host_storage_permutation_preserves_orientation_and_all_bits(tmp_path, native_cxx):
    root = Path(__file__).resolve().parents[2]
    canonical = fan_li15_native_plan()
    basis = CartesianMonomialBasis(tuple(reversed(FAN_LI15_INDICES)))
    permuted = fan_li15_native_plan(basis=basis)
    historical = subprocess.run(["git", "show", "3f5a5502:tests/cpp/support/generated_fan_li15.hpp"],
                                cwd=root, capture_output=True, text=True, check=True).stdout
    historical = historical.replace("namespace test_fan_li15", "namespace historical")
    (tmp_path / "historical.hpp").write_text(historical)
    source = ("#include <pops/numerics/moments/normalized_hermite.hpp>\n#include \"historical.hpp\"\n#include <pops/numerics/moments/normalized_moment_path.hpp>\n#include <array>\n"
              + "\n".join(emit_moment_path_kernel(canonical, "Canonical")) + "\n"
              + "\n".join(emit_moment_path_kernel(permuted, "Permuted")) + r'''
int main() {
  double a[15]={2,0,2,0,6,0,0,0,0,2,0,2,0,0,6};
  double b[15]={1,0.1,1.01,0.301,3.0601,-0.2,-0.02,-0.202,-0.0602,1.04,0.104,1.0504,-0.608,-0.0608,3.2416};
  double pa[15], pb[15]; for(int k=0;k<15;++k) {pa[14-k]=a[k];pb[14-k]=b[k];}
  auto old=historical::Kernel{}.path_integral(a,b,{0.7,-0.2});
  auto oldf=historical::Kernel{}.path_directional_flux(a,{0.7,-0.2});
  auto x=Canonical{}.path_integral(a,b,{0.7,-0.2});
  auto y=Permuted{}.path_integral(pa,pb,{0.7,-0.2});
  auto f=Canonical{}.path_directional_flux(a,{0.7,-0.2});
  auto g=Permuted{}.path_directional_flux(pa,{0.7,-0.2});
  if (!x.succeeded() || !y.succeeded() || !f.succeeded() || !g.succeeded()) return 1;
  if (x.speed_bound!=y.speed_bound) return 2;
  for(int k=0;k<15;++k) {
    if(x.integral[k]!=y.integral[14-k] || f.flux.values[k]!=g.flux.values[14-k]) return 3;
    if (std::abs(x.integral[k]-old.integral[k]) > 512*std::numeric_limits<double>::epsilon()*(1+std::abs(old.integral[k]))) return 4;
    if (std::abs(f.flux.values[k]-oldf.flux.values[k]) > 512*std::numeric_limits<double>::epsilon()*(1+std::abs(oldf.flux.values[k]))) return 5;
  }
}
''')
    cpp, exe = tmp_path / "permutation.cpp", tmp_path / "permutation"
    cpp.write_text(source)
    command = [native_cxx, "-std=c++20", "-DPOPS_NATIVE_DIM=2", "-O2", "-fno-fast-math", "-ffp-contract=off",
               "-I", str(root / "include"), str(cpp), "-o", str(exe)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(exe)], capture_output=True).returncode == 0


def test_basis_order_two_integrand_degree_six_is_realized_without_truncation(tmp_path, native_cxx):
    basis = CartesianMonomialBasis(((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)))
    inputs = NormalizedPathInputs(basis, polynomial=True)
    plan = normalized_polynomial_path(basis, flux=(0,)*6,
        integrands=(inputs.normalized((1, 0))**6 * derivative(inputs.normalized((1, 0))), 0, 0, 0, 0, 0), factors=(1,)*6, speed=1)
    assert plan["order"] == 2 and plan["polynomial_degree"] == 6
    body = "\n".join(emit_moment_path_kernel(plan, "DegreeSix"))
    assert "integrate_normalized_moment_path<2, 6>" in body
    source = "#include <pops/numerics/moments/normalized_moment_path.hpp>\n#include <array>\n" + body + r"""
int main() {
  const double a[6]={1,0,1,0,0,1}, b[6]={1,1,2,0,0,1};
  auto r=DegreeSix{}.path_integral(a,b,{1,0});
  // Integral_0^1 u^6 du = 1/7; coefficient six must survive, even
  // though the canonical numerical orientation evaluates (1-s)^6.
  // Its integer binomial coefficients are exact. Seven divisions/products
  // and four Kahan operations per coefficient give gamma_42 times the
  // absolute weighted coefficient sum sum_k C(6,k)/(k+1)=127/7.
  const double e=std::numeric_limits<double>::epsilon();
  const double bound=(42*e/(1-42*e))*(127.0/7.0);
  if(!r.succeeded() || std::abs(r.integral[0]-1.0/7.0)>bound) return 1;
}
"""
    root = Path(__file__).resolve().parents[2]
    cpp, exe = tmp_path / "degree6.cpp", tmp_path / "degree6"
    cpp.write_text(source)
    result = subprocess.run([native_cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                             "-I", str(root / "include"), str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(exe)]).returncode == 0
