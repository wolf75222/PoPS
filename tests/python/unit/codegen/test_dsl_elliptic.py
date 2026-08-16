"""Test du codegen elliptique (emit_cpp_elliptic) : meme mecanique que source / flux.

La brique de second membre generee (rhs(U)) doit reproduire pops::ChargeDensity ecrite a la main.
Pur Python ; gate sur compilateur + en-tetes pops, sinon skip propre. Emission uniquement apres
abaissement Module + ProviderPack exact.
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

Q = -1.0
INCLUDE = repo_include()
# Canonical Module-lowered cache identity for the charge-density elliptic brick.
GOLDEN_ELLIPTIC_HASH = "789e8b4e8d72cb54c1a1adaaf502b15a5117bf84d827e69f21ff3422c32832f2"


def _lowered(model):
    """Emit only after the canonical Module resolves its exact provider packs."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert emit_model is model
    assert source_module is model.module
    assert type(emit_model._m._auxiliary_provider_pack) is ProviderPack
    return emit_model


def build_charge():
    e = Model("charge")
    (rho,) = e.conservative_vars("rho")
    e.elliptic_rhs(Q * rho)   # densite de charge f = q n
    return e


def emit_elliptic(model, *, name):
    return _lowered(model)._m.emit_cpp_elliptic(name=name)


def canonical_hash(model):
    return _lowered(model)._model_hash()


HARNESS = r"""
#include <pops/physics/bricks/bricks.hpp>
%s
#include <cstdio>
#include <cmath>

int main() {
  pops::ChargeDensity ref; ref.q = %r;
  pops_generated::GenCharge gen;
  const double S[] = {0.0, 0.5, 1.0, 2.5, -0.3};
  double maxdiff = 0.0;
  for (int k=0;k<5;++k){
    pops::StateVec<1> u{}; u[0]=S[k];
    double d = std::fabs(ref.rhs(u) - gen.rhs(u));
    if (d>maxdiff) maxdiff=d;
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
    e = build_charge()
    src = emit_elliptic(e, name="GenCharge")
    assert "struct GenCharge" in src and "rhs(const State& U)" in src, "forme du struct inattendue"
    assert canonical_hash(e) == GOLDEN_ELLIPTIC_HASH, canonical_hash(e)
    print("OK  emit_cpp_elliptic : struct genere (%d lignes)" % src.count("\n"))

    out = _compile_and_run(HARNESS % (src, Q), "ell")
    if out is None:
        print("test_dsl_elliptic : OK (forme du struct seulement)")
        return

    d = float(out.strip())
    assert d < 1e-12, "elliptique genere != pops::ChargeDensity (ecart max %.2e)" % d
    print("OK  GenCharge::rhs == pops::ChargeDensity{%g} (ecart max %.1e)" % (Q, d))
    print("test_dsl_elliptic : tout est vert")


if __name__ == "__main__":
    main()
