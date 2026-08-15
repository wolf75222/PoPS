"""Test du codegen C++ du mini-DSL via la brique physique canonique.

Verifie : (1) ``emit_cpp_brick()`` apres abaissement ``Module`` produit la brique exact-rank
(signatures flux, dimensions, provider ABI, locals, assignations) ; (2) si un compilateur C++ et
les en-tetes du depot sont disponibles, la brique est compilee, executee sur des etats deterministes,
et son resultat compare a l'interprete numpy (meme arbre, deux backends).
"""
import os
import subprocess
import tempfile

import numpy as np

import test_dsl_compose as B


def main():
    model = B.build_euler()
    src = B.emit_brick(model, name="EulerGen")

    # (1) la source generee a la bonne forme (sans compilateur)
    assert "struct EulerGen {" in src, "brique attendue absente"
    assert "static constexpr int dimension = 2;" in src, "rang spatial exact absent"
    assert "template <int Axis>\n  POPS_HD State flux(const State& U, const auto&) const" in src
    assert "template <class Providers>\n  POPS_HD State flux(const State& U, const Providers& a, int axis) const" in src
    assert "serialize_exact_parameters(pops::ExactContractBuilder& contract) const" in src
    assert "static constexpr int n_flux_providers = 0;" in src
    assert "const pops::Real rho = U[0];" in src and "const pops::Real E = U[3];" in src, "locals cons absents"
    assert "const pops::Real p = " in src and "const pops::Real u = " in src, "primitives absentes"
    assert src.count("F[") == 8, "attendu 4 composantes x 2 directions"
    assert "std::pow" not in src, "Euler ne devrait pas produire de pow"
    print("OK  emit_cpp_brick : source C++ generee (%d lignes)" % src.count("\n"))

    cxx = B.header_only_cxx()
    if cxx is None:
        return

    # (2) etats deterministes (rho > 0, p > 0) ; le main genere imprime F en pleine precision
    states = [(1.0, 0.2, -0.1, 2.5), (2.0, 0.5, 0.3, 6.0),
              (0.5, -0.2, 0.1, 1.8), (1.5, 0.0, 0.0, 3.0)]
    lits = ",".join("{%s}" % ",".join(repr(float(x)) for x in s) for s in states)
    main_cpp = src + (
        "#include <cstdio>\n"
        "int main() {\n"
        "  using Model = pops_generated::EulerGen;\n"
        "  static_assert(pops::HyperbolicModel<Model>);\n"
        "  pops::FluxProviderValues<Model> provider_values{};\n"
        "  const auto providers = pops::bind_flux_providers<Model>(provider_values);\n"
        "  const double S[%d][4] = {%s};\n"
        "  for (int k = 0; k < %d; ++k) {\n"
        "    Model::State U{};\n"
        "    for (int i = 0; i < 4; ++i) U[i] = S[k][i];\n"
        "    for (int d = 0; d < 2; ++d) {\n"
        "      const auto F = Model{}.flux(U, providers, d);\n"
        "      printf(\"%%.17g %%.17g %%.17g %%.17g\\n\", static_cast<double>(F[0]), static_cast<double>(F[1]), static_cast<double>(F[2]), static_cast<double>(F[3]));\n"
        "    }\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    ) % (len(states), lits, len(states))

    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "gen.cpp")
        exe = os.path.join(tmp, "gen")
        with open(cpp, "w") as f:
            f.write(main_cpp)
        subprocess.run(
            [cxx, *B.header_only_flags(), cpp, "-o", exe],
            check=True,
        )
        out = subprocess.run(
            [exe], capture_output=True, text=True, check=True,
        ).stdout

    rows = [list(map(float, line.split())) for line in out.strip().splitlines()]
    k = 0
    for s in states:
        U = np.array(s, dtype=float).reshape(4, 1, 1)
        for d in (0, 1):
            f_interp = model._m.flux(U, {}, d).reshape(4)
            f_cpp = np.array(rows[k])
            k += 1
            assert np.allclose(f_interp, f_cpp, rtol=1e-12, atol=1e-12), \
                "flux C++ != interprete (etat %s, dir %d) : %s vs %s" % (s, d, f_interp, f_cpp)
    print("OK  flux C++ genere == interprete numpy (%d etats x 2 directions, compile %s)"
          % (len(states), os.path.basename(cxx)))
    print("test_dsl_codegen : tout est vert")


if __name__ == "__main__":
    main()
