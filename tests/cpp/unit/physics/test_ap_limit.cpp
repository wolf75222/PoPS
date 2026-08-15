// Limite asymptotic-preserving QUANTIFIEE : on balaie la raideur eps sur 8 decades a pas de
// temps FIXE et on montre que l'IMEX reste UNIFORMEMENT borne (stabilite independante de la
// raideur) et capture de mieux en mieux l'equilibre quand eps -> 0. C'est la propriete qui
// definit un schema AP, au-dela d'une seule valeur de eps (cf. test_imex_ap).
//
//   du/dt = (u_eq - u)/eps ,  IMEX implicite : u^{n+1} = (u^n + (dt/eps) u_eq)/(1 + dt/eps).
// Erreur apres n pas : |u_n - u_eq| = |u0 - u_eq| / (1 + dt/eps)^n  -> 0 quand eps -> 0.
// L'explicite, lui, a un facteur |1 - dt/eps| >> 1 des que dt >> eps : il explose.

#include <gtest/gtest.h>

#include <pops/numerics/time/schemes/imex.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <vector>

using namespace pops;

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
void no_transport(MultiFab<Dim>&, Real) {}

// n pas d'IMEX implicite sur la relaxation, renvoie |u_n - u_eq|.
template <int Dim>
static double imex_err(double eps, double u0, double u_eq, double dt, int n) {
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(4));
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> ranks(Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  MultiFab<Dim> U(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});
  U.set_val(u0);
  auto simpl = [=](MultiFab<Dim>& V, Real h) {
    const Real c = h / eps;
    for (std::size_t local = 0; local < V.local_size(); ++local) {
      const auto values = V.fab(local).view();
      for_each_cell(V.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        values(cell, 0) = (values(cell, 0) + c * u_eq) / (Real(1) + c);
      });
    }
  };
  for (int s = 0; s < n; ++s)
    imex_euler_step(U, dt, no_transport<Dim>, simpl);
  const auto& fab = U.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return std::fabs(host(0) - u_eq);
}

// explicite naif : facteur d'amplification |1 - dt/eps| par pas.
static double explicit_val(double eps, double u0, double u_eq, double dt, int n) {
  double u = u0;
  for (int s = 0; s < n; ++s)
    u = u + dt * (u_eq - u) / eps;
  return u;
}

template <int Dim>
void expect_uniform_stability_across_stiffness() {
  const double u0 = 2.0, u_eq = 1.0, dt = 0.1;
  const int n = 50;
  const std::vector<double> epss = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

  double prev = 1e30, worst = 0;
  bool monotone = true;
  for (double eps : epss) {
    const double e = imex_err<Dim>(eps, u0, u_eq, dt, n);
    worst = std::fmax(worst, e);
    if (!std::isfinite(e))
      monotone = false;
    if (e > prev + 1e-14)
      monotone = false;  // erreur non croissante quand eps decroit
    prev = e;
  }
  const double e_stiff = imex_err<Dim>(1e-8, u0, u_eq, dt, n);
  const double e_expl = explicit_val(1e-6, u0, u_eq, dt, 10);

  // stabilite UNIFORME : borne independante de la raideur (jamais > |u0 - u_eq|).
  EXPECT_TRUE(std::isfinite(worst) && worst <= std::fabs(u0 - u_eq) + 1e-12)
      << "AP_borne_uniforme (worst=" << worst << ")";
  // capture de l'equilibre dans la limite raide.
  EXPECT_TRUE(e_stiff < 1e-6) << "AP_capture_equilibre (e_stiff=" << e_stiff << ")";
  // l'erreur ne croit pas quand on durcit (de plus en plus AP).
  EXPECT_TRUE(monotone) << "AP_monotone_en_raideur";
  // contraste : l'explicite explose au meme regime.
  EXPECT_TRUE(!std::isfinite(e_expl) || std::fabs(e_expl) > 1e3)
      << "explicite_explose (e_expl=" << e_expl << ")";
}

TEST(ApLimit, UniformStabilityAcrossStiffness) {
  expect_uniform_stability_across_stiffness<1>();
  expect_uniform_stability_across_stiffness<2>();
  expect_uniform_stability_across_stiffness<3>();
}
