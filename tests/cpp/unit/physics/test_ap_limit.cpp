// Limite asymptotic-preserving QUANTIFIEE : on balaie la raideur eps sur 8 decades a pas de
// temps FIXE et on montre que l'IMEX reste UNIFORMEMENT borne (stabilite independante de la
// raideur) et capture de mieux en mieux l'equilibre quand eps -> 0. C'est la propriete qui
// definit un schema AP, au-dela d'une seule valeur de eps (cf. test_imex_ap).
//
//   du/dt = (u_eq - u)/eps ,  IMEX implicite : u^{n+1} = (u^n + (dt/eps) u_eq)/(1 + dt/eps).
// Erreur apres n pas : |u_n - u_eq| = |u0 - u_eq| / (1 + dt/eps)^n  -> 0 quand eps -> 0.
// L'explicite, lui, a un facteur |1 - dt/eps| >> 1 des que dt >> eps : il explose.

#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/schemes/imex.hpp>

#include <cmath>
#include <vector>

using namespace pops;

namespace {

constexpr int kDim = kNativeDimension;
using Field = MultiFab<kDim>;

Field relaxation_field(Real value) {
  Extent<kDim> shape{};
  Extent<kDim> one_rank{};
  for (int axis = 0; axis < kDim; ++axis) {
    shape[axis] = 4;
    one_rank[axis] = 1;
  }
  const Box<kDim> domain = Box<kDim>::from_extents(shape);
  const mesh::BoxArray<kDim> layout(std::vector<Box<kDim>>{domain});
  const mesh::RankSpace<kDim> ranks(Index<kDim>{}, one_rank);
  Field state(layout, mesh::Distribution<kDim>::replicated(layout, ranks), Index<kDim>{}, 1,
              Extent<kDim>{});
  state.set_val(value);
  return state;
}

void no_transport(Field&, Real) {}

// n pas d'IMEX implicite sur la relaxation, renvoie |u_n - u_eq|.
static double imex_err(double eps, double u0, double u_eq, double dt, int n) {
  Field state = relaxation_field(Real(u0));
  const auto simpl = [=](Field& candidate, Real h) {
    const Real c = h / eps;
    for (std::size_t local = 0; local < candidate.local_size(); ++local) {
      const FieldView<Real, kDim> values = candidate.fab(local).view();
      for_each_cell(candidate.box(local), [values, c, u_eq] POPS_HD(const Index<kDim>& index) {
        values(index) = (values(index) + c * u_eq) / (1 + c);
      });
    }
  };
  for (int s = 0; s < n; ++s)
    imex_euler_step(state, dt, no_transport, simpl);
  pops::sync_host();
  return std::fabs(state.fab(0).view()(state.box(0).lo) - u_eq);
}

// explicite naif : facteur d'amplification |1 - dt/eps| par pas.
static double explicit_val(double eps, double u0, double u_eq, double dt, int n) {
  double u = u0;
  for (int s = 0; s < n; ++s)
    u = u + dt * (u_eq - u) / eps;
  return u;
}

TEST(ApLimit, UniformStabilityAcrossStiffness) {
  const double u0 = 2.0, u_eq = 1.0, dt = 0.1;
  const int n = 50;
  const std::vector<double> epss = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

  double prev = 1e30, worst = 0;
  bool monotone = true;
  for (double eps : epss) {
    const double e = imex_err(eps, u0, u_eq, dt, n);
    worst = std::fmax(worst, e);
    if (!std::isfinite(e))
      monotone = false;
    if (e > prev + 1e-14)
      monotone = false;  // erreur non croissante quand eps decroit
    prev = e;
  }
  const double e_stiff = imex_err(1e-8, u0, u_eq, dt, n);
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

}  // namespace
