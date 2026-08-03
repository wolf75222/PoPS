// Le constructeur de fermetures de bloc (pops/runtime/block_builder.hpp) est extrait de System::Impl
// pour etre instanciable HORS de l'unite system.cpp : c'est la fondation du backend AOT (un modele
// genere par le DSL, compile ahead-of-time, entre dans le System par le chemin de PRODUCTION
// template -- HLLC, ordre 2 -- et non plus par le seul chemin hote virtuel du bloc dynamique).
//
// Ce test exerce le chemin EXTERNE : on assemble un CompositeModel arbitraire et un GridContext a la
// main (sans System), puis on verifie que make_block / make_max_speed / make_poisson_rhs produisent
// exactement le residu / la vitesse d'onde / le second membre de Poisson du chemin direct, et que
// l'avance SSPRK2 conserve la masse. Si ca compile et passe, un .so genere peut faire de meme.
#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, NoSource, GravityForce, GravityCoupling, IsothermalFlux
#include <pops/physics/fluids/euler.hpp>  // Euler (brique hyperbolique compressible)
#include <pops/runtime/builders/block/block_builder.hpp>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>

using namespace pops;

namespace {
constexpr double kPi = 3.14159265358979323846;

// euler_poisson compile : Euler + force de gravite + couplage GravityCoupling. Modele arbitraire
// assemble a la main, comme le ferait une unite generee AOT.
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

}  // namespace

// Pipeline sequentiel : (1) residu, (2) max_speed, (3) poisson_rhs, (4) avance -- toutes les etapes
// s'appuient sur le MEME U (etape 4 le mutant par avance), donc elles restent un seul TEST.
TEST(test_block_builder, external_seam_matches_direct_path_and_advances) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique

  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 1);
  aux.set_val(0.0);  // grad phi nul ici (on teste le cablage, pas la physique du couplage)

  Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};

  {  // bulle de densite + vitesse douce
    Array4 a = U.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i) - 0.5, y = geom.y_cell(j) - 0.5;
      const double rho = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02);
      a(i, j, 0) = rho;
      a(i, j, 1) = 0.1 * rho * std::sin(2 * kPi * geom.x_cell(i));
      a(i, j, 2) = 0.0;
      a(i, j, 3) = 1.0 / (1.4 - 1.0) + 0.5 * a(i, j, 1) * a(i, j, 1) / rho;
    });
  }

  const GridContext ctx{dom, bc, geom, &aux};

  // (1) make_block (minmod + HLLC, primitif) : son residu == assemble_rhs direct du meme schema.
  BlockClosures clo = make_block(model, "minmod", "hllc", ctx, /*recon_prim=*/true);
  MultiFab R1(ba, dm, 4, 0), R2(ba, dm, 4, 0);
  clo.rhs_into(U, R1);
  fill_ghosts(U, dom, bc);
  assemble_rhs<Minmod, HLLCFlux>(model, U, aux, geom, R2, /*recon_prim=*/true);
  double dres = 0, nrm = 0;
  for (int c = 0; c < 4; ++c) {
    const ConstArray4 r1 = R1.fab(0).const_array(), r2 = R2.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        dres = std::fmax(dres, std::fabs(r1(i, j, c) - r2(i, j, c)));
        nrm = std::fmax(nrm, std::fabs(r2(i, j, c)));
      }
  }
  EXPECT_LT(dres, 1e-14) << "rhs_into == assemble_rhs direct (HLLC ordre 2)";
  EXPECT_GT(nrm, 1e-6) << "residu non trivial";

  // (2) make_max_speed == max_wave_speed_mf direct.
  auto max_speed = make_max_speed(model, ctx);
  EXPECT_LT(std::fabs(max_speed(U) - max_wave_speed_mf(model, U, aux)), 1e-14) << "make_max_speed";

  // (3) make_poisson_rhs : rhs += elliptic_rhs(U) cellule par cellule.
  auto poisson_rhs = make_poisson_rhs(model);
  MultiFab rhs(ba, dm, 1, 0);
  rhs.set_val(0.0);
  poisson_rhs(U, rhs);
  double dpo = 0;
  {
    const ConstArray4 rr = rhs.fab(0).const_array(), u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        dpo =
            std::fmax(dpo, std::fabs(rr(i, j, 0) - model.elliptic_rhs(load_state<Model>(u, i, j))));
  }
  EXPECT_LT(dpo, 1e-14) << "make_poisson_rhs == elliptic_rhs";

  // (4) l'avance SSPRK2 tourne (chemin de production) et conserve la masse sur un etat lisse.
  const double mass0 = sum(U);
  run_explicit_substeps<SSPRK2Step>(
      [&](MultiFab& state, MultiFab& residual) { clo.rhs_into(state, residual); }, U, Real(2e-3),
      10);
  double mn = 1e300;
  {
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        mn = std::fmin(mn, u(i, j, 0));
  }
  EXPECT_LT(std::fabs(sum(U) - mass0), 1e-9) << "avance conserve la masse";
  EXPECT_TRUE(mn > 0.0 && std::isfinite(mn)) << "etat physique apres avance";
}

// (5) REFUS : un transport isotherme sans capability HLLC est refuse before template selection.
TEST(test_block_builder, isothermal_model_without_hllc_capability_is_rejected) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  const GridContext ctx{dom, bc, geom, &aux};

  using IsoModel = CompositeModel<IsothermalFlux, NoSource, BackgroundDensity>;
  IsoModel iso{IsothermalFlux{0.5}, NoSource{}, BackgroundDensity{0.0, 0.0}};
  MultiFab Us(ba, dm, 1, 2), Rs(ba, dm, 1, 0);
  Us.set_val(1.0);
  auto refused_with = [&](const char* riem, const char* frag) {
    try {
      make_block(iso, "minmod", riem, ctx, false);
      return false;
    } catch (const std::runtime_error& e) {
      return std::string(e.what()).find(frag) != std::string::npos;
    }
  };
  EXPECT_TRUE(refused_with("hllc", "capability"))
      << "isotherme + hllc refuse (nomme la capability)";
}

TEST(test_block_builder, cell_primitive_conversion_consumes_prepared_recovery_outcome) {
  const Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};
  const auto conversion = make_cell_convert(model);

  // rho=1, (u,v)=(0.2,-0.1), p=1 -> E=p/(gamma-1)+rho*(u^2+v^2)/2=2.525.
  const std::array<double, 4> conservative{1.0, 0.2, -0.1, 2.525};
  const std::array<double, 4> authored_primitive{1.0, 0.2, -0.1, 1.0};
  std::array<double, 4> forward_candidate{-9.0, -9.0, -9.0, -9.0};
  EXPECT_NO_THROW(conversion.first(authored_primitive.data(), forward_candidate.data()));
  for (std::size_t component = 0; component < conservative.size(); ++component)
    EXPECT_NEAR(forward_candidate[component], conservative[component], 1e-14);

  // Forward conversion is also a candidate transaction: the conservative result must survive the
  // same prepared inverse authority before any output component is published.
  const std::array<double, 4> invalid_primitive{0.0, 0.0, 0.0, 1.0};
  const std::array<double, 4> forward_sentinel{1.25, -2.5, 3.75, -5.0};
  forward_candidate = forward_sentinel;
  EXPECT_THROW(conversion.first(invalid_primitive.data(), forward_candidate.data()),
               std::runtime_error);
  EXPECT_EQ(forward_candidate, forward_sentinel);

  std::array<double, 4> primitive{-9.0, -9.0, -9.0, -9.0};
  const RecoveryReport success = conversion.second(conservative.data(), primitive.data());
  EXPECT_TRUE(success.recovered());
  EXPECT_EQ(success.status, RecoveryStatus::kRecovered);
  EXPECT_EQ(success.cause, RecoveryCause::kNone);
  EXPECT_EQ(success.attempted_methods, 1);
  EXPECT_EQ(success.selected_method, 0);
  EXPECT_EQ(success.selected_method_kind, RecoveryMethodKind::kClosedForm);
  EXPECT_EQ(success.last_method_kind, RecoveryMethodKind::kClosedForm);
  EXPECT_DOUBLE_EQ(primitive[0], 1.0);
  EXPECT_DOUBLE_EQ(primitive[1], 0.2);
  EXPECT_DOUBLE_EQ(primitive[2], -0.1);
  EXPECT_NEAR(primitive[3], 1.0, 1e-14);

  // The Euler closed form produces non-finite velocity/pressure for rho=0. The common prepared
  // authority rejects that candidate and the type-erased closure must leave output byte-exact.
  const std::array<double, 4> invalid_conservative{0.0, 0.0, 0.0, 0.0};
  const std::array<double, 4> sentinel{1.25, -2.5, 3.75, -5.0};
  primitive = sentinel;
  const RecoveryReport failure = conversion.second(invalid_conservative.data(), primitive.data());
  EXPECT_FALSE(failure.publication_permitted());
  EXPECT_EQ(failure.status, RecoveryStatus::kInvalidContract);
  EXPECT_EQ(failure.cause, RecoveryCause::kNonFiniteCandidate);
  EXPECT_EQ(failure.attempted_methods, 1);
  EXPECT_EQ(failure.selected_method_kind, RecoveryMethodKind::kUnknown);
  EXPECT_EQ(failure.last_method_kind, RecoveryMethodKind::kClosedForm);
  EXPECT_GE(failure.failing_component, 1);
  EXPECT_EQ(primitive, sentinel);
}

TEST(test_block_builder, primitive_to_conservative_publication_roundtrips_before_commit) {
  const Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};
  const auto conversion = make_cell_convert(model);

  const std::array<double, 4> authored_primitive{1.0, 0.2, -0.1, 1.0};
  const std::array<double, 4> expected_conservative{1.0, 0.2, -0.1, 2.525};
  std::array<double, 4> published{-9.0, -9.0, -9.0, -9.0};
  EXPECT_NO_THROW(conversion.first(authored_primitive.data(), published.data()));
  for (std::size_t component = 0; component < published.size(); ++component)
    EXPECT_NEAR(published[component], expected_conservative[component], 1e-14);
}
