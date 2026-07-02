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

#include <cmath>
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
  BlockClosures clo = make_block(model, "minmod", "hllc", ctx, /*imex=*/false, /*recon_prim=*/true);
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
  for (int s = 0; s < 10; ++s)
    clo.advance(U, 2e-3, 1);
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

// (5) ADC-590 -- BIT-IDENTITY du flux : sur le vrai Euler, le chemin GENERIQUE (HLLCFlux / RoeFlux
// via HasHLLCStructure / HasRoeDissipation, cabl par les nouvelles capabilites de la brique) donne
// EXACTEMENT le meme flux que le chemin EXPLICITE (EulerHLLCFlux2D / EulerRoeFlux2D). C'est la preuve
// au niveau flux que la conversion de la brique Euler en modele a-capabilites ne bouge aucun bit.
TEST(test_block_builder, generic_flux_bit_identical_to_explicit_on_composite_euler) {
  Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};
  static_assert(HasHLLCStructure<Model>,
                "CompositeModel<Euler,..> doit exposer HasHLLCStructure (ADC-590)");
  static_assert(HasRoeDissipation<Model>,
                "CompositeModel<Euler,..> doit exposer HasRoeDissipation (ADC-590)");
  HLLCFlux ghllc;
  EulerHLLCFlux2D ehllc;
  RoeFlux groe;
  EulerRoeFlux2D eroe;
  const Model::State UL{1.0, 0.1, 0.02, 2.5}, UR{1.3, 0.05, -0.01, 2.7};
  const Aux A{};
  double dh = 0, dr = 0;
  for (int d = 0; d < 2; ++d) {
    const auto fh_g = ghllc(model, UL, A, UR, A, d), fh_e = ehllc(model, UL, A, UR, A, d);
    const auto fr_g = groe(model, UL, A, UR, A, d), fr_e = eroe(model, UL, A, UR, A, d);
    for (int c = 0; c < 4; ++c) {
      dh = std::fmax(dh, std::fabs(fh_g[c] - fh_e[c]));
      dr = std::fmax(dr, std::fabs(fr_g[c] - fr_e[c]));
    }
  }
  EXPECT_EQ(dh, 0.0) << "HLLCFlux generique == EulerHLLCFlux2D explicite (bit-identique, ADC-590)";
  EXPECT_EQ(dr, 0.0) << "RoeFlux generique == EulerRoeFlux2D explicite (bit-identique, ADC-590)";
}

// (6) route EXPLICITE euler_hllc / euler_roe : make_block construit avec EulerHLLCFlux2D /
// EulerRoeFlux2D et, sur le vrai Euler, le residu == celui du chemin generique hllc / roe.
TEST(test_block_builder, explicit_euler_hllc_route_matches_generic_hllc) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;
  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  {
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
  Model model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}};
  const GridContext ctx{dom, bc, geom, &aux};

  BlockClosures ceh = make_block(model, "minmod", "euler_hllc", ctx, false, true);
  BlockClosures ch = make_block(model, "minmod", "hllc", ctx, false, true);
  MultiFab Reh(ba, dm, 4, 0), Rh(ba, dm, 4, 0);
  ceh.rhs_into(U, Reh);
  ch.rhs_into(U, Rh);
  double de = 0;
  for (int c = 0; c < 4; ++c) {
    const ConstArray4 a = Reh.fab(0).const_array(), b = Rh.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        de = std::fmax(de, std::fabs(a(i, j, c) - b(i, j, c)));
  }
  EXPECT_EQ(de, 0.0) << "make_block euler_hllc == hllc sur le vrai Euler (bit-identique)";
}

// (7) REFUS : un transport isotherme 3-var (sans pression, sans capability HLLC) est REFUSE par
// hllc ET par euler_hllc, avec un message qui NOMME les capabilites / la couche Euler canonique.
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
      make_block(iso, "minmod", riem, ctx, false, false);
      return false;
    } catch (const std::runtime_error& e) {
      return std::string(e.what()).find(frag) != std::string::npos;
    }
  };
  EXPECT_TRUE(refused_with("hllc", "capability"))
      << "isotherme + hllc refuse (nomme la capability, ADC-590)";
  EXPECT_TRUE(refused_with("euler_hllc", "Euler 2D"))
      << "isotherme + euler_hllc refuse (nomme la couche Euler canonique)";
}
