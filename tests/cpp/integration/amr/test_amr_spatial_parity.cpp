// Parite spatiale AmrSystem <-> System : le coeur spatial du chemin AMR
// (compute_face_fluxes<Limiter, Flux> + recon_prim, consomme par le reflux du
// ProgramGraph AMR) reproduit EXACTEMENT le residu d'assemble_rhs<Limiter, Flux> du
// chemin System, sous reconstruction PRIMITIVE et flux HLLC puis Roe. C'est la
// preuve que la facade raffinee accepte les memes parametres de schema que System
// et les applique a chaque niveau/patch.
//
// Le coeur spatial (bit-identique) prouve que div(compute_face_fluxes) == assemble_rhs pour HLLC et
// Roe en reconstruction primitive (minmod, ordre 2), et que le primitif differe du conservatif.
// L'integration temporelle et la conservation AMR sont couvertes par les tests ProgramGraph/reflux.

#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, CompressibleFlux, NoSource, ChargeDensity
#include <pops/numerics/fv/numerical_flux.hpp>  // HLLCFlux, RoeFlux
#include <pops/numerics/fv/reconstruction.hpp>  // Minmod
#include <pops/numerics/spatial_operator.hpp>   // assemble_rhs, compute_face_fluxes, load_state

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>  // sum
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

TEST(test_amr_spatial_parity, Runs) {
  const int n = 32;
  const double L = 1.0;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, L, 0.0, L};
  const double dx = geom.dx(), dy = geom.dy();
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const BCRec bc;  // periodique

  // Euler compressible pur (sans source) : le residu vient entierement du flux
  // hyperbolique reconstruit -> isole la parite reconstruction x flux Riemann.
  using Model = CompositeModel<CompressibleFlux, NoSource, ChargeDensity>;
  const Model model{CompressibleFlux{1.4}, NoSource{}, ChargeDensity{1.0}};

  MultiFab U0(ba, dm, 4, 2), aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  {  // bulle de densite + champ de vitesse doux (etat lisse, positif)
    Array4 a = U0.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      const double x = geom.x_cell(i) - 0.5, y = geom.y_cell(j) - 0.5;
      const double rho = 1.0 + 0.4 * std::exp(-(x * x + y * y) / 0.02);
      a(i, j, 0) = rho;
      a(i, j, 1) = 0.2 * rho * std::sin(2 * kPi * geom.x_cell(i));
      a(i, j, 2) = 0.1 * rho * std::cos(2 * kPi * geom.y_cell(j));
      const double ke = 0.5 * (a(i, j, 1) * a(i, j, 1) + a(i, j, 2) * a(i, j, 2)) / rho;
      a(i, j, 3) = 1.0 / (1.4 - 1.0) + ke;
    });
  }

  // residu d'assemble_rhs (chemin System) sur les cellules valides.
  auto rhs_system = [&](auto flux_tag, bool prim, MultiFab& R) {
    using Flux = decltype(flux_tag);
    MultiFab U = U0;
    fill_ghosts(U, dom, bc);
    assemble_rhs<Minmod, Flux>(model, U, aux, geom, R, prim);
  };
  // residu reconstitue du chemin AMR : -div des flux de face de compute_face_fluxes
  // (NoSource -> pas de terme S). Memes (Limiter, Flux, recon_prim) que System.
  auto rhs_amr = [&](auto flux_tag, bool prim, MultiFab& R) {
    using Flux = decltype(flux_tag);
    MultiFab U = U0;
    fill_ghosts(U, dom, bc);
    MultiFab Fx(BoxArray(std::vector<Box2D>{xface_box(dom)}), dm, 4, 0);
    MultiFab Fy(BoxArray(std::vector<Box2D>{yface_box(dom)}), dm, 4, 0);
    compute_face_fluxes<Minmod, Flux>(model, U, aux, Fx, Fy, dx, dy, prim);
    Array4 r = R.fab(0).array();
    const ConstArray4 fx = Fx.fab(0).const_array(), fy = Fy.fab(0).const_array();
    for_each_cell(dom, [=] POPS_HD(int i, int j) {
      for (int c = 0; c < 4; ++c)
        r(i, j, c) = -((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy);
    });
  };

  auto maxdiff = [&](const MultiFab& A, const MultiFab& B) {
    double d = 0;
    const ConstArray4 a = A.fab(0).const_array(), b = B.fab(0).const_array();
    for (int c = 0; c < 4; ++c)
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          d = std::fmax(d, std::fabs(a(i, j, c) - b(i, j, c)));
    return d;
  };

  // Coeur spatial AMR == coeur spatial System (bit-identique).
  {
    MultiFab Rs(ba, dm, 4, 0), Ra(ba, dm, 4, 0), Rc(ba, dm, 4, 0);

    // HLLC + primitif : div(face fluxes) == assemble_rhs, exactement.
    rhs_system(HLLCFlux{}, true, Rs);
    rhs_amr(HLLCFlux{}, true, Ra);
    EXPECT_TRUE(maxdiff(Rs, Ra) < 1e-13)
        << "HLLC+primitif : div(compute_face_fluxes) == assemble_rhs";
    EXPECT_TRUE(norm_inf(Rs) > 1e-6) << "HLLC+primitif : residu non trivial";

    // le primitif change le residu vs le conservatif (la reconstruction joue).
    rhs_system(HLLCFlux{}, false, Rc);
    EXPECT_TRUE(maxdiff(Rs, Rc) > 1e-9) << "HLLC : recon primitif != conservatif";

    // Roe + primitif : meme parite bit-identique.
    rhs_system(RoeFlux{}, true, Rs);
    rhs_amr(RoeFlux{}, true, Ra);
    EXPECT_TRUE(maxdiff(Rs, Ra) < 1e-13)
        << "Roe+primitif : div(compute_face_fluxes) == assemble_rhs";
    EXPECT_TRUE(norm_inf(Rs) > 1e-6) << "Roe+primitif : residu non trivial";
  }
}
