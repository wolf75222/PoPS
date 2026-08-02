// Chantier "Aux extensible", increment 2 : PEUPLEMENT d'un champ aux supplementaire cote
// coupleur. Le Coupler mono-bloc alloue le canal aux a la largeur DU MODELE (aux_comps) et
// remplit la composante B_z depuis une fonction B_z(x, y) fournie par l'utilisateur. On
// exerce le chemin SPATIAL Poisson -> aux -> B_z -> residual :
//   modele jouet n_aux=4, flux nul, elliptic_rhs nul (phi = 0), source S = B_z * u.
//   -> R(U) = B_z U ; B_z constant c et U=1 donnent R=c.

#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/coupling/single/coupler.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>

using namespace pops;

// Modele jouet : croissance pilotee par B_z. flux nul, pas de couplage elliptique (phi = 0).
struct BzGrow {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrow>, "BzGrow modele PhysicalModel");
static_assert(aux_comps<BzGrow>() == 4, "BzGrow declare n_aux = 4");

TEST(AuxCouplerBz, SpatialResidualConsumesPreparedBz) {
  const int n = 16;
  const Real L = 1.0, c = 0.5;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique (transport et Poisson)

  BzGrow model;
  // active vide ; B_z = constante c.
  Coupler<BzGrow> cpl(model, geom, ba, bc, bc, {}, constant_scalar_field_provider(c));

  MultiFab U(ba, dm, 1, 1);
  U.set_val(1.0);

  // --- (A) le canal aux est alloue a la largeur du modele (4) et B_z est peuple ---
  ASSERT_TRUE(cpl.aux().fab(0).ncomp() == 4) << "aux_width_is_model_width";
  cpl.solve_fields(U);  // Poisson (f=0 -> phi=0) puis derive aux
  {
    double maxphi = 0, maxbz = 0;
    const MultiFab& A = cpl.aux();
    for (int li = 0; li < A.local_size(); ++li) {
      const ConstArray4 a = A.fab(li).const_array();
      const Box2D v = A.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxphi = std::max(maxphi, std::fabs(a(i, j, 0)));    // phi = 0
          maxbz = std::max(maxbz, std::fabs(a(i, j, 3) - c));  // B_z = c
        }
    }
    EXPECT_TRUE(maxphi < 1e-12) << "phi_is_zero (max|phi|=" << maxphi << ")";
    EXPECT_TRUE(maxbz < 1e-14) << "Bz_populated (max|B_z - c|=" << maxbz << ")";
  }

  // --- (B) le residu spatial consomme le B_z prepare, sans choisir un schema temporel ---
  MultiFab residual(ba, dm, 1, 0);
  cpl.assemble_residual(U, residual);
  {
    double maxerr = 0, val = 0;
    for (int li = 0; li < residual.local_size(); ++li) {
      const ConstArray4 r = residual.fab(li).const_array();
      const Box2D v = residual.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxerr = std::max(maxerr, std::fabs(r(i, j, 0) - c));
          val = r(i, j, 0);
        }
    }
    EXPECT_TRUE(maxerr < 1e-12) << "Bz_drives_spatial_residual (R=" << val << " attendu=" << c
                                << " err=" << maxerr << ")";
  }
}
