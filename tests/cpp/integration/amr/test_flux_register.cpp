// Contrat du type FluxRegister (revue, point 2 : role promu en type). Verifie l'indexation
// sur une region a origine, la semantique set (ecrasement) vs add (accumulation bornee), la
// lecture at, et que add hors region est un no-op. Le gather distribue est couvert par les
// tests MPI de reflux (test_mpi_amr_multipatch / _multipatch3, np=1/2/4 bit-identiques) ;
// ici on fige les mecaniques locales du registre, independamment de l'integration AMR.

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::FluxRegister
#include <pops/mesh/index/box2d.hpp>

using namespace pops;

TEST(test_flux_register, Runs) {
  // region a origine non nulle : [10,20] x [5,8], 3 composantes.
  const Box2D region{{10, 5}, {20, 8}};
  FluxRegister fr(region, 3);
  EXPECT_TRUE(fr.NX == 11 && fr.NY == 4 && fr.nc == 3) << "dims";
  EXPECT_TRUE(fr.in(10, 5) && fr.in(20, 8) && !fr.in(9, 5) && !fr.in(21, 8) && !fr.in(15, 4))
      << "in_bounds";

  // index a origine : (I0,J0,0) -> 0, monotone en I puis J puis k.
  EXPECT_EQ(fr.idx(10, 5, 0), 0) << "idx_origin";
  EXPECT_TRUE(fr.idx(11, 5, 0) == 3 && fr.idx(10, 6, 0) == 33 && fr.idx(10, 5, 2) == 2)
      << "idx_layout";

  // set = ecrasement ; at relit la valeur.
  fr.set(12, 7, 1, 4.0);
  EXPECT_EQ(fr.at(12, 7, 1), 4.0) << "set_at";
  fr.set(12, 7, 1, -2.0);
  EXPECT_EQ(fr.at(12, 7, 1), -2.0) << "set_overwrite";

  // add = accumulation ; deux contributions s'additionnent.
  fr.add(15, 6, 2, 1.5);
  fr.add(15, 6, 2, 2.5);
  EXPECT_EQ(fr.at(15, 6, 2), 4.0) << "add_accumulate";

  // add hors region : no-op (pas de crash, valeurs inchangees).
  const double before = fr.at(20, 8, 0);
  fr.add(100, 100, 0, 999.0);  // hors borne
  fr.add(-5, 5, 0, 999.0);     // hors borne
  EXPECT_EQ(fr.at(20, 8, 0), before) << "add_out_of_region_noop";

  // gather en serie = identite (all_reduce sur 1 rang).
  const double v = fr.at(12, 7, 1);
  fr.gather();
  EXPECT_EQ(fr.at(12, 7, 1), v) << "gather_serial_identity";
}
