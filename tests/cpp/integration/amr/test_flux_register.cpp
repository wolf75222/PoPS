// Contrat du type FluxRegister (revue, point 2 : role promu en type). Verifie l'indexation
// sur une region a origine, la semantique set (ecrasement) vs add (accumulation bornee), la
// lecture at, et que add hors region est un no-op. Le gather distribue est couvert par les
// tests MPI de reflux (test_mpi_amr_multipatch / _multipatch3, np=1/2/4 bit-identiques) ;
// ici on fige les mecaniques locales du registre, independamment de l'integration AMR.

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::FluxRegister
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <vector>

using namespace pops;

namespace {
struct FillCompactRegisterOnDevice {
  FluxRegisterView output;
  POPS_HD void operator()(int i, int j) const {
    output.set(i, j, 0, static_cast<Real>(100 * j + i));
  }
};

struct FillFineFromParentIndex {
  Array4 values;
  Real offset = Real(0);
  POPS_HD void operator()(int i, int j) const {
    const int I = floor_div(i, kAmrRefRatio);
    const int J = floor_div(j, kAmrRefRatio);
    values(i, j, 0) = static_cast<Real>(10 * I + J) + offset;
  }
};
}  // namespace

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

TEST(test_flux_register, device_view_preserves_negative_origin_and_compact_holes) {
  const std::vector<Box2D> regions{Box2D{{-4, -3}, {-3, 0}}, Box2D{{-1, -3}, {0, 0}}};
  FluxRegister correction(regions, 1);
  for_each_cell(Box2D{{-4, -3}, {0, 0}}, FillCompactRegisterOnDevice{correction.view()});
  correction.gather();  // also orders the device writes before MPI/host access

  EXPECT_TRUE(correction.in(-4, -3));
  EXPECT_TRUE(correction.in(0, 0));
  EXPECT_FALSE(correction.in(-2, -1)) << "the gap between compact regions remains a hole";
  EXPECT_EQ(correction.at(-4, -3, 0), Real(-304));
  EXPECT_EQ(correction.at(0, 0, 0), Real(0));

  const Box2D coarse_box{{-4, -3}, {0, 0}};
  MultiFab coarse(BoxArray(std::vector<Box2D>{coarse_box}), DistributionMapping(1, 1), 1, 0);
  coarse.set_val(Real(7));
  for_each_cell(coarse_box, detail::ApplyRefluxRegisterKernel{
                                coarse.fab(0).array(), correction.view(), 1});
  device_fence();
  EXPECT_EQ(coarse.fab(0)(-4, -3, 0), Real(7 - 304));
  EXPECT_EQ(coarse.fab(0)(-2, -1, 0), Real(7)) << "a multibox hole is never overwritten";
  EXPECT_EQ(coarse.fab(0)(-1, 0, 0), Real(6));
}

TEST(test_flux_register, sparse_lookup_storage_is_independent_of_bounding_box_span) {
  const std::vector<Box2D> regions{Box2D{{-1000000000, -7}, {-999999999, -7}},
                                      Box2D{{999999999, 11}, {1000000000, 11}}};
  FluxRegister correction(regions, 2);

  EXPECT_EQ(correction.covered_cell_count(), 4u);
  EXPECT_LE(correction.lookup_capacity(), 8u);
  EXPECT_TRUE(correction.in(-1000000000, -7));
  EXPECT_TRUE(correction.in(1000000000, 11));
  EXPECT_FALSE(correction.in(0, 0));

  correction.set(-1000000000, -7, 1, Real(3));
  correction.add(1000000000, 11, 0, Real(5));
  EXPECT_EQ(correction.at(-1000000000, -7, 1), Real(3));
  EXPECT_EQ(correction.at(1000000000, 11, 0), Real(5));
}

TEST(test_flux_register, multifab_box_lookup_is_sparse_across_far_apart_patches) {
  const std::vector<Box2D> boxes{Box2D{{-1000000000, -2}, {-999999999, -2}},
                                 Box2D{{999999999, 3}, {1000000000, 3}}};
  MultiFab values(BoxArray(boxes), DistributionMapping(2, 1), 1, 0);
  MfBoxLookup lookup(values);

  EXPECT_LE(lookup.lookup_capacity(), 8u);
  EXPECT_EQ(lookup.find(-1000000000, -2), 0);
  EXPECT_EQ(lookup.find(1000000000, 3), 1);
  EXPECT_EQ(lookup.find(0, 0), -1);

  std::size_t local = std::numeric_limits<std::size_t>::max();
  EXPECT_TRUE(lookup.view().locate(999999999, 3, local));
  EXPECT_EQ(local, 1u);
}

TEST(test_flux_register, average_down_device_path_preserves_multibox_holes) {
  const std::vector<Box2D> fine_boxes{Box2D{{-8, -4}, {-5, -1}},
                                      Box2D{{-2, -4}, {1, -1}}};
  MultiFab fine(BoxArray(fine_boxes), DistributionMapping(2, 1), 1, 0);
  for (int local = 0; local < fine.local_size(); ++local)
    for_each_cell(fine.box(local), FillFineFromParentIndex{fine.fab(local).array()});

  const Box2D coarse_box{{-4, -2}, {0, -1}};
  MultiFab coarse(BoxArray(std::vector<Box2D>{coarse_box}), DistributionMapping(1, 1), 1, 0);
  coarse.set_val(Real(99));
  auto workspace = PreparedAverageDownWorkspace::prepare(fine, coarse, 17);
  workspace.apply(fine, coarse, 17, world_communicator_view());

  EXPECT_EQ(coarse.fab(0)(-4, -2, 0), Real(-42));
  EXPECT_EQ(coarse.fab(0)(-3, -1, 0), Real(-31));
  EXPECT_EQ(coarse.fab(0)(-2, -2, 0), Real(99)) << "unrefined parent hole is untouched";
  EXPECT_EQ(coarse.fab(0)(-1, -2, 0), Real(-12));
  EXPECT_EQ(coarse.fab(0)(0, -1, 0), Real(-1));

  // Stable replay reuses the same register/coverage storage and refreshes only values.
  for (int local = 0; local < fine.local_size(); ++local)
    for_each_cell(fine.box(local),
                  FillFineFromParentIndex{fine.fab(local).array(), Real(100)});
  coarse.set_val(Real(99));
  workspace.apply(fine, coarse, 17, world_communicator_view());
  EXPECT_EQ(coarse.fab(0)(-4, -2, 0), Real(58));
  EXPECT_EQ(coarse.fab(0)(-2, -2, 0), Real(99));
  EXPECT_THROW(workspace.apply(fine, coarse, 18, world_communicator_view()),
               std::invalid_argument);
}
