/// @file
/// @brief Distributed two-box Cartesian cut-cell residual and uniform-ratio fraction transfer.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/numerics/spatial/embedded_boundary/characteristic.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> filled_real(double value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = static_cast<pops::Real>(value);
  return result;
}

template <int Dim>
pops::RealVector<Dim> unit_velocity() {
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  return velocity;
}

template <int Dim>
struct SumVolumeWeightedResidual {
  pops::FieldView<const pops::Real, Dim> residual{};
  pops::FieldView<const pops::Real, Dim> active{};
  pops::FieldView<const pops::Real, Dim> inverse_volume{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    if (active(cell) < pops::Real(0.5))
      return pops::Real(0);
    return residual(cell) / inverse_volume(cell);
  }
};

template <int Dim>
struct SampleFinePhi {
  pops::FieldView<pops::Real, Dim> phi{};

  POPS_HD void operator()(const pops::Index<Dim>& index) const {
    phi(index) = (static_cast<pops::Real>(index[0]) + pops::Real(0.5)) / pops::Real(4) -
                 pops::Real(0.5);
  }
};

void prove_uniform_ratio_cut_cell_transfer() {
  const pops::Box<2> fine_domain = pops::Box<2>::from_extents(filled_extent<2>(4));
  const pops::Box<2> coarse_domain = pops::Box<2>::from_extents(filled_extent<2>(2));
  pops::Fab<2> fine_phi(fine_domain, 1, filled_extent<2>(1));
  pops::Fab<2> coarse_volume(coarse_domain, 1);
  pops::Fab<2> fine_volume(fine_domain, 1);
  pops::Fab<2> aperture_residual(coarse_domain, 1);
  pops::for_each_cell(fine_phi.grown_box(), SampleFinePhi<2>{fine_phi.view()});
  pops::generated_amr_detail::apply_generated_amr_cut_cell_fraction_transfer(
      std::as_const(fine_phi).view(), coarse_volume.view(), fine_volume.view(),
      aperture_residual.view(), coarse_domain, pops::amr::RefinementRatio<2>(2, 2));
  pops::sync_host();
  pops::nd::CutCellFractions<2> children[4]{};
  for (int child = 0; child < 4; ++child) {
    pops::Index<2> fine{};
    fine[0] = child % 2;
    fine[1] = child / 2;
    children[child] =
        pops::nd::cut_cell_fractions_from_phi_cell(std::as_const(fine_phi).view(), fine);
  }
  const auto restricted = pops::nd::restrict_cut_cell_fractions(children, 4);
  EXPECT_DOUBLE_EQ(std::as_const(coarse_volume).view()(pops::Index<2>{}),
                   restricted.volume_fraction);
}

int run_mpi_cutcell_multibox(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      auto lane = pops::ExecutionLane::duplicate_world_collectively("test/mpi-cutcell-multibox");
      const int ranks = lane.size();
      if (ranks != 1 && ranks != 2)
        throw std::runtime_error("test_mpi_cutcell_multibox requires 1 or 2 MPI ranks");

      prove_uniform_ratio_cut_cell_transfer();

      pops::Extent<kDim> cells{};
      for (int axis = 0; axis < kDim; ++axis)
        cells[axis] = axis == 0 ? 8 : 4;
      const auto domain = pops::Box<kDim>::from_extents(cells);
      const auto geometry =
          pops::Geometry<kDim>::from_bounds(domain, filled_real<kDim>(0.0), filled_real<kDim>(1.0));
      pops::Box<kDim> left = domain;
      pops::Box<kDim> right = domain;
      left.hi[0] = 3;
      right.lo[0] = 4;
      const pops::mesh::BoxArray<kDim> layout{std::vector<pops::Box<kDim>>{left, right}};
      pops::Extent<kDim> rank_extent = filled_extent<kDim>(1);
      rank_extent[0] = ranks;
      const pops::mesh::RankSpace<kDim> rank_space{pops::Index<kDim>{}, rank_extent};
      std::vector<pops::Index<kDim>> owners{rank_space.coordinate(0),
                                            rank_space.coordinate(ranks == 1 ? 0 : 1)};
      const auto distribution =
          pops::mesh::Distribution<kDim>::partitioned(layout, rank_space, owners);
      const pops::Index<kDim> local_rank = rank_space.coordinate(static_cast<std::size_t>(lane.rank()));
      const pops::MultiFab<kDim> prototype{layout, distribution, local_rank, 1,
                                           filled_extent<kDim>(1)};
      const auto prepared =
          pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
              {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, geometry,
              pops::BoundaryTopology<kDim>::physical(), prototype,
              pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{},
              7, lane);

      pops::runtime::system::require_prepared_eb_active_mask_matches_phi(*prepared, lane);

      pops::RealVector<kDim> lengths{};
      for (int axis = 0; axis < kDim; ++axis)
        lengths[axis] = geometry.spacing(axis);
      const auto metric = pops::prepare_metric_provider(
          domain, pops::CartesianCoordinateMap<kDim>::make(pops::RealVector<kDim>{}, lengths));
      const auto model = pops::nd::ScalarAdvection<kDim>::prepare(unit_velocity<kDim>());
      const auto op = pops::nd::prepare_embedded_boundary_operator(model, metric);

      pops::MultiFab<kDim> state{layout, distribution, local_rank, 1, filled_extent<kDim>(1)};
      pops::MultiFab<kDim> residual{layout, distribution, local_rank, 1, pops::Extent<kDim>{}};
      state.set_val(pops::Real(1));
      residual.set_val(pops::Real(0));

      pops::nd::BoundaryFaceOmission<kDim> omission{};
      omission.domain = domain;
      for (int axis = 0; axis < kDim; ++axis) {
        omission.lower[axis] = true;
        omission.upper[axis] = true;
      }

      bool refused_without_authority = false;
      try {
        op.assemble_residual(state, prepared->active_mask(), prepared->inverse_volume_fraction(),
                             prepared->face_aperture_lower(), prepared->face_aperture_upper(),
                             residual, omission);
      } catch (const std::invalid_argument& error) {
        refused_without_authority =
            std::string(error.what()).find("halo authority") != std::string::npos;
      }
      if (ranks == 2) {
        EXPECT_EQ(pops::all_reduce_sum(refused_without_authority ? 1L : 0L),
                  static_cast<long>(ranks));
      } else {
        EXPECT_FALSE(refused_without_authority);
      }

      pops::runtime::system::fill_prepared_eb_transport_state_ghosts(state, *prepared, lane);
      pops::runtime::system::require_prepared_eb_active_mask_matches_phi(*prepared, lane);
      op.assemble_residual(state, prepared->active_mask(), prepared->inverse_volume_fraction(),
                           prepared->face_aperture_lower(), prepared->face_aperture_upper(),
                           residual, omission, pops::nd::PreparedEbPartitionHalo{&lane});

      pops::Real local_balance = pops::Real(0);
      for (std::size_t local = 0; local < residual.local_size(); ++local) {
        local_balance += pops::for_each_cell_reduce_sum(
            residual.box(local),
            SumVolumeWeightedResidual<kDim>{
                std::as_const(residual.fab(local)).view(),
                prepared->active_mask().fab(local).view(),
                prepared->inverse_volume_fraction().fab(local).view()});
      }
      const double global_balance = pops::all_reduce_sum(static_cast<double>(local_balance));
      EXPECT_NEAR(global_balance, 0.0, 2e-10);

      pops::RealVector<kDim> char_velocity{};
      char_velocity[0] = pops::Real(1);
      if constexpr (kDim >= 2)
        char_velocity[1] = pops::Real(-2);
      const auto char_model = pops::nd::ScalarAdvection<kDim>::prepare(char_velocity);
      typename std::remove_cvref_t<decltype(char_model)>::State reference{};
      pops::MultiFab<kDim> exterior{layout, distribution, local_rank, 1, pops::Extent<kDim>{}};
      pops::fill_prepared_embedded_characteristic_no_inflow(
          char_model, state, prepared->phi(), prepared->active_mask(), reference, exterior);

      pops::sync_host();
      int local_cut_checks = 0;
      for (std::size_t local = 0; local < prepared->phi().local_size(); ++local) {
        const auto box = prepared->phi().box(local);
        const auto phi = prepared->phi().fab(local).view();
        const auto mask = prepared->active_mask().fab(local).view();
        const auto ghost = std::as_const(exterior.fab(local)).view();
        for (int i0 = box.lo[0]; i0 <= box.hi[0]; ++i0) {
          const int i1_lo = kDim > 1 ? box.lo[1] : 0;
          const int i1_hi = kDim > 1 ? box.hi[1] : 0;
          for (int i1 = i1_lo; i1 <= i1_hi; ++i1) {
            const int i2_lo = kDim > 2 ? box.lo[2] : 0;
            const int i2_hi = kDim > 2 ? box.hi[2] : 0;
            for (int i2 = i2_lo; i2 <= i2_hi; ++i2) {
              pops::Index<kDim> cell{};
              cell[0] = i0;
              if constexpr (kDim > 1)
                cell[1] = i1;
              if constexpr (kDim > 2)
                cell[2] = i2;
              if (mask(cell) < pops::Real(0.5))
                continue;
              const auto fractions = pops::nd::cut_cell_fractions_from_phi_cell(phi, cell);
              pops::RealVector<kDim> normal{};
              if (!pops::nd::cut_cell_interface_normal(fractions, normal))
                continue;
              typename std::remove_cvref_t<decltype(char_model)>::State interior{};
              interior[0] = pops::Real(1);
              typename std::remove_cvref_t<decltype(char_model)>::State expected{};
              EXPECT_TRUE(pops::nd::apply_characteristic_no_inflow_on_normal<kDim>(
                  char_model, interior, reference, normal, expected));
              EXPECT_NEAR(ghost(cell), expected[0], 1e-12);
              ++local_cut_checks;
            }
          }
        }
      }
      EXPECT_GT(pops::all_reduce_sum(static_cast<long>(local_cut_checks)), 0L);

      for (std::size_t local = 0; local < prepared->active_mask().local_size(); ++local) {
        const auto mask = prepared->active_mask().fab(local).view();
        const auto phi = prepared->phi().fab(local).view();
        const auto grown = prepared->active_mask().fab(local).grown_box();
        for (int interface_i : {3, 4}) {
          pops::Index<kDim> interface{};
          interface[0] = interface_i;
          if (!grown.contains(interface))
            continue;
          const pops::Real expected =
              phi(interface) < pops::Real(0) ? pops::Real(1) : pops::Real(0);
          EXPECT_DOUBLE_EQ(mask(interface), expected);
        }
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d test_mpi_cutcell_multibox failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_cutcell_multibox np=%d dim=%d\n", pops::n_ranks(), kDim);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_cutcell_multibox, DistributedZeroFluxAndFractionTransfer) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_cutcell_multibox, "test_mpi_cutcell_multibox"), 0);
}
