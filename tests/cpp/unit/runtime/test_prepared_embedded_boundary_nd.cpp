/// @file
/// @brief Exact 1D/2D/3D and transactional proofs for prepared EB geometry.

#include <gtest/gtest.h>

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <cmath>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

template <int Dim>
struct SumField {
  pops::FieldView<const pops::Real, Dim> view;

  POPS_HD pops::Real operator()(const pops::Index<Dim>& index) const { return view(index); }
};

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
struct Fixture {
  pops::Box<Dim> domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  pops::Geometry<Dim> geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  pops::mesh::Distribution<Dim> distribution =
      pops::mesh::Distribution<Dim>::replicated(layout, rank_space);
  pops::MultiFab<Dim> prototype{layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{}};
  pops::ExecutionLane lane = pops::ExecutionLane::world("test/prepared-eb");
};

template <int Dim>
void prove_ranked_cut_geometry() {
  pops::RealVector<Dim> lower_samples{};
  pops::RealVector<Dim> upper_samples{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower_samples[axis] = pops::Real(-0.3);
    upper_samples[axis] = pops::Real(-0.3);
    spacing[axis] = pops::Real(0.2 * (axis + 1));
  }
  for (int cut_axis = 0; cut_axis < Dim; ++cut_axis) {
    auto axis_samples = upper_samples;
    axis_samples[cut_axis] = pops::Real(0.1);
    const auto cut = pops::nd::cut_cell_fractions_from_samples<Dim>(pops::Real(-0.1), lower_samples,
                                                                    axis_samples);
    static_assert(decltype(cut)::dimension == Dim);
    for (int axis = 0; axis < Dim; ++axis) {
      EXPECT_DOUBLE_EQ(cut.lower[axis], pops::Real(1));
      EXPECT_DOUBLE_EQ(cut.upper[axis], axis == cut_axis ? pops::Real(0.5) : pops::Real(1));
    }
    if constexpr (Dim < 3)
      EXPECT_DOUBLE_EQ(cut.volume_fraction, pops::Real(0.75));
    else {
      EXPECT_GT(cut.volume_fraction, pops::Real(0));
      EXPECT_LE(cut.volume_fraction, pops::Real(1));
      EXPECT_DOUBLE_EQ(cut.volume_fraction, pops::Real(1));
    }
    for (int axis = 0; axis < Dim; ++axis) {
      EXPECT_GE(cut.face_lower[axis], pops::Real(0));
      EXPECT_LE(cut.face_lower[axis], pops::Real(1));
      EXPECT_GE(cut.face_upper[axis], pops::Real(0));
      EXPECT_LE(cut.face_upper[axis], pops::Real(1));
      if constexpr (Dim < 3) {
        if (axis == cut_axis) {
          EXPECT_DOUBLE_EQ(cut.face_lower[axis], pops::Real(1));
          EXPECT_DOUBLE_EQ(cut.face_upper[axis], pops::Real(0));
        } else {
          EXPECT_DOUBLE_EQ(cut.face_lower[axis], pops::Real(1));
          EXPECT_DOUBLE_EQ(cut.face_upper[axis], pops::Real(1));
        }
      } else if (axis == cut_axis) {
        EXPECT_DOUBLE_EQ(cut.face_lower[axis], pops::Real(1));
      } else {
        EXPECT_DOUBLE_EQ(cut.face_lower[axis], pops::Real(1));
        EXPECT_DOUBLE_EQ(cut.face_upper[axis], pops::Real(1));
      }
    }

    const auto stencil = pops::nd::shortley_weller_stencil(cut, spacing);
    pops::Real expected_diagonal = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const pops::Real lower_distance = cut.lower[axis] * spacing[axis];
      const pops::Real upper_distance = cut.upper[axis] * spacing[axis];
      const pops::Real span = lower_distance + upper_distance;
      EXPECT_DOUBLE_EQ(stencil.lower[axis], pops::Real(2) / (lower_distance * span));
      EXPECT_DOUBLE_EQ(stencil.upper[axis], pops::Real(2) / (upper_distance * span));
      expected_diagonal += pops::Real(2) / (lower_distance * upper_distance);
    }
    EXPECT_DOUBLE_EQ(stencil.diagonal, expected_diagonal);
  }

  for (int cut_axis = 0; cut_axis < Dim; ++cut_axis) {
    auto grazing_samples = upper_samples;
    grazing_samples[cut_axis] = pops::Real(0.2);
    const auto grazing = pops::nd::cut_cell_fractions_from_samples<Dim>(
        pops::Real(-1.0e-8), lower_samples, grazing_samples);
    EXPECT_DOUBLE_EQ(grazing.upper[cut_axis], pops::kEbCutFractionFloor);
  }
}

TEST(PreparedEmbeddedBoundaryND, CutGeometryUsesOneAxisLoopInOneTwoAndThreeDimensions) {
  static_assert(pops::nd::CutCellFractions<1>::dimension == 1);
  static_assert(pops::nd::CutCellFractions<2>::dimension == 2);
  static_assert(pops::nd::CutCellFractions<3>::dimension == 3);
  prove_ranked_cut_geometry<1>();
  prove_ranked_cut_geometry<2>();
  prove_ranked_cut_geometry<3>();
}

TEST(PreparedEmbeddedBoundaryND, Dim3VolumeIsPlaneReconstructionNotAxisProduct) {
  pops::RealVector<3> lower_samples{-0.5, -0.5, -0.5};
  pops::RealVector<3> upper_samples{0.3, 0.3, 0.3};
  const auto cut = pops::nd::cut_cell_fractions_from_samples<3>(pops::Real(-0.1), lower_samples,
                                                                upper_samples);
  pops::Real product = pops::Real(1);
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_DOUBLE_EQ(cut.lower[axis], pops::Real(1));
    EXPECT_DOUBLE_EQ(cut.upper[axis], pops::Real(0.25));
    product *= pops::Real(0.5) * (cut.lower[axis] + cut.upper[axis]);
    EXPECT_GE(cut.face_lower[axis], pops::Real(0));
    EXPECT_LE(cut.face_lower[axis], pops::Real(1));
    EXPECT_GE(cut.face_upper[axis], pops::Real(0));
    EXPECT_LE(cut.face_upper[axis], pops::Real(1));
    EXPECT_GT(cut.face_lower[axis], cut.face_upper[axis]);
  }
  EXPECT_NEAR(product, pops::Real(0.244140625), 1e-15);
  EXPECT_GT(cut.volume_fraction, pops::Real(0));
  EXPECT_LE(cut.volume_fraction, pops::Real(1));
  EXPECT_NEAR(cut.volume_fraction, pops::Real(131) / pops::Real(192), 1e-12);
  EXPECT_GT(std::abs(cut.volume_fraction - product), pops::Real(0.1));
  EXPECT_NEAR(cut.face_lower[0], pops::Real(0.96875), 1e-12);
  EXPECT_NEAR(cut.face_upper[0], pops::Real(0.28125), 1e-12);
  EXPECT_NEAR(cut.face_lower[0], cut.face_lower[1], 1e-12);
  EXPECT_NEAR(cut.face_lower[0], cut.face_lower[2], 1e-12);
  EXPECT_NEAR(cut.face_upper[0], cut.face_upper[1], 1e-12);
  EXPECT_NEAR(cut.face_upper[0], cut.face_upper[2], 1e-12);
}

TEST(PreparedEmbeddedBoundaryND, Dim12VolumeStaysAxisProductBitCompatible) {
  pops::RealVector<1> lower1{-0.4};
  pops::RealVector<1> upper1{0.2};
  const auto cut1 =
      pops::nd::cut_cell_fractions_from_samples<1>(pops::Real(-0.1), lower1, upper1);
  EXPECT_DOUBLE_EQ(cut1.volume_fraction, pops::Real(0.5) * (cut1.lower[0] + cut1.upper[0]));
  EXPECT_DOUBLE_EQ(cut1.face_lower[0], pops::Real(1));
  EXPECT_DOUBLE_EQ(cut1.face_upper[0], pops::Real(0));

  pops::RealVector<2> lower2{-0.4, -0.4};
  pops::RealVector<2> upper2{0.2, -0.4};
  const auto cut2 =
      pops::nd::cut_cell_fractions_from_samples<2>(pops::Real(-0.1), lower2, upper2);
  EXPECT_DOUBLE_EQ(cut2.volume_fraction, pops::Real(0.5) * (cut2.lower[0] + cut2.upper[0]) *
                                             pops::Real(0.5) * (cut2.lower[1] + cut2.upper[1]));
}

TEST(PreparedEmbeddedBoundaryND, Dim3NonplanarSamplesUseCubeTriangulationNotSinglePlane) {
  pops::RealVector<3> lower_samples{-0.8, -0.1, -0.9};
  pops::RealVector<3> upper_samples{0.4, 0.6, 0.2};
  const auto cut = pops::nd::cut_cell_fractions_from_samples<3>(pops::Real(-0.2), lower_samples,
                                                                upper_samples);
  pops::RealVector<3> gradient{};
  pops::Real intercept = pops::Real(0);
  pops::nd::cut_geometry_detail::linear_level_set_from_samples<3>(
      pops::Real(-0.2), lower_samples, upper_samples, gradient, intercept);
  pops::Real g[3]{gradient[0], gradient[1], gradient[2]};
  const pops::Real plane =
      pops::nd::cut_geometry_detail::unit_box_negative_halfspace<3>(g, intercept);
  EXPECT_GT(cut.volume_fraction, pops::Real(0));
  EXPECT_LE(cut.volume_fraction, pops::Real(1));
  EXPECT_GT(std::abs(cut.volume_fraction - plane), pops::Real(1e-4));
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_GE(cut.face_lower[axis], pops::Real(0));
    EXPECT_LE(cut.face_lower[axis], pops::Real(1));
    EXPECT_GE(cut.face_upper[axis], pops::Real(0));
    EXPECT_LE(cut.face_upper[axis], pops::Real(1));
  }
}

TEST(PreparedEmbeddedBoundaryND, Dim3AxisAlignedPlaneHasConservativeFaceApertures) {
  // Neighbour-centre samples of phi = z - 0.7. Volume of {z < 0.7} is 0.7, not the 1-D product 0.6.
  pops::RealVector<3> lower_samples{-0.2, -0.2, -1.2};
  pops::RealVector<3> upper_samples{-0.2, -0.2, 0.8};
  const auto cut = pops::nd::cut_cell_fractions_from_samples<3>(pops::Real(-0.2), lower_samples,
                                                                upper_samples);
  const pops::Real product = pops::Real(0.5) * (cut.lower[2] + cut.upper[2]);
  EXPECT_DOUBLE_EQ(cut.lower[2], pops::Real(1));
  EXPECT_DOUBLE_EQ(cut.upper[2], pops::Real(0.2));
  EXPECT_NEAR(product, pops::Real(0.6), 1e-15);
  EXPECT_NEAR(cut.volume_fraction, pops::Real(0.7), 1e-12);
  EXPECT_GT(cut.volume_fraction, pops::Real(0));
  EXPECT_LE(cut.volume_fraction, pops::Real(1));
  EXPECT_DOUBLE_EQ(cut.face_lower[2], pops::Real(1));
  EXPECT_DOUBLE_EQ(cut.face_upper[2], pops::Real(0));
  EXPECT_NEAR(cut.face_lower[0], pops::Real(0.7), 1e-12);
  EXPECT_NEAR(cut.face_upper[0], pops::Real(0.7), 1e-12);
  EXPECT_NEAR(cut.face_lower[1], pops::Real(0.7), 1e-12);
  EXPECT_NEAR(cut.face_upper[1], pops::Real(0.7), 1e-12);
}

TEST(PreparedEmbeddedBoundaryND, ConservativeAmrRestrictProlongRefluxStayOnOneMetric) {
  pops::nd::CutCellFractions<2> fine[4]{};
  for (auto& child : fine) {
    child.volume_fraction = pops::Real(0.5);
    child.face_lower[0] = pops::Real(0.25);
    child.face_upper[0] = pops::Real(0.75);
  }
  const auto coarse = pops::nd::restrict_cut_cell_fractions(fine, 4);
  EXPECT_DOUBLE_EQ(coarse.volume_fraction, pops::Real(0.5));
  EXPECT_DOUBLE_EQ(coarse.face_lower[0], pops::Real(0.25));
  const auto injected = pops::nd::prolong_cut_cell_fractions(coarse);
  EXPECT_DOUBLE_EQ(injected.volume_fraction, coarse.volume_fraction);
  const pops::Real fine_faces[2] = {pops::Real(0.2), pops::Real(0.3)};
  EXPECT_DOUBLE_EQ(pops::nd::reflux_cut_face_aperture(pops::Real(0.4), fine_faces, 2),
                   pops::Real(0.1));
}

template <int Dim>
struct SampleFinePhi {
  pops::FieldView<pops::Real, Dim> phi{};

  POPS_HD void operator()(const pops::Index<Dim>& index) const {
    phi(index) = (static_cast<pops::Real>(index[0]) + pops::Real(0.5)) / pops::Real(4) -
                 pops::Real(0.5);
  }
};

TEST(PreparedEmbeddedBoundaryND, CutCellFractionTransferIsInvokedOnUniformRatioBoundary) {
  const pops::Box<2> fine_domain = pops::Box<2>::from_extents(filled_extent<2>(4));
  const pops::Box<2> coarse_domain = pops::Box<2>::from_extents(filled_extent<2>(2));
  pops::Fab<2> fine_phi(fine_domain, 1, filled_extent<2>(1));
  pops::Fab<2> coarse_volume(coarse_domain, 1);
  pops::Fab<2> fine_volume(fine_domain, 1);
  pops::Fab<2> aperture_residual(coarse_domain, 1);
  pops::for_each_cell(fine_phi.grown_box(), SampleFinePhi<2>{fine_phi.view()});

  const pops::amr::RefinementRatio<2> ratio(2, 2);
  pops::nd::apply_cut_cell_fraction_amr_transfer(
      std::as_const(fine_phi).view(), coarse_volume.view(), fine_volume.view(),
      aperture_residual.view(), coarse_domain, ratio);

  pops::sync_host();
  const pops::Index<2> coarse{};
  pops::nd::CutCellFractions<2> children[4]{};
  for (int child = 0; child < 4; ++child) {
    pops::Index<2> fine{};
    fine[0] = child % 2;
    fine[1] = child / 2;
    children[child] = pops::nd::cut_cell_fractions_from_phi_cell(std::as_const(fine_phi).view(),
                                                                 fine);
  }
  const auto restricted = pops::nd::restrict_cut_cell_fractions(children, 4);
  EXPECT_DOUBLE_EQ(std::as_const(coarse_volume).view()(coarse), restricted.volume_fraction);
  const auto injected = pops::nd::prolong_cut_cell_fractions(restricted);
  EXPECT_DOUBLE_EQ(std::as_const(fine_volume).view()(pops::Index<2>{}), injected.volume_fraction);
  EXPECT_DOUBLE_EQ(std::as_const(fine_volume).view()(pops::Index<2>{1, 1}),
                   injected.volume_fraction);

  pops::Real covering[2]{};
  int covering_count = 0;
  for (int child = 0; child < 4; ++child) {
    if ((child % 2) != 0)
      continue;
    covering[covering_count++] = children[child].face_lower[0];
  }
  EXPECT_DOUBLE_EQ(std::as_const(aperture_residual).view()(coarse),
                   pops::nd::reflux_cut_face_aperture(restricted.face_lower[0], covering, 2));
}

template <int Dim>
void prove_staircase() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<Dim> fixture;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase, pops::EbThresholds{}, 1,
      fixture.lane);
  ASSERT_TRUE(prepared);
  EXPECT_EQ(prepared->mode(), pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase);
  EXPECT_EQ(prepared->generation(), 1U);
  EXPECT_EQ(prepared->digest().size(),
            std::string("pops.prepared-eb-geometry.v1:sha256:").size() + 64U);
  EXPECT_EQ(prepared->phi().layout(), fixture.layout);
  EXPECT_EQ(prepared->active_mask().ghosts(), filled_extent<Dim>(1));
  EXPECT_EQ(prepared->volume_fraction().ghosts(), pops::Extent<Dim>{});
  EXPECT_EQ(prepared->face_aperture_lower().ncomp(), Dim);
  EXPECT_EQ(prepared->face_aperture_upper().ncomp(), Dim);
  EXPECT_EQ(prepared->face_aperture_lower().ghosts(), pops::Extent<Dim>{});
  EXPECT_EQ(prepared->face_aperture_upper().ghosts(), pops::Extent<Dim>{});

  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  const auto mask = prepared->active_mask().fab(0).view();
  const auto kappa = prepared->volume_fraction().fab(0).view();
  const auto inverse = prepared->inverse_volume_fraction().fab(0).view();
  const auto face_lower = prepared->face_aperture_lower().fab(0).view();
  const auto face_upper = prepared->face_aperture_upper().fab(0).view();
  pops::Index<Dim> active{};
  pops::Index<Dim> inactive{};
  for (int axis = 0; axis < Dim; ++axis) {
    active[axis] = 2;
    inactive[axis] = 6;
  }
  EXPECT_LT(phi(active), pops::Real(0));
  EXPECT_EQ(mask(active), pops::Real(1));
  EXPECT_GT(kappa(active), pops::Real(0));
  EXPECT_GE(inverse(active), pops::Real(1));
  EXPECT_GT(phi(inactive), pops::Real(0));
  EXPECT_EQ(mask(inactive), pops::Real(0));
  EXPECT_EQ(kappa(inactive), pops::Real(0));
  EXPECT_EQ(inverse(inactive), pops::Real(0));
  const auto expected_fractions = pops::nd::cut_cell_fractions_from_phi_cell(phi, active);
  for (int axis = 0; axis < Dim; ++axis) {
    EXPECT_DOUBLE_EQ(face_lower(active, axis), expected_fractions.face_lower[axis]);
    EXPECT_DOUBLE_EQ(face_upper(active, axis), expected_fractions.face_upper[axis]);
    EXPECT_DOUBLE_EQ(face_lower(inactive, axis), pops::Real(0));
    EXPECT_DOUBLE_EQ(face_upper(inactive, axis), pops::Real(0));
  }

  double active_count = 0.0;
  for (std::size_t local = 0; local < prepared->active_mask().local_size(); ++local)
    active_count +=
        pops::for_each_cell_reduce_sum(prepared->active_mask().box(local),
                                       SumField<Dim>{prepared->active_mask().fab(local).view()});
  std::size_t expected = 4;
  for (int axis = 1; axis < Dim; ++axis)
    expected *= 8;
  EXPECT_EQ(active_count, static_cast<double>(expected));
}

TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInOneDimension) {
  prove_staircase<1>();
}
TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInTwoDimensions) {
  prove_staircase<2>();
}
TEST(PreparedEmbeddedBoundaryND, StaircaseIsExactInThreeDimensions) {
  prove_staircase<3>();
}

template <int Dim>
void prove_prepared_cut_cell_stores_continuous_apertures() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<Dim> fixture;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "y", "add", "constant", "sub"}, {0.0, 0.0, 0.0, 0.4, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 8,
      fixture.lane);
  ASSERT_TRUE(prepared);
  pops::sync_host();
  bool found_partial = false;
  for (std::size_t local = 0; local < prepared->face_aperture_lower().local_size(); ++local) {
    const auto box = prepared->face_aperture_lower().box(local);
    const auto phi = prepared->phi().fab(local).view();
    const auto lower = prepared->face_aperture_lower().fab(local).view();
    const auto upper = prepared->face_aperture_upper().fab(local).view();
    const auto mask = prepared->active_mask().fab(local).view();
    for (int i0 = box.lo[0]; i0 <= box.hi[0]; ++i0) {
      const int i1_lo = Dim > 1 ? box.lo[1] : 0;
      const int i1_hi = Dim > 1 ? box.hi[1] : 0;
      for (int i1 = i1_lo; i1 <= i1_hi; ++i1) {
        const int i2_lo = Dim > 2 ? box.lo[2] : 0;
        const int i2_hi = Dim > 2 ? box.hi[2] : 0;
        for (int i2 = i2_lo; i2 <= i2_hi; ++i2) {
          pops::Index<Dim> cell{};
          cell[0] = i0;
          if constexpr (Dim > 1)
            cell[1] = i1;
          if constexpr (Dim > 2)
            cell[2] = i2;
          if (mask(cell) < pops::Real(0.5))
            continue;
          const auto expected = pops::nd::cut_cell_fractions_from_phi_cell(phi, cell);
          for (int axis = 0; axis < Dim; ++axis) {
            EXPECT_NEAR(lower(cell, axis), expected.face_lower[axis], 1e-12);
            EXPECT_NEAR(upper(cell, axis), expected.face_upper[axis], 1e-12);
            if ((expected.face_lower[axis] > pops::Real(0) &&
                 expected.face_lower[axis] < pops::Real(1)) ||
                (expected.face_upper[axis] > pops::Real(0) &&
                 expected.face_upper[axis] < pops::Real(1)))
              found_partial = true;
          }
        }
      }
    }
  }
  if constexpr (Dim >= 2)
    EXPECT_TRUE(found_partial);
}

TEST(PreparedEmbeddedBoundaryND, CutCellStoresContinuousAperturesInTwoDimensions) {
  prove_prepared_cut_cell_stores_continuous_apertures<2>();
}
TEST(PreparedEmbeddedBoundaryND, CutCellStoresContinuousAperturesInThreeDimensions) {
  prove_prepared_cut_cell_stores_continuous_apertures<3>();
}

template <int Dim>
void prove_periodic_halo() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<Dim> fixture;
  std::array<bool, Dim> periodic{};
  periodic[0] = true;
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.25, 0.0}, fixture.geometry,
      pops::BoundaryTopology<Dim>::axis_periodic(periodic), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 2,
      fixture.lane);
  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  pops::Index<Dim> low_ghost{};
  pops::Index<Dim> high_valid{};
  pops::Index<Dim> high_ghost{};
  pops::Index<Dim> low_valid{};
  for (int axis = 0; axis < Dim; ++axis) {
    low_ghost[axis] = 3;
    high_valid[axis] = 3;
    high_ghost[axis] = 3;
    low_valid[axis] = 3;
  }
  low_ghost[0] = -1;
  high_valid[0] = 7;
  high_ghost[0] = 8;
  low_valid[0] = 0;
  EXPECT_EQ(phi(low_ghost), phi(high_valid));
  EXPECT_EQ(phi(high_ghost), phi(low_valid));
}

TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInOneDimension) {
  prove_periodic_halo<1>();
}
TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInTwoDimensions) {
  prove_periodic_halo<2>();
}
TEST(PreparedEmbeddedBoundaryND, PeriodicHaloUsesExactTopologyInThreeDimensions) {
  prove_periodic_halo<3>();
}

template <int Dim>
void prove_sparse_in_domain_ghosts_are_analytic() {
  auto world = pops::ExecutionLane::world("test/prepared-eb-sparse-serial");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";

  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(16));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::Box<Dim> patch{};
  for (int axis = 0; axis < Dim; ++axis) {
    patch.lo[axis] = 4;
    patch.hi[axis] = 7;
  }
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{patch}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  const pops::MultiFab<Dim> prototype{layout, distribution, pops::Index<Dim>{}, 1,
                                      pops::Extent<Dim>{}};
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, geometry, pops::BoundaryTopology<Dim>::physical(),
      prototype, pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase,
      pops::EbThresholds{}, 7, world);

  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = 4;
    upper[axis] = 7;
  }
  lower[0] = 3;
  upper[0] = 8;
  EXPECT_DOUBLE_EQ(phi(lower), pops::Real(3.5 / 16.0 - 0.5));
  EXPECT_DOUBLE_EQ(phi(upper), pops::Real(8.5 / 16.0 - 0.5));
}

TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInOneDimension) {
  prove_sparse_in_domain_ghosts_are_analytic<1>();
}
TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInTwoDimensions) {
  prove_sparse_in_domain_ghosts_are_analytic<2>();
}
TEST(PreparedEmbeddedBoundaryND, SparseInDomainGhostsAreAnalyticInThreeDimensions) {
  prove_sparse_in_domain_ghosts_are_analytic<3>();
}

TEST(PreparedEmbeddedBoundaryND, NonFiniteReplacementRollsBackAcceptedOwner) {
  auto world = pops::ExecutionLane::world("test/prepared-eb-serial-preflight");
  if (world.size() != 1)
    GTEST_SKIP() << "serial exact-rank fixture";
  Fixture<2> fixture;
  std::shared_ptr<const pops::runtime::system::PreparedEmbeddedBoundaryGeometry<2>> accepted;
  pops::runtime::system::replace_prepared_embedded_boundary_geometry_collectively(
      accepted, {"x", "constant", "sub"}, {0.0, 0.5, 0.0}, fixture.geometry,
      pops::BoundaryTopology<2>::physical(), fixture.prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase, pops::EbThresholds{}, 3,
      fixture.lane);
  ASSERT_TRUE(accepted);
  const auto* original = accepted.get();
  const std::string digest = accepted->digest();
  EXPECT_THROW(pops::runtime::system::replace_prepared_embedded_boundary_geometry_collectively(
                   accepted, {"x", "x", "sub", "constant", "div"}, {0.0, 0.0, 0.0, 0.0, 0.0},
                   fixture.geometry, pops::BoundaryTopology<2>::physical(), fixture.prototype,
                   pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell,
                   pops::EbThresholds{}, 4, fixture.lane),
               std::domain_error);
  EXPECT_EQ(accepted.get(), original);
  EXPECT_EQ(accepted->digest(), digest);
}

TEST(PreparedEmbeddedBoundaryND, RankDivergentRequestFailsBeforePublication) {
  auto lane = pops::ExecutionLane::world("test/prepared-eb-mismatch");
  if (lane.size() < 2)
    GTEST_SKIP() << "requires at least two MPI ranks";

  constexpr int Dim = 1;
  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  pops::Extent<Dim> rank_extent{};
  rank_extent[0] = lane.size();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  std::vector<pops::Index<Dim>> owners{pops::Index<Dim>{0}};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(layout, ranks, owners);
  pops::Index<Dim> local_rank{};
  local_rank[0] = lane.rank();
  const pops::MultiFab<Dim> prototype{layout, distribution, local_rank, 1, pops::Extent<Dim>{}};
  const double split = lane.rank() == 0 ? 0.4 : 0.6;
  EXPECT_THROW((void)pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
                   {"x", "constant", "sub"}, {0.0, split, 0.0}, geometry,
                   pops::BoundaryTopology<Dim>::physical(), prototype,
                   pops::runtime::system::PreparedEmbeddedBoundaryMode::staircase,
                   pops::EbThresholds{}, 5, lane),
               std::runtime_error);
}

TEST(PreparedEmbeddedBoundaryND, DistributedPeriodicHaloUsesOwningExecutionLane) {
  constexpr int Dim = 1;
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test/prepared-eb-periodic-mpi");
  if (lane.size() != 2)
    GTEST_SKIP() << "requires exactly two MPI ranks";

  const auto domain = pops::Box<Dim>::from_extents(filled_extent<Dim>(8));
  const auto geometry =
      pops::Geometry<Dim>::from_bounds(domain, filled_real<Dim>(0.0), filled_real<Dim>(1.0));
  pops::Box<Dim> left = domain;
  pops::Box<Dim> right = domain;
  left.hi[0] = 3;
  right.lo[0] = 4;
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{left, right}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(2)};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(
      layout, ranks, {pops::Index<Dim>{0}, pops::Index<Dim>{1}});
  const pops::Index<Dim> local_rank{lane.rank()};
  const pops::MultiFab<Dim> prototype{layout, distribution, local_rank, 1, pops::Extent<Dim>{}};
  std::array<bool, Dim> periodic{true};
  const auto prepared = pops::runtime::system::prepare_embedded_boundary_geometry_collectively(
      {"x", "constant", "sub"}, {0.0, 0.25, 0.0}, geometry,
      pops::BoundaryTopology<Dim>::axis_periodic(periodic), prototype,
      pops::runtime::system::PreparedEmbeddedBoundaryMode::cut_cell, pops::EbThresholds{}, 6, lane);

  ASSERT_EQ(prepared->phi().local_size(), 1U);
  pops::sync_host();
  const auto phi = prepared->phi().fab(0).view();
  if (lane.rank() == 0) {
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{-1}), pops::Real(0.6875));
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{4}), pops::Real(0.3125));
  } else {
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{3}), pops::Real(0.1875));
    EXPECT_DOUBLE_EQ(phi(pops::Index<Dim>{8}), pops::Real(-0.1875));
  }
}

}  // namespace
