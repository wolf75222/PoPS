#pragma once

#include "load_balance_test_authority.hpp"

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace pops::test {

struct PeriodicRegridCrossingEvidence {
  int axis = 0;
  int regrid_count = 0;
  bool multi_patch_parent = false;
  bool periodic_low_and_high_covered = false;
  bool transverse_boundary_covered = false;
  bool layout_consensus = false;
  Real maximum_composite_mass_error = std::numeric_limits<Real>::infinity();
  Real maximum_injection_error = std::numeric_limits<Real>::infinity();

  [[nodiscard]] bool passed(Real tolerance = Real(1e-11)) const {
    return regrid_count == 3 && multi_patch_parent && periodic_low_and_high_covered &&
           transverse_boundary_covered && layout_consensus &&
           maximum_composite_mass_error <= tolerance && maximum_injection_error <= tolerance;
  }
};

namespace periodic_regrid_detail {

inline Real periodic_gaussian(int i, int j, int axis, int periodic_center, int transverse_center,
                              int extent) {
  const int periodic_coordinate = axis == 0 ? i : j;
  const int transverse_coordinate = axis == 0 ? j : i;
  const int direct_distance = std::abs(periodic_coordinate - periodic_center);
  const int periodic_distance = std::min(direct_distance, extent - direct_distance);
  const int transverse_distance = transverse_coordinate - transverse_center;
  constexpr Real sigma = Real(1.25);
  const Real radius_squared =
      Real(periodic_distance * periodic_distance + transverse_distance * transverse_distance);
  return std::exp(-radius_squared / (Real(2) * sigma * sigma));
}

inline void write_exact_profile(MultiFab& field, int level, int axis, int periodic_center,
                                int transverse_center, int extent, int refinement_ratio) {
  field.sync_host();
  int scale = 1;
  for (int current = 0; current < level; ++current)
    scale *= refinement_ratio;
  for (int local = 0; local < field.local_size(); ++local) {
    Fab2D& fab = field.fab(local);
    const Box2D& valid = fab.box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        fab(i, j, 0) = periodic_gaussian(i / scale, j / scale, axis, periodic_center,
                                         transverse_center, extent);
  }
  field.sync_device();
}

inline bool parent_cell_is_refined(const BoxArray& fine_boxes, int i, int j, int refinement_ratio) {
  return std::any_of(fine_boxes.boxes().begin(), fine_boxes.boxes().end(), [=](const Box2D& fine) {
    return fine.coarsen(refinement_ratio).contains(i, j);
  });
}

inline Real composite_mass(const AmrHierarchy& hierarchy) {
  const Real coarse_mass = sum(hierarchy.data(0));
  if (hierarchy.num_levels() == 1)
    return coarse_mass;
  const int ratio = hierarchy.ref_ratio();
  const BoxArray& fine_boxes = hierarchy.boxes(1);
  const MultiFab& coarse = hierarchy.data(0);
  coarse.sync_host();
  Real local_covered_mass = Real(0);
  for (int local = 0; local < coarse.local_size(); ++local) {
    const Fab2D& fab = coarse.fab(local);
    const Box2D& valid = fab.box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        if (parent_cell_is_refined(fine_boxes, i, j, ratio))
          local_covered_mass += fab(i, j, 0);
  }
  const Real covered_mass =
      static_cast<Real>(all_reduce_sum(static_cast<double>(local_covered_mass)));
  return coarse_mass - covered_mass + sum(hierarchy.data(1)) / Real(ratio * ratio);
}

inline Real maximum_injection_error(const AmrHierarchy& hierarchy, int axis, int periodic_center,
                                    int transverse_center, int extent) {
  const int ratio = hierarchy.ref_ratio();
  const MultiFab& fine = hierarchy.data(1);
  fine.sync_host();
  Real local_error = Real(0);
  for (int local = 0; local < fine.local_size(); ++local) {
    const Fab2D& fab = fine.fab(local);
    const Box2D& valid = fab.box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Real expected = periodic_gaussian(i / ratio, j / ratio, axis, periodic_center,
                                                transverse_center, extent);
        local_error = std::max(local_error, std::fabs(fab(i, j, 0) - expected));
      }
  }
  return static_cast<Real>(
      all_reduce_max(static_cast<double>(local_error), world_communicator_view()));
}

inline bool layout_has_collective_consensus(const BoxArray& boxes) {
  const CommunicatorView communicator = world_communicator_view();
  const long count = boxes.size();
  if (all_reduce_min(count, communicator) != all_reduce_max(count, communicator))
    return false;
  for (const Box2D& box : boxes.boxes())
    for (const int coordinate : std::array<int, 4>{box.lo[0], box.lo[1], box.hi[0], box.hi[1]})
      if (all_reduce_min(static_cast<long>(coordinate), communicator) !=
          all_reduce_max(static_cast<long>(coordinate), communicator))
        return false;
  return true;
}

inline TagBox expected_grown_tags(const Box2D& domain, int axis, int periodic_center,
                                  int transverse_center, Real threshold,
                                  RegridPeriodicity periodicity) {
  TagBox tags(domain);
  for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
    for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
      if (periodic_gaussian(i, j, axis, periodic_center, transverse_center, domain.length(axis)) >
          threshold)
        tags(i, j) = 1;
  return grow_regrid_tags(tags, /*radius=*/1, domain, periodicity);
}

inline bool every_expected_tag_is_refined(const TagBox& expected, const BoxArray& fine_boxes,
                                          int refinement_ratio) {
  for (int j = expected.box.lo[1]; j <= expected.box.hi[1]; ++j)
    for (int i = expected.box.lo[0]; i <= expected.box.hi[0]; ++i)
      if (expected.tagged(i, j) && !parent_cell_is_refined(fine_boxes, i, j, refinement_ratio))
        return false;
  return true;
}

}  // namespace periodic_regrid_detail

inline PeriodicRegridCrossingEvidence run_periodic_regrid_crossing_evidence(int axis) {
  using namespace periodic_regrid_detail;
  constexpr int extent = 16;
  constexpr Real threshold = Real(0.25);
  const Box2D domain = Box2D::from_extents(extent, extent);
  const RegridPeriodicity periodicity{axis == 0, axis == 1};
  // x periodic: exercise a transverse physical wall. y periodic: straddle the x=7/8 parent-patch
  // seam. Together the oriented cases prove that physical support and same-level patch tiling do
  // not erase a valid periodic refinement request.
  const int transverse_center = axis == 0 ? domain.lo[1] : extent / 2 - 1;
  const RegridPhysicalGhostSupport physical_support{/*provided_depth=*/1,
                                                    /*fills_all_requested_depth=*/false};
  const RegridPhysicalGhostSupport* support = axis == 0 ? &physical_support : nullptr;
  AmrHierarchy hierarchy(domain, /*max_grid_size=*/8, /*ncomp=*/1, /*ngrow=*/1,
                         prepare_test_space_filling_curve_load_balance(),
                         /*ref_ratio=*/2);
  const amr::BergerRigoutsosProvider clustering(ClusterParams{});
  const RegridProlongation prolongation = [](const MultiFab& coarse, MultiFab& fine, int,
                                             int refinement_ratio, bool,
                                             const CommunicatorView& communicator) {
    interpolate(coarse, fine, refinement_ratio, communicator);
  };
  const HierarchyRegridOptions options{/*tag_buffer=*/1,
                                       /*nesting_margin=*/1, periodicity, support};
  const auto criterion = [] POPS_HD(const ConstArray4& values, int i, int j) {
    return values(i, j, 0) > threshold;
  };

  PeriodicRegridCrossingEvidence evidence;
  evidence.axis = axis;
  evidence.multi_patch_parent = hierarchy.boxes(0).size() == 4;
  evidence.periodic_low_and_high_covered = true;
  evidence.transverse_boundary_covered = true;
  evidence.layout_consensus = true;
  evidence.maximum_composite_mass_error = Real(0);
  evidence.maximum_injection_error = Real(0);
  Real reference_mass = std::numeric_limits<Real>::quiet_NaN();

  // high -> low -> high crosses both oriented faces and forces two replacement regrids after the
  // initial publication. The fine state is advanced to the same exact injected profile before each
  // topology change, so carry-over and prolongation have one conservation oracle.
  for (const int center : std::array<int, 3>{extent - 2, 1, extent - 2}) {
    write_exact_profile(hierarchy.data(0), /*level=*/0, axis, center, transverse_center, extent,
                        hierarchy.ref_ratio());
    if (hierarchy.num_levels() > 1)
      write_exact_profile(hierarchy.data(1), /*level=*/1, axis, center, transverse_center, extent,
                          hierarchy.ref_ratio());
    const Real coarse_mass = sum(hierarchy.data(0));
    if (!std::isfinite(reference_mass))
      reference_mass = coarse_mass;
    evidence.maximum_composite_mass_error =
        std::max(evidence.maximum_composite_mass_error, std::fabs(coarse_mass - reference_mass));

    const bool published =
        regrid_hierarchy_level(hierarchy, /*coarse_level=*/0, criterion, options, clustering,
                               prolongation, world_communicator_view());
    if (!published || hierarchy.num_levels() != 2)
      continue;
    ++evidence.regrid_count;

    const TagBox expected =
        expected_grown_tags(domain, axis, center, transverse_center, threshold, periodicity);
    const BoxArray& fine_boxes = hierarchy.boxes(1);
    const int low_i = axis == 0 ? domain.lo[0] : transverse_center;
    const int low_j = axis == 0 ? transverse_center : domain.lo[1];
    const int high_i = axis == 0 ? domain.hi[0] : transverse_center;
    const int high_j = axis == 0 ? transverse_center : domain.hi[1];
    evidence.periodic_low_and_high_covered =
        evidence.periodic_low_and_high_covered && expected.tagged(low_i, low_j) &&
        expected.tagged(high_i, high_j) &&
        parent_cell_is_refined(fine_boxes, low_i, low_j, hierarchy.ref_ratio()) &&
        parent_cell_is_refined(fine_boxes, high_i, high_j, hierarchy.ref_ratio()) &&
        every_expected_tag_is_refined(expected, fine_boxes, hierarchy.ref_ratio());

    const std::array<std::array<int, 2>, 2> transverse_probes =
        axis == 0
            ? std::array<std::array<int, 2>, 2>{std::array<int, 2>{domain.lo[0], domain.lo[1]},
                                                std::array<int, 2>{domain.hi[0], domain.lo[1]}}
            : std::array<std::array<int, 2>, 2>{std::array<int, 2>{extent / 2 - 1, domain.lo[1]},
                                                std::array<int, 2>{extent / 2, domain.lo[1]}};
    for (const auto& probe : transverse_probes)
      evidence.transverse_boundary_covered =
          evidence.transverse_boundary_covered && expected.tagged(probe[0], probe[1]) &&
          parent_cell_is_refined(fine_boxes, probe[0], probe[1], hierarchy.ref_ratio());

    evidence.layout_consensus =
        evidence.layout_consensus && layout_has_collective_consensus(fine_boxes);
    evidence.maximum_composite_mass_error =
        std::max(evidence.maximum_composite_mass_error,
                 std::fabs(composite_mass(hierarchy) - reference_mass));
    evidence.maximum_injection_error =
        std::max(evidence.maximum_injection_error,
                 maximum_injection_error(hierarchy, axis, center, transverse_center, extent));
  }
  return evidence;
}

}  // namespace pops::test
