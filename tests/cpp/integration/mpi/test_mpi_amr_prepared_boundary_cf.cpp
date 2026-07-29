// Distributed qualification of the sole prepared transport-boundary authority at a moving AMR
// hierarchy.
//
// A real regrid first removes and then recreates the fine level.  The recreated patch remains
// strictly inside the physical domain, so every uncovered fine ghost is a coarse/fine interface
// ghost, never a physical-boundary ghost.  The persistent PreparedGridBoundarySession must execute
// the conservative coarse/fine producer before same-level/MPI and physical-face production.  A
// large fixed physical value makes any accidental patch-edge-as-domain-edge routing immediately
// visible.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "amr_transfer_test_authority.hpp"
#include "gtest_compat.hpp"
#include "load_balance_test_authority.hpp"

#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr Real kCoarseValue = Real(7.25);
constexpr Real kFineValue = Real(3.0);
constexpr Real kPhysicalValue = Real(40.0);
constexpr Real kUntouchedGhost = Real(-901.0);
constexpr Real kExpectedPhysicalGhost = Real(2) * kPhysicalValue - kCoarseValue;

bool covered_by(const BoxArray& boxes, int i, int j) {
  return std::any_of(boxes.boxes().begin(), boxes.boxes().end(),
                     [=](const Box2D& box) { return box.contains(i, j); });
}

POPS_HD Real native_exp(Real value) {
#if defined(POPS_HAS_KOKKOS)
  return Kokkos::exp(value);
#else
  return std::exp(value);
#endif
}

void seed_centered_refinement_marker(MultiFab& state, int resolution) {
  state.set_val(Real(1));
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    const Box2D valid = state.box(local);
    for_each_cell(valid, [=] POPS_HD(int i, int j) {
      const Real x = (Real(i) + Real(0.5)) / Real(resolution);
      const Real y = (Real(j) + Real(0.5)) / Real(resolution);
      const Real r2 = (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5));
      values(i, j, 0) = Real(1) + Real(0.8) * native_exp(-r2 / Real(0.0064));
    });
  }
  device_fence();
}

int run_prepared_boundary_cf_regrid(int me, int np) {
  constexpr int n = 16;
  const Geometry geometry{Box2D::from_extents(n, n), Real(0), Real(1), Real(0), Real(1)};
  const BoxArray coarse_boxes = BoxArray::from_domain(geometry.domain, n / 2);
  const Box2D initial_fine_patch = Box2D{{4, 4}, {11, 11}}.refine(2);

  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->push_back(
      AmrLevelMP{MultiFab(coarse_boxes, DistributionMapping(coarse_boxes.size(), n_ranks()), 1, 1),
                 nullptr, geometry.dx(), geometry.dy()});
  levels->push_back(
      AmrLevelMP{MultiFab(BoxArray({initial_fine_patch}), DistributionMapping({0}), 1, 1), nullptr,
                 geometry.dx() / Real(2), geometry.dy() / Real(2)});

  MultiFab& coarse_seed = levels->front().U;
  seed_centered_refinement_marker(coarse_seed, n);
  levels->back().U.set_val(Real(1));

  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrHierarchyLayout hierarchy = AmrHierarchyLayout::from_levels(*levels, load_balance);

  const std::string state_identity = "test://adc749/mpi-amr/block/tracer/state/U";
  AmrRuntimeBlock block;
  block.name = "tracer";
  block.state_identity = state_identity;
  block.ncomp = 1;
  block.levels = levels;
  block.add_elliptic_rhs = [](const MultiFab&, MultiFab&) {};
  block.max_speed = [](const MultiFab&, const MultiFab&) { return Real(0); };
  block.boundary_plan = std::make_shared<PreparedBoundaryPlan>(
      "test://adc749/mpi-amr/block/tracer/boundary", 1,
      prepare_hyperbolic_boundary<2>({"dirichlet", "dirichlet", "dirichlet", "dirichlet"},
                                     std::vector<double>(4, static_cast<double>(kPhysicalValue)),
                                     {"test://adc749/mpi-amr/xlo", "test://adc749/mpi-amr/xhi",
                                      "test://adc749/mpi-amr/ylo", "test://adc749/mpi-amr/yhi"},
                                     {"Scalar"}),
      std::vector<int>{}, state_identity);
  block.boundary_field_registry = std::make_shared<GridContext::BoundaryFieldRegistryFactory>();
  block.level_rhs_core_at_point_prepared =
      [](const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& state,
         const MultiFab&, const Geometry&, MultiFab& residual,
         const PreparedGridBoundarySession& boundary) {
        boundary.fill(state, point);
        residual.set_val(Real(0));
      };
  block.level_boundary_residual_at_point_prepared =
      [](const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, const MultiFab&,
         const Geometry&, MultiFab&, const PreparedGridBoundarySession&) {};
  block.level_rhs_at_point = [](const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                                const MultiFab&, const Geometry&, MultiFab&) {
    throw std::runtime_error("legacy AMR boundary fallback was selected");
  };

  BCRec poisson_boundary;
  poisson_boundary.xlo = poisson_boundary.xhi = BCType::Foextrap;
  poisson_boundary.ylo = poisson_boundary.yhi = BCType::Foextrap;
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(std::move(block));
  AmrRuntime runtime(geometry, std::move(hierarchy), poisson_boundary, std::move(blocks),
                     Periodicity{false, false}, /*replicated_coarse=*/false);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  runtime.set_parent_child_temporal_relations({amr::ParentChildClockRelation(
      0, 1, amr::Rational(2, 1), amr::RemainderPolicy::IntegralOnly)});
  runtime.install_boundary_storage_routes({});

  // Exercise both topology transitions.  Boundary sessions and coarse/fine workspaces must be
  // destroyed with the removed level and rematerialized for the new distributed fine layout.
  test::install_prepared_threshold_decisions(
      runtime, {{0, 0, Real(1e9), test::PreparedThresholdRelation::Above}},
      {{0, 0, Real(1e9), test::PreparedThresholdRelation::Below}}, "test::adc749::remove-fine@1");
  runtime.regrid();
  const bool removed = runtime.nlev() == 1;

  // Fine-to-coarse removal legitimately averages the former fine state into the parent. Re-seed
  // the coarse tagging field so this fixture requests a second, independent topology transition.
  seed_centered_refinement_marker(runtime.level_state(0, 0), n);
  test::install_prepared_threshold_decisions(
      runtime, {{0, 0, Real(1.05), test::PreparedThresholdRelation::Above}},
      {{0, 0, Real(1.05), test::PreparedThresholdRelation::Below}}, "test::adc749::regrow-fine@1");
  runtime.regrid();
  const bool regrown = runtime.nlev() == 2 && runtime.n_patches() > 0;

  // Do not enter a boundary fill collectively unless every rank published both topology
  // transitions. A broken regrid consensus must fail this fixture, never strand only a subset of
  // ranks inside the MPI halo exchange exercised below.
  const bool topology_ready = all_reduce_max(removed && regrown ? 0L : 1L) == 0L;
  long local_failures = topology_ready ? 0L : 1L;
  long local_cf_ghosts = 0;
  long local_fine_physical_touches = 0;
  long local_physical_face_ghosts = 0;
  if (topology_ready) {
    MultiFab& coarse = runtime.level_state(0, 0);
    MultiFab& fine = runtime.level_state(0, 1);
    coarse.set_val(kCoarseValue);
    fine.set_val(kUntouchedGhost);
    for (int local = 0; local < fine.local_size(); ++local) {
      const Array4 values = fine.fab(local).array();
      for_each_cell(fine.box(local), [=] POPS_HD(int i, int j) { values(i, j, 0) = kFineValue; });
    }
    device_fence();

    MultiFab residual(fine.box_array(), fine.dmap(), fine.ncomp(), 0);
    const runtime::multiblock::BoundaryEvaluationPoint point{
        "clock.adc749-amr-boundary", 0, 1, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
    runtime.level_rhs_into_at(0, 1, point, fine, residual);
    device_fence();
    fine.sync_host();

    const Box2D fine_domain = runtime.level_geom(1).domain;
    const BoxArray& valid_boxes = fine.box_array();
    for (const Box2D& box : valid_boxes.boxes())
      if (box.lo[0] == fine_domain.lo[0] || box.hi[0] == fine_domain.hi[0] ||
          box.lo[1] == fine_domain.lo[1] || box.hi[1] == fine_domain.hi[1])
        ++local_fine_physical_touches;

    for (int local = 0; local < fine.local_size(); ++local) {
      const Fab2D& values = fine.fab(local);
      const Box2D grown = values.grown_box();
      for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
        for (int i = grown.lo[0]; i <= grown.hi[0]; ++i) {
          if (!fine_domain.contains(i, j) || covered_by(valid_boxes, i, j))
            continue;
          ++local_cf_ghosts;
          if (std::fabs(values(i, j, 0) - kCoarseValue) > Real(1e-12))
            ++local_failures;
        }
    }

    // The same rematerialized plan must still own actual base-domain faces.  This companion check
    // prevents an inert boundary plan from making the internal-interface assertion vacuous.
    MultiFab coarse_residual(coarse.box_array(), coarse.dmap(), coarse.ncomp(), 0);
    const runtime::multiblock::BoundaryEvaluationPoint coarse_point{
        "clock.adc749-amr-boundary", 0, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
    runtime.level_rhs_into_at(0, 0, coarse_point, coarse, coarse_residual);
    device_fence();
    coarse.sync_host();
    const Box2D coarse_domain = runtime.level_geom(0).domain;
    for (int local = 0; local < coarse.local_size(); ++local) {
      const Fab2D& values = coarse.fab(local);
      const Box2D valid = coarse.box(local);
      auto check = [&](int i, int j) {
        ++local_physical_face_ghosts;
        if (std::fabs(values(i, j, 0) - kExpectedPhysicalGhost) > Real(1e-12))
          ++local_failures;
      };
      if (valid.lo[0] == coarse_domain.lo[0])
        for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
          check(coarse_domain.lo[0] - 1, j);
      if (valid.hi[0] == coarse_domain.hi[0])
        for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
          check(coarse_domain.hi[0] + 1, j);
      if (valid.lo[1] == coarse_domain.lo[1])
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          check(i, coarse_domain.lo[1] - 1);
      if (valid.hi[1] == coarse_domain.hi[1])
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          check(i, coarse_domain.hi[1] + 1);
    }
  }

  const long failures = all_reduce_sum(local_failures);
  const long cf_ghosts = all_reduce_sum(local_cf_ghosts);
  const long fine_physical_touches = all_reduce_max(local_fine_physical_touches);
  const long physical_face_ghosts = all_reduce_sum(local_physical_face_ghosts);
  const double patch_spread = all_reduce_max(static_cast<double>(runtime.n_patches())) -
                              (-all_reduce_max(-static_cast<double>(runtime.n_patches())));
  const bool qualified = topology_ready && failures == 0 && cf_ghosts > 0 &&
                         fine_physical_touches == 0 && physical_face_ghosts > 0 &&
                         patch_spread == 0.0 && runtime.regrid_count() == 2;

  if (me == 0)
    std::printf(
        "ADC749_BOUNDARY_CF np=%d | removed=%d regrown=%d regrids=%d patches=%d | "
        "cf_ghosts=%ld fine_physical_touches=%ld physical_face_ghosts=%ld failures=%ld "
        "spread=%.1f\n",
        np, removed ? 1 : 0, regrown ? 1 : 0, runtime.regrid_count(), runtime.n_patches(),
        cf_ghosts, fine_physical_touches, physical_face_ghosts, failures, patch_spread);
  return qualified ? 0 : 1;
}

int pops_run_test_mpi_amr_prepared_boundary_cf(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int result = run_prepared_boundary_cf_regrid(my_rank(), n_ranks());
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_amr_prepared_boundary_cf, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_amr_prepared_boundary_cf,
                                    "test_mpi_amr_prepared_boundary_cf"),
            0);
}
