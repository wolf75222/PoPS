#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
struct FillCoordinate {
  FieldView<Real, Dim> values{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    values(index, 0) = static_cast<Real>(index[0]);
  }
};

template <int Dim>
struct SetSample {
  FieldView<Real, Dim> values{};
  Real sample = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) = sample; }
};

template <int Dim>
mesh::RankSpace<Dim> rank_space() {
  Index<Dim> origin{};
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  extent[0] = 2;
  return {origin, extent};
}

template <int Dim>
Box<Dim> domain() {
  Index<Dim> lower{};
  Index<Dim> upper{};
  upper[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    upper[axis] = 1;
  return {lower, upper};
}

template <int Dim>
mesh::BoxArray<Dim> patches() {
  Box<Dim> lower = domain<Dim>();
  Box<Dim> upper = lower;
  lower.hi[0] = 1;
  upper.lo[0] = 2;
  return mesh::BoxArray<Dim>(std::vector<Box<Dim>>{lower, upper});
}

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
runtime::amr::PreparedTaggingProgram<Dim> tagging_program(std::string identity) {
  using Program = runtime::amr::PreparedTaggingProgram<Dim>;
  Program program;
  program.leaves = {
      typename Program::Leaf{0, 0, POPS_TAGGING_ABOVE_V1, 1.5, POPS_TAGGING_NO_STENCIL_V1}};
  program.refine_ops = {POPS_TAGGING_ABOVE_V1};
  program.refine_args = {0};
  program.clock_identity = "test.mpi.prepared-tagging.clock";
  program.provider_identity = std::move(identity);
  program.prepared = true;
  return program;
}

template <int Dim>
runtime::amr::PreparedTaggingExecutionBudget budget(std::size_t owned_patches,
                                                    std::size_t owned_cells, bool replicated) {
  const std::size_t cells_per_patch = static_cast<std::size_t>(patches<Dim>()[0].numPts());
  return {{2, owned_patches, cells_per_patch, owned_cells, owned_cells, 1U << 20},
          owned_cells,
          replicated ? owned_cells * 2u : 0u};
}

template <int Dim, class Require>
void exercise_partitioned(Require&& require) {
  const auto boxes = patches<Dim>();
  const auto ranks = rank_space<Dim>();
  const std::vector<Index<Dim>> owners{rank_coordinate<Dim>(0), rank_coordinate<Dim>(1)};
  const auto distribution = mesh::Distribution<Dim>::partitioned(boxes, ranks, owners);
  const amr::hierarchy::LevelLayout<Dim> layout(0, domain<Dim>(), boxes, distribution,
                                                amr::RefinementRatio<Dim>{},
                                                mesh::BoxArrayValidationBudget{2, 1});
  const Index<Dim> local_rank = rank_coordinate<Dim>(my_rank());
  Extent<Dim> ghosts{};
  MultiFab<Dim> state(boxes, distribution, local_rank, 1, ghosts);
  require(state.local_size() == 1U);
  for_each_cell(state.fab(0).box(), FillCoordinate<Dim>{state.fab(0).view()});
  device_fence();

  using Field =
      runtime::amr::PreparedTaggingField<Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  using Plan = runtime::amr::PreparedTaggingExecutionPlan<Dim>;
  const std::vector<std::vector<Field>> fields{{{"state/U", &state}}};
  const std::vector<amr::hierarchy::LevelLayout<Dim>> layouts{layout};
  const std::size_t local_cells = static_cast<std::size_t>(state.box(0).numPts());
  const std::vector<runtime::amr::PreparedTaggingExecutionBudget> budgets{
      budget<Dim>(1, local_cells, false)};
  std::array<Real, Dim> spacing{};
  spacing.fill(Real(1));

  auto plan =
      Plan::prepare(tagging_program<Dim>("test.mpi.prepared-tagging"), fields, layouts, budgets, 7);
  const auto& accepted = plan.execute(0, layout, spacing, 7);
  const long local_count = static_cast<long>(accepted.refine.count());
  const long tangent_cells = static_cast<long>(local_cells / 2u);
  require(local_count == (my_rank() == 0 ? 0L : local_cells));
  require(all_reduce_sum(local_count) == 2L * tangent_cells);

  auto drifted = tagging_program<Dim>(my_rank() == 0 ? "test.mpi.prepared-tagging"
                                                     : "test.mpi.prepared-tagging.drift");
  bool rejected_drift = false;
  try {
    (void)Plan::prepare(drifted, fields, layouts, budgets, 8);
  } catch (const std::invalid_argument&) {
    rejected_drift = true;
  }
  require(all_reduce_min(rejected_drift ? 1L : 0L) == 1L);

  MultiFab<Dim> component_drift_state(boxes, distribution, local_rank, 1 + my_rank(), ghosts);
  const std::vector<std::vector<Field>> component_drift_fields{
      {{"state/U", &component_drift_state}}};
  bool rejected_component_drift = false;
  try {
    (void)Plan::prepare(tagging_program<Dim>("test.mpi.prepared-tagging"), component_drift_fields,
                        layouts, budgets, 8);
  } catch (const std::invalid_argument&) {
    rejected_component_drift = true;
  }
  require(all_reduce_min(rejected_component_drift ? 1L : 0L) == 1L);

  Extent<Dim> drifted_ghosts{};
  drifted_ghosts[0] = my_rank();
  MultiFab<Dim> ghost_drift_state(boxes, distribution, local_rank, 1, drifted_ghosts);
  const std::vector<std::vector<Field>> ghost_drift_fields{{{"state/U", &ghost_drift_state}}};
  bool rejected_ghost_drift = false;
  try {
    (void)Plan::prepare(tagging_program<Dim>("test.mpi.prepared-tagging"), ghost_drift_fields,
                        layouts, budgets, 8);
  } catch (const std::invalid_argument&) {
    rejected_ghost_drift = true;
  }
  require(all_reduce_min(rejected_ghost_drift ? 1L : 0L) == 1L);

  Box<Dim> topology_domain = domain<Dim>();
  std::vector<Box<Dim>> topology_patch_values = boxes.boxes();
  if (my_rank() == 1) {
    topology_domain.lo[0] += 8;
    topology_domain.hi[0] += 8;
    for (Box<Dim>& patch : topology_patch_values) {
      patch.lo[0] += 8;
      patch.hi[0] += 8;
    }
  }
  const mesh::BoxArray<Dim> topology_boxes(std::move(topology_patch_values));
  const auto topology_distribution =
      mesh::Distribution<Dim>::partitioned(topology_boxes, ranks, owners);
  const amr::hierarchy::LevelLayout<Dim> topology_layout(
      0, topology_domain, topology_boxes, topology_distribution, amr::RefinementRatio<Dim>{},
      mesh::BoxArrayValidationBudget{2, 1});
  MultiFab<Dim> topology_state(topology_boxes, topology_distribution, local_rank, 1, ghosts);
  const std::vector<std::vector<Field>> topology_fields{{{"state/U", &topology_state}}};
  const std::vector<amr::hierarchy::LevelLayout<Dim>> topology_layouts{topology_layout};
  bool rejected_topology_drift = false;
  try {
    (void)Plan::prepare(tagging_program<Dim>("test.mpi.prepared-tagging"), topology_fields,
                        topology_layouts, budgets, 8);
  } catch (const std::invalid_argument&) {
    rejected_topology_drift = true;
  }
  require(all_reduce_min(rejected_topology_drift ? 1L : 0L) == 1L);

  if (my_rank() == 1) {
    const Index<Dim> bad = state.box(0).lo;
    for_each_cell(Box<Dim>{bad, bad},
                  SetSample<Dim>{state.fab(0).view(), std::numeric_limits<Real>::quiet_NaN()});
    device_fence();
  }
  bool rejected_non_finite = false;
  try {
    (void)plan.execute(0, layout, spacing, 7);
  } catch (const std::runtime_error&) {
    rejected_non_finite = true;
  }
  require(all_reduce_min(rejected_non_finite ? 1L : 0L) == 1L);
  require(static_cast<long>(accepted.refine.count()) == local_count);
}

template <int Dim, class Require>
void exercise_replicated_consensus(Require&& require) {
  const auto boxes = patches<Dim>();
  const auto ranks = rank_space<Dim>();
  const auto distribution = mesh::Distribution<Dim>::replicated(boxes, ranks);
  const amr::hierarchy::LevelLayout<Dim> layout(0, domain<Dim>(), boxes, distribution,
                                                amr::RefinementRatio<Dim>{},
                                                mesh::BoxArrayValidationBudget{2, 1});
  const Index<Dim> local_rank = rank_coordinate<Dim>(my_rank());
  Extent<Dim> ghosts{};
  MultiFab<Dim> state(boxes, distribution, local_rank, 1, ghosts);
  state.set_val(Real(0));
  if (my_rank() == 1) {
    const Index<Dim> divergent = state.box(0).lo;
    for_each_cell(Box<Dim>{divergent, divergent}, SetSample<Dim>{state.fab(0).view(), Real(3)});
    device_fence();
  }

  using Field =
      runtime::amr::PreparedTaggingField<Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  using Plan = runtime::amr::PreparedTaggingExecutionPlan<Dim>;
  const std::vector<std::vector<Field>> fields{{{"state/U", &state}}};
  const std::vector<amr::hierarchy::LevelLayout<Dim>> layouts{layout};
  const std::size_t local_cells = static_cast<std::size_t>(domain<Dim>().numPts());
  const std::vector<runtime::amr::PreparedTaggingExecutionBudget> budgets{
      budget<Dim>(2, local_cells, true)};
  std::array<Real, Dim> spacing{};
  spacing.fill(Real(1));
  auto plan = Plan::prepare(tagging_program<Dim>("test.mpi.prepared-tagging.replica"), fields,
                            layouts, budgets, 9);
  bool rejected = false;
  try {
    (void)plan.execute(0, layout, spacing, 9);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(all_reduce_min(rejected ? 1L : 0L) == 1L);
}

int run_prepared_tagging_execution(int argc, char** argv) {
  comm_init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);
  long failures = n_ranks() == 2 ? 0 : 1;
  const auto require = [&failures](bool condition, const std::source_location where =
                                                       std::source_location::current()) {
    if (!condition) {
      std::cerr << "prepared tagging MPI check failed on rank " << my_rank() << " at "
                << where.file_name() << ':' << where.line() << '\n';
      ++failures;
    }
  };

  exercise_partitioned<1>(require);
  exercise_partitioned<2>(require);
  exercise_partitioned<3>(require);
  exercise_replicated_consensus<1>(require);
  exercise_replicated_consensus<2>(require);
  exercise_replicated_consensus<3>(require);

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_prepared_tagging_execution, ExactRankedCollectivesFailClosed) {
  EXPECT_EQ(pops::test::RunTestBody(&run_prepared_tagging_execution,
                                    "test_mpi_prepared_tagging_execution"),
            0);
}
