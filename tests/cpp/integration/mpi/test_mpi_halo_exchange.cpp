#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/system.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace pops;
using namespace pops::mesh;

namespace pops {

/// A generic conservation-law package with identically zero transport.  It is an infrastructure
/// fixture, not a time-evolving advection case.
template <int Dim>
struct HaloZeroFluxScalar {
  using Law = nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-halo-exchange.zero-flux-scalar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(Real{0});
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real{0}; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }

 private:
  Law law = Law::prepare(RealVector<Dim>{});
};

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr Real kGhost = Real{-777};

template <int Dim>
struct HaloCase {
  Box<Dim> domain{};
  BoxArray<Dim> layout{};
  Distribution<Dim> distribution{};
  Extent<Dim> ghosts{};
  BoundaryTopology<Dim> topology{};
};

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> coordinate{};
  coordinate[0] = rank;
  return coordinate;
}

template <int Dim>
HaloCase<Dim> make_case(int ranks, bool replicated) {
  constexpr int boxes_per_rank = 2;
  Index<Dim> lower{};
  Index<Dim> upper{};
  upper[0] = ranks * boxes_per_rank * 2 - 1;
  for (int axis = 1; axis < Dim; ++axis)
    upper[axis] = 2;
  const Box<Dim> domain{lower, upper};

  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  boxes.reserve(static_cast<std::size_t>(ranks * boxes_per_rank));
  owners.reserve(static_cast<std::size_t>(ranks * boxes_per_rank));
  for (int box = 0; box < ranks * boxes_per_rank; ++box) {
    Index<Dim> box_lower = lower;
    Index<Dim> box_upper = upper;
    box_lower[0] = 2 * box;
    box_upper[0] = 2 * box + 1;
    boxes.push_back(Box<Dim>{box_lower, box_upper});
    owners.push_back(rank_coordinate<Dim>(box / boxes_per_rank));
  }
  const BoxArray<Dim> layout(std::move(boxes));

  Extent<Dim> rank_extent{};
  rank_extent[0] = ranks;
  for (int axis = 1; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution =
      replicated ? Distribution<Dim>::replicated(layout, rank_space)
                 : Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));

  Extent<Dim> ghosts{};
  ghosts[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    ghosts[axis] = 1;
  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  return HaloCase<Dim>{domain, layout, distribution, ghosts,
                       BoundaryTopology<Dim>::axis_periodic(periodic)};
}

template <int Dim>
HaloScheduleBudget budget(std::size_t boxes) {
  constexpr std::size_t images = 64;
  return HaloScheduleBudget{{boxes, boxes * (boxes - 1) / 2},
                            boxes * boxes * images,
                            boxes * boxes * images * static_cast<std::size_t>(2 * Dim),
                            images,
                            boxes,
                            1'000'000,
                            1'000'000,
                            1'000'000};
}

template <int Dim>
Index<Dim> index_from_cell(const Box<Dim>& box, std::size_t cell) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(cell % extent);
    cell /= extent;
  }
  return index;
}

template <int Dim>
Real encoded_value(const Index<Dim>& index, int component, Real bias) {
  Real value = bias + static_cast<Real>(component * 10'000);
  Real scale = Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    value += scale * static_cast<Real>(index[axis]);
    scale *= Real{97};
  }
  return value;
}

template <int Dim>
void fill_valid(MultiFab<Dim>& fields, Real bias) {
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        host(static_cast<std::size_t>(component) * cells + cell) =
            fab.box().contains(index) ? encoded_value(index, component, bias) : kGhost;
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void overwrite_valid(MultiFab<Dim>& fields, Real bias) {
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        if (fab.box().contains(index))
          host(static_cast<std::size_t>(component) * cells + cell) =
              encoded_value(index, component, bias);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
std::vector<Real> snapshot(const MultiFab<Dim>& fields) {
  std::vector<Real> result;
  for (const std::size_t global_box : fields.local_global_indices()) {
    const auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      result.push_back(host(element));
  }
  return result;
}

template <int Dim>
Real value_at(const MultiFab<Dim>& fields, std::size_t global_box, const Index<Dim>& index,
              int component) {
  const auto& fab = fields.fab_global(global_box);
  const Box<Dim>& grown = fab.grown_box();
  std::size_t stride = 1;
  std::size_t cell = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(static_cast<std::size_t>(component) * stride + cell);
}

template <int Dim>
void expect_job_published(const HaloSchedule<Dim>& schedule, const MultiFab<Dim>& fields,
                          const HaloJob<Dim>& job, Real bias) {
  for (int component = 0; component < schedule.ncomp(); ++component)
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(job.destination_region.numPts());
         ++cell) {
      const Index<Dim> destination = index_from_cell(job.destination_region, cell);
      Index<Dim> source = destination;
      for (int axis = 0; axis < Dim; ++axis)
        source[axis] += job.source_from_destination[axis];
      EXPECT_EQ(value_at(fields, job.destination_box, destination, component),
                encoded_value(source, component, bias));
    }
}

template <int Dim>
void expect_schedule_published(const HaloSchedule<Dim>& schedule, const MultiFab<Dim>& fields,
                               Real bias) {
  for (const HaloJob<Dim>& job : schedule.local_jobs())
    expect_job_published(schedule, fields, job, bias);
  for (const HaloPeerPlan<Dim>& plan : schedule.receive_plans())
    for (const HaloJob<Dim>& job : plan.jobs)
      expect_job_published(schedule, fields, job, bias);
}

template <int Dim>
void expect_plan_witnesses(const HaloSchedule<Dim>& schedule, const ExecutionLane& lane) {
  bool local = !schedule.local_jobs().empty();
  bool multi_job_receive = false;
  bool nonzero_offset = false;
  bool periodic_remote = false;
  for (const HaloPeerPlan<Dim>& plan : schedule.receive_plans()) {
    multi_job_receive = multi_job_receive || plan.jobs.size() > 1;
    for (const HaloJob<Dim>& job : plan.jobs) {
      nonzero_offset = nonzero_offset || job.offset != 0;
      for (int axis = 0; axis < Dim; ++axis)
        periodic_remote = periodic_remote || job.source_from_destination[axis] != 0;
    }
  }
  EXPECT_EQ(all_reduce_max(local ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(multi_job_receive ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(nonzero_offset ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(periodic_remote ? 1L : 0L, lane.communicator()), 1L);
}

template <int Dim>
void expect_success(int ranks, int rank, bool replicated, const ExecutionLane& lane,
                    std::uint64_t generation) {
  const HaloCase<Dim> definition = make_case<Dim>(ranks, replicated);
  MultiFab<Dim> fields(definition.layout, definition.distribution, rank_coordinate<Dim>(rank), 3,
                       definition.ghosts);
  fill_valid(fields, Real{0});
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<Dim>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  HaloExchange<Dim> exchange(schedule, lane, context);

  if (!replicated && ranks >= 2) {
    EXPECT_TRUE(schedule.has_remote_jobs());
    expect_plan_witnesses(schedule, lane);
  }
  const std::vector<Real> before = snapshot(fields);
  exchange.begin(fields, lane);
  EXPECT_TRUE(exchange.in_flight());
  EXPECT_EQ(snapshot(fields), before);
  overwrite_valid(fields, Real{500'000});
  EXPECT_EQ(all_reduce_max(0L, lane.communicator()), 0L);
  exchange.complete(fields, lane);
  EXPECT_FALSE(exchange.in_flight());
  EXPECT_FALSE(exchange.sealed());
  EXPECT_EQ(exchange.live_request_count(), 0U);
  expect_schedule_published(schedule, fields, Real{0});

  fill_valid(fields, Real{1'000'000});
  const std::vector<Real> second_before = snapshot(fields);
  fill_boundary(fields, exchange, lane);
  EXPECT_NE(snapshot(fields), second_before);
  expect_schedule_published(schedule, fields, Real{1'000'000});
}

template <int Dim>
void expect_one_shot(int ranks, int rank, const ExecutionLane& lane, std::uint64_t generation) {
  const HaloCase<Dim> definition = make_case<Dim>(ranks, false);
  MultiFab<Dim> fields(definition.layout, definition.distribution, rank_coordinate<Dim>(rank), 2,
                       definition.ghosts);
  fill_valid(fields, Real{2'000'000});
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<Dim>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  fill_boundary(fields, definition.domain, definition.topology,
                budget<Dim>(definition.layout.size()), lane, context);
  expect_schedule_published(schedule, fields, Real{2'000'000});
}

/// Regression for the underlying periodic transport session: one 3D Cartesian rank per octant,
/// no physical boundary and no time integration.  The prepared session must publish face, edge and
/// corner periodic ghosts through its owned HaloExchange on every rank.
void expect_periodic_cartesian_3d_transport(const ExecutionLane& lane) {
  constexpr int Dim = 3;
  constexpr int cells_per_axis = 32;
  constexpr int ranks_per_axis = 2;
  const Extent<Dim> rank_extent{ranks_per_axis, ranks_per_axis, ranks_per_axis};
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  ASSERT_EQ(lane.size(), static_cast<int>(rank_space.size()));

  const Box<Dim> domain = Box<Dim>::from_extents(
      Extent<Dim>{cells_per_axis, cells_per_axis, cells_per_axis});
  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  boxes.reserve(rank_space.size());
  owners.reserve(rank_space.size());
  for (std::size_t linear = 0; linear < rank_space.size(); ++linear) {
    const Index<Dim> owner = rank_space.coordinate(linear);
    Index<Dim> lower{};
    Index<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = owner[axis] * cells_per_axis / ranks_per_axis;
      upper[axis] = (owner[axis] + 1) * cells_per_axis / ranks_per_axis - 1;
    }
    boxes.emplace_back(lower, upper);
    owners.push_back(owner);
  }
  const BoxArray<Dim> layout(std::move(boxes));
  const Distribution<Dim> distribution =
      Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  const Index<Dim> local_rank = rank_space.coordinate(static_cast<std::size_t>(lane.rank()));
  const Extent<Dim> ghosts{1, 1, 1};
  MultiFab<Dim> fields(layout, distribution, local_rank, 1, ghosts);
  fill_valid(fields, Real{0});

  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real{1};
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, lower, upper);
  const auto session = runtime::program::PreparedScalarBoundarySession<Dim>::prepare_block(
      geometry, BoundaryTopology<Dim>::axis_periodic(periodic), fields, lane, 1701);
  ASSERT_TRUE(session);
  session->fill(fields);

  for (const std::size_t global_box : fields.local_global_indices()) {
    const Fab<Dim>& fab = fields.fab_global(global_box);
    const Box<Dim>& grown = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(grown.numPts()); ++ordinal) {
      const Index<Dim> destination = index_from_cell(grown, ordinal);
      if (fab.box().contains(destination))
        continue;
      Index<Dim> source = destination;
      for (int axis = 0; axis < Dim; ++axis) {
        source[axis] %= cells_per_axis;
        if (source[axis] < 0)
          source[axis] += cells_per_axis;
      }
      EXPECT_EQ(value_at(fields, global_box, destination, 0), encoded_value(source, 0, Real{0}));
    }
  }
}

/// Regression for the Uniform Program/System route.  The test deliberately uses a zero-flux
/// scalar contract: it performs one RHS evaluation only, never advances a numerical solution.
/// The eight spatial boxes form a 2x2x2 tiling; the companion transport assertion above inspects
/// its face, edge and corner ghosts.  The manual closures make the legacy local-only route fail
/// deterministically; the same test then exercises the real generated package with ZeroFluxScalar.
template <int Dim>
void expect_periodic_program_rhs_transport(const ExecutionLane& lane) {
  static_assert(Dim == 3);
  constexpr int cells_per_axis = 32;
  constexpr int ranks_per_axis = 2;
  const Extent<Dim> octants{ranks_per_axis, ranks_per_axis, ranks_per_axis};
  ASSERT_EQ(lane.size(), static_cast<int>(ranks_per_axis * ranks_per_axis * ranks_per_axis));

  SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = Real{0};
    config.upper[axis] = Real{1};
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  for (int z = 0; z < ranks_per_axis; ++z)
    for (int y = 0; y < ranks_per_axis; ++y)
      for (int x = 0; x < ranks_per_axis; ++x) {
        Index<Dim> lower{};
        Index<Dim> upper{};
        lower[0] = x * cells_per_axis / ranks_per_axis;
        lower[1] = y * cells_per_axis / ranks_per_axis;
        lower[2] = z * cells_per_axis / ranks_per_axis;
        upper[0] = (x + 1) * cells_per_axis / ranks_per_axis - 1;
        upper[1] = (y + 1) * cells_per_axis / ranks_per_axis - 1;
        upper[2] = (z + 1) * cells_per_axis / ranks_per_axis - 1;
        config.boxes.emplace_back(lower, upper);
      }

  System<Dim> system(config);
  system.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "test.mpi-halo-exchange.program-periodic-route@1")));
  system.install_block_state_route("u", "test.mpi-halo-exchange.state.u@1");

  const Box<Dim> domain = config.index_domain();
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, config.lower, config.upper);
  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(periodic);
  const auto local_only_rhs = [geometry, topology](const auto&, MultiFab<Dim>& state,
                                                    MultiFab<Dim>& residual) {
    const HaloSchedule<Dim> schedule =
        prepare_halo_schedule(state, geometry.domain(), topology, budget<Dim>(state.layout().size()));
    fill_boundary(state, schedule);  // Must never be selected for distributed periodic Program RHS.
    residual.set_val(Real{0});
  };
  const auto transport_calls = std::make_shared<int>(0);
  const auto prepare_calls = std::make_shared<int>(0);
  const auto transport_rhs = [transport_calls](
                                 const auto&, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                                 const ExecutionLane&,
                                 const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
    transport.fill(state);
    residual.set_val(Real{0});
    ++*transport_calls;
  };

  PreparedSystemBlock<Dim> prepared;
  prepared.name = "u";
  prepared.provider_identity = "test.mpi-halo-exchange.zero-flux-scalar@1";
  prepared.ncomp = 1;
  prepared.conservative_variables =
      {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  prepared.primitive_variables = {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  prepared.gamma = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    prepared.ghosts[axis] = 1;
  const auto zero = [](MultiFab<Dim>&, MultiFab<Dim>& residual) { residual.set_val(Real{0}); };
  prepared.closures.rhs_into = zero;
  prepared.closures.rhs_flux_only = zero;
  prepared.closures.source_only = zero;
  prepared.closures.source_only_masked = zero;
  prepared.closures.rhs_at_point = local_only_rhs;
  prepared.closures.rhs_flux_only_at_point = local_only_rhs;
  prepared.closures.rhs_without_prepared_interfaces = local_only_rhs;
  prepared.closures.rhs_flux_only_without_prepared_interfaces = local_only_rhs;
  prepared.closures.rhs_core_at_point = local_only_rhs;
  prepared.closures.rhs_flux_only_core_at_point = local_only_rhs;
  prepared.closures.rhs_core_at_point_prepared =
      [](const auto&, MultiFab<Dim>&, MultiFab<Dim>& residual, const auto&) {
        residual.set_val(Real{0});
      };
  prepared.closures.rhs_flux_only_core_at_point_prepared =
      prepared.closures.rhs_core_at_point_prepared;
  prepared.closures.transport_rhs_at_point_prepared = transport_rhs;
  prepared.closures.transport_flux_at_point_prepared = transport_rhs;
  prepared.closures.prepare_generated_state_at_point = [](const auto&, MultiFab<Dim>&) {};
  prepared.closures.prepare_generated_state_at_point_prepared =
      [](const auto&, MultiFab<Dim>&, const auto&) {};
  prepared.closures.prepare_generated_state_with_transport_prepared =
      [](const auto&, MultiFab<Dim>&, const auto&, const ExecutionLane&, const auto&) {};
  prepared.closures.transport_prepare_generated_state_at_point_prepared =
      [prepare_calls](const auto&, MultiFab<Dim>& state, const ExecutionLane&,
         const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        transport.fill(state);
        ++*prepare_calls;
      };
  prepared.closures.external_ghost_boundary =
      std::make_shared<typename SystemBlockClosures<Dim>::ExternalGhostBoundary>(
          [](const auto&, MultiFab<Dim>&, const auto&, const ExecutionLane&) {});
  prepared.maximum_speed = [](const MultiFab<Dim>&, const ExecutionLane&) { return Real{0}; };
  prepared.poisson_rhs = [](const MultiFab<Dim>&, MultiFab<Dim>& rhs) { rhs.set_val(Real{0}); };
  prepared.primitive_to_conservative = [](const double* primitive, double* conservative) {
    conservative[0] = primitive[0];
  };
  prepared.conservative_to_primitive = [](const double* conservative, double* primitive) {
    primitive[0] = conservative[0];
    RecoveryReport report;
    report.status = RecoveryStatus::kRecovered;
    report.attempted_methods = 1;
    return report;
  };
  prepared.batch_conservative_to_primitive = [](const std::vector<double>& conservative,
                                                std::vector<double>& primitive) {
    primitive = conservative;
    UniformRecoveryBatchReport report;
    report.recovery.status = RecoveryStatus::kRecovered;
    report.recovery.attempted_methods = 1;
    report.cell_count = conservative.size();
    report.recovered_cells = conservative.size();
    report.published = true;
    return report;
  };
  system.install_prepared_block(std::move(prepared));
  system.set_program_block_map({0});

  MultiFab<Dim>& state = system.block_state(0);
  ASSERT_EQ(state.layout().size(), static_cast<std::size_t>(octants[0] * octants[1] * octants[2]));
  ASSERT_EQ(state.local_size(), 1U);
  fill_valid(state, Real{0});
  MultiFab<Dim> rhs(state.layout(), state.distribution(), state.local_rank(), state.ncomp(),
                    state.ghosts());
  runtime::program::ProgramContext<Dim> context(&system);
  context.configure_primary_clock("test.mpi-halo-exchange.clock@1");
  context.begin_step(Real{0.125});
  context.set_stage_time(0, 1);
  EXPECT_TRUE(system.requires_block_boundary_session(0));
  const std::vector<Real> state_before = snapshot(state);
  EXPECT_NO_THROW(context.rhs_into(0, state, rhs, 0));
  EXPECT_EQ(*transport_calls, 1);
  EXPECT_EQ(snapshot(state), state_before);
  const std::vector<Real> rhs_values = snapshot(rhs);
  EXPECT_TRUE(std::all_of(rhs_values.begin(), rhs_values.end(),
                          [](Real value) { return value == Real{0}; }));
  EXPECT_NO_THROW(context.neg_div_flux_default_into(0, state, rhs, 0));
  EXPECT_EQ(*transport_calls, 2);
  const std::vector<Real> flux_values = snapshot(rhs);
  EXPECT_TRUE(std::all_of(flux_values.begin(), flux_values.end(),
                          [](Real value) { return value == Real{0}; }));
  EXPECT_NO_THROW(context.prepare_generated_state(0, state, 0));
  EXPECT_EQ(*prepare_calls, 1);
  EXPECT_THROW(system.block_rhs_group(runtime::multiblock::BoundaryEvaluationPoint{}, {0}, {&state},
                                      {&rhs}, {0}),
               std::runtime_error);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(system.time(), 0.0);

  // Exercise the actual generated-system package wiring as well as the public ProgramContext
  // route above.  This has no time advance and a zero velocity on every axis.
  System<Dim> generated(config);
  generated.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "test.mpi-halo-exchange.generated-periodic-route@1")));
  add_compiled_model(generated, "u", HaloZeroFluxScalar<Dim>{}, "none", "rusanov",
                     "conservative", "explicit");
  generated.set_program_block_map({0});
  MultiFab<Dim>& generated_state = generated.block_state(0);
  fill_valid(generated_state, Real{0});
  MultiFab<Dim> generated_rhs(generated_state.layout(), generated_state.distribution(),
                              generated_state.local_rank(), generated_state.ncomp(),
                              generated_state.ghosts());
  runtime::program::ProgramContext<Dim> generated_context(&generated);
  generated_context.configure_primary_clock("test.mpi-halo-exchange.generated-clock@1");
  generated_context.begin_step(Real{0.125});
  generated_context.set_stage_time(0, 1);
  EXPECT_TRUE(generated.requires_block_boundary_session(0));
  EXPECT_NO_THROW(generated_context.rhs_into(0, generated_state, generated_rhs, 0));
  const std::vector<Real> generated_values = snapshot(generated_rhs);
  EXPECT_TRUE(std::all_of(generated_values.begin(), generated_values.end(),
                          [](Real value) { return value == Real{0}; }));
  EXPECT_EQ(generated.macro_step(), 0);
  EXPECT_EQ(generated.time(), 0.0);
  const std::vector<Real> generated_before_step = snapshot(generated_state);
  generated_context.install([&generated_context](double dt) {
    generated_context.begin_step(dt);
    generated_context.set_stage_time(0, 1);
    MultiFab<Dim>& live = generated_context.state(0);
    MultiFab<Dim> rate = generated_context.rhs_scratch_like(live);
    generated_context.rhs_into(0, live, rate, 0);
    generated_context.axpy(live, Real(dt), rate);
  });
  generated.set_program_block_map({0});
  EXPECT_NO_THROW(generated.step(0.125));
  EXPECT_EQ(snapshot(generated_state), generated_before_step);
  EXPECT_EQ(generated.macro_step(), 1);
  EXPECT_EQ(generated.time(), 0.125);
}

void expect_collective_constructor_failures(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));

  HaloExchangeContext mismatch{};
  mismatch.context_generation = static_cast<std::uint64_t>(rank == 0 ? 701 : 703);
  mismatch.schedule_generation = 709;
  bool mismatch_threw = false;
  try {
    HaloExchange<1> exchange(schedule, lane, mismatch);
    (void)exchange;
  } catch (const std::exception&) {
    mismatch_threw = true;
  }
  EXPECT_EQ(all_reduce_max(mismatch_threw ? 0L : 1L, lane.communicator()), 0L);

  HaloExchangeContext allocation{};
  allocation.context_generation = 719;
  allocation.schedule_generation = 727;
  allocation.fail_allocation_rank = 0;
  bool allocation_threw = false;
  try {
    HaloExchange<1> exchange(schedule, lane, allocation);
    (void)exchange;
  } catch (const std::exception&) {
    allocation_threw = true;
  }
  EXPECT_EQ(all_reduce_max(allocation_threw ? 0L : 1L, lane.communicator()), 0L);
}

void expect_sealed_failure(int ranks, int rank, const ExecutionLane& lane,
                           HaloExchangeDiagnosticStage stage, std::uint64_t generation) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  fill_valid(fields, Real{0});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  if (stage == HaloExchangeDiagnosticStage::receive_post)
    context.fail_receive_post_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::send_post)
    context.fail_send_post_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::wait)
    context.fail_wait_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::staging)
    context.fail_staging_rank = 0;
  else
    throw std::invalid_argument("unsupported injected halo failure stage");

  HaloExchange<1> exchange(schedule, lane, context);
  const std::vector<Real> before = snapshot(fields);
  bool threw = false;
  try {
    exchange.begin(fields, lane);
    EXPECT_EQ(snapshot(fields), before);
    exchange.complete(fields, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), stage);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(fields), before);

  bool reuse_threw = false;
  try {
    exchange.begin(fields, lane);
  } catch (const std::exception&) {
    reuse_threw = true;
  }
  EXPECT_EQ(all_reduce_max(reuse_threw ? 0L : 1L, lane.communicator()), 0L);
}

void expect_complete_requires_exact_field(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> packed(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  MultiFab<1> impostor(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                       definition.ghosts);
  fill_valid(packed, Real{0});
  fill_valid(impostor, Real{10'000'000});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      packed, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = 809;
  context.schedule_generation = 811;
  HaloExchange<1> exchange(schedule, lane, context);
  const std::vector<Real> packed_before = snapshot(packed);
  const std::vector<Real> impostor_before = snapshot(impostor);
  exchange.begin(packed, lane);
  bool threw = false;
  try {
    exchange.complete(impostor, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), HaloExchangeDiagnosticStage::binding);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(packed), packed_before);
  EXPECT_EQ(snapshot(impostor), impostor_before);
}

void expect_complete_rejects_rebound_storage(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  MultiFab<1> replacement(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                          definition.ghosts);
  fill_valid(fields, Real{0});
  fill_valid(replacement, Real{20'000'000});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = 821;
  context.schedule_generation = 823;
  HaloExchange<1> exchange(schedule, lane, context);
  exchange.begin(fields, lane);
  fields = std::move(replacement);
  const std::vector<Real> rebound_before = snapshot(fields);
  bool threw = false;
  try {
    exchange.complete(fields, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), HaloExchangeDiagnosticStage::binding);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(fields), rebound_before);
}

int run_mpi_halo_exchange(int argc, char** argv) {
  comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const int rank = my_rank();
    const int ranks = n_ranks();
    auto lane = ExecutionLane::duplicate_world_collectively("production-halo-exchange");

    expect_success<1>(ranks, rank, false, lane, 101);
    expect_success<2>(ranks, rank, false, lane, 211);
    expect_success<3>(ranks, rank, false, lane, 307);
    expect_success<1>(ranks, rank, true, lane, 401);
    expect_success<2>(ranks, rank, true, lane, 503);
    expect_success<3>(ranks, rank, true, lane, 601);
    expect_one_shot<2>(ranks, rank, lane, 659);

    if (ranks == 8) {
      expect_periodic_cartesian_3d_transport(lane);
      if constexpr (kNativeDimension == 3)
        expect_periodic_program_rhs_transport<3>(lane);
    }

    if (ranks >= 2) {
      expect_collective_constructor_failures(ranks, rank, lane);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::receive_post, 907);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::send_post, 919);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::wait, 929);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::staging, 937);
      expect_complete_requires_exact_field(ranks, rank, lane);
      expect_complete_rejects_rebound_storage(ranks, rank, lane);
    }
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_halo_exchange, RunsExactRankedProductionMatrix) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_halo_exchange, "test_mpi_halo_exchange"), 0);
}
