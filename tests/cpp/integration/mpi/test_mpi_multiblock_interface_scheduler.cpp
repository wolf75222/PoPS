#include <gtest/gtest.h>

#include "gtest_compat.hpp"

#include <pops/parallel/comm.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using namespace pops::runtime::multiblock;

namespace {

constexpr int Dim = 3;
using Field = MultiFab<Dim>;
using Route = AxisAlignedInterface<Dim>;
using Scheduler = InterfaceFluxScheduler<Dim>;

Extent<Dim> rank_extent(int ranks) {
  Extent<Dim> result{};
  result[0] = ranks;
  result[1] = 1;
  result[2] = 1;
  return result;
}

PopsExecutionContextV1 mpi_execution() {
  return {sizeof(PopsExecutionContextV1),
          1u,
          "test::mpi-nd-interface",
          POPS_MEMORY_SPACE_HOST_V1,
          "pops.runtime-backend-manifest.v1:sha256:mpi-nd-interface",
          "host",
          POPS_SCALAR_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          0,
          "host::synchronous",
          static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD)),
          static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)),
          "MPI_COMM_WORLD",
          "MPI_DOUBLE"};
}

Field make_field(const std::vector<Box<Dim>>& boxes, int components, int owner_shift = 0) {
  mesh::BoxArray<Dim> layout(boxes);
  mesh::RankSpace<Dim> ranks(Index<Dim>{}, rank_extent(n_ranks()));
  std::vector<Index<Dim>> owners;
  owners.reserve(boxes.size());
  for (std::size_t box = 0; box < boxes.size(); ++box)
    owners.push_back(ranks.coordinate((box + static_cast<std::size_t>(owner_shift)) %
                                      static_cast<std::size_t>(n_ranks())));
  auto distribution = mesh::Distribution<Dim>::partitioned(layout, ranks, std::move(owners));
  return Field(std::move(layout), std::move(distribution), ranks.coordinate(my_rank()), components,
               Extent<Dim>{});
}

Geometry<Dim> geometry(const Box<Dim>& domain, Real xlo, Real xhi) {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  lower[0] = xlo;
  upper[0] = xhi;
  lower[1] = Real(0);
  upper[1] = Real(4);
  lower[2] = Real(0);
  upper[2] = Real(2);
  return Geometry<Dim>::from_bounds(domain, lower, upper);
}

std::size_t host_offset(const Fab<Dim>& fab, const Index<Dim>& index, int component) {
  const Box<Dim>& grown = fab.grown_box();
  std::size_t stride = 1;
  std::size_t offset = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return offset + static_cast<std::size_t>(component) * stride;
}

void set_cell(Field& field, const Index<Dim>& index, int component, Real value) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    if (!fab.box().contains(index))
      continue;
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    host(host_offset(fab, index, component)) = value;
    fab.copy_from_host(host);
    return;
  }
}

Real get_cell(const Field& field, const Index<Dim>& index, int component) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    if (!fab.box().contains(index))
      continue;
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    return host(host_offset(fab, index, component));
  }
  throw std::out_of_range("MPI interface test cell is not local");
}

bool local_field_is_zero(const Field& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t value = 0; value < host.size(); ++value)
      if (host(value) != Real(0))
        return false;
  }
  return true;
}

void authenticate(Route& route) {
  route.left_trace_projection_identity = route.identity + ".left.trace";
  route.right_trace_projection_identity = route.identity + ".right.trace";
  route.left_trace_provider_identity = "test.mpi.left";
  route.right_trace_provider_identity = "test.mpi.right";
  route.left_trace_operation = InterfaceTraceOperation::CellAverage;
  route.right_trace_operation = InterfaceTraceOperation::CellAverage;
  route.left_trace_required_depth = 1;
  route.right_trace_required_depth = 1;
}

int run_mpi_interface_scheduler() {
  static Kokkos::ScopeGuard guard;
  comm_init();
  int failures = 0;
  try {
    if (n_ranks() != 2)
      throw std::runtime_error("MPI ND interface proof requires exactly two ranks");
    const Box<Dim> left_domain(Index<Dim>(0, 0, 0), Index<Dim>(1, 3, 1));
    const Box<Dim> right_domain(Index<Dim>(5, 0, 0), Index<Dim>(6, 3, 1));
    const std::vector<Box<Dim>> left_boxes{Box<Dim>(Index<Dim>(0, 0, 0), Index<Dim>(1, 1, 1)),
                                           Box<Dim>(Index<Dim>(0, 2, 0), Index<Dim>(1, 3, 1))};
    const std::vector<Box<Dim>> right_boxes{Box<Dim>(Index<Dim>(5, 0, 0), Index<Dim>(6, 1, 1)),
                                            Box<Dim>(Index<Dim>(5, 2, 0), Index<Dim>(6, 3, 1))};
    Field left = make_field(left_boxes, 2);
    Field right = make_field(right_boxes, 2);
    Field left_rhs = make_field(left_boxes, 2);
    Field right_rhs = make_field(right_boxes, 2);
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 4; ++y) {
        const int face = y + 4 * z;
        set_cell(left, Index<Dim>(1, y, z), 0, Real(100 + face));
        set_cell(left, Index<Dim>(1, y, z), 1, Real(200 + face));
        set_cell(right, Index<Dim>(5, y, z), 1, Real(300 + face));
        set_cell(right, Index<Dim>(5, y, z), 0, Real(400 + face));
      }

    Route route;
    route.identity = "mpi.nd.interface";
    route.left_block = 0;
    route.right_block = 1;
    route.left_axis = route.right_axis = 0;
    route.left_side = InterfaceSide::High;
    route.right_side = InterfaceSide::Low;
    route.right_component_for_left = {1, 0};
    authenticate(route);
    Scheduler scheduler;
    int calls = 0;
    scheduler.install(route, left, geometry(left_domain, Real(0), Real(1)), right,
                      geometry(right_domain, Real(1), Real(2)), mpi_execution(),
                      [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                        ++calls;
                        if (batch.face_count != 8 || batch.component_count != 2)
                          throw std::runtime_error("MPI ND interface batch shape is invalid");
                        for (int face = 0; face < batch.face_count; ++face) {
                          if (batch.left_state[2 * face] != Real(100 + face) ||
                              batch.right_state[2 * face] != Real(300 + face) ||
                              batch.left_state[2 * face + 1] != Real(200 + face) ||
                              batch.right_state[2 * face + 1] != Real(400 + face))
                            throw std::runtime_error("MPI ND interface trace is incomplete");
                          batch.shared_flux[2 * face] = Real(face + 1);
                          batch.shared_flux[2 * face + 1] = Real(face + 11);
                        }
                      });
    const BoundaryEvaluationPoint point{"mpi.nd.clock", 1, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
    scheduler.apply(point, std::vector<Field*>{&left, &right},
                    std::vector<Field*>{&left_rhs, &right_rhs});
    if (calls != 1 || scheduler.evaluation_count(route.identity, 0) != 1)
      ++failures;
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 4; ++y) {
        const int face = y + 4 * z;
        const bool local = left.local_index_of(y < 2 ? 0u : 1u) != Field::not_local;
        if (!local)
          continue;
        if (get_cell(left_rhs, Index<Dim>(1, y, z), 0) != Real(-2 * (face + 1)) ||
            get_cell(right_rhs, Index<Dim>(5, y, z), 1) != Real(2 * (face + 1)) ||
            get_cell(left_rhs, Index<Dim>(1, y, z), 1) != Real(-2 * (face + 11)) ||
            get_cell(right_rhs, Index<Dim>(5, y, z), 0) != Real(2 * (face + 11)))
          ++failures;
      }

    left_rhs.set_val(Real(0));
    right_rhs.set_val(Real(0));
    BoundaryEvaluationPoint divergent_point = point;
    divergent_point.tick += my_rank();
    bool point_rejected = false;
    try {
      scheduler.apply(divergent_point, std::vector<Field*>{&left, &right},
                      std::vector<Field*>{&left_rhs, &right_rhs});
    } catch (const std::runtime_error& error) {
      point_rejected =
          std::string(error.what()).find("BoundaryEvaluationPoint differs") != std::string::npos;
    }
    if (!point_rejected || calls != 1 || !local_field_is_zero(left_rhs) ||
        !local_field_is_zero(right_rhs))
      ++failures;

    const std::vector<Field*> divergent_states =
        my_rank() == 0 ? std::vector<Field*>{&left, &right} : std::vector<Field*>{nullptr, nullptr};
    const std::vector<Field*> divergent_rhs = my_rank() == 0
                                                  ? std::vector<Field*>{&left_rhs, &right_rhs}
                                                  : std::vector<Field*>{nullptr, nullptr};
    bool active_rejected = false;
    try {
      scheduler.apply(point, divergent_states, divergent_rhs);
    } catch (const std::runtime_error& error) {
      active_rejected = std::string(error.what()).find("active mask differs") != std::string::npos;
    }
    if (!active_rejected || calls != 1 || !local_field_is_zero(left_rhs) ||
        !local_field_is_zero(right_rhs))
      ++failures;

    Scheduler divergent;
    divergent.install(route, left, geometry(left_domain, Real(0), Real(1)), right,
                      geometry(right_domain, Real(1), Real(2)), mpi_execution(),
                      [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                        for (int value = 0; value < batch.face_count * batch.component_count;
                             ++value)
                          batch.shared_flux[value] = Real(value + my_rank());
                      });
    bool rejected = false;
    try {
      divergent.apply(point, std::vector<Field*>{&left, &right},
                      std::vector<Field*>{&left_rhs, &right_rhs});
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("rank-dependent shared flux") != std::string::npos;
    }
    if (!rejected || !local_field_is_zero(left_rhs) || !local_field_is_zero(right_rhs) ||
        divergent.evaluation_count(route.identity, 0) != 0)
      ++failures;

    // A rank-local evaluator exception is already collective for every admitted MPI route,
    // with unique patch ownership; omitting sample capture must preserve failure/retry behavior.
    auto failure_left = make_field(left_boxes, 2);
    auto failure_right = make_field(right_boxes, 2);
    auto failure_left_rhs = make_field(left_boxes, 2);
    auto failure_right_rhs = make_field(right_boxes, 2);
    failure_left.set_val(Real(1));
    failure_right.set_val(Real(2));
    failure_left_rhs.set_val(Real(0));
    failure_right_rhs.set_val(Real(0));
    Scheduler evaluator_failure;
    bool inject_failure = true;
    evaluator_failure.install(
        route, failure_left, geometry(left_domain, Real(0), Real(1)), failure_right,
        geometry(right_domain, Real(1), Real(2)), mpi_execution(),
        [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
          if (inject_failure && my_rank() == 1)
            throw std::runtime_error("injected evaluator failure");
          for (int value = 0; value < batch.face_count * batch.component_count; ++value)
            batch.shared_flux[value] = Real(0);
        });
    bool evaluator_rejected = false;
    try {
      evaluator_failure.apply(point, std::vector<Field*>{&failure_left, &failure_right},
                              std::vector<Field*>{&failure_left_rhs, &failure_right_rhs});
    } catch (const std::runtime_error& error) {
      evaluator_rejected =
          std::string(error.what()).find("evaluator failed on one or more ranks") !=
          std::string::npos;
    }
    if (all_reduce_sum(evaluator_rejected ? 1L : 0L) != 2 ||
        !local_field_is_zero(failure_left_rhs) || !local_field_is_zero(failure_right_rhs) ||
        evaluator_failure.evaluation_count(route.identity, 0) != 0)
      ++failures;
    inject_failure = false;
    evaluator_failure.apply(point, std::vector<Field*>{&failure_left, &failure_right},
                            std::vector<Field*>{&failure_left_rhs, &failure_right_rhs});
    if (evaluator_failure.evaluation_count(route.identity, 0) != 1 ||
        !local_field_is_zero(failure_left_rhs) || !local_field_is_zero(failure_right_rhs))
      ++failures;

    // Preserve one earned physical sample across an ownership-only transition, while
    // requiring a different exact live route on every rank after redistribution.
    auto sample_left = make_field(left_boxes, 2), sample_right = make_field(left_boxes, 2);
    auto sample_left_rhs = make_field(left_boxes, 2), sample_right_rhs = make_field(left_boxes, 2);
    sample_left.set_val(Real(1));
    sample_right.set_val(Real(2));
    sample_left_rhs.set_val(Real(0));
    sample_right_rhs.set_val(Real(0));
    auto sample_route = route;
    sample_route.identity = "mpi.retained.physical-sampling";
    sample_route.sampling_provider_identity = "test.artifact-v1/parameters-v1/endpoints-v1";
    sample_route.affine_mapping_identity = "test.periodic-normal-map";
    sample_route.right_normal_translation = Real(1);
    const auto sample_geometry = geometry(left_domain, Real(0), Real(1));
    const auto sample_evaluator = [](const BoundaryEvaluationPoint&,
                                     const InterfaceFluxBatch& batch) {
      for (int value = 0; value < batch.face_count * batch.component_count; ++value)
        batch.shared_flux[value] = Real(value + 1) / 8;
    };
    Scheduler sample_scheduler;
    sample_scheduler.install(sample_route, sample_left, sample_geometry, sample_right,
                             sample_geometry, mpi_execution(), sample_evaluator);
    auto sample_point = point;
    sample_point.graph_identity = "test.retained-program";
    sample_point.rate_identity = "shared-rhs/1";
    sample_point.application_identity = "program-rhs-group";
    std::vector<InterfaceFluxSample> captured;
    sample_scheduler.apply(sample_point, std::vector<Field*>{&sample_left, &sample_right},
                           std::vector<Field*>{&sample_left_rhs, &sample_right_rhs}, nullptr,
                           &captured);
    if (captured.size() != 1)
      throw std::runtime_error("shared capture did not return one physical sample");
    const auto retained = captured.front();
    const std::string old_live(sample_scheduler.authenticate_sample(retained));
    auto redistributed_left = make_field(left_boxes, 2, 1);
    auto redistributed_right = make_field(left_boxes, 2, 1);
    auto redistributed = sample_scheduler.rematerialized(
        1,
        [&](std::size_t block, int) -> Field& {
          return block == 0 ? redistributed_left : redistributed_right;
        },
        [&](int) { return sample_geometry; });
    const std::string new_live(redistributed.authenticate_sample(retained));
    if (old_live == new_live || captured.front().flux_density != retained.flux_density ||
        captured.front().route_contract != retained.route_contract)
      ++failures;
    Scheduler fresh_provider;
    fresh_provider.install(sample_route, redistributed_left, sample_geometry, redistributed_right,
                           sample_geometry, mpi_execution(), sample_evaluator);
    if (fresh_provider.authenticate_sample(retained) != new_live ||
        fresh_provider.evaluation_count(sample_route.identity, 0) != 0)
      ++failures;
    // Altering only the provider parameters must invalidate the retained physical sample.
    auto altered_route = sample_route;
    altered_route.sampling_provider_identity += "/other-parameters";
    Scheduler altered_provider;
    altered_provider.install(altered_route, redistributed_left, sample_geometry,
                             redistributed_right, sample_geometry, mpi_execution(),
                             sample_evaluator);
    bool altered_rejected = false;
    try {
      (void)altered_provider.authenticate_sample(retained);
    } catch (const std::invalid_argument&) {
      altered_rejected = true;
    }
    if (all_reduce_sum(altered_rejected ? 1L : 0L) != 2)
      ++failures;

    Scheduler factory_failure;
    int factory_calls = 0;
    bool factory_rejected = false;
    try {
      factory_failure.install(route, left, geometry(left_domain, Real(0), Real(1)), right,
                              geometry(right_domain, Real(1), Real(2)), mpi_execution(),
                              InterfaceFluxEvaluatorFactory([&]() -> InterfaceFluxEvaluator {
                                ++factory_calls;
                                if (my_rank() == 1)
                                  throw std::runtime_error("rank-local factory failure");
                                return [](const BoundaryEvaluationPoint&,
                                          const InterfaceFluxBatch&) {};
                              }));
    } catch (const std::runtime_error&) {
      factory_rejected = true;
    }
    if (!factory_rejected || factory_calls != 1 || factory_failure.size() != 0)
      ++failures;
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "rank " << my_rank() << ": " << error.what() << '\n';
  }
  failures = static_cast<int>(all_reduce_sum(static_cast<long>(failures)));
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_multiblock_interface_scheduler,
     ReconstructsExactThreeDimensionalDistributedTracesAndCommitsConservatively) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_interface_scheduler,
                                    "test_mpi_multiblock_interface_scheduler"),
            0);
}
