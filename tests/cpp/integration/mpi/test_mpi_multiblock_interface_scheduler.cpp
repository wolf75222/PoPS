#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using namespace pops::runtime::multiblock;

namespace {

PopsExecutionContextV1 mpi_world_execution() {
  return {sizeof(PopsExecutionContextV1),
          1u,
          "test::mpi-multiblock-execution",
          POPS_MEMORY_SPACE_HOST_V1,
          "pops.runtime-backend-manifest.v1:sha256:test-mpi-multiblock",
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

Real left_value(int face, int component) {
  return Real(100 + 10 * face + component);
}

Real right_value(int face, int component) {
  return Real(200 + 10 * face + component);
}

Real shared_flux(int face, int component) {
  return Real(0.25) * (left_value(face, component) + right_value(face, component));
}

void initialize_left(MultiFab& field) {
  for (int local = 0; local < field.local_size(); ++local) {
    const Box2D box = field.box(local);
    Array4 values = field.fab(local).array();
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int component = 0; component < field.ncomp(); ++component)
          values(i, j, component) = left_value(j, component);
  }
}

void initialize_right(MultiFab& field, const std::vector<int>& right_component_for_left) {
  for (int local = 0; local < field.local_size(); ++local) {
    const Box2D box = field.box(local);
    Array4 values = field.fab(local).array();
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int component = 0; component < field.ncomp(); ++component)
          values(i, j, right_component_for_left[static_cast<std::size_t>(component)]) =
              right_value(j, component);
  }
}

bool field_is_zero(const MultiFab& field) {
  for (int local = 0; local < field.local_size(); ++local) {
    const Box2D box = field.box(local);
    const ConstArray4 values = field.fab(local).const_array();
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int component = 0; component < field.ncomp(); ++component)
          if (values(i, j, component) != Real(0))
            return false;
  }
  return true;
}

int run_mpi_multiblock_interface_scheduler(int argc, char** argv) {
  comm_init(&argc, &argv);
  long failures = 0;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };

  {
    try {
      require(n_ranks() == 2);

      const Box2D left_domain{{0, 0}, {1, 3}};
      const Box2D right_domain{{2, 0}, {3, 3}};
      const BoxArray left_boxes(std::vector<Box2D>{{{0, 0}, {1, 1}}, {{0, 2}, {1, 3}}});
      const BoxArray right_boxes(std::vector<Box2D>{{{2, 0}, {3, 1}}, {{2, 2}, {3, 3}}});
      // Opposite assignments prove that neither trace can be reconstructed by assuming that the
      // two interface sides are co-located on the same rank.
      const DistributionMapping left_owners(std::vector<int>{0, 1});
      const DistributionMapping right_owners(std::vector<int>{1, 0});
      MultiFab left_state(left_boxes, left_owners, 2, 0);
      MultiFab right_state(right_boxes, right_owners, 2, 0);
      MultiFab left_rhs(left_boxes, left_owners, 2, 0);
      MultiFab right_rhs(right_boxes, right_owners, 2, 0);
      left_rhs.set_val(Real(0));
      right_rhs.set_val(Real(0));

      AxisAlignedInterface route;
      route.identity = "mpi-two-rank.shared-flux";
      route.left_block = 0;
      route.right_block = 1;
      route.left_axis = route.right_axis = InterfaceAxis::X;
      route.left_side = InterfaceSide::High;
      route.right_side = InterfaceSide::Low;
      route.right_component_for_left = {1, 0};
      initialize_left(left_state);
      initialize_right(right_state, route.right_component_for_left);

      const Geometry left_geometry{left_domain, Real(0), Real(1), Real(0), Real(1)};
      const Geometry right_geometry{right_domain, Real(1), Real(2), Real(0), Real(1)};
      const PopsExecutionContextV1 execution = mpi_world_execution();
      const BoundaryEvaluationPoint point{"clock.mpi-interface", 3,     0,    0, 1,
                                          amr::Rational(1, 1),   0.125, 0.375};

      InterfaceFluxScheduler scheduler;
      int evaluator_calls = 0;
      bool complete_traces = true;
      scheduler.install(
          route, left_state, left_geometry, right_state, right_geometry, execution,
          [&](const BoundaryEvaluationPoint& actual_point, const InterfaceFluxBatch& batch) {
            ++evaluator_calls;
            complete_traces = complete_traces && actual_point == point && batch.face_count == 4 &&
                              batch.component_count == 2;
            for (int face = 0; face < batch.face_count; ++face)
              for (int component = 0; component < batch.component_count; ++component) {
                const std::size_t offset =
                    static_cast<std::size_t>(face) * 2 + static_cast<std::size_t>(component);
                complete_traces = complete_traces &&
                                  batch.left_state[offset] == left_value(face, component) &&
                                  batch.right_state[offset] == right_value(face, component);
                batch.shared_flux[offset] = shared_flux(face, component);
              }
          });
      std::vector<MultiFab*> states{&left_state, &right_state};
      std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
      scheduler.apply(point, states, rhs);

      require(evaluator_calls == 1);
      require(complete_traces);
      require(scheduler.evaluation_count(route.identity, 0) == 1u);
      for (int local = 0; local < left_rhs.local_size(); ++local) {
        const Box2D box = left_rhs.box(local);
        const ConstArray4 values = left_rhs.fab(local).const_array();
        for (int j = box.lo[1]; j <= box.hi[1]; ++j)
          for (int i = box.lo[0]; i <= box.hi[0]; ++i)
            for (int component = 0; component < left_rhs.ncomp(); ++component) {
              const Real expected = i == left_domain.hi[0]
                                        ? -shared_flux(j, component) / left_geometry.dx()
                                        : Real(0);
              require(values(i, j, component) == expected);
            }
      }
      for (int local = 0; local < right_rhs.local_size(); ++local) {
        const Box2D box = right_rhs.box(local);
        const ConstArray4 values = right_rhs.fab(local).const_array();
        for (int j = box.lo[1]; j <= box.hi[1]; ++j)
          for (int i = box.lo[0]; i <= box.hi[0]; ++i)
            for (int component = 0; component < right_rhs.ncomp(); ++component) {
              int canonical_component = -1;
              for (int candidate = 0; candidate < 2; ++candidate)
                if (route.right_component_for_left[static_cast<std::size_t>(candidate)] ==
                    component)
                  canonical_component = candidate;
              const Real expected = i == right_domain.lo[0]
                                        ? shared_flux(j, canonical_component) / right_geometry.dx()
                                        : Real(0);
              require(canonical_component >= 0 && values(i, j, component) == expected);
            }
      }

      left_rhs.set_val(Real(0));
      right_rhs.set_val(Real(0));

      // A rank-local structural error reaches one failure consensus before any rank prepares a
      // component.  The failing rank retains its exact diagnostic, while its peer exits the same
      // phase instead of entering a later collective alone.
      InterfaceFluxScheduler invalid_route_scheduler;
      AxisAlignedInterface invalid_route = route;
      if (my_rank() == 1)
        invalid_route.identity.clear();
      int invalid_route_factory_calls = 0;
      bool invalid_route_rejected = false;
      try {
        invalid_route_scheduler.install(
            invalid_route, left_state, left_geometry, right_state, right_geometry, execution,
            InterfaceFluxEvaluatorFactory([&]() {
              ++invalid_route_factory_calls;
              return InterfaceFluxEvaluator(
                  [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch&) {});
            }));
      } catch (const std::exception& error) {
        const std::string message(error.what());
        invalid_route_rejected =
            my_rank() == 1
                ? message.find("identity/ownership is invalid") != std::string::npos
                : message.find("preflight failed on another MPI rank") != std::string::npos;
      }
      require(invalid_route_rejected);
      require(invalid_route_factory_calls == 0);
      require(invalid_route_scheduler.size() == 0);

      // Two locally valid but different routes are rejected by the exact canonical payload
      // consensus before component preparation or registry mutation.
      InterfaceFluxScheduler divergent_route_scheduler;
      AxisAlignedInterface divergent_route = route;
      divergent_route.identity = my_rank() == 0 ? "route.rank-zero" : "route.rank-one";
      int divergent_route_factory_calls = 0;
      bool divergent_route_rejected = false;
      try {
        divergent_route_scheduler.install(
            divergent_route, left_state, left_geometry, right_state, right_geometry, execution,
            InterfaceFluxEvaluatorFactory([&]() {
              ++divergent_route_factory_calls;
              return InterfaceFluxEvaluator(
                  [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch&) {});
            }));
      } catch (const std::runtime_error& error) {
        divergent_route_rejected =
            std::string(error.what()).find("route/layout differs across MPI ranks") !=
            std::string::npos;
      }
      require(divergent_route_rejected);
      require(divergent_route_factory_calls == 0);
      require(divergent_route_scheduler.size() == 0);

      // A factory failure on one rank is also a transactional collective failure: no peer commits
      // the prepared route and the original component exception remains visible where it occurred.
      InterfaceFluxScheduler factory_failure_scheduler;
      int factory_calls = 0;
      bool factory_failure_rejected = false;
      try {
        factory_failure_scheduler.install(
            AxisAlignedInterface(route), left_state, left_geometry, right_state, right_geometry,
            execution, InterfaceFluxEvaluatorFactory([&]() -> InterfaceFluxEvaluator {
              ++factory_calls;
              if (my_rank() == 1)
                throw std::runtime_error("rank-local factory failure");
              return [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch&) {};
            }));
      } catch (const std::runtime_error& error) {
        const std::string message(error.what());
        factory_failure_rejected =
            my_rank() == 1 ? message.find("rank-local factory failure") != std::string::npos
                           : message.find("evaluator preparation failed on another MPI rank") !=
                                 std::string::npos;
      }
      require(factory_failure_rejected);
      require(factory_calls == 1);
      require(factory_failure_scheduler.size() == 0);

      // Point identity and sparse active masks are collective control-flow authorities.  Rank
      // disagreement is rejected before evaluator invocation and before either RHS is touched.
      InterfaceFluxScheduler control_flow_scheduler;
      int control_flow_evaluator_calls = 0;
      control_flow_scheduler.install(
          AxisAlignedInterface(route), left_state, left_geometry, right_state, right_geometry,
          execution, [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
            ++control_flow_evaluator_calls;
            for (int offset = 0; offset < batch.face_count * batch.component_count; ++offset)
              batch.shared_flux[offset] = Real(0);
          });
      BoundaryEvaluationPoint divergent_point = point;
      divergent_point.tick += my_rank();
      bool point_rejected = false;
      try {
        control_flow_scheduler.apply(divergent_point, states, rhs);
      } catch (const std::runtime_error& error) {
        point_rejected =
            std::string(error.what()).find("BoundaryEvaluationPoint differs") != std::string::npos;
      }
      require(point_rejected);
      require(control_flow_evaluator_calls == 0);
      require(control_flow_scheduler.evaluation_count(route.identity, 0) == 0u);
      require(field_is_zero(left_rhs) && field_is_zero(right_rhs));

      std::vector<MultiFab*> divergent_states =
          my_rank() == 0 ? states : std::vector<MultiFab*>{nullptr, nullptr};
      std::vector<MultiFab*> divergent_rhs =
          my_rank() == 0 ? rhs : std::vector<MultiFab*>{nullptr, nullptr};
      bool active_mask_rejected = false;
      try {
        control_flow_scheduler.apply(point, divergent_states, divergent_rhs);
      } catch (const std::runtime_error& error) {
        active_mask_rejected =
            std::string(error.what()).find("active mask differs") != std::string::npos;
      }
      require(active_mask_rejected);
      require(control_flow_evaluator_calls == 0);
      require(control_flow_scheduler.evaluation_count(route.identity, 0) == 0u);
      require(field_is_zero(left_rhs) && field_is_zero(right_rhs));

      // Rank-dependent component output is never scattered: every rank compares against the same
      // rank-0 native result before committing either side of the interface.
      InterfaceFluxScheduler divergent_scheduler;
      divergent_scheduler.install(
          AxisAlignedInterface(route), left_state, left_geometry, right_state, right_geometry,
          execution, [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
            for (int offset = 0; offset < batch.face_count * batch.component_count; ++offset)
              batch.shared_flux[offset] = Real(offset + my_rank());
          });
      bool divergence_rejected = false;
      try {
        divergent_scheduler.apply(point, states, rhs);
      } catch (const std::runtime_error& error) {
        divergence_rejected =
            std::string(error.what()).find("rank-dependent shared flux") != std::string::npos;
      }
      require(divergence_rejected);
      require(divergent_scheduler.evaluation_count(route.identity, 0) == 0u);
      require(field_is_zero(left_rhs) && field_is_zero(right_rhs));
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "rank " << my_rank()
                << ": unexpected multi-block MPI scheduler failure: " << error.what() << '\n';
    }
  }

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_multiblock_interface_scheduler,
     ReconstructsCompleteRemoteTracesAndCommitsOneConservativeFlux) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_multiblock_interface_scheduler,
                                    "test_mpi_multiblock_interface_scheduler"),
            0);
}
