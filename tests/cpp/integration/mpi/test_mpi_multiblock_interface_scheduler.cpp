#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "amr_transfer_test_authority.hpp"
#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using namespace pops::runtime::multiblock;

namespace {

using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;

ExBModel scalar_model() {
  return ExBModel{ExBVelocity{Real(1)}, NoSource{}, ChargeDensity{Real(0)}};
}

void authenticate_cell_average_trace(AxisAlignedInterface& route) {
  route.left_trace_projection_identity = route.identity + ".left-trace";
  route.right_trace_projection_identity = route.identity + ".right-trace";
  route.left_trace_provider_identity = "limiter.none";
  route.right_trace_provider_identity = "limiter.none";
  route.left_trace_operation = InterfaceTraceOperation::CellAverage;
  route.right_trace_operation = InterfaceTraceOperation::CellAverage;
  route.left_trace_required_depth = 1;
  route.right_trace_required_depth = 1;
}

class ScopedMpiCommunicator {
 public:
  explicit ScopedMpiCommunicator(MPI_Comm source) {
    if (MPI_Comm_dup(source, &communicator_) != MPI_SUCCESS)
      throw std::runtime_error("MPI_Comm_dup failed for the interface scheduler test lane");
    if (MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
      MPI_Comm_free(&communicator_);
      throw std::runtime_error(
          "MPI_Comm_set_errhandler failed for the interface scheduler test lane");
    }
  }

  ~ScopedMpiCommunicator() {
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
  }

  ScopedMpiCommunicator(const ScopedMpiCommunicator&) = delete;
  ScopedMpiCommunicator& operator=(const ScopedMpiCommunicator&) = delete;

  MPI_Comm get() const { return communicator_; }

 private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
};

PopsExecutionContextV1 mpi_lane_execution(MPI_Comm communicator) {
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
          static_cast<std::int64_t>(MPI_Comm_c2f(communicator)),
          static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)),
          "test::mpi-multiblock-interface-lane",
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

template <class Value>
void append_exact(std::string& bytes, const Value& value) {
  bytes.append(reinterpret_cast<const char*>(&value), sizeof(Value));
}

void append_exact_text(std::string& bytes, const std::string& value) {
  append_exact(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.append(value);
}

std::string exact_layout_identity(const MultiFab& field) {
  std::string bytes;
  const auto& boxes = field.box_array().boxes();
  const auto& ranks = field.dmap().ranks();
  append_exact(bytes, static_cast<std::uint64_t>(boxes.size()));
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const Box2D& box = boxes[index];
    append_exact(bytes, box.lo[0]);
    append_exact(bytes, box.lo[1]);
    append_exact(bytes, box.hi[0]);
    append_exact(bytes, box.hi[1]);
    append_exact(bytes, ranks[index]);
  }
  return bytes;
}

template <class Fragment>
std::string exact_fragment_identity(const Fragment& fragment) {
  std::string bytes;
  append_exact_text(bytes, fragment.key.interface_identity);
  append_exact(bytes, fragment.key.topology_epoch);
  append_exact(bytes, fragment.key.coarse_level);
  append_exact(bytes, fragment.key.fine_level);
  append_exact(bytes, fragment.key.clock.level);
  append_exact(bytes, fragment.key.clock.macro_step);
  append_exact(bytes, fragment.key.clock.phase.numerator);
  append_exact(bytes, fragment.key.clock.phase.denominator);
  append_exact(bytes, fragment.key.clock.physical_time);
  append_exact_text(bytes, fragment.key.stage_identity);
  append_exact(bytes, fragment.key.interval.begin.level);
  append_exact(bytes, fragment.key.interval.begin.macro_step);
  append_exact(bytes, fragment.key.interval.begin.phase.numerator);
  append_exact(bytes, fragment.key.interval.begin.phase.denominator);
  append_exact(bytes, fragment.key.interval.begin.physical_time);
  append_exact(bytes, fragment.key.interval.end.level);
  append_exact(bytes, fragment.key.interval.end.macro_step);
  append_exact(bytes, fragment.key.interval.end.phase.numerator);
  append_exact(bytes, fragment.key.interval.end.phase.denominator);
  append_exact(bytes, fragment.key.interval.end.physical_time);
  append_exact(bytes, fragment.key.orientation);
  append_exact(bytes, fragment.key.left_block);
  append_exact(bytes, fragment.key.right_block);
  append_exact(bytes, fragment.measure.stage_weight.numerator);
  append_exact(bytes, fragment.measure.stage_weight.denominator);
  append_exact(bytes, fragment.measure.stage_weight_resolved);
  append_exact(bytes, fragment.measure.substep_duration);
  append_exact(bytes, fragment.measure.face_measure);
  append_exact(bytes, static_cast<std::uint64_t>(fragment.payload.size()));
  for (const Real value : fragment.payload)
    append_exact(bytes, value);
  return bytes;
}

struct OneShotRematerializationFailure {
  std::shared_ptr<bool> fail_next_copy;
  std::array<int, 2>* evaluator_calls = nullptr;
  int level = 0;

  OneShotRematerializationFailure(std::shared_ptr<bool> fail, std::array<int, 2>* calls, int k)
      : fail_next_copy(std::move(fail)), evaluator_calls(calls), level(k) {}

  OneShotRematerializationFailure(const OneShotRematerializationFailure& other)
      : fail_next_copy(other.fail_next_copy),
        evaluator_calls(other.evaluator_calls),
        level(other.level) {
    if (*fail_next_copy && my_rank() == 1) {
      *fail_next_copy = false;
      throw std::runtime_error("injected rank-local interface rematerialization failure");
    }
  }

  OneShotRematerializationFailure(OneShotRematerializationFailure&&) noexcept = default;
  OneShotRematerializationFailure& operator=(const OneShotRematerializationFailure&) = default;
  OneShotRematerializationFailure& operator=(OneShotRematerializationFailure&&) noexcept = default;

  void operator()(const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) const {
    ++(*evaluator_calls)[static_cast<std::size_t>(level)];
    for (int face = 0; face < batch.face_count; ++face)
      for (int component = 0; component < batch.component_count; ++component)
        batch.shared_flux[static_cast<std::size_t>(face) * batch.component_count + component] =
            Real(level + face + component + 1);
  }
};

AmrRuntime make_dynamic_mpi_interface_runtime(
    std::array<int, 2>& evaluator_calls,
    const std::shared_ptr<bool>& fail_next_rematerialization_copy) {
  constexpr int cells = 4;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 1;
  params.mesh.distribute_coarse = true;
  params.mesh.coarse_max_grid = 2;
  params.poisson.bc = BCRec{};
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 2);
  layout.ba[1] = BoxArray(std::vector<Box2D>{layout.geom.domain.refine(kAmrRefRatio)});
  layout.dm[1] = layout.load_balance->distribute(layout.ba[1], n_ranks());

  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"left", "right"}) {
    AmrRuntimeBlock block = detail::dispatch_amr_block(
        scalar_model(), "none", "rusanov", layout, name,
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false, 1);
    block.state_identity = std::string("test://dynamic-mpi-interface/block/") + name + "/state/U";
    const auto omit_local_interface = [](MultiFab&, const MultiFab&, const Geometry&, MultiFab& fx,
                                         MultiFab& fy, MultiFab& rhs) {
      fx.set_val(Real(0));
      fy.set_val(Real(0));
      rhs.set_val(Real(0));
    };
    block.level_flux_capture = omit_local_interface;
    block.level_flux_capture_neg_div = omit_local_interface;
    block.level_rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                                     const MultiFab&, const Geometry&,
                                                     MultiFab& rhs) { rhs.set_val(Real(0)); };
    block.level_neg_div_flux_without_prepared_interfaces =
        block.level_rhs_without_prepared_interfaces;
    blocks.push_back(std::move(block));
  }

  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  runtime.set_parent_child_temporal_relations({amr::ParentChildClockRelation(
      0, 1, amr::Rational(2, 1), amr::RemainderPolicy::IntegralOnly)});
  runtime.set_regrid(/*every=*/1, /*grow=*/0, /*margin=*/0);

  const PopsExecutionContextV1 execution = mpi_world_execution();
  for (int level = 0; level < 2; ++level) {
    AxisAlignedInterface route;
    route.identity = "mpi-two-rank.dynamic-refined-shared-flux";
    route.left_block = 0;
    route.right_block = 1;
    route.level = level;
    route.left_axis = route.right_axis = InterfaceAxis::X;
    route.left_side = InterfaceSide::High;
    route.right_side = InterfaceSide::Low;
    route.right_component_for_left = {0};
    route.affine_mapping_identity = "periodic-x-translation";
    route.right_normal_translation = Real(1);
    authenticate_cell_average_trace(route);
    runtime.install_level_interface_flux(
        level, std::move(route), execution,
        InterfaceFluxEvaluator(OneShotRematerializationFailure{fail_next_rematerialization_copy,
                                                               &evaluator_calls, level}));
  }
  runtime.require_complete_active_level_interfaces();
  return runtime;
}

long exercise_dynamic_refined_interface_rematerialization() {
  long failures = 0;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };

  std::array<int, 2> evaluator_calls{0, 0};
  auto fail_next_rematerialization_copy = std::make_shared<bool>(false);
  AmrRuntime runtime =
      make_dynamic_mpi_interface_runtime(evaluator_calls, fail_next_rematerialization_copy);
  const std::string interface_identity = "mpi-two-rank.dynamic-refined-shared-flux";
  const std::string accepted_layout = exact_layout_identity(runtime.level_state(0, 1));
  const std::uint64_t accepted_epoch = runtime.topology_epoch();

  const auto evaluate_level = [&](std::int64_t tick) {
    MultiFab& left = runtime.level_state(0, 1);
    MultiFab& right = runtime.level_state(1, 1);
    MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
    MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
    const BoundaryEvaluationPoint point{"clock.dynamic-mpi-interface", tick, 1,          0, 0,
                                        amr::Rational(0, 1),           0.05, 0.05 * tick};
    runtime.level_rhs_with_interfaces(1, point, {&left, &right}, {&left_rhs, &right_rhs});
    require(all_reduce_sum(field_is_zero(left_rhs) ? 0L : 1L) > 0);
    require(all_reduce_sum(field_is_zero(right_rhs) ? 0L : 1L) > 0);
  };

  evaluate_level(1);
  require(evaluator_calls == (std::array<int, 2>{0, 1}));
  require(runtime.interface_evaluation_count(interface_identity, 1) == 1u);

  runtime.set_clustering(/*min_efficiency=*/1.0, /*min_box_size=*/1, /*max_box_size=*/2);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(0.5)}, {1, 0, Real(0.5)}},
                                         "test::dynamic-mpi-interface-full-domain@1");
  *fail_next_rematerialization_copy = my_rank() == 1;
  bool collective_failure_observed = false;
  try {
    runtime.regrid();
  } catch (const std::runtime_error& error) {
    const std::string message(error.what());
    collective_failure_observed =
        my_rank() == 1
            ? message.find("injected rank-local interface rematerialization failure") !=
                  std::string::npos
            : message.find("replacement route/layout preflight failed on another MPI rank") !=
                  std::string::npos;
  }
  require(collective_failure_observed);
  require(runtime.topology_epoch() == accepted_epoch);
  require(runtime.regrid_count() == 0);
  require(exact_layout_identity(runtime.level_state(0, 1)) == accepted_layout);
  require(runtime.interface_evaluation_count(interface_identity, 1) == 1u);
  runtime.require_complete_active_level_interfaces();

  evaluate_level(2);
  require(evaluator_calls == (std::array<int, 2>{0, 2}));
  require(runtime.interface_evaluation_count(interface_identity, 1) == 2u);

  runtime.regrid();
  require(runtime.topology_epoch() > accepted_epoch);
  require(runtime.regrid_count() == 1);
  const std::string replacement_layout = exact_layout_identity(runtime.level_state(0, 1));
  require(replacement_layout != accepted_layout);
  require(all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("dynamic-refined-layout"), std::string_view(replacement_layout)}}));
  runtime.require_complete_active_level_interfaces();

  MultiFab& left = runtime.level_state(0, 1);
  MultiFab& right = runtime.level_state(1, 1);
  MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
  MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
  left_rhs.set_val(Real(0));
  right_rhs.set_val(Real(0));
  const BoundaryEvaluationPoint point{"clock.dynamic-mpi-interface", 3,    1,    0, 1,
                                      amr::Rational(1, 2),           0.05, 0.125};
  InterfaceFluxFragmentLedger ledger(runtime.topology_epoch());
  ledger.begin();
  const amr::ClockWindow interval{{1, 3, amr::Rational(0, 1), 0.1},
                                  {1, 3, amr::Rational(1, 1), 0.15}};
  InterfaceFluxFragmentPublication publication{&ledger,
                                               runtime.topology_epoch(),
                                               2,
                                               amr::ClockStamp{1, 3, amr::Rational(1, 2), 0.125},
                                               "program.group.dynamic-refined-mpi",
                                               interval,
                                               amr::Rational(1, 2)};
  runtime.publish_level_interface_flux_fragments(1, point, {0, 1}, {&left, &right},
                                                 {&left_rhs, &right_rhs}, publication);
  require(evaluator_calls == (std::array<int, 2>{0, 3}));
  require(runtime.interface_evaluation_count(interface_identity, 1) == 3u);
  require(ledger.pending_size() == 1u);
  if (ledger.pending_size() == 1u) {
    const auto& fragment = ledger.pending_entries().front();
    require(fragment.key.interface_identity == interface_identity);
    require(fragment.key.topology_epoch == runtime.topology_epoch());
    require(fragment.key.coarse_level == 0 && fragment.key.fine_level == 1);
    require(fragment.key.clock == publication.clock);
    require(fragment.key.stage_identity == publication.stage_identity);
    require(fragment.key.interval.begin == interval.begin &&
            fragment.key.interval.end == interval.end);
    require(fragment.key.orientation == amr::InterfaceFluxOrientation::FineOutward);
    require(fragment.measure.stage_weight == amr::Rational(1, 2));
    require(fragment.measure.stage_weight_resolved);
    require(fragment.measure.substep_duration == point.dt);
    const std::string fragment_identity = exact_fragment_identity(fragment);
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("dynamic-refined-fragment"), std::string_view(fragment_identity)}}));
  }
  ledger.commit();
  require(ledger.published_size() == 1u);
  require(all_reduce_sum(field_is_zero(left_rhs) ? 0L : 1L) > 0);
  require(all_reduce_sum(field_is_zero(right_rhs) ? 0L : 1L) > 0);
  return failures;
}

int run_mpi_multiblock_interface_scheduler(int argc, char** argv) {
  comm_init(&argc, &argv);
  long failures = 0;
  const auto require = [&failures](bool condition, const std::source_location where =
                                                       std::source_location::current()) {
    if (!condition) {
      ++failures;
      std::cerr << "rank " << my_rank() << ": failed requirement at " << where.file_name() << ':'
                << where.line() << '\n';
    }
  };

  {
    try {
      require(n_ranks() == 2);
      const ScopedMpiCommunicator interface_lane(MPI_COMM_WORLD);
      int world_relation = MPI_UNEQUAL;
      require(MPI_Comm_compare(interface_lane.get(), MPI_COMM_WORLD, &world_relation) ==
              MPI_SUCCESS);
      require(world_relation == MPI_CONGRUENT);

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
      authenticate_cell_average_trace(route);
      initialize_left(left_state);
      initialize_right(right_state, route.right_component_for_left);

      const Geometry left_geometry{left_domain, Real(0), Real(1), Real(0), Real(1)};
      const Geometry right_geometry{right_domain, Real(1), Real(2), Real(0), Real(1)};
      const PopsExecutionContextV1 execution = mpi_lane_execution(interface_lane.get());
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
      bool implicit_mpi_rejected = false;
      try {
        scheduler.require_exact_jacvec_pair(0, 0, 1);
      } catch (const std::runtime_error& error) {
        implicit_mpi_rejected =
            std::string(error.what()).find("serial rank-one") != std::string::npos;
      }
      require(implicit_mpi_rejected);
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

      // The same prepared scheduler owns the L0 and L1 collective routes. Publishing the refined
      // canonical flux fragment proves that MPI execution does not fall back to two independent
      // endpoint fluxes when the Program ledger qualifies the fine side of L0/L1.
      const Box2D fine_left_domain = left_domain.refine(2);
      const Box2D fine_right_domain = right_domain.refine(2);
      const BoxArray fine_left_boxes(std::vector<Box2D>{{{0, 0}, {3, 3}}, {{0, 4}, {3, 7}}});
      const BoxArray fine_right_boxes(std::vector<Box2D>{{{4, 0}, {7, 3}}, {{4, 4}, {7, 7}}});
      MultiFab fine_left_state(fine_left_boxes, left_owners, 2, 0);
      MultiFab fine_right_state(fine_right_boxes, right_owners, 2, 0);
      MultiFab fine_left_rhs(fine_left_boxes, left_owners, 2, 0);
      MultiFab fine_right_rhs(fine_right_boxes, right_owners, 2, 0);
      fine_left_rhs.set_val(Real(0));
      fine_right_rhs.set_val(Real(0));
      initialize_left(fine_left_state);
      initialize_right(fine_right_state, route.right_component_for_left);

      AxisAlignedInterface fine_route = route;
      fine_route.level = 1;
      const Geometry fine_left_geometry = left_geometry.refine(2);
      const Geometry fine_right_geometry = right_geometry.refine(2);
      const BoundaryEvaluationPoint fine_point{"clock.mpi-interface", 4,     1,     0, 2,
                                               amr::Rational(1, 2),   0.125, 0.3125};
      int fine_evaluator_calls = 0;
      bool fine_traces_complete = true;
      scheduler.install(
          fine_route, fine_left_state, fine_left_geometry, fine_right_state, fine_right_geometry,
          execution,
          [&](const BoundaryEvaluationPoint& actual_point, const InterfaceFluxBatch& batch) {
            ++fine_evaluator_calls;
            fine_traces_complete = fine_traces_complete && actual_point == fine_point &&
                                   batch.face_count == 8 && batch.component_count == 2;
            for (int face = 0; face < batch.face_count; ++face)
              for (int component = 0; component < batch.component_count; ++component) {
                const std::size_t offset =
                    static_cast<std::size_t>(face) * 2 + static_cast<std::size_t>(component);
                fine_traces_complete = fine_traces_complete &&
                                       batch.left_state[offset] == left_value(face, component) &&
                                       batch.right_state[offset] == right_value(face, component);
                batch.shared_flux[offset] = shared_flux(face, component);
              }
          });
      InterfaceFluxFragmentLedger fine_ledger(19);
      fine_ledger.begin();
      const amr::ClockWindow fine_interval{{1, 4, amr::Rational(0, 1), 0.25},
                                           {1, 4, amr::Rational(1, 1), 0.375}};
      const amr::ClockStamp fine_clock{1, 4, amr::Rational(1, 2), 0.3125};
      InterfaceFluxFragmentPublication fine_publication{
          &fine_ledger,       19, 2, fine_clock, "program.group.refined-mpi", fine_interval,
          amr::Rational(1, 1)};
      std::vector<MultiFab*> fine_states{&fine_left_state, &fine_right_state};
      std::vector<MultiFab*> fine_rhs{&fine_left_rhs, &fine_right_rhs};
      scheduler.apply(fine_point, fine_states, fine_rhs, &fine_publication);

      require(fine_evaluator_calls == 1);
      require(fine_traces_complete);
      require(scheduler.size() == 2u);
      require(scheduler.evaluation_count(route.identity, 0) == 1u);
      require(scheduler.evaluation_count(fine_route.identity, 1) == 1u);
      require(fine_ledger.pending_size() == 1u);
      fine_ledger.commit();
      require(fine_ledger.published_size() == 1u);
      const auto& fine_fragment = fine_ledger.published_entries().front();
      require(fine_fragment.key.interface_identity == fine_route.identity);
      require(fine_fragment.key.coarse_level == 0 && fine_fragment.key.fine_level == 1);
      require(fine_fragment.key.clock.level == 1);
      require(fine_fragment.key.orientation == amr::InterfaceFluxOrientation::FineOutward);
      require(fine_fragment.payload.size() == 16u);
      for (int face = 0; face < 8; ++face)
        for (int component = 0; component < 2; ++component) {
          const std::size_t offset =
              static_cast<std::size_t>(face) * 2u + static_cast<std::size_t>(component);
          require(fine_fragment.payload[offset] == shared_flux(face, component));
        }
      require(!field_is_zero(fine_left_rhs) && !field_is_zero(fine_right_rhs));
      require(fine_left_domain == fine_left_state.box_array().bounding_box());
      require(fine_right_domain == fine_right_state.box_array().bounding_box());

      // Exercise the actual AMR publication entry point, not only the detached scheduler. The
      // middle level of a three-level prefix must append one fragment to each adjacent pair under
      // the same MPI_COMM_WORLD collective identity.
      AmrBuildParams amr_params;
      amr_params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
      amr_params.mesh.periodicity = Periodicity{true, true};
      amr_params.mesh.n = 4;
      amr_params.mesh.L = 1.0;
      amr_params.mesh.regrid_every = 0;
      amr_params.mesh.distribute_coarse = true;
      amr_params.mesh.coarse_max_grid = 2;
      amr_params.poisson.bc = BCRec{};
      detail::SharedAmrLayout amr_layout = detail::make_shared_amr_layout_levels(amr_params, 3);
      for (int level = 1, refinement = kAmrRefRatio; level < 3;
           ++level, refinement *= kAmrRefRatio) {
        amr_layout.ba[static_cast<std::size_t>(level)] =
            BoxArray(std::vector<Box2D>{amr_layout.geom.domain.refine(refinement)});
        amr_layout.dm[static_cast<std::size_t>(level)] = amr_layout.load_balance->distribute(
            amr_layout.ba[static_cast<std::size_t>(level)], n_ranks());
      }
      std::vector<AmrRuntimeBlock> amr_blocks;
      for (const char* name : {"left", "right"}) {
        AmrRuntimeBlock block =
            detail::dispatch_amr_block(scalar_model(), "none", "rusanov", amr_layout, name,
                                       std::vector<double>(16, 1.0), true, 1.4, 1, false, 1);
        block.state_identity =
            std::string("test://mpi-three-level-interface/block/") + name + "/state/U";
        const auto omit_local_interface = [](MultiFab&, const MultiFab&, const Geometry&,
                                             MultiFab& fx, MultiFab& fy, MultiFab& rhs) {
          fx.set_val(Real(0));
          fy.set_val(Real(0));
          rhs.set_val(Real(0));
        };
        block.level_flux_capture = omit_local_interface;
        block.level_flux_capture_neg_div = omit_local_interface;
        block.level_rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                                         const MultiFab&, const Geometry&,
                                                         MultiFab& rhs) { rhs.set_val(Real(0)); };
        block.level_neg_div_flux_without_prepared_interfaces =
            block.level_rhs_without_prepared_interfaces;
        amr_blocks.push_back(std::move(block));
      }
      AmrRuntime amr_runtime(amr_layout.geom, amr_layout.runtime_hierarchy(), amr_layout.poisson_bc,
                             std::move(amr_blocks), amr_layout.base_per,
                             amr_layout.replicated_coarse, amr_layout.wall);
      test::install_second_order_amr_transfer_authorities(amr_runtime, 2);
      amr_runtime.set_parent_child_temporal_relations(
          {amr::ParentChildClockRelation(0, 1, amr::Rational(2, 1),
                                         amr::RemainderPolicy::IntegralOnly),
           amr::ParentChildClockRelation(1, 2, amr::Rational(2, 1),
                                         amr::RemainderPolicy::IntegralOnly)});
      std::array<int, 3> amr_evaluator_calls{0, 0, 0};
      for (int level = 0; level < 3; ++level) {
        AxisAlignedInterface amr_route;
        amr_route.identity = "mpi-two-rank.three-level-shared-flux";
        amr_route.left_block = 0;
        amr_route.right_block = 1;
        amr_route.level = level;
        amr_route.left_axis = amr_route.right_axis = InterfaceAxis::X;
        amr_route.left_side = InterfaceSide::High;
        amr_route.right_side = InterfaceSide::Low;
        amr_route.right_component_for_left = {0};
        amr_route.affine_mapping_identity = "periodic-x-translation";
        amr_route.right_normal_translation = Real(1);
        authenticate_cell_average_trace(amr_route);
        amr_runtime.install_level_interface_flux(
            level, amr_route, execution,
            [&amr_evaluator_calls, level](const BoundaryEvaluationPoint&,
                                          const InterfaceFluxBatch& batch) {
              ++amr_evaluator_calls[static_cast<std::size_t>(level)];
              for (int face = 0; face < batch.face_count; ++face)
                batch.shared_flux[face] = Real(level + face + 1);
            });
      }
      amr_runtime.require_complete_active_level_interfaces();
      MultiFab& amr_left = amr_runtime.level_state(0, 1);
      MultiFab& amr_right = amr_runtime.level_state(1, 1);
      MultiFab amr_left_rhs(amr_left.box_array(), amr_left.dmap(), 1, 0);
      MultiFab amr_right_rhs(amr_right.box_array(), amr_right.dmap(), 1, 0);
      amr_left_rhs.set_val(Real(0));
      amr_right_rhs.set_val(Real(0));
      const BoundaryEvaluationPoint amr_point{"clock.mpi-three-level", 5,   1,   0, 4,
                                              amr::Rational(1, 2),     0.1, 0.45};
      InterfaceFluxFragmentLedger amr_ledger(amr_runtime.topology_epoch());
      amr_ledger.begin();
      const amr::ClockWindow amr_interval{{1, 5, amr::Rational(0, 1), 0.4},
                                          {1, 5, amr::Rational(1, 1), 0.5}};
      InterfaceFluxFragmentPublication amr_publication{
          &amr_ledger,
          amr_runtime.topology_epoch(),
          3,
          amr::ClockStamp{1, 5, amr::Rational(1, 2), 0.45},
          "program.group.mpi-three-level",
          amr_interval,
          amr::Rational(1, 1)};
      amr_runtime.publish_level_interface_flux_fragments(
          1, amr_point, {0, 1}, {&amr_left, &amr_right}, {&amr_left_rhs, &amr_right_rhs},
          amr_publication);
      require(amr_evaluator_calls == (std::array<int, 3>{0, 1, 0}));
      require(amr_ledger.pending_size() == 2u);
      bool saw_lower_pair = false;
      bool saw_upper_pair = false;
      for (const auto& fragment : amr_ledger.pending_entries()) {
        require(fragment.key.interface_identity == "mpi-two-rank.three-level-shared-flux");
        require(fragment.key.topology_epoch == amr_runtime.topology_epoch());
        require(fragment.key.clock.level == 1);
        saw_lower_pair = saw_lower_pair ||
                         (fragment.key.coarse_level == 0 && fragment.key.fine_level == 1 &&
                          fragment.key.orientation == amr::InterfaceFluxOrientation::FineOutward);
        saw_upper_pair = saw_upper_pair ||
                         (fragment.key.coarse_level == 1 && fragment.key.fine_level == 2 &&
                          fragment.key.orientation == amr::InterfaceFluxOrientation::CoarseOutward);
      }
      require(saw_lower_pair && saw_upper_pair);
      require(all_reduce_sum(field_is_zero(amr_left_rhs) ? 0L : 1L) > 0);
      require(all_reduce_sum(field_is_zero(amr_right_rhs) ? 0L : 1L) > 0);
      amr_ledger.commit();

      // Cross the capability boundary that RegridOnRestart uses: keep L0->L1 unchanged, recluster
      // only the finest L1->L2 transition, and require the distributed interface registry to be
      // rematerialized over the new boxes/owners before any Program flux can run again.
      const auto accepted_middle_boxes = amr_runtime.level_state(0, 1).box_array().boxes();
      const auto accepted_finest_boxes = amr_runtime.level_state(0, 2).box_array().boxes();
      const std::uint64_t accepted_epoch = amr_runtime.topology_epoch();
      const auto accepted_counts = amr_evaluator_calls;
      amr_runtime.set_clustering(/*min_efficiency=*/1.0, /*min_box_size=*/1,
                                 /*max_box_size=*/4);
      test::install_prepared_threshold_union(amr_runtime, {{0, 0, Real(0.5)}, {1, 0, Real(0.5)}},
                                             "test::mpi-dynamic-interface-finest@1");
      amr_runtime.require_restart_regrid_supported();
      amr_runtime.regrid();
      require(amr_runtime.nlev() == 3);
      require(amr_runtime.level_state(0, 1).box_array().boxes() == accepted_middle_boxes);
      require(amr_runtime.level_state(0, 2).box_array().boxes() != accepted_finest_boxes);
      require(amr_runtime.topology_epoch() > accepted_epoch);
      require(amr_evaluator_calls == accepted_counts);
      amr_runtime.require_complete_active_level_interfaces();

      const auto rematerialized_boxes = amr_runtime.level_state(0, 2).box_array().boxes();
      const auto rematerialized_owners = amr_runtime.level_state(0, 2).dmap().ranks();
      const long rematerialized_count = static_cast<long>(rematerialized_boxes.size());
      const long rematerialized_owner_count = static_cast<long>(rematerialized_owners.size());
      const long minimum_rematerialized_count = all_reduce_min(rematerialized_count);
      const long maximum_rematerialized_count = all_reduce_max(rematerialized_count);
      const long minimum_rematerialized_owner_count = all_reduce_min(rematerialized_owner_count);
      const long maximum_rematerialized_owner_count = all_reduce_max(rematerialized_owner_count);
      const bool stable_rematerialized_cardinality =
          minimum_rematerialized_count == maximum_rematerialized_count &&
          minimum_rematerialized_owner_count == maximum_rematerialized_owner_count &&
          rematerialized_count == rematerialized_owner_count;
      require(stable_rematerialized_cardinality);
      if (!stable_rematerialized_cardinality)
        throw std::runtime_error(
            "rematerialized MPI hierarchy has rank-divergent box/owner cardinality");
      for (std::size_t patch = 0; patch < rematerialized_boxes.size(); ++patch) {
        const Box2D& box = rematerialized_boxes[patch];
        for (const int coordinate : {box.lo[0], box.lo[1], box.hi[0], box.hi[1]})
          require(all_reduce_min(static_cast<long>(coordinate)) ==
                  all_reduce_max(static_cast<long>(coordinate)));
        require(all_reduce_min(static_cast<long>(rematerialized_owners[patch])) ==
                all_reduce_max(static_cast<long>(rematerialized_owners[patch])));
      }

      MultiFab& rematerialized_left = amr_runtime.level_state(0, 2);
      MultiFab& rematerialized_right = amr_runtime.level_state(1, 2);
      MultiFab rematerialized_left_rhs(rematerialized_left.box_array(), rematerialized_left.dmap(),
                                       1, 0);
      MultiFab rematerialized_right_rhs(rematerialized_right.box_array(),
                                        rematerialized_right.dmap(), 1, 0);
      rematerialized_left_rhs.set_val(Real(0));
      rematerialized_right_rhs.set_val(Real(0));
      const BoundaryEvaluationPoint rematerialized_point{"clock.mpi-three-level", 6,    2,    0, 5,
                                                         amr::Rational(1, 2),     0.05, 0.525};
      InterfaceFluxFragmentLedger rematerialized_ledger(amr_runtime.topology_epoch());
      rematerialized_ledger.begin();
      const amr::ClockWindow rematerialized_interval{{2, 6, amr::Rational(0, 1), 0.5},
                                                     {2, 6, amr::Rational(1, 1), 0.55}};
      InterfaceFluxFragmentPublication rematerialized_publication{
          &rematerialized_ledger,
          amr_runtime.topology_epoch(),
          3,
          amr::ClockStamp{2, 6, amr::Rational(1, 2), 0.525},
          "program.group.mpi-rematerialized",
          rematerialized_interval,
          amr::Rational(1, 1)};
      amr_runtime.publish_level_interface_flux_fragments(
          2, rematerialized_point, {0, 1}, {&rematerialized_left, &rematerialized_right},
          {&rematerialized_left_rhs, &rematerialized_right_rhs}, rematerialized_publication);
      require(amr_evaluator_calls == (std::array<int, 3>{0, 1, 1}));
      require(rematerialized_ledger.pending_size() == 1u);
      const auto& rematerialized_fragment = rematerialized_ledger.pending_entries().front();
      require(rematerialized_fragment.key.topology_epoch == amr_runtime.topology_epoch());
      require(rematerialized_fragment.key.coarse_level == 1);
      require(rematerialized_fragment.key.fine_level == 2);
      require(rematerialized_fragment.key.orientation ==
              amr::InterfaceFluxOrientation::FineOutward);
      rematerialized_ledger.commit();

      fine_left_rhs.set_val(Real(0));
      fine_right_rhs.set_val(Real(0));
      InterfaceFluxFragmentLedger divergent_publication_ledger(20);
      divergent_publication_ledger.begin();
      InterfaceFluxFragmentPublication divergent_publication{
          &divergent_publication_ledger,
          20,
          2,
          fine_clock,
          my_rank() == 0 ? "program.group.rank-zero" : "program.group.rank-one",
          fine_interval,
          amr::Rational(1, 1)};
      bool divergent_publication_rejected = false;
      try {
        scheduler.apply(fine_point, fine_states, fine_rhs, &divergent_publication);
      } catch (const std::runtime_error& error) {
        divergent_publication_rejected =
            std::string(error.what()).find("fragment publication differs") != std::string::npos;
      }
      require(divergent_publication_rejected);
      require(fine_evaluator_calls == 1);
      require(divergent_publication_ledger.pending_size() == 0u);
      require(field_is_zero(fine_left_rhs) && field_is_zero(fine_right_rhs));
      divergent_publication_ledger.rollback();

      InterfaceFluxFragmentLedger sparse_publication_ledger(21);
      sparse_publication_ledger.begin();
      InterfaceFluxFragmentPublication sparse_publication{&sparse_publication_ledger,
                                                          21,
                                                          2,
                                                          fine_clock,
                                                          "program.group.sparse-publication",
                                                          fine_interval,
                                                          amr::Rational(1, 1)};
      bool sparse_publication_rejected = false;
      try {
        scheduler.apply(fine_point, fine_states, fine_rhs,
                        my_rank() == 0 ? &sparse_publication : nullptr);
      } catch (const std::runtime_error& error) {
        sparse_publication_rejected =
            std::string(error.what()).find("publication presence differs") != std::string::npos;
      }
      require(sparse_publication_rejected);
      require(fine_evaluator_calls == 1);
      require(sparse_publication_ledger.pending_size() == 0u);
      require(field_is_zero(fine_left_rhs) && field_is_zero(fine_right_rhs));
      sparse_publication_ledger.rollback();

      InterfaceFluxFragmentLedger divergent_transaction_ledger(22);
      divergent_transaction_ledger.begin();
      if (my_rank() == 0)
        divergent_transaction_ledger.begin();
      InterfaceFluxFragmentPublication divergent_transaction_publication{
          &divergent_transaction_ledger,
          22,
          2,
          fine_clock,
          "program.group.divergent-transaction",
          fine_interval,
          amr::Rational(1, 1)};
      bool divergent_transaction_rejected = false;
      try {
        scheduler.apply(fine_point, fine_states, fine_rhs, &divergent_transaction_publication);
      } catch (const std::runtime_error& error) {
        divergent_transaction_rejected =
            std::string(error.what()).find("fragment publication differs") != std::string::npos;
      }
      require(divergent_transaction_rejected);
      require(fine_evaluator_calls == 1);
      require(divergent_transaction_ledger.pending_size() == 0u);
      require(field_is_zero(fine_left_rhs) && field_is_zero(fine_right_rhs));
      if (my_rank() == 0)
        divergent_transaction_ledger.rollback();
      divergent_transaction_ledger.rollback();

      // Even if an externally constructed replicated ledger has already diverged behind identical
      // transaction coordinates, one rank-local duplicate cannot let a peer scatter/publish alone.
      // The enclosing attempt then rolls both ledgers back to their common savepoint.
      InterfaceFluxFragmentLedger accumulation_failure_ledger(23);
      accumulation_failure_ledger.begin();
      amr::InterfaceFluxFragmentKey existing_key{
          fine_route.identity,
          23,
          0,
          1,
          fine_clock,
          my_rank() == 0 ? "program.group.duplicate" : "program.group.other",
          fine_interval,
          amr::InterfaceFluxOrientation::FineOutward,
          fine_route.left_block,
          fine_route.right_block};
      accumulation_failure_ledger.accumulate(std::move(existing_key),
                                             {amr::Rational(1, 1), 0.125, 0.125},
                                             InterfaceFluxFragmentPayload(16, Real(0)));
      InterfaceFluxFragmentPublication accumulation_failure_publication{
          &accumulation_failure_ledger, 23, 2, fine_clock, "program.group.duplicate", fine_interval,
          amr::Rational(1, 1)};
      bool accumulation_failure_rejected = false;
      try {
        scheduler.apply(fine_point, fine_states, fine_rhs, &accumulation_failure_publication);
      } catch (const std::runtime_error& error) {
        const std::string message(error.what());
        accumulation_failure_rejected =
            message.find("duplicate stage/clock fragment identity") != std::string::npos ||
            message.find("accumulation failed on another MPI rank") != std::string::npos;
      }
      require(accumulation_failure_rejected);
      require(fine_evaluator_calls == 2);
      require(scheduler.evaluation_count(fine_route.identity, 1) == 1u);
      require(field_is_zero(fine_left_rhs) && field_is_zero(fine_right_rhs));
      accumulation_failure_ledger.rollback();
      require(accumulation_failure_ledger.pending_size() == 0u);

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
      divergent_route.left_trace_projection_identity =
          my_rank() == 0 ? "trace.rank-zero" : "trace.rank-one";
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

      // A rank-local incomplete active-level prefix must close its structural status reduction
      // before any rank enters exact registry consensus. Deliberately destroy the accepted L1 route
      // on rank one only; every rank must return from the same preflight rather than deadlocking.
      if (my_rank() == 1)
        scheduler.rollback_installations(1);
      bool incomplete_registry_rejected = false;
      try {
        scheduler.require_runtime_rematerialization_ready(2);
      } catch (const std::runtime_error& error) {
        const std::string message(error.what());
        incomplete_registry_rejected =
            my_rank() == 1
                ? message.find("registry is incomplete") != std::string::npos
                : message.find("preflight failed on another MPI rank") != std::string::npos;
      }
      require(incomplete_registry_rejected);

      failures += exercise_dynamic_refined_interface_rematerialization();
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
