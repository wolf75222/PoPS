#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/parallel/comm.hpp>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

constexpr std::string_view kExactReplicaValidationFailure =
    "replicated vector values failed exact collective validation";

namespace {

inline constexpr int kDim = 2;
using TestBox = Box<kDim>;
using TestLayout = mesh::BoxArray<kDim>;
using TestRankSpace = mesh::RankSpace<kDim>;
using TestDistribution = mesh::Distribution<kDim>;
using TestField = pops::MultiFab<kDim>;
using TestVectorDistribution = pops::PreparedVectorDistribution<kDim>;
using TestApplyFn = pops::ApplyFn<kDim>;
using TestAffineOperatorProvider = pops::PreparedAffineOperatorProvider<kDim>;
using TestAffineOperatorCallbacks = pops::PreparedAffineOperatorSessionCallbacks<kDim>;
using TestLinearPreconditioner = pops::PreparedLinearPreconditioner<kDim>;
using TestLinearPreconditionerProvider = pops::PreparedLinearPreconditionerProvider<kDim>;
using TestLinearPreconditionerCallbacks = pops::PreparedLinearPreconditionerSessionCallbacks<kDim>;
using TestAffineProblem = pops::PreparedAffineLinearProblem<kDim>;
using TestKrylovMethod = pops::PreparedKrylovMethod<kDim>;
using TestKrylovMethodProvider = pops::PreparedKrylovMethodProvider<kDim>;
using TestKrylovSolveContext = pops::PreparedKrylovSolveContext<kDim>;
using TestKrylovWorkspace = pops::KrylovWorkspace<kDim>;
using TestKrylovFootprint = pops::KrylovFootprint<kDim>;
using TestKrylovControls = pops::KrylovControls<kDim>;
using TestKrylovProblemFacts = pops::KrylovMethodProblemFacts<kDim>;
using TestKrylovWorkspaceRequest = pops::KrylovWorkspaceRequest<kDim>;
using TestVectorMetric = pops::PreparedVectorMetric<kDim>;
using TestNullspacePlan = pops::FieldNullspacePlan<kDim>;
using TestNullspacePolicy = pops::PreparedNullspacePolicy<kDim>;

Index<kDim> rank_coordinate(int rank) {
  return Index<kDim>{rank, 0};
}

TestRankSpace world_rank_space() {
  return TestRankSpace{Index<kDim>{}, Extent<kDim>{n_ranks(), 1}};
}

Extent<kDim> ghost_extent(int width) {
  return Extent<kDim>{width, width};
}

TestLayout test_layout() {
  return TestLayout(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}},
                                         TestBox{Index<kDim>{2, 0}, Index<kDim>{3, 1}}});
}

TestDistribution partitioned_distribution(const TestLayout& boxes) {
  return TestDistribution::partitioned(
      boxes, world_rank_space(), std::vector<Index<kDim>>{rank_coordinate(0), rank_coordinate(1)});
}

OperatorEvaluationSnapshot snapshot_for(
    const TestField& field,
    TestVectorDistribution ownership = TestVectorDistribution::Distributed) {
  OperatorEvaluationSnapshot snapshot;
  snapshot.authority = {1, 2, 3, 4};
  snapshot.revision = 1;
  snapshot.macro_step = 0;
  snapshot.stage_numerator = 0;
  snapshot.stage_denominator = 1;
  snapshot.dt_bits = std::bit_cast<std::uint64_t>(0.1);
  snapshot.physical_time_bits = std::bit_cast<std::uint64_t>(0.0);
  snapshot.topology_revision = 1;
  snapshot.topology = detail::layout_fingerprint(field, ownership);
  snapshot.resources = {5, 6, 7, 8};
  return snapshot;
}

TestField make_field(int components = 1, int ghosts = 0) {
  const TestLayout boxes = test_layout();
  const TestDistribution distribution = partitioned_distribution(boxes);
  TestField field(boxes, distribution, rank_coordinate(my_rank()), components,
                  ghost_extent(ghosts));
  field.set_val(Real(0));
  return field;
}

TestField make_replicated_field(int components = 1, int ghosts = 0) {
  const TestLayout boxes = test_layout();
  const TestDistribution distribution = TestDistribution::replicated(boxes, world_rank_space());
  TestField field(boxes, distribution, rank_coordinate(my_rank()), components,
                  ghost_extent(ghosts));
  field.set_val(Real(0));
  return field;
}

struct ReplicatedMeanZeroRhsKernel {
  FieldView<Real, kDim> values;
  POPS_HD void operator()(const Index<kDim>& index) const {
    values(index, 0) = index[0] < 2 ? Real(1) : Real(-1);
  }
};

void fill_replicated_mean_zero_rhs(TestField& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), ReplicatedMeanZeroRhsKernel{field.fab(local).view()});
}

void fill_isometric_rank_permutation(TestField& field) {
  field.set_val(Real(0));
  FieldView<Real, kDim> values = field.fab(0).view();
  if (my_rank() == 0)
    values(Index<kDim>{0, 0}, 0) = Real(1);
  else
    values(Index<kDim>{1, 0}, 0) = Real(1);
}

struct AlternatingDiagonalKernel {
  FieldView<Real, kDim> output;
  FieldView<const Real, kDim> input;

  POPS_HD void operator()(const Index<kDim>& index) const {
    output(index, 0) = (index[0] % 2 == 0 ? Real(1) : Real(2)) * input(index, 0);
  }
};

void apply_alternating_diagonal(TestField& output, const TestField& input) {
  for (std::size_t local = 0; local < output.local_size(); ++local)
    for_each_cell(output.box(local),
                  AlternatingDiagonalKernel{output.fab(local).view(), input.fab(local).view()});
}

/// A real non-default metric provider: a positive scalar multiple of the Euclidean product. The
/// scale participates in the exact contract, and every global/local/robust route represents the
/// same product. This exercises the provider protocol without teaching Krylov a metric name.
struct ScaledEuclideanMetricSource {
  Real scale = Real(1);

  static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.vector-metric.scaled-euclidean", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const { contract.scalar(scale); }

  constexpr std::size_t robust_payload_width() const noexcept {
    return detail::PreparedFieldAlgebra::kRobustDotPayloadWidth;
  }

  Real inner_product(const TestField& left, const TestField& right,
                     const TestVectorDistribution& distribution, std::span<double> scratch,
                     const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return scale * detail::PreparedFieldAlgebra::dot(left, right, distribution, scratch, lane);
    });
  }

  Real norm(const TestField& value, const TestVectorDistribution& distribution,
            std::span<double> scratch, const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return std::sqrt(scale) *
             detail::PreparedFieldAlgebra::norm(value, distribution, scratch, lane);
    });
  }

  Real absolute_inner_product(const TestField& left, const TestField& right,
                              const TestVectorDistribution& distribution, std::span<double> scratch,
                              const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return scale *
             detail::PreparedFieldAlgebra::absolute_dot(left, right, distribution, scratch, lane);
    });
  }

  Real nullspace_inner_product(const TestField& left, const TestField& right, Real cell_measure,
                               const TestVectorDistribution& distribution,
                               std::span<double> scratch,
                               const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return scale * detail::PreparedFieldAlgebra::nullspace_pairing(
                         left, right, cell_measure, false, distribution, scratch, lane);
    });
  }

  Real nullspace_absolute_inner_product(const TestField& left, const TestField& right,
                                        Real cell_measure,
                                        const TestVectorDistribution& distribution,
                                        std::span<double> scratch,
                                        const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return scale * detail::PreparedFieldAlgebra::nullspace_pairing(
                         left, right, cell_measure, true, distribution, scratch, lane);
    });
  }

  Real local_inner_product(const TestField& left, const TestField& right) const noexcept {
    return detail::prepared_metric_value_noexcept(
        [&] { return scale * detail::PreparedFieldAlgebra::local_dot(left, right); });
  }

  void local_robust_inner_product_payload(const TestField& left, const TestField& right,
                                          std::span<double> payload) const noexcept {
    detail::prepared_metric_payload_noexcept(payload, [&] {
      detail::PreparedFieldAlgebra::local_robust_dot_payload(left, right, payload.data());
    });
  }

  Real inner_product_from_global_robust_payload(std::span<const double> payload) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return scale * detail::PreparedFieldAlgebra::dot_from_global_robust_payload(payload.data());
    });
  }
};

struct RankLocalThrowingMetricSource : ScaledEuclideanMetricSource {
  Real inner_product(const TestField&, const TestField&, const TestVectorDistribution&,
                     std::span<double>, const ExecutionLane&) const {
    throw std::runtime_error("rank-local metric exception");
  }
};

static_assert(PreparedVectorMetricSource<kDim, ScaledEuclideanMetricSource>);
static_assert(!PreparedVectorMetricSource<kDim, RankLocalThrowingMetricSource>);

/// Adversarial external methods for the common post-provider boundary. Both complete the same
/// provider-owned collective trace first. The common Krylov wrapper must then turn either a
/// rank-local exception or a divergent report into one uniform InvalidEvaluation result.
class CollectiveBoundaryKrylovProvider final : public TestKrylovMethodProvider {
 public:
  enum class Behavior { kThrowOnRankZero, kDivergentReport, kThrowOnRankZeroAndRetryOnRankOne };

  explicit CollectiveBoundaryKrylovProvider(
      Behavior behavior, std::shared_ptr<std::atomic<bool>> mixed_apply_armed = {})
      : behavior_(behavior), mixed_apply_armed_(std::move(mixed_apply_armed)) {}

  std::string_view identity() const noexcept override {
    if (behavior_ == Behavior::kDivergentReport)
      return "pops.test.krylov.collective-divergent-report";
    return behavior_ == Behavior::kThrowOnRankZero ? "pops.test.krylov.collective-throw"
                                                   : "pops.test.krylov.collective-mixed-failure";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    if (behavior_ == Behavior::kDivergentReport)
      return "pops.test.krylov.collective-divergent-report@1";
    return behavior_ == Behavior::kThrowOnRankZero ? "pops.test.krylov.collective-throw@1"
                                                   : "pops.test.krylov.collective-mixed-failure@1";
  }
  KrylovMethodValidation validate_controls(const KrylovMethodControls& controls,
                                           const PreparedProviderOptions&) const noexcept override {
    return detail::validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(const TestKrylovProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "collective probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const TestKrylovWorkspaceRequest& request, const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("collective probe is unpreconditioned");
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(TestKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    const ExecutionLane& lane = context.execution_lane();
    (void)all_reduce_sum(1L, lane);
    if (behavior_ == Behavior::kThrowOnRankZeroAndRetryOnRankOne) {
      if (!mixed_apply_armed_)
        throw std::logic_error("mixed-failure probe has no apply arm");
      mixed_apply_armed_->store(true, std::memory_order_release);
      context.apply_linear(context.field(0), context.initial_residual());
      mixed_apply_armed_->store(false, std::memory_order_release);
    }
    if ((behavior_ == Behavior::kThrowOnRankZero ||
         behavior_ == Behavior::kThrowOnRankZeroAndRetryOnRankOne) &&
        lane.rank() == 0)
      throw std::runtime_error("rank-local provider failure after complete trace");
    const int iterations = behavior_ == Behavior::kDivergentReport && lane.rank() != 0 ? 2 : 1;
    return context.report(context.initial_physical_residual(), iterations,
                          SolveStatus::kIterationLimit);
  }

 private:
  Behavior behavior_;
  std::shared_ptr<std::atomic<bool>> mixed_apply_armed_;
};

/// MPI oracle for the exact, unbounded provider-report consensus boundary. The divergent case
/// differs by one byte beyond the legacy 4096-byte limit, so a truncated comparison cannot pass.
class LongReasonCollectiveKrylovProvider final : public TestKrylovMethodProvider {
 public:
  enum class Behavior { kIdentical, kDivergentAfterLegacyLimit };
  static constexpr std::size_t kReasonBytes = 16 * 1024 + 37;
  static constexpr std::size_t kDivergenceOffset = 4096 + 73;

  explicit LongReasonCollectiveKrylovProvider(Behavior behavior) : behavior_(behavior) {}

  std::string_view identity() const noexcept override {
    return behavior_ == Behavior::kIdentical ? "pops.test.krylov.collective-long-reason-identical"
                                             : "pops.test.krylov.collective-long-reason-divergent";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return behavior_ == Behavior::kIdentical
               ? "pops.test.krylov.collective-long-reason-identical@1"
               : "pops.test.krylov.collective-long-reason-divergent@1";
  }
  KrylovMethodValidation validate_controls(const KrylovMethodControls& controls,
                                           const PreparedProviderOptions&) const noexcept override {
    return detail::validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(const TestKrylovProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "long-reason probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const TestKrylovWorkspaceRequest& request, const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("long-reason probe is unpreconditioned");
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(TestKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    const ExecutionLane& lane = context.execution_lane();
    (void)all_reduce_sum(1L, lane);
    SolveReport report =
        context.report(context.initial_physical_residual(), 1, SolveStatus::kIterationLimit);
    report.reason.assign(kReasonBytes, 'r');
    if (behavior_ == Behavior::kDivergentAfterLegacyLimit && lane.rank() != 0)
      report.reason[kDivergenceOffset] = 'x';
    return report;
  }

 private:
  Behavior behavior_;
};

/// External method oracle for sticky apply failures. It deliberately performs two operator applies
/// before its first scientific reduction; the second callback may write finite data, but the common
/// workspace latch must re-poison that output and publish one uniform InvalidEvaluation result.
class TwoApplyBeforeReductionKrylovProvider final : public TestKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override {
    return "pops.test.krylov.two-apply-before-reduction";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.krylov.two-apply-before-reduction@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions& options) const noexcept override {
    if (options.schema_identity != "pops.test.krylov.two-apply-before-reduction.options@1" ||
        !options.values.empty())
      return KrylovMethodValidation::reject(1, "two-apply options contract is invalid");
    return detail::validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(const TestKrylovProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "two-apply probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const TestKrylovWorkspaceRequest& request, const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("two-apply probe is unpreconditioned");
    return {.field_count = 3, .initial_residual_field = 1};
  }
  SolveReport solve(TestKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    TestField& scratch = context.field(2);
    context.apply_linear(scratch, context.initial_residual());
    context.apply_linear(scratch, context.initial_residual());
    const Real residual = context.residual_norm(scratch);
    return context.report(residual, 1,
                          std::isfinite(static_cast<double>(residual))
                              ? SolveStatus::kIterationLimit
                              : SolveStatus::kInvalidEvaluation);
  }
};

enum class ExceptionKind : long { kNone, kInvalidArgument, kLogicError, kOther };

template <class Operation>
bool uniformly_rejected(Operation&& operation, std::string_view expected_fragment) {
  bool threw = false;
  ExceptionKind kind = ExceptionKind::kNone;
  std::string message;
  try {
    operation();
  } catch (const std::invalid_argument& error) {
    threw = true;
    kind = ExceptionKind::kInvalidArgument;
    message = error.what();
  } catch (const std::logic_error& error) {
    threw = true;
    kind = ExceptionKind::kLogicError;
    message = error.what();
  } catch (const std::exception& error) {
    threw = true;
    kind = ExceptionKind::kOther;
    message = error.what();
  }
  const long threw_count = all_reduce_sum(threw ? 1L : 0L);
  const long local_kind = static_cast<long>(kind);
  const bool same_kind = all_reduce_min(local_kind) == all_reduce_max(local_kind);
  const bool same_message = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("krylov-collective-exception"), std::string_view(message)}});
  return threw_count == static_cast<long>(n_ranks()) && same_kind && same_message &&
         message.find(expected_fragment) != std::string::npos;
}

TestLinearPreconditioner authenticated_preconditioner(
    const TestField& prototype, std::string_view implementation, TestApplyFn apply,
    std::string exact_parameters = {}, std::function<void(const ExecutionLane&)> prepare = {},
    TestVectorDistribution distribution = TestVectorDistribution::Distributed) {
  return TestLinearPreconditioner(
      prototype,
      TestLinearPreconditionerProvider::trusted_extension(
          {implementation, 1}, std::move(exact_parameters),
          [prepare = std::move(prepare), apply = std::move(apply)](const ExecutionLane& lane) {
            PreparedResourceFn session_prepare;
            if (prepare)
              session_prepare = [prepare, &lane] { prepare(lane); };
            return TestLinearPreconditionerCallbacks{std::move(session_prepare), apply,
                                                     [] { return std::size_t{0}; }};
          }),
      distribution);
}

TestAffineOperatorProvider reentrant_operator(TestApplyFn apply) {
  return TestAffineOperatorProvider::trusted_reentrant(std::move(apply),
                                                       [] { return std::size_t{0}; });
}

int run_krylov_collective_contract(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  long failures = n_ranks() == 2 ? 0 : 1;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };
  const int rank = my_rank();

  // Replication is a vector-space property, not a nullspace shortcut. A nonsingular identity
  // problem exercises the complete GMRES Arnoldi and DGKS reduction path over rank-local complete
  // copies. Its L2 metric counts the physical replica once, then a rank-dependent mutation is
  // rejected collectively instead of being silently averaged.
  {
    const TestVectorDistribution ownership = TestVectorDistribution::Replicated;
    TestField iterate = make_replicated_field();
    TestField rhs = make_replicated_field();
    rhs.set_val(Real(2));
    const TestKrylovFootprint footprint{1, ghost_extent(0), true};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, ownership);
    TestAffineOperatorProvider operator_provider = reentrant_operator(
        [](TestField& out, const TestField& in) { PureFieldAlgebra::copy(out, in); });
    TestLinearPreconditioner preconditioner = authenticated_preconditioner(
        iterate, "pops.test.krylov-collective.replicated-copy",
        [](TestField& out, const TestField& in) { PureFieldAlgebra::copy(out, in); }, {}, {},
        ownership);
    const AllocationEventStats before_problem_construction = allocation_event_stats();
    TestAffineProblem problem(
        iterate, std::move(operator_provider), std::move(preconditioner),
        LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
        [&] { return snapshot; }, {}, ownership);
    const AllocationEventStats after_problem_construction = allocation_event_stats();
    require(after_problem_construction.communication_calls ==
            before_problem_construction.communication_calls + 1);
    require(after_problem_construction.communication_bytes ==
            before_problem_construction.communication_bytes +
                ownership.validation_scratch_byte_count());
    TestKrylovWorkspace workspace(iterate, gmres_krylov_method<kDim>(3), footprint, ownership);
    problem.prepare(snapshot);
    const AllocationEventStats before_reprepare = allocation_event_stats();
    problem.prepare(snapshot);
    const AllocationEventStats after_reprepare = allocation_event_stats();
    require(after_reprepare.communication_calls == before_reprepare.communication_calls);
    require(after_reprepare.communication_bytes == before_reprepare.communication_bytes);
    workspace.bind(problem);
    require(problem.vector_distribution() == ownership);
    require(std::abs(static_cast<double>(problem.inner_product(rhs, rhs)) - 32.0) < 1e-12);
    require(std::abs(static_cast<double>(problem.residual_norm(rhs)) - std::sqrt(32.0)) < 1e-12);
    const SolveReport report = detail::solve_prepared_affine_in_place(
        problem, workspace, iterate, rhs,
        TestKrylovControls{gmres_krylov_method<kDim>(3), Real(1e-12), Real(0), 4});
    require(report.status == SolveStatus::kSolved);
    TestField error = make_replicated_field();
    PureFieldAlgebra::lincomb(error, Real(1), iterate, Real(-1), rhs);
    require(PureFieldAlgebra::max_abs(error, ownership) < Real(1e-12));

    fill_isometric_rank_permutation(rhs);
    require(uniformly_rejected([&] { (void)problem.residual_norm(rhs); },
                               kExactReplicaValidationFailure));
    TestField stable_left = make_replicated_field();
    stable_left.set_val(Real(1));
    require(uniformly_rejected([&] { (void)problem.inner_product(stable_left, rhs); },
                               kExactReplicaValidationFailure));
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(
              problem, workspace, iterate, rhs,
              TestKrylovControls{gmres_krylov_method<kDim>(3), Real(1e-12), Real(0), 4});
        },
        kExactReplicaValidationFailure));

    rhs.set_val(Real(2));
    fill_isometric_rank_permutation(iterate);
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(
              problem, workspace, iterate, rhs,
              TestKrylovControls{gmres_krylov_method<kDim>(3), Real(1e-12), Real(0), 4});
        },
        kExactReplicaValidationFailure));
  }

  // A distributed problem has no replica-validation footprint at all. Its construction must not
  // enter comm_allocator merely to materialize an empty persistent span.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    TestAffineOperatorProvider operator_provider = reentrant_operator(
        [](TestField& out, const TestField& in) { PureFieldAlgebra::copy(out, in); });
    const AllocationEventStats before_problem_construction = allocation_event_stats();
    TestAffineProblem problem(prototype, std::move(operator_provider),
                              TestLinearPreconditioner::identity(),
                              LinearOperatorProperties::symmetric_positive_definite(), footprint,
                              TestNullspacePolicy::nonsingular(), [&] { return snapshot; });
    const AllocationEventStats after_problem_construction = allocation_event_stats();
    require(after_problem_construction.communication_calls ==
            before_problem_construction.communication_calls);
    require(after_problem_construction.communication_bytes ==
            before_problem_construction.communication_bytes);
  }

  // Keep layout, snapshot and callable type fixed while varying first the implementation identity,
  // then one exact parameter. Both cases must fail before the provider can be applied: the
  // collective preflight authenticates the complete provider contract, not just presence.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), true};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    const auto require_provider_rejection = [&](std::string_view implementation,
                                                std::string exact_parameters) {
      std::atomic<long> apply_calls{0};
      TestAffineProblem problem(prototype,
                                reentrant_operator([](TestField& out, const TestField& in) {
                                  PureFieldAlgebra::copy(out, in);
                                }),
                                authenticated_preconditioner(
                                    prototype, implementation,
                                    [&](TestField& out, const TestField& in) {
                                      apply_calls.fetch_add(1, std::memory_order_relaxed);
                                      PureFieldAlgebra::copy(out, in);
                                    },
                                    std::move(exact_parameters)),
                                LinearOperatorProperties::general(), footprint,
                                TestNullspacePolicy::nonsingular(), [&] { return snapshot; });
      require(uniformly_rejected([&] { problem.prepare(snapshot); }, "provider contract differs"));
      require(apply_calls.load(std::memory_order_relaxed) == 0);
    };
    require_provider_rejection(rank == 0 ? "pops.test.krylov-collective.identity-a"
                                         : "pops.test.krylov-collective.identity-b",
                               exact_provider_parameters(std::int32_t{7}));
    require_provider_rejection("pops.test.krylov-collective.parameterized-copy",
                               exact_provider_parameters(std::int32_t{rank}));
  }

  // Verified extension callbacks cannot publish rank-dependent isometric outputs. The constant
  // response is prepared while the callback is deterministic; divergence is enabled only for the
  // solve matvec so this specifically exercises the post-apply trust boundary.
  {
    const TestVectorDistribution distribution = TestVectorDistribution::Replicated;
    TestField iterate = make_replicated_field();
    TestField rhs = make_replicated_field();
    rhs.set_val(Real(1));
    std::atomic<bool> diverge{false};
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, distribution);
    TestAffineProblem problem(
        iterate, reentrant_operator([&](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
          if (diverge.load(std::memory_order_acquire))
            fill_isometric_rank_permutation(out);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        TestNullspacePolicy::nonsingular(), [&] { return snapshot; }, {}, distribution);
    TestKrylovWorkspace workspace(iterate, cg_krylov_method<kDim>(), footprint, distribution);
    problem.prepare(snapshot);
    workspace.bind(problem);
    diverge.store(true, std::memory_order_release);
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(
              problem, workspace, iterate, rhs,
              TestKrylovControls{cg_krylov_method<kDim>(), Real(1e-12), Real(0), 4});
        },
        "prepared Krylov solve failed terminally on at least one communicator rank"));
  }

  // The same ownership contract supports a real singular prepared solve. A projection operator is
  // SPD on the constant-nullspace complement; two warm starts with different constant offsets
  // reuse the prepared nullspace certificate, persistent metric bases and workspace, and both publish
  // the same mean-zero representative.
  {
    const TestVectorDistribution ownership = TestVectorDistribution::Replicated;
    TestField iterate = make_replicated_field();
    TestField rhs = make_replicated_field();
    fill_replicated_mean_zero_rhs(rhs);
    auto constant_mode = std::make_shared<TestField>(make_replicated_field());
    constant_mode->set_val(Real(1));
    TestAffineOperatorProvider project_mean = TestAffineOperatorProvider::trusted_extension(
        {"pops.test.krylov-collective.project-mean", 1},
        std::string(ownership.collective_contract()),
        [constant_mode, ownership](const ExecutionLane& lane) {
          auto reduction_scratch =
              std::make_shared<std::vector<double>>(ownership.reduction_scratch_value_count(
                  detail::PreparedFieldAlgebra::kRobustDotPayloadWidth));
          return TestAffineOperatorCallbacks{{},
                                             [constant_mode, ownership, reduction_scratch, &lane](
                                                 TestField& out, const TestField& in) {
                                               PureFieldAlgebra::copy(out, in);
                                               const Real mean = detail::PreparedFieldAlgebra::dot(
                                                                     in, *constant_mode, ownership,
                                                                     *reduction_scratch, lane) /
                                                                 Real(8);
                                               PureFieldAlgebra::axpy(out, -mean, *constant_mode);
                                             },
                                             [] { return std::size_t{0}; }};
        });
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, ownership);
    TestVectorMetric metric(iterate, ownership, ScaledEuclideanMetricSource{Real(3)});
    TestNullspacePlan nullspace =
        constant_mean_zero_nullspace<kDim>("test://krylov/replicated-persistent-nullspace@1",
                                           "replicated constant mode for prepared Krylov", Real(1));
    TestAffineProblem problem(
        iterate, std::move(project_mean), TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        TestNullspacePolicy::preserving(std::move(nullspace)), [&] { return snapshot; }, {},
        ownership, metric);
    TestKrylovWorkspace workspace(iterate, cg_krylov_method<kDim>(), footprint, ownership, metric);
    const AllocationEventStats before_cold_nullspace_prepare = allocation_event_stats();
    problem.prepare(snapshot);
    const AllocationEventStats after_cold_nullspace_prepare = allocation_event_stats();
    require(after_cold_nullspace_prepare.communication_calls ==
            before_cold_nullspace_prepare.communication_calls);
    require(after_cold_nullspace_prepare.communication_bytes ==
            before_cold_nullspace_prepare.communication_bytes);
    workspace.bind(problem);
    const TestKrylovControls controls{cg_krylov_method<kDim>(), Real(1e-12), Real(0), 4};
    for (const Real offset : {Real(3), Real(-7)}) {
      iterate.set_val(offset);
      const SolveReport report =
          detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
      require(report.solved());
      require(std::abs(static_cast<double>(problem.inner_product(iterate, *constant_mode))) <
              1e-12);
      TestField error = make_replicated_field();
      PureFieldAlgebra::lincomb(error, Real(1), iterate, Real(-1), rhs);
      require(PureFieldAlgebra::max_abs(error, ownership) < Real(1e-12));
    }
  }

  // Declaring a genuinely distributed layout as replicated is rejected while the typed vector
  // metric is bound to its exact vector space, before an operator callback or scientific reduction
  // can double-count it.
  {
    const TestVectorDistribution ownership = TestVectorDistribution::Replicated;
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype, ownership);
    require(uniformly_rejected(
        [&] {
          (void)TestAffineProblem(
              prototype, reentrant_operator([](TestField& out, const TestField& in) {
                PureFieldAlgebra::copy(out, in);
              }),
              TestLinearPreconditioner::identity(),
              LinearOperatorProperties::symmetric_positive_definite(), footprint,
              TestNullspacePolicy::nonsingular(), [&] { return snapshot; }, {}, ownership);
        },
        "received invalid construction arguments on at least one communicator rank"));
  }

  // A prepared single-vector problem consumes the explicitly selected absolute level. Extra
  // hierarchy metadata is provider-owned and must not be rejected through a closed scope enum.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    TestNullspacePlan hierarchy = constant_mean_zero_nullspace<kDim>(
        "test://krylov/selected-level@1", "provider-owned hierarchy mode");
    hierarchy.bases[0].cell_measure.push_back(Real(0.25));
    TestAffineProblem problem(
        prototype, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        TestNullspacePolicy::preserving(std::move(hierarchy)), [&] { return snapshot; });
    try {
      problem.prepare(snapshot);
    } catch (...) {
      ++failures;
    }
  }

  // Metrics are typed providers, exact vector-space contracts and part of workspace binding. A
  // scaled metric changes both inner products and norms coherently while the Krylov code remains
  // implementation-agnostic; rank divergence and a differently-bound workspace fail uniformly.
  {
    TestField iterate = make_field();
    TestField rhs = make_field();
    rhs.set_val(Real(2));
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
    TestVectorMetric metric(iterate, TestVectorDistribution::Distributed,
                            ScaledEuclideanMetricSource{Real(3)});
    TestAffineProblem problem(
        iterate, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        TestNullspacePolicy::nonsingular(), [&] { return snapshot; }, {},
        TestVectorDistribution::Distributed, metric);
    TestKrylovWorkspace workspace(iterate, cg_krylov_method<kDim>(), footprint,
                                  TestVectorDistribution::Distributed, metric);
    problem.prepare(snapshot);
    workspace.bind(problem);
    require(std::abs(static_cast<double>(problem.inner_product(rhs, rhs)) - 96.0) < 1e-12);
    require(std::abs(static_cast<double>(problem.residual_norm(rhs)) - std::sqrt(96.0)) < 1e-12);
    const SolveReport report = detail::solve_prepared_affine_in_place(
        problem, workspace, iterate, rhs,
        TestKrylovControls{cg_krylov_method<kDim>(), Real(1e-12), Real(0), 2});
    require(report.solved());

    TestVectorMetric different_metric(iterate, TestVectorDistribution::Distributed,
                                      ScaledEuclideanMetricSource{Real(4)});
    TestKrylovWorkspace different_workspace(iterate, cg_krylov_method<kDim>(), footprint,
                                            TestVectorDistribution::Distributed, different_metric);
    require(
        uniformly_rejected([&] { different_workspace.bind(problem); }, "metric is incompatible"));
  }

  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    TestVectorMetric divergent_metric(prototype, TestVectorDistribution::Distributed,
                                      ScaledEuclideanMetricSource{rank == 0 ? Real(3) : Real(4)});
    TestAffineProblem problem(
        prototype, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        TestNullspacePolicy::nonsingular(), [&] { return snapshot; }, {},
        TestVectorDistribution::Distributed, std::move(divergent_metric));
    require(
        uniformly_rejected([&] { problem.prepare(snapshot); }, "vector metric contract differs"));
  }

  // A future schema growth must fail uniformly after the fixed min/max exchange.  It must never
  // terminate a rank or write past the stack payload before the other ranks reach the boundary.
  {
    detail::KrylovCollectivePayload payload;
    for (std::size_t index = 0; index < detail::KrylovCollectivePayload::kCapacity; ++index)
      payload.append(static_cast<std::uint64_t>(index));
    require(uniformly_rejected([&] { (void)detail::collective_payload_agrees(payload); },
                               "exceeded its fixed internal capacity"));
  }

  // The problem contract is authenticated before any preparation callback or nullspace Gram work.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    TestAffineProblem problem(
        prototype, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        rank == 0 ? LinearOperatorProperties::general()
                  : LinearOperatorProperties::symmetric_positive_definite(),
        footprint, TestNullspacePolicy::nonsingular(), [&] { return snapshot; });
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "problem contract differs"));
  }

  // Direct policy preparation is itself collective-safe. One rank cannot return through the
  // nonsingular branch while another enters a nullspace Gram reduction.
  {
    TestField layout = make_field();
    TestNullspacePolicy policy =
        rank == 0 ? TestNullspacePolicy::nonsingular()
                  : TestNullspacePolicy::preserving(constant_mean_zero_nullspace<kDim>(
                        "test://rank-divergent-policy@1", "rank-divergent direct policy"));
    require(uniformly_rejected([&] { policy.prepare(layout, TestVectorDistribution::Distributed); },
                               "collective preflight rejected"));
  }

  // A callback may throw on one rank only after completing the same callback-owned MPI trace.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    TestAffineProblem problem(
        prototype, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        TestNullspacePolicy::nonsingular(), [&] { return snapshot; },
        [&] {
          (void)all_reduce_sum(1L);
          if (rank == 0)
            throw std::runtime_error("rank-local freeze failure after collective");
        });
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "resource freeze failed"));
  }

  // The extension boundary is generic and collective-safe for arbitrary method providers: after a
  // complete provider trace, neither a rank-local exception nor a rank-dependent SolveReport may
  // escape as split control flow. Both become the same typed invalid result on every rank.
  for (const auto behavior : {
           CollectiveBoundaryKrylovProvider::Behavior::kThrowOnRankZero,
           CollectiveBoundaryKrylovProvider::Behavior::kDivergentReport,
       }) {
    TestField method_iterate = make_field();
    TestField method_rhs = make_field();
    method_rhs.set_val(Real(1));
    const TestKrylovFootprint method_footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot method_snapshot = snapshot_for(method_iterate);
    const TestKrylovMethod method(
        std::make_shared<const CollectiveBoundaryKrylovProvider>(behavior),
        PreparedProviderOptions{"pops.test.krylov.collective-probe.options@1", {}});
    TestAffineProblem method_problem(
        method_iterate, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), method_footprint,
        TestNullspacePolicy::nonsingular(), [&] { return method_snapshot; });
    TestKrylovWorkspace method_workspace(method_iterate, method, method_footprint);
    method_problem.prepare(method_snapshot);
    method_workspace.bind(method_problem);
    const SolveReport invalid = detail::solve_prepared_affine_in_place(
        method_problem, method_workspace, method_iterate, method_rhs,
        TestKrylovControls{method, Real(1e-12), Real(0), 2});
    const std::string_view expected_reason =
        behavior == CollectiveBoundaryKrylovProvider::Behavior::kThrowOnRankZero
            ? "prepared Krylov provider failed after its collective solve trace"
            : "prepared Krylov provider report differs between communicator ranks";
    require(invalid.status == SolveStatus::kInvalidEvaluation);
    require(invalid.action == SolveAction::kFailRun);
    require(invalid.reason == expected_reason);
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("external-krylov-invalid-report"), std::string_view(invalid.reason)}}));
  }

  // Infrastructure failure is terminal even when another rank records a retryable numerical
  // failure in the same completed provider trace. The wrapper must not let max-enum ordering turn
  // this mixed state into RejectAttempt.
  {
    TestField method_iterate = make_field();
    TestField method_rhs = make_field();
    method_rhs.set_val(Real(1));
    const TestKrylovFootprint method_footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot method_snapshot = snapshot_for(method_iterate);
    auto mixed_apply_armed = std::make_shared<std::atomic<bool>>(false);
    std::atomic<bool> retry_emitted{false};
    const TestKrylovMethod method(
        std::make_shared<const CollectiveBoundaryKrylovProvider>(
            CollectiveBoundaryKrylovProvider::Behavior::kThrowOnRankZeroAndRetryOnRankOne,
            mixed_apply_armed),
        PreparedProviderOptions{"pops.test.krylov.collective-probe.options@1", {}});
    TestAffineProblem method_problem(
        method_iterate,
        reentrant_operator([&, mixed_apply_armed](TestField& out, const TestField& in) {
          if (mixed_apply_armed->load(std::memory_order_acquire) && my_rank() == 1) {
            retry_emitted.store(true, std::memory_order_release);
            throw FluxEvaluationFailure(EvaluationStatus::kRetry, UINT32_C(0x4d495845),
                                        "mixed_provider_failure");
          }
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), method_footprint,
        TestNullspacePolicy::nonsingular(), [&] { return method_snapshot; });
    TestKrylovWorkspace method_workspace(method_iterate, method, method_footprint);
    method_problem.prepare(method_snapshot);
    method_workspace.bind(method_problem);
    const SolveReport invalid = detail::solve_prepared_affine_in_place(
        method_problem, method_workspace, method_iterate, method_rhs,
        TestKrylovControls{method, Real(1e-12), Real(0), 2});
    require(all_reduce_sum(retry_emitted.load(std::memory_order_acquire) ? 1L : 0L) == 1);
    require(invalid.status == SolveStatus::kInvalidEvaluation);
    require(invalid.action == SolveAction::kFailRun);
    require(invalid.reason == "prepared Krylov provider failed after its collective solve trace");
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("mixed-provider-failure-priority"), std::string_view(invalid.reason)}}));
  }

  // Exercise the real np=2 publication boundary with a reason four times larger than the removed
  // fixed limit. Identical reasons survive byte-for-byte; one rank-local byte after offset 4096 is
  // rejected uniformly instead of being silently truncated.
  for (const auto behavior : {
           LongReasonCollectiveKrylovProvider::Behavior::kIdentical,
           LongReasonCollectiveKrylovProvider::Behavior::kDivergentAfterLegacyLimit,
       }) {
    TestField method_iterate = make_field();
    TestField method_rhs = make_field();
    method_rhs.set_val(Real(1));
    const TestKrylovFootprint method_footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot method_snapshot = snapshot_for(method_iterate);
    const TestKrylovMethod method(
        std::make_shared<const LongReasonCollectiveKrylovProvider>(behavior),
        PreparedProviderOptions{"pops.test.krylov.collective-long-reason.options@1", {}});
    TestAffineProblem method_problem(
        method_iterate, reentrant_operator([](TestField& out, const TestField& in) {
          PureFieldAlgebra::copy(out, in);
        }),
        TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), method_footprint,
        TestNullspacePolicy::nonsingular(), [&] { return method_snapshot; });
    TestKrylovWorkspace method_workspace(method_iterate, method, method_footprint);
    method_problem.prepare(method_snapshot);
    method_workspace.bind(method_problem);

    const SolveReport report = detail::solve_prepared_affine_in_place(
        method_problem, method_workspace, method_iterate, method_rhs,
        TestKrylovControls{method, Real(1e-12), Real(0), 2});
    if (behavior == LongReasonCollectiveKrylovProvider::Behavior::kIdentical) {
      require(report.status == SolveStatus::kIterationLimit);
      require(report.reason == std::string(LongReasonCollectiveKrylovProvider::kReasonBytes, 'r'));
    } else {
      require(report.status == SolveStatus::kInvalidEvaluation);
      require(report.action == SolveAction::kFailRun);
      require(report.reason ==
              "prepared Krylov provider report differs between communicator ranks");
    }
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("external-krylov-long-report-result"),
          std::string_view(report.reason)}}));
  }

  // A successful nullspace certificate is reusable only after the fixed collective preflight.
  // A later rank-symmetric stage failure must leave the policy uniformly re-preparable: the retry
  // still performs freeze/preconditioner/A(0), but it does not rebuild the immutable nullspace
  // Gram certificate or its persistent metric-basis storage.
  {
    TestField prototype = make_field();
    const TestKrylovFootprint footprint{1, ghost_extent(0), true};
    OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    auto mutable_mask = std::make_shared<TestField>(make_field());
    mutable_mask->set_val(Real(1));
    TestNullspacePlan nullspace = constant_mean_zero_nullspace<kDim>(
        "test://collective-contract/nullspace-certificate@1", "constant test mode");
    nullspace.bases[0].masks = {mutable_mask};
    std::atomic<bool> fail_preconditioner{true};
    std::atomic<long> freeze_calls{0};
    std::atomic<long> preconditioner_prepare_calls{0};
    std::atomic<long> operator_calls{0};
    TestLinearPreconditioner preconditioner = authenticated_preconditioner(
        prototype, "pops.test.krylov-collective.retryable-copy",
        [](TestField& out, const TestField& in) { PureFieldAlgebra::copy(out, in); }, {},
        [&](const ExecutionLane& lane) {
          preconditioner_prepare_calls.fetch_add(1, std::memory_order_relaxed);
          (void)all_reduce_sum(1L, lane);
          if (fail_preconditioner.load(std::memory_order_acquire) && lane.rank() == 0)
            throw std::runtime_error("retryable prepared preconditioner failure");
        });
    TestAffineProblem problem(
        prototype, reentrant_operator([&](TestField& out, const TestField& in) {
          operator_calls.fetch_add(1, std::memory_order_relaxed);
          PureFieldAlgebra::copy(out, in);
        }),
        std::move(preconditioner),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        TestNullspacePolicy::preserving(std::move(nullspace)), [&] { return snapshot; },
        [&] {
          freeze_calls.fetch_add(1, std::memory_order_relaxed);
          (void)all_reduce_sum(1L);
        });

    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "preconditioner setup failed"));
    fail_preconditioner.store(false, std::memory_order_release);
    problem.prepare(snapshot);
    require(problem.prepared());
    require(freeze_calls.load(std::memory_order_relaxed) == 2);
    require(preconditioner_prepare_calls.load(std::memory_order_relaxed) == 2);
    require(operator_calls.load(std::memory_order_relaxed) == 1);

    mutable_mask->set_val(Real(0));
    snapshot.resources[0] += 1;
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "nullspace setup failed"));
    require(operator_calls.load(std::memory_order_relaxed) == 1);
  }

  TestField iterate = make_field();
  TestField rhs = make_field();
  rhs.set_val(Real(1));
  const TestKrylovFootprint footprint{1, ghost_extent(0), false};
  const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  std::atomic<bool> throw_after_collective{false};
  std::atomic<long> operator_calls{0};
  TestAffineOperatorProvider collective_operator = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov-collective.collective-operator", 1}, {}, [&](const ExecutionLane& lane) {
        return TestAffineOperatorCallbacks{
            {},
            [&](TestField& out, const TestField& in) {
              operator_calls.fetch_add(1, std::memory_order_relaxed);
              (void)all_reduce_sum(1L, lane);
              if (throw_after_collective.load(std::memory_order_acquire) && lane.rank() == 0)
                throw FluxEvaluationFailure(EvaluationStatus::kRetry, UINT32_C(0x4d504952),
                                            "rank_local_collective_flux");
              PureFieldAlgebra::copy(out, in);
            },
            [] { return std::size_t{0}; }};
      });
  TestAffineProblem problem(iterate, std::move(collective_operator),
                            TestLinearPreconditioner::identity(),
                            LinearOperatorProperties::symmetric_positive_definite(), footprint,
                            TestNullspacePolicy::nonsingular(), [&] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, cg_krylov_method<kDim>(), footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const TestKrylovControls valid{cg_krylov_method<kDim>(), Real(1e-8), Real(0), 4};

  {
    TestField incompatible = make_field(rank == 0 ? 2 : 1);
    const TestField& local = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected([&] { (void)problem.inner_product(local, rhs); },
                               "incompatible vector space"));
  }
  {
    TestField output = make_field();
    TestField& local_output = rank == 0 ? rhs : output;
    const long calls_before = operator_calls.load(std::memory_order_relaxed);
    require(uniformly_rejected([&] { problem.apply_linear(local_output, rhs); },
                               "output aliases an input field"));
    require(operator_calls.load(std::memory_order_relaxed) == calls_before);
  }
  if (n_ranks() > 1) {
    TestField output = make_field();
    require(uniformly_rejected(
        [&] {
          if (rank == 0)
            problem.apply_linear(output, rhs);
          else
            problem.true_residual(output, rhs, iterate);
        },
        "prepared affine public operations differ across communicator ranks"));
  }
  {
    TestField incompatible = make_field(rank == 0 ? 2 : 1);
    const TestField& local = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected([&] { (void)problem.residual_norm(local); },
                               "incompatible vector space"));
  }

  {
    TestKrylovControls controls = valid;
    if (rank == 0)
      controls.max_iterations = 0;
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
        },
        "collective contract differs"));
  }
  {
    TestKrylovControls controls = valid;
    if (rank == 0)
      controls.rel_tol = Real(1e-7);
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
        },
        "collective contract differs"));
  }
  {
    const TestField& local_rhs = rank == 0 ? iterate : rhs;
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, local_rhs,
                                                       valid);
        },
        "distinct storage"));
  }
  {
    TestField incompatible = make_field(rank == 0 ? 2 : 1);
    const TestField& local_rhs = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, local_rhs,
                                                       valid);
        },
        "incompatible vector space"));
  }
  {
    TestKrylovWorkspace unbound(iterate, cg_krylov_method<kDim>(), footprint);
    TestKrylovWorkspace& local_workspace = rank == 0 ? unbound : workspace;
    require(uniformly_rejected(
        [&] {
          (void)detail::solve_prepared_affine_in_place(problem, local_workspace, iterate, rhs,
                                                       valid);
        },
        "not bound"));
  }
  {
    require(uniformly_rejected(
        [&] {
          (void)TestKrylovWorkspace(
              iterate, rank == 0 ? cg_krylov_method<kDim>() : richardson_krylov_method<kDim>(),
              footprint);
        },
        "workspace requirements differ"));
  }

  const auto require_sticky_failure_report = [&](const SolveReport& report,
                                                 std::string_view label) {
    require(report.status == SolveStatus::kInvalidEvaluation);
    require(report.action == SolveAction::kFailRun);
    require(report.reason ==
            "prepared operator or preconditioner application failed during Krylov recurrence: "
            "prepared callback raised an unknown exception");
    require(all_ranks_agree_exact_ordered_byte_pairs({{label, std::string_view(report.reason)}}));
  };

  // GMRES applies A(v), then the prepared preconditioner, before Arnoldi's reduction. Rank zero's
  // operator failure is followed by a preconditioner callback that deliberately writes finite data;
  // the workspace-level sticky status must re-poison it and survive to the common report boundary.
  {
    TestField sticky_iterate = make_field();
    TestField sticky_rhs = make_field();
    sticky_rhs.set_val(Real(1));
    const TestKrylovFootprint sticky_footprint{1, ghost_extent(0), true};
    const OperatorEvaluationSnapshot sticky_snapshot = snapshot_for(sticky_iterate);
    std::atomic<long> operator_target{std::numeric_limits<long>::max()};
    std::atomic<long> sticky_operator_calls{0};
    std::atomic<long> sticky_preconditioner_calls{0};
    std::atomic<bool> operator_failed_locally{false};
    TestAffineOperatorProvider sticky_operator = TestAffineOperatorProvider::trusted_extension(
        {"pops.test.krylov.sticky-gmres-operator", 1}, {}, [&](const ExecutionLane& callback_lane) {
          return TestAffineOperatorCallbacks{
              {},
              [&, lane = &callback_lane](TestField& out, const TestField& in) {
                const long call = sticky_operator_calls.fetch_add(1, std::memory_order_relaxed) + 1;
                (void)all_reduce_sum(1L, *lane);
                if (call == operator_target.load(std::memory_order_acquire) && lane->rank() == 0) {
                  operator_failed_locally.store(true, std::memory_order_release);
                  throw std::runtime_error("rank-local GMRES operator failure");
                }
                PureFieldAlgebra::copy(out, in);
              },
              [] { return std::size_t{0}; }};
        });
    TestLinearPreconditioner sticky_preconditioner = authenticated_preconditioner(
        sticky_iterate, "pops.test.krylov.sticky-gmres-preconditioner",
        [&](TestField& out, const TestField& in) {
          sticky_preconditioner_calls.fetch_add(1, std::memory_order_relaxed);
          if (operator_failed_locally.load(std::memory_order_acquire))
            PureFieldAlgebra::fill_valid(out, Real(0));
          else
            PureFieldAlgebra::copy(out, in);
        });
    TestAffineProblem sticky_problem(
        sticky_iterate, std::move(sticky_operator), std::move(sticky_preconditioner),
        LinearOperatorProperties::general(), sticky_footprint, TestNullspacePolicy::nonsingular(),
        [&] { return sticky_snapshot; });
    const TestKrylovMethod sticky_method = gmres_krylov_method<kDim>(3);
    TestKrylovWorkspace sticky_workspace(sticky_iterate, sticky_method, sticky_footprint);
    sticky_problem.prepare(sticky_snapshot);
    sticky_workspace.bind(sticky_problem);
    const long operator_calls_before = sticky_operator_calls.load(std::memory_order_relaxed);
    const long preconditioner_calls_before =
        sticky_preconditioner_calls.load(std::memory_order_relaxed);
    operator_target.store(operator_calls_before + 2, std::memory_order_release);
    const SolveReport sticky_report = detail::solve_prepared_affine_in_place(
        sticky_problem, sticky_workspace, sticky_iterate, sticky_rhs,
        TestKrylovControls{sticky_method, Real(1e-12), Real(0), 6});
    require_sticky_failure_report(sticky_report, "sticky-gmres-op-preconditioner");
    require(sticky_preconditioner_calls.load(std::memory_order_relaxed) >=
            preconditioner_calls_before + 2);
    require(all_reduce_sum(operator_failed_locally.load(std::memory_order_acquire) ? 1L : 0L) == 1);
  }

  // On the second BiCGStab preconditioner application the solve scale is already fixed, so the
  // following operator callback still runs. It rewrites finite output on the failing rank; only the
  // cross-provider workspace latch can keep that earlier preconditioner failure authoritative.
  {
    TestField sticky_iterate = make_field();
    TestField sticky_rhs = make_field();
    sticky_rhs.set_val(Real(1));
    const TestKrylovFootprint sticky_footprint{1, ghost_extent(0), true};
    const OperatorEvaluationSnapshot sticky_snapshot = snapshot_for(sticky_iterate);
    std::atomic<long> preconditioner_target{std::numeric_limits<long>::max()};
    std::atomic<long> sticky_operator_calls{0};
    std::atomic<long> sticky_preconditioner_calls{0};
    std::atomic<bool> preconditioner_failed_locally{false};
    TestAffineOperatorProvider sticky_operator = TestAffineOperatorProvider::trusted_extension(
        {"pops.test.krylov.sticky-bicgstab-operator", 1}, {},
        [&](const ExecutionLane& callback_lane) {
          return TestAffineOperatorCallbacks{
              {},
              [&, lane = &callback_lane](TestField& out, const TestField& in) {
                sticky_operator_calls.fetch_add(1, std::memory_order_relaxed);
                (void)all_reduce_sum(1L, *lane);
                if (preconditioner_failed_locally.load(std::memory_order_acquire))
                  PureFieldAlgebra::fill_valid(out, Real(0));
                else
                  apply_alternating_diagonal(out, in);
              },
              [] { return std::size_t{0}; }};
        });
    TestLinearPreconditioner sticky_preconditioner = authenticated_preconditioner(
        sticky_iterate, "pops.test.krylov.sticky-bicgstab-preconditioner",
        [&](TestField& out, const TestField& in) {
          const long call = sticky_preconditioner_calls.fetch_add(1, std::memory_order_relaxed) + 1;
          if (call == preconditioner_target.load(std::memory_order_acquire) && my_rank() == 0) {
            preconditioner_failed_locally.store(true, std::memory_order_release);
            throw std::runtime_error("rank-local BiCGStab preconditioner failure");
          }
          PureFieldAlgebra::copy(out, in);
        });
    TestAffineProblem sticky_problem(
        sticky_iterate, std::move(sticky_operator), std::move(sticky_preconditioner),
        LinearOperatorProperties::general(), sticky_footprint, TestNullspacePolicy::nonsingular(),
        [&] { return sticky_snapshot; });
    const TestKrylovMethod sticky_method = bicgstab_krylov_method<kDim>();
    TestKrylovWorkspace sticky_workspace(sticky_iterate, sticky_method, sticky_footprint);
    sticky_problem.prepare(sticky_snapshot);
    sticky_workspace.bind(sticky_problem);
    const long operator_calls_before = sticky_operator_calls.load(std::memory_order_relaxed);
    const long preconditioner_calls_before =
        sticky_preconditioner_calls.load(std::memory_order_relaxed);
    preconditioner_target.store(preconditioner_calls_before + 2, std::memory_order_release);
    const SolveReport sticky_report = detail::solve_prepared_affine_in_place(
        sticky_problem, sticky_workspace, sticky_iterate, sticky_rhs,
        TestKrylovControls{sticky_method, Real(1e-12), Real(0), 6});
    require_sticky_failure_report(sticky_report, "sticky-bicgstab-preconditioner-op");
    require(sticky_operator_calls.load(std::memory_order_relaxed) >= operator_calls_before + 3);
    require(all_reduce_sum(
                preconditioner_failed_locally.load(std::memory_order_acquire) ? 1L : 0L) == 1);
  }

  // A fifth-party method has no method-name branch in the wrapper. Its first failed apply must stay
  // latched across a second successful, finite-writing apply until the provider's first reduction.
  {
    TestField sticky_iterate = make_field();
    TestField sticky_rhs = make_field();
    sticky_rhs.set_val(Real(1));
    const TestKrylovFootprint sticky_footprint{1, ghost_extent(0), false};
    const OperatorEvaluationSnapshot sticky_snapshot = snapshot_for(sticky_iterate);
    std::atomic<long> operator_target{std::numeric_limits<long>::max()};
    std::atomic<long> sticky_operator_calls{0};
    std::atomic<bool> operator_failed_locally{false};
    TestAffineOperatorProvider sticky_operator = TestAffineOperatorProvider::trusted_reentrant(
        [&](TestField& out, const TestField& in) {
          const long call = sticky_operator_calls.fetch_add(1, std::memory_order_relaxed) + 1;
          if (call == operator_target.load(std::memory_order_acquire) && my_rank() == 0) {
            operator_failed_locally.store(true, std::memory_order_release);
            throw std::runtime_error("rank-local custom-method operator failure");
          }
          if (operator_failed_locally.load(std::memory_order_acquire))
            PureFieldAlgebra::fill_valid(out, Real(0));
          else
            PureFieldAlgebra::copy(out, in);
        },
        [] { return std::size_t{0}; });
    TestAffineProblem sticky_problem(
        sticky_iterate, std::move(sticky_operator), TestLinearPreconditioner::identity(),
        LinearOperatorProperties::general(), sticky_footprint, TestNullspacePolicy::nonsingular(),
        [&] { return sticky_snapshot; });
    const TestKrylovMethod sticky_method(
        std::make_shared<TwoApplyBeforeReductionKrylovProvider>(),
        PreparedProviderOptions{"pops.test.krylov.two-apply-before-reduction.options@1", {}});
    TestKrylovWorkspace sticky_workspace(sticky_iterate, sticky_method, sticky_footprint);
    sticky_problem.prepare(sticky_snapshot);
    sticky_workspace.bind(sticky_problem);
    const long operator_calls_before = sticky_operator_calls.load(std::memory_order_relaxed);
    operator_target.store(operator_calls_before + 2, std::memory_order_release);
    const SolveReport sticky_report = detail::solve_prepared_affine_in_place(
        sticky_problem, sticky_workspace, sticky_iterate, sticky_rhs,
        TestKrylovControls{sticky_method, Real(1e-12), Real(0), 4});
    require_sticky_failure_report(sticky_report, "sticky-custom-two-apply");
    require(sticky_operator_calls.load(std::memory_order_relaxed) >= operator_calls_before + 3);
    require(all_reduce_sum(operator_failed_locally.load(std::memory_order_acquire) ? 1L : 0L) == 1);
  }

  {
    throw_after_collective.store(true, std::memory_order_release);
    const SolveReport invalid =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, valid);
    require(invalid.status == SolveStatus::kInvalidEvaluation);
    require(invalid.action == SolveAction::kRejectAttempt);
    require(
        invalid.reason ==
        "prepared operator application failed before Krylov recurrence: numerical flux evaluation "
        "retry during rank_local_collective_flux: reason_code=0x4d504952");
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("rank-local-hot-apply-failure"), std::string_view(invalid.reason)}}));
  }

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_krylov_collective_contract, RankLocalFailuresAreUniformBeforeScientificCollectives) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_krylov_collective_contract, "test_krylov_collective_contract"),
      0);
}
