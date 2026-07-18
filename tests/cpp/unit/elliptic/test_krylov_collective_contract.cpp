#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/parallel/comm.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

OperatorEvaluationSnapshot snapshot_for(
    const MultiFab& field,
    PreparedVectorDistribution ownership = PreparedVectorDistribution::Distributed) {
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

MultiFab make_field(int components = 1, int ghosts = 0) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {3, 1}}});
  MultiFab field(boxes, DistributionMapping(std::vector<int>{0, 1}), components, ghosts);
  field.set_val(Real(0));
  return field;
}

MultiFab make_replicated_field(int components = 1, int ghosts = 0) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {3, 1}}});
  MultiFab field(boxes, DistributionMapping(std::vector<int>(boxes.size(), my_rank())), components,
                 ghosts);
  field.set_val(Real(0));
  return field;
}

struct ReplicatedMeanZeroRhsKernel {
  Array4 values;
  POPS_HD void operator()(int i, int j) const { values(i, j) = i < 2 ? Real(1) : Real(-1); }
};

void fill_replicated_mean_zero_rhs(MultiFab& field) {
  for (int local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), ReplicatedMeanZeroRhsKernel{field.fab(local).array()});
}

void fill_isometric_rank_permutation(MultiFab& field) {
  field.set_val(Real(0));
  field.sync_host();
  Array4 values = field.fab(0).array();
  if (my_rank() == 0)
    values(0, 0) = Real(1);
  else
    values(1, 0) = Real(1);
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

  Real inner_product(const MultiFab& left, const MultiFab& right,
                     const PreparedVectorDistribution& distribution,
                     std::span<double> scratch) const {
    return scale * detail::PreparedFieldAlgebra::dot(left, right, distribution, scratch);
  }

  Real norm(const MultiFab& value, const PreparedVectorDistribution& distribution,
            std::span<double> scratch) const {
    return std::sqrt(scale) * detail::PreparedFieldAlgebra::norm(value, distribution, scratch);
  }

  Real absolute_inner_product(const MultiFab& left, const MultiFab& right,
                              const PreparedVectorDistribution& distribution,
                              std::span<double> scratch) const {
    return scale * detail::PreparedFieldAlgebra::absolute_dot(left, right, distribution, scratch);
  }

  Real nullspace_inner_product(const MultiFab& left, const MultiFab& right, Real cell_measure,
                               const PreparedVectorDistribution& distribution,
                               std::span<double> scratch) const {
    return scale * detail::PreparedFieldAlgebra::nullspace_pairing(left, right, cell_measure, false,
                                                                   distribution, scratch);
  }

  Real nullspace_absolute_inner_product(const MultiFab& left, const MultiFab& right,
                                        Real cell_measure,
                                        const PreparedVectorDistribution& distribution,
                                        std::span<double> scratch) const {
    return scale * detail::PreparedFieldAlgebra::nullspace_pairing(left, right, cell_measure, true,
                                                                   distribution, scratch);
  }

  Real local_inner_product(const MultiFab& left, const MultiFab& right) const {
    return scale * detail::PreparedFieldAlgebra::local_dot(left, right);
  }

  void local_robust_inner_product_payload(const MultiFab& left, const MultiFab& right,
                                          std::span<double> payload) const {
    detail::PreparedFieldAlgebra::local_robust_dot_payload(left, right, payload.data());
  }

  Real inner_product_from_global_robust_payload(std::span<const double> payload) const {
    return scale * detail::PreparedFieldAlgebra::dot_from_global_robust_payload(payload.data());
  }
};

/// Adversarial external methods for the common post-provider boundary. Both complete the same
/// provider-owned collective trace first. The common Krylov wrapper must then turn either a
/// rank-local exception or a divergent report into one uniform InvalidEvaluation result.
class CollectiveBoundaryKrylovProvider final : public PreparedKrylovMethodProvider {
 public:
  enum class Behavior { kThrowOnRankZero, kDivergentReport };

  explicit CollectiveBoundaryKrylovProvider(Behavior behavior) : behavior_(behavior) {}

  std::string_view identity() const noexcept override {
    return behavior_ == Behavior::kThrowOnRankZero
               ? "pops.test.krylov.collective-throw"
               : "pops.test.krylov.collective-divergent-report";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return behavior_ == Behavior::kThrowOnRankZero
               ? "pops.test.krylov.collective-throw@1"
               : "pops.test.krylov.collective-divergent-report@1";
  }
  KrylovMethodValidation validate_controls(
      const KrylovMethodControls& controls,
      const PreparedProviderOptions&) const noexcept override {
    return detail::validate_common_krylov_controls(controls);
  }
  KrylovMethodValidation validate_problem(
      const KrylovMethodProblemFacts& facts,
      const PreparedProviderOptions&) const noexcept override {
    if (const KrylovMethodValidation common = detail::validate_generic_problem_facts(facts);
        !common.accepted())
      return common;
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "collective probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest& request,
      const PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("collective probe is unpreconditioned");
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    (void)all_reduce_sum(1L);
    if (behavior_ == Behavior::kThrowOnRankZero && my_rank() == 0)
      throw std::runtime_error("rank-local provider failure after complete trace");
    const int iterations =
        behavior_ == Behavior::kDivergentReport && my_rank() != 0 ? 2 : 1;
    return context.report(context.initial_physical_residual(), iterations,
                          SolveStatus::kIterationLimit);
  }

 private:
  Behavior behavior_;
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

PreparedLinearPreconditioner authenticated_preconditioner(
    const MultiFab& prototype, std::string_view implementation, ApplyFn apply,
    std::string exact_parameters = {}, PreparedResourceFn prepare = {},
    PreparedVectorDistribution distribution = PreparedVectorDistribution::Distributed) {
  return PreparedLinearPreconditioner(
      prototype,
      PreparedLinearPreconditionerProvider::trusted_extension(
          {implementation, 1}, std::move(exact_parameters), std::move(prepare),
          std::move(apply)),
      distribution);
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
    const PreparedVectorDistribution ownership = PreparedVectorDistribution::Replicated;
    MultiFab iterate = make_replicated_field();
    MultiFab rhs = make_replicated_field();
    rhs.set_val(Real(2));
    const KrylovFootprint footprint{1, 0, true};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, ownership);
    PreparedAffineLinearProblem problem(
        iterate, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        authenticated_preconditioner(
            iterate, "pops.test.krylov-collective.replicated-copy",
            [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); }, {}, {},
            ownership),
        LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
        [&] { return snapshot; }, {}, ownership);
    KrylovWorkspace workspace(iterate, gmres_krylov_method(3), footprint, ownership);
    problem.prepare(snapshot);
    workspace.bind(problem);
    require(problem.vector_distribution() == ownership);
    require(std::abs(static_cast<double>(problem.inner_product(rhs, rhs)) - 32.0) < 1e-12);
    require(std::abs(static_cast<double>(problem.residual_norm(rhs)) - std::sqrt(32.0)) < 1e-12);
    const SolveReport report = solve_prepared_affine(
        problem, workspace, iterate, rhs,
        KrylovControls{gmres_krylov_method(3), Real(1e-12), Real(0), 4});
    require(report.status == SolveStatus::kSolved);
    MultiFab error = make_replicated_field();
    PureFieldAlgebra::lincomb(error, Real(1), iterate, Real(-1), rhs);
    require(PureFieldAlgebra::max_abs(error, ownership) < Real(1e-12));

    fill_isometric_rank_permutation(rhs);
    require(uniformly_rejected([&] { (void)problem.residual_norm(rhs); },
                               kExactReplicaValidationFailure));
    MultiFab stable_left = make_replicated_field();
    stable_left.set_val(Real(1));
    require(uniformly_rejected([&] { (void)problem.inner_product(stable_left, rhs); },
                               kExactReplicaValidationFailure));
    require(uniformly_rejected(
        [&] {
          (void)solve_prepared_affine(
              problem, workspace, iterate, rhs,
              KrylovControls{gmres_krylov_method(3), Real(1e-12), Real(0), 4});
        },
        kExactReplicaValidationFailure));

    rhs.set_val(Real(2));
    fill_isometric_rank_permutation(iterate);
    require(uniformly_rejected(
        [&] {
          (void)solve_prepared_affine(
              problem, workspace, iterate, rhs,
              KrylovControls{gmres_krylov_method(3), Real(1e-12), Real(0), 4});
        },
        kExactReplicaValidationFailure));
  }

  // Keep layout, snapshot and callable type fixed while varying first the implementation identity,
  // then one exact parameter. Both cases must fail before the provider can be applied: the
  // collective preflight authenticates the complete provider contract, not just presence.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, true};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    const auto require_provider_rejection = [&](std::string_view implementation,
                                                std::string exact_parameters) {
      long apply_calls = 0;
      PreparedAffineLinearProblem problem(
          prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
          authenticated_preconditioner(
              prototype, implementation,
              [&](MultiFab& out, const MultiFab& in) {
                ++apply_calls;
                PureFieldAlgebra::copy(out, in);
              },
              std::move(exact_parameters)),
          LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
          [&] { return snapshot; });
      require(uniformly_rejected([&] { problem.prepare(snapshot); },
                                 "provider contract differs"));
      require(apply_calls == 0);
    };
    require_provider_rejection(
        rank == 0 ? "pops.test.krylov-collective.identity-a"
                  : "pops.test.krylov-collective.identity-b",
        exact_provider_parameters(std::int32_t{7}));
    require_provider_rejection("pops.test.krylov-collective.parameterized-copy",
                               exact_provider_parameters(std::int32_t{rank}));
  }

  // Verified extension callbacks cannot publish rank-dependent isometric outputs. The constant
  // response is prepared while the callback is deterministic; divergence is enabled only for the
  // solve matvec so this specifically exercises the post-apply trust boundary.
  {
    const PreparedVectorDistribution distribution = PreparedVectorDistribution::Replicated;
    MultiFab iterate = make_replicated_field();
    MultiFab rhs = make_replicated_field();
    rhs.set_val(Real(1));
    bool diverge = false;
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, distribution);
    PreparedAffineLinearProblem problem(
        iterate,
        [&](MultiFab& out, const MultiFab& in) {
          PureFieldAlgebra::copy(out, in);
          if (diverge)
            fill_isometric_rank_permutation(out);
        },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; }, {}, distribution);
    KrylovWorkspace workspace(iterate, cg_krylov_method(), footprint, distribution);
    problem.prepare(snapshot);
    workspace.bind(problem);
    diverge = true;
    require(uniformly_rejected(
        [&] {
          (void)solve_prepared_affine(
              problem, workspace, iterate, rhs,
              KrylovControls{cg_krylov_method(), Real(1e-12), Real(0), 4});
        },
        kExactReplicaValidationFailure));
  }

  // The same ownership contract supports a real singular prepared solve. A projection operator is
  // SPD on the constant-nullspace complement; two warm starts with different constant offsets
  // reuse the prepared nullspace certificate, persistent metric bases and workspace, and both publish
  // the same mean-zero representative.
  {
    const PreparedVectorDistribution ownership = PreparedVectorDistribution::Replicated;
    MultiFab iterate = make_replicated_field();
    MultiFab rhs = make_replicated_field();
    fill_replicated_mean_zero_rhs(rhs);
    auto constant_mode = std::make_shared<MultiFab>(make_replicated_field());
    constant_mode->set_val(Real(1));
    const ApplyFn project_mean = [constant_mode, ownership](MultiFab& out, const MultiFab& in) {
      PureFieldAlgebra::copy(out, in);
      const Real mean = PureFieldAlgebra::dot(in, *constant_mode, ownership) / Real(8);
      PureFieldAlgebra::axpy(out, -mean, *constant_mode);
    };
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate, ownership);
    PreparedVectorMetric metric(iterate, ownership, ScaledEuclideanMetricSource{Real(3)});
    FieldNullspacePlan nullspace = constant_mean_zero_nullspace(
        "test://krylov/replicated-persistent-nullspace@1",
        "replicated constant mode for prepared Krylov", Real(1));
    PreparedAffineLinearProblem problem(
        iterate, project_mean, PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        PreparedNullspacePolicy::preserving(std::move(nullspace)), [&] { return snapshot; }, {},
        ownership, metric);
    KrylovWorkspace workspace(iterate, cg_krylov_method(), footprint, ownership, metric);
    problem.prepare(snapshot);
    workspace.bind(problem);
    const KrylovControls controls{cg_krylov_method(), Real(1e-12), Real(0), 4};
    for (const Real offset : {Real(3), Real(-7)}) {
      iterate.set_val(offset);
      const SolveReport report = solve_prepared_affine(problem, workspace, iterate, rhs, controls);
      require(report.solved());
      require(std::abs(static_cast<double>(problem.inner_product(iterate, *constant_mode))) <
              1e-12);
      MultiFab error = make_replicated_field();
      PureFieldAlgebra::lincomb(error, Real(1), iterate, Real(-1), rhs);
      require(PureFieldAlgebra::max_abs(error, ownership) < Real(1e-12));
    }
  }

  // Declaring a genuinely distributed layout as replicated is rejected while the typed vector
  // metric is bound to its exact vector space, before an operator callback or scientific reduction
  // can double-count it.
  {
    const PreparedVectorDistribution ownership = PreparedVectorDistribution::Replicated;
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype, ownership);
    require(uniformly_rejected(
        [&] {
          (void)PreparedAffineLinearProblem(
              prototype,
              [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
              PreparedLinearPreconditioner::identity(),
              LinearOperatorProperties::symmetric_positive_definite(), footprint,
              PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; }, {}, ownership);
        },
        "incoherent field layout"));
  }

  // A prepared single-vector problem consumes the explicitly selected absolute level. Extra
  // hierarchy metadata is provider-owned and must not be rejected through a closed scope enum.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    FieldNullspacePlan hierarchy = constant_mean_zero_nullspace(
        "test://krylov/selected-level@1", "provider-owned hierarchy mode");
    hierarchy.bases[0].cell_measure.push_back(Real(0.25));
    PreparedAffineLinearProblem problem(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        PreparedNullspacePolicy::preserving(std::move(hierarchy)), [&] { return snapshot; });
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
    MultiFab iterate = make_field();
    MultiFab rhs = make_field();
    rhs.set_val(Real(2));
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
    PreparedVectorMetric metric(iterate, PreparedVectorDistribution::Distributed,
                                ScaledEuclideanMetricSource{Real(3)});
    PreparedAffineLinearProblem problem(
        iterate, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; }, {},
        PreparedVectorDistribution::Distributed, metric);
    KrylovWorkspace workspace(iterate, cg_krylov_method(), footprint,
                              PreparedVectorDistribution::Distributed, metric);
    problem.prepare(snapshot);
    workspace.bind(problem);
    require(std::abs(static_cast<double>(problem.inner_product(rhs, rhs)) - 96.0) < 1e-12);
    require(std::abs(static_cast<double>(problem.residual_norm(rhs)) - std::sqrt(96.0)) < 1e-12);
    const SolveReport report = solve_prepared_affine(
        problem, workspace, iterate, rhs,
              KrylovControls{cg_krylov_method(), Real(1e-12), Real(0), 2});
    require(report.solved());

    PreparedVectorMetric different_metric(iterate, PreparedVectorDistribution::Distributed,
                                          ScaledEuclideanMetricSource{Real(4)});
    KrylovWorkspace different_workspace(iterate, cg_krylov_method(), footprint,
                                        PreparedVectorDistribution::Distributed, different_metric);
    require(
        uniformly_rejected([&] { different_workspace.bind(problem); }, "metric is incompatible"));
  }

  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    PreparedVectorMetric divergent_metric(
        prototype, PreparedVectorDistribution::Distributed,
        ScaledEuclideanMetricSource{rank == 0 ? Real(3) : Real(4)});
    PreparedAffineLinearProblem problem(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; }, {},
        PreparedVectorDistribution::Distributed, std::move(divergent_metric));
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
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    PreparedAffineLinearProblem problem(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        rank == 0 ? LinearOperatorProperties::general()
                  : LinearOperatorProperties::symmetric_positive_definite(),
        footprint, PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; });
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "problem contract differs"));
  }

  // Direct policy preparation is itself collective-safe. One rank cannot return through the
  // nonsingular branch while another enters a nullspace Gram reduction.
  {
    MultiFab layout = make_field();
    PreparedNullspacePolicy policy =
        rank == 0 ? PreparedNullspacePolicy::nonsingular()
                  : PreparedNullspacePolicy::preserving(constant_mean_zero_nullspace(
                        "test://rank-divergent-policy@1", "rank-divergent direct policy"));
    require(uniformly_rejected(
        [&] { policy.prepare(layout, PreparedVectorDistribution::Distributed); },
        "collective preflight rejected"));
  }

  // A callback may throw on one rank only after completing the same callback-owned MPI trace.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    PreparedAffineLinearProblem problem(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; },
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
    MultiFab method_iterate = make_field();
    MultiFab method_rhs = make_field();
    method_rhs.set_val(Real(1));
    const KrylovFootprint method_footprint{1, 0, false};
    const OperatorEvaluationSnapshot method_snapshot = snapshot_for(method_iterate);
    const PreparedKrylovMethod method(
        std::make_shared<const CollectiveBoundaryKrylovProvider>(behavior),
        PreparedProviderOptions{"pops.test.krylov.collective-probe.options@1", {}});
    PreparedAffineLinearProblem method_problem(
        method_iterate,
        [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), method_footprint,
        PreparedNullspacePolicy::nonsingular(), [&] { return method_snapshot; });
    KrylovWorkspace method_workspace(method_iterate, method, method_footprint);
    method_problem.prepare(method_snapshot);
    method_workspace.bind(method_problem);
    const SolveReport invalid = solve_prepared_affine(
        method_problem, method_workspace, method_iterate, method_rhs,
        KrylovControls{method, Real(1e-12), Real(0), 2});
    const std::string_view expected_reason =
        behavior == CollectiveBoundaryKrylovProvider::Behavior::kThrowOnRankZero
            ? "prepared Krylov provider failed after its collective solve trace"
            : "prepared Krylov provider report differs between communicator ranks";
    require(invalid.status == SolveStatus::kInvalidEvaluation);
    require(invalid.action == SolveAction::kFailRun);
    require(invalid.reason == expected_reason);
    require(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("external-krylov-invalid-report"),
          std::string_view(invalid.reason)}}));
  }

  // A successful nullspace certificate is reusable only after the fixed collective preflight.
  // A later rank-symmetric stage failure must leave the policy uniformly re-preparable: the retry
  // still performs freeze/preconditioner/A(0), but it does not rebuild the immutable nullspace
  // Gram certificate or its persistent metric-basis storage.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, true};
    OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    auto mutable_mask = std::make_shared<MultiFab>(make_field());
    mutable_mask->set_val(Real(1));
    FieldNullspacePlan nullspace = constant_mean_zero_nullspace(
        "test://collective-contract/nullspace-certificate@1", "constant test mode");
    nullspace.bases[0].masks = {mutable_mask};
    bool fail_preconditioner = true;
    long freeze_calls = 0;
    long preconditioner_prepare_calls = 0;
    long operator_calls = 0;
    PreparedLinearPreconditioner preconditioner = authenticated_preconditioner(
        prototype, "pops.test.krylov-collective.retryable-copy",
        [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); }, {},
        [&] {
          ++preconditioner_prepare_calls;
          (void)all_reduce_sum(1L);
          if (fail_preconditioner && rank == 0)
            throw std::runtime_error("retryable prepared preconditioner failure");
        });
    PreparedAffineLinearProblem problem(
        prototype,
        [&](MultiFab& out, const MultiFab& in) {
          ++operator_calls;
          PureFieldAlgebra::copy(out, in);
        },
        std::move(preconditioner),
        LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
        PreparedNullspacePolicy::preserving(std::move(nullspace)), [&] { return snapshot; },
        [&] {
          ++freeze_calls;
          (void)all_reduce_sum(1L);
        });

    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "preconditioner setup failed"));
    fail_preconditioner = false;
    problem.prepare(snapshot);
    require(problem.prepared());
    require(freeze_calls == 2);
    require(preconditioner_prepare_calls == 2);
    require(operator_calls == 1);

    mutable_mask->set_val(Real(0));
    snapshot.resources[0] += 1;
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "nullspace setup failed"));
    require(operator_calls == 1);
  }

  MultiFab iterate = make_field();
  MultiFab rhs = make_field();
  rhs.set_val(Real(1));
  const KrylovFootprint footprint{1, 0, false};
  const OperatorEvaluationSnapshot snapshot = snapshot_for(iterate);
  bool throw_after_collective = false;
  long operator_calls = 0;
  PreparedAffineLinearProblem problem(
      iterate,
      [&](MultiFab& out, const MultiFab& in) {
        ++operator_calls;
        (void)all_reduce_sum(1L);
        if (throw_after_collective && rank == 0)
          throw std::runtime_error("rank-local apply failure after collective");
        PureFieldAlgebra::copy(out, in);
      },
      PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; });
  KrylovWorkspace workspace(iterate, cg_krylov_method(), footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const KrylovControls valid{cg_krylov_method(), Real(1e-8), Real(0), 4};

  {
    MultiFab incompatible = make_field(rank == 0 ? 2 : 1);
    const MultiFab& local = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected([&] { (void)problem.inner_product(local, rhs); },
                               "incompatible vector space"));
  }
  {
    MultiFab output = make_field();
    MultiFab& local_output = rank == 0 ? rhs : output;
    const long calls_before = operator_calls;
    require(uniformly_rejected([&] { problem.apply_linear(local_output, rhs); },
                               "output aliases an input field"));
    require(operator_calls == calls_before);
  }
  {
    MultiFab incompatible = make_field(rank == 0 ? 2 : 1);
    const MultiFab& local = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected([&] { (void)problem.residual_norm(local); },
                               "incompatible vector space"));
  }

  {
    KrylovControls controls = valid;
    if (rank == 0)
      controls.max_iterations = 0;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, workspace, iterate, rhs, controls); },
        "collective contract differs"));
  }
  {
    KrylovControls controls = valid;
    if (rank == 0)
      controls.rel_tol = Real(1e-7);
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, workspace, iterate, rhs, controls); },
        "collective contract differs"));
  }
  {
    const MultiFab& local_rhs = rank == 0 ? iterate : rhs;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, workspace, iterate, local_rhs, valid); },
        "distinct storage"));
  }
  {
    MultiFab incompatible = make_field(rank == 0 ? 2 : 1);
    const MultiFab& local_rhs = rank == 0 ? incompatible : rhs;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, workspace, iterate, local_rhs, valid); },
        "incompatible vector space"));
  }
  {
    KrylovWorkspace unbound(iterate, cg_krylov_method(), footprint);
    KrylovWorkspace& local_workspace = rank == 0 ? unbound : workspace;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, local_workspace, iterate, rhs, valid); },
        "not bound"));
  }
  {
    require(uniformly_rejected(
        [&] {
          (void)KrylovWorkspace(
              iterate, rank == 0 ? cg_krylov_method() : richardson_krylov_method(), footprint);
        },
        "workspace requirements differ"));
  }
  {
    throw_after_collective = true;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, workspace, iterate, rhs, valid); },
        "operator callback failed"));
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
