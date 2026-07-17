#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/parallel/comm.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

OperatorEvaluationSnapshot snapshot_for(const MultiFab& field) {
  OperatorEvaluationSnapshot snapshot;
  snapshot.authority = {1, 2, 3, 4};
  snapshot.revision = 1;
  snapshot.macro_step = 0;
  snapshot.stage_numerator = 0;
  snapshot.stage_denominator = 1;
  snapshot.dt_bits = std::bit_cast<std::uint64_t>(0.1);
  snapshot.physical_time_bits = std::bit_cast<std::uint64_t>(0.0);
  snapshot.topology_revision = 1;
  snapshot.topology = detail::layout_fingerprint(field);
  snapshot.resources = {5, 6, 7, 8};
  return snapshot;
}

MultiFab make_field(int components = 1, int ghosts = 0) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {3, 1}}});
  MultiFab field(boxes, DistributionMapping(std::vector<int>{0, 1}), components, ghosts);
  field.set_val(Real(0));
  return field;
}

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
    const KrylovFootprint footprint{1, 0, 0, false};
    const OperatorEvaluationSnapshot snapshot = snapshot_for(prototype);
    PreparedAffineLinearProblem problem(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
        PreparedLinearPreconditioner::identity(),
        rank == 0 ? LinearOperatorProperties::general()
                  : LinearOperatorProperties::symmetric_positive_definite(),
        footprint, PreparedNullspacePolicy::nonsingular(), [&] { return snapshot; });
    require(uniformly_rejected([&] { problem.prepare(snapshot); }, "problem contract differs"));
  }

  // A callback may throw on one rank only after completing the same callback-owned MPI trace.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, 0, false};
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

  // A successful nullspace certificate is reusable only after the fixed collective preflight.
  // A later rank-symmetric stage failure must leave the policy uniformly re-preparable: the retry
  // still performs freeze/preconditioner/A(0), but it does not rebuild the immutable nullspace
  // Gram certificate or its persistent moment storage.
  {
    MultiFab prototype = make_field();
    const KrylovFootprint footprint{1, 0, 0, true};
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
    PreparedLinearPreconditioner preconditioner(
        prototype, [](MultiFab& out, const MultiFab& in) { PureFieldAlgebra::copy(out, in); },
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
  const KrylovFootprint footprint{1, 0, 0, false};
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
  KrylovWorkspace workspace(iterate, KrylovMethod::kCg, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const KrylovControls valid{KrylovMethod::kCg, Real(1e-8), Real(0), 4, 0, Real(1)};

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
        "max_iterations"));
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
    KrylovWorkspace unbound(iterate, KrylovMethod::kCg, footprint);
    KrylovWorkspace& local_workspace = rank == 0 ? unbound : workspace;
    require(uniformly_rejected(
        [&] { (void)solve_prepared_affine(problem, local_workspace, iterate, rhs, valid); },
        "not bound"));
  }
  {
    KrylovWorkspace divergent(iterate, rank == 0 ? KrylovMethod::kCg : KrylovMethod::kRichardson,
                              footprint);
    require(uniformly_rejected([&] { divergent.bind(problem); }, "bind contract differs"));
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
